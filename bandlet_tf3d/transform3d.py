from __future__ import annotations
import copy
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from .blocks3d import assemble_blocks_3d, crop_to_shape_3d, extract_blocks_3d, pad_to_multiple_3d
from .config3d import Bandlet3DConfig
from .directional_ops3d import analyze_blocks_planewise, soft_threshold_packed_3d, synthesize_blocks_planewise
from .directional_spec3d import build_packed_plane_spec_3d, PackedPlaneSpec3D
from .haar3d import HaarLevel3D, dwt3_haar, idwt3_haar
from .packing3d import export_template_meta_3d, pack_encoded_3d, unpack_encoded_3d
from .types3d import EncodedBandlet3D, EncodedSubband3D

_SUBBAND_NAMES = ("LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")

class BandletTransform3D(nn.Module):
    def __init__(self, config: Bandlet3DConfig | None = None):
        super().__init__()
        self.config = config or Bandlet3DConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = getattr(torch, self.config.dtype)
        self._spec_cache = {}

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _to_tensor(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not x.is_floating_point():
            x = x.float()
        x = x.to(device=self.device, dtype=self.dtype)
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim == 5:
            pass
        else:
            raise ValueError("Expected input shape [D,H,W], [B,D,H,W], or [B,C,D,H,W]")
        return x.contiguous()

    def _pad_volume_for_haar(self, x: torch.Tensor):
        mult = 2 ** self.config.levels
        return pad_to_multiple_3d(x, mult, mult, mult, mode=self.config.pad_mode_image)

    def _get_plane_spec(self, block_size: int, device, dtype) -> PackedPlaneSpec3D:
        key = (block_size, tuple(self.config.normals), self.config.plane_h, self.config.plane_w, str(device), str(dtype))
        if key not in self._spec_cache:
            self._spec_cache[key] = build_packed_plane_spec_3d(
                block_size=block_size,
                normals=self.config.normals,
                plane_h=self.config.plane_h,
                plane_w=self.config.plane_w,
                device=device,
                dtype=dtype,
            )
        return self._spec_cache[key]

    def _encode_subband(self, band: torch.Tensor, level: int, subband_name: str) -> EncodedSubband3D:
        blocks, orig_shape, padded_shape, nd, nh, nw = extract_blocks_3d(band, self.config.block_size, pad_mode=self.config.pad_mode_block)
        spec = self._get_plane_spec(self.config.block_size, blocks.device, blocks.dtype)
        packed = analyze_blocks_planewise(blocks, spec)
        return EncodedSubband3D(
            level=level,
            subband=subband_name,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
            num_blocks_d=nd,
            num_blocks_h=nh,
            num_blocks_w=nw,
            block_size=self.config.block_size,
            num_normals=len(self.config.normals),
            packed=packed,
        )

    def _decode_subband(self, encoded: EncodedSubband3D) -> torch.Tensor:
        spec = self._get_plane_spec(encoded.block_size, encoded.device, encoded.dtype)
        rec_blocks = synthesize_blocks_planewise(encoded.packed, spec, encoded.block_size)
        band_pad = assemble_blocks_3d(rec_blocks, encoded.padded_shape, (encoded.num_blocks_d, encoded.num_blocks_h, encoded.num_blocks_w), encoded.block_size)
        return crop_to_shape_3d(band_pad, encoded.orig_shape)

    def encode(self, x) -> EncodedBandlet3D:
        x = self._to_tensor(x)
        x_pad, orig_shape = self._pad_volume_for_haar(x)
        approx, coeffs = dwt3_haar(x_pad, self.config.levels)
        detail_bands = []
        for level_idx, lvl in enumerate(coeffs, start=1):
            bands = (lvl.llh, lvl.lhl, lvl.lhh, lvl.hll, lvl.hlh, lvl.hhl, lvl.hhh)
            septet = tuple(self._encode_subband(b, level_idx, name) for b, name in zip(bands, _SUBBAND_NAMES))
            detail_bands.append(septet)
        meta = {
            "orig_volume_shape": orig_shape,
            "padded_volume_shape": tuple(x_pad.shape[-3:]),
            "input_shape_5d": tuple(x.shape),
            "levels": self.config.levels,
            "block_size": self.config.block_size,
            "normals": tuple(self.config.normals),
            "plane_h": self.config.plane_h,
            "plane_w": self.config.plane_w,
        }
        return EncodedBandlet3D(approx=approx, detail_bands=detail_bands, meta=meta)

    def reconstruct(self, enc: EncodedBandlet3D) -> torch.Tensor:
        coeffs = []
        for septet in enc.detail_bands:
            recs = [self._decode_subband(s) for s in septet]
            coeffs.append(HaarLevel3D(
                llh=recs[0], lhl=recs[1], lhh=recs[2],
                hll=recs[3], hlh=recs[4], hhl=recs[5], hhh=recs[6],
            ))
        x_pad = idwt3_haar(enc.approx, coeffs)
        x = crop_to_shape_3d(x_pad, tuple(enc.meta["orig_volume_shape"]))
        return x

    def threshold(self, enc: EncodedBandlet3D, tau: float) -> EncodedBandlet3D:
        out = enc.clone()
        refreshed = []
        for septet in out.detail_bands:
            new_septet = []
            for sub in septet:
                packed = soft_threshold_packed_3d(sub.packed, tau=tau, keep_dc=self.config.keep_dc_on_threshold)
                new_septet.append(EncodedSubband3D(
                    level=sub.level,
                    subband=sub.subband,
                    orig_shape=sub.orig_shape,
                    padded_shape=sub.padded_shape,
                    num_blocks_d=sub.num_blocks_d,
                    num_blocks_h=sub.num_blocks_h,
                    num_blocks_w=sub.num_blocks_w,
                    block_size=sub.block_size,
                    num_normals=sub.num_normals,
                    packed=packed,
                ))
            refreshed.append(tuple(new_septet))
        out.detail_bands = refreshed
        return out

    def denoise(self, x, tau: float) -> torch.Tensor:
        return self.reconstruct(self.threshold(self.encode(x), tau=tau))

    def stats(self, enc: EncodedBandlet3D) -> Dict[str, Any]:
        detail_energy = 0.0
        detail_coeffs = 0
        for septet in enc.detail_bands:
            for sub in septet:
                c = sub.packed.coeffs
                detail_energy += float((c**2).sum().item())
                detail_coeffs += c.numel()
        approx_energy = float((enc.approx**2).sum().item())
        return {
            "approx_shape": tuple(enc.approx.shape),
            "detail_coeffs": detail_coeffs,
            "approx_energy": approx_energy,
            "detail_energy": detail_energy,
            "total_energy": approx_energy + detail_energy,
        }

    def pack(self, enc: EncodedBandlet3D) -> torch.Tensor:
        return pack_encoded_3d(enc)

    def export_template_meta(self, enc: EncodedBandlet3D) -> Dict[str, Any]:
        return export_template_meta_3d(enc)

    def unpack(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> EncodedBandlet3D:
        return unpack_encoded_3d(vec, template_meta, device=vec.device, dtype=vec.dtype)

    def encode_packed(self, x):
        enc = self.encode(x)
        return self.pack(enc), self.export_template_meta(enc)

    def decode_packed(self, vec: torch.Tensor, template_meta: Dict[str, Any]) -> torch.Tensor:
        return self.reconstruct(self.unpack(vec, template_meta))
