from __future__ import annotations
from typing import Any, Dict, List
import torch
from .types3d import EncodedBandlet3D, EncodedSubband3D, PackedPlaneCoeffs3D

_SUBBAND_ORDER = ("LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")

def pack_encoded_3d(enc: EncodedBandlet3D) -> torch.Tensor:
    parts = [enc.approx.reshape(enc.approx.shape[0], -1)]
    for septet in enc.detail_bands:
        for sub in septet:
            parts.append(sub.packed.coeffs.reshape(sub.packed.coeffs.shape[0], -1))
    return torch.cat(parts, dim=1)

def export_template_meta_3d(enc: EncodedBandlet3D) -> Dict[str, Any]:
    subs = []
    for septet in enc.detail_bands:
        lvl = []
        for sub in septet:
            lvl.append({
                "level": sub.level,
                "subband": sub.subband,
                "orig_shape": sub.orig_shape,
                "padded_shape": sub.padded_shape,
                "num_blocks_d": sub.num_blocks_d,
                "num_blocks_h": sub.num_blocks_h,
                "num_blocks_w": sub.num_blocks_w,
                "block_size": sub.block_size,
                "num_normals": sub.num_normals,
                "coeff_shape": tuple(sub.packed.coeffs.shape),
                "valid_mask": sub.packed.valid_mask.detach().cpu(),
                "coeff_mask": sub.packed.coeff_mask.detach().cpu(),
                "plane_area_mask": sub.packed.plane_area_mask.detach().cpu(),
                "plane_count": sub.packed.plane_count,
                "plane_h": sub.packed.plane_h,
                "plane_w": sub.packed.plane_w,
                "tight_scale": sub.packed.tight_scale,
            })
        subs.append(lvl)
    return {
        "approx_shape": tuple(enc.approx.shape),
        "detail_bands": subs,
        "meta": dict(enc.meta),
    }

def unpack_encoded_3d(vec: torch.Tensor, template_meta: Dict[str, Any], device=None, dtype=None) -> EncodedBandlet3D:
    if device is None:
        device = vec.device
    if dtype is None:
        dtype = vec.dtype
    B = vec.shape[0]
    offset = 0
    approx_numel = int(torch.tensor(template_meta["approx_shape"][1:]).prod().item())
    approx = vec[:, offset:offset + approx_numel].reshape((B, *template_meta["approx_shape"][1:])).to(device=device, dtype=dtype)
    offset += approx_numel
    detail_bands = []
    for septet_meta in template_meta["detail_bands"]:
        septet = []
        for sm in septet_meta:
            shape = sm["coeff_shape"]
            numel = 1
            for v in shape[1:]:
                numel *= v
            coeffs = vec[:, offset:offset + numel].reshape((B, *shape[1:])).to(device=device, dtype=dtype)
            offset += numel
            packed = PackedPlaneCoeffs3D(
                coeffs=coeffs,
                valid_mask=sm["valid_mask"].to(device=device),
                coeff_mask=sm["coeff_mask"].to(device=device),
                plane_area_mask=sm["plane_area_mask"].to(device=device),
                plane_count=sm["plane_count"],
                plane_h=sm["plane_h"],
                plane_w=sm["plane_w"],
                tight_scale=sm["tight_scale"],
            )
            septet.append(EncodedSubband3D(
                level=sm["level"],
                subband=sm["subband"],
                orig_shape=tuple(sm["orig_shape"]),
                padded_shape=tuple(sm["padded_shape"]),
                num_blocks_d=sm["num_blocks_d"],
                num_blocks_h=sm["num_blocks_h"],
                num_blocks_w=sm["num_blocks_w"],
                block_size=sm["block_size"],
                num_normals=sm["num_normals"],
                packed=packed,
            ))
        detail_bands.append(tuple(septet))
    return EncodedBandlet3D(approx=approx, detail_bands=detail_bands, meta=dict(template_meta["meta"]))
