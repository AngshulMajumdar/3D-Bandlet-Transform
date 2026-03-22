from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple
import torch

@dataclass
class PackedPlaneCoeffs3D:
    coeffs: torch.Tensor                  # [B, C, N, K, P, Hf, Wf]
    valid_mask: torch.Tensor             # [K, P, Hf, Wf]
    coeff_mask: torch.Tensor             # [K, P, Hf, Wf]
    plane_area_mask: torch.Tensor        # [K, P, Hf, Wf]
    plane_count: int
    plane_h: int
    plane_w: int
    tight_scale: float

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.coeffs.shape)

    @property
    def device(self) -> torch.device:
        return self.coeffs.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs.dtype

    def clone(self) -> "PackedPlaneCoeffs3D":
        return PackedPlaneCoeffs3D(
            coeffs=self.coeffs.clone(),
            valid_mask=self.valid_mask.clone(),
            coeff_mask=self.coeff_mask.clone(),
            plane_area_mask=self.plane_area_mask.clone(),
            plane_count=self.plane_count,
            plane_h=self.plane_h,
            plane_w=self.plane_w,
            tight_scale=self.tight_scale,
        )

    def to(self, device=None, dtype=None) -> "PackedPlaneCoeffs3D":
        return PackedPlaneCoeffs3D(
            coeffs=self.coeffs.to(device=device, dtype=dtype),
            valid_mask=self.valid_mask.to(device=device),
            coeff_mask=self.coeff_mask.to(device=device),
            plane_area_mask=self.plane_area_mask.to(device=device),
            plane_count=self.plane_count,
            plane_h=self.plane_h,
            plane_w=self.plane_w,
            tight_scale=self.tight_scale,
        )

@dataclass
class EncodedSubband3D:
    level: int
    subband: str
    orig_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    num_blocks_d: int
    num_blocks_h: int
    num_blocks_w: int
    block_size: int
    num_normals: int
    packed: PackedPlaneCoeffs3D

    @property
    def coeff_shape(self) -> Tuple[int, ...]:
        return tuple(self.packed.coeffs.shape)

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.packed.dtype

    def clone(self) -> "EncodedSubband3D":
        return replace(self, packed=self.packed.clone())

    def to(self, device=None, dtype=None) -> "EncodedSubband3D":
        return replace(self, packed=self.packed.to(device=device, dtype=dtype))

@dataclass
class EncodedBandlet3D:
    approx: torch.Tensor
    detail_bands: List[
        Tuple[
            EncodedSubband3D, EncodedSubband3D, EncodedSubband3D,
            EncodedSubband3D, EncodedSubband3D, EncodedSubband3D,
            EncodedSubband3D,
        ]
    ]
    meta: Dict[str, Any]

    @property
    def device(self) -> torch.device:
        return self.approx.device

    @property
    def dtype(self) -> torch.dtype:
        return self.approx.dtype

    def clone(self) -> "EncodedBandlet3D":
        return EncodedBandlet3D(
            approx=self.approx.clone(),
            detail_bands=[tuple(sub.clone() for sub in septet) for septet in self.detail_bands],
            meta=dict(self.meta),
        )

    def to(self, device=None, dtype=None) -> "EncodedBandlet3D":
        return EncodedBandlet3D(
            approx=self.approx.to(device=device, dtype=dtype),
            detail_bands=[tuple(sub.to(device=device, dtype=dtype) for sub in septet) for septet in self.detail_bands],
            meta=dict(self.meta),
        )
