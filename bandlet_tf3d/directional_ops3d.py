from __future__ import annotations
import functools
import torch
from .directional_spec3d import PackedPlaneSpec3D
from .types3d import PackedPlaneCoeffs3D

@functools.lru_cache(maxsize=64)
def _cached_dct_mats(n: int, device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    mat = torch.cos(torch.pi * (i + 0.5) * k / n)
    mat[0] *= (1.0 / n) ** 0.5
    if n > 1:
        mat[1:] *= (2.0 / n) ** 0.5
    return mat

def _dct_matrix(n: int, device, dtype):
    return _cached_dct_mats(n, str(device), str(dtype).split(".")[-1])

def _apply_dct2(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = torch.einsum("ab,...bc->...ac", Dh, x)
    y = torch.einsum("...ab,bc->...ac", y, Dw.t())
    return y

def _apply_idct2(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = torch.einsum("ab,...bc->...ac", Dh.t(), x)
    y = torch.einsum("...ab,bc->...ac", y, Dw)
    return y

def analyze_blocks_planewise(blocks: torch.Tensor, spec: PackedPlaneSpec3D) -> PackedPlaneCoeffs3D:
    B, C, N, b1, b2, b3 = blocks.shape
    assert b1 == b2 == b3 == spec.block_size
    V = spec.block_size ** 3
    K, P, Hf, Wf = spec.gather_idx.shape
    blocks_flat = blocks.reshape(B, C, N, V)
    idx = spec.gather_idx.reshape(1, 1, 1, K * P * Hf * Wf).expand(B, C, N, -1)
    gathered = torch.gather(blocks_flat, dim=-1, index=idx)
    patches = gathered.reshape(B, C, N, K, P, Hf, Wf)
    patches = patches * spec.valid_mask.view(1, 1, 1, K, P, Hf, Wf).to(blocks.dtype)
    coeffs = _apply_dct2(patches) * spec.tight_scale
    return PackedPlaneCoeffs3D(
        coeffs=coeffs,
        valid_mask=spec.valid_mask,
        coeff_mask=spec.coeff_mask,
        plane_area_mask=spec.plane_area_mask,
        plane_count=spec.plane_count,
        plane_h=spec.plane_h,
        plane_w=spec.plane_w,
        tight_scale=spec.tight_scale,
    )

def synthesize_blocks_planewise(packed: PackedPlaneCoeffs3D, spec: PackedPlaneSpec3D, block_size: int) -> torch.Tensor:
    coeffs = packed.coeffs
    B, C, N, K, P, Hf, Wf = coeffs.shape
    V = block_size ** 3
    patches = _apply_idct2(coeffs / packed.tight_scale)
    patches = patches * spec.valid_mask.view(1, 1, 1, K, P, Hf, Wf).to(coeffs.dtype)
    patches_flat = patches.reshape(B, C, N, K * P * Hf * Wf)
    idx = spec.scatter_idx.reshape(1, 1, 1, K * P * Hf * Wf).expand(B, C, N, -1)
    valmask = spec.valid_mask.reshape(1, 1, 1, K * P * Hf * Wf).to(coeffs.dtype)
    accum = torch.zeros((B, C, N, V), device=coeffs.device, dtype=coeffs.dtype)
    accum.scatter_add_(dim=-1, index=idx, src=patches_flat * valmask)
    rec_flat = accum / spec.voxel_norm.view(1, 1, 1, V)
    return rec_flat.reshape(B, C, N, block_size, block_size, block_size)

def soft_threshold_packed_3d(packed: PackedPlaneCoeffs3D, tau: float, keep_dc: bool = True) -> PackedPlaneCoeffs3D:
    coeffs = packed.coeffs.clone()
    if tau > 0:
        coeffs = torch.sign(coeffs) * torch.clamp(coeffs.abs() - tau, min=0.0)
    if keep_dc:
        coeffs[..., 0, 0] = packed.coeffs[..., 0, 0]
    coeffs = coeffs * packed.coeff_mask.view(1, 1, 1, *packed.coeff_mask.shape).to(coeffs.dtype)
    return PackedPlaneCoeffs3D(
        coeffs=coeffs,
        valid_mask=packed.valid_mask,
        coeff_mask=packed.coeff_mask,
        plane_area_mask=packed.plane_area_mask,
        plane_count=packed.plane_count,
        plane_h=packed.plane_h,
        plane_w=packed.plane_w,
        tight_scale=packed.tight_scale,
    )
