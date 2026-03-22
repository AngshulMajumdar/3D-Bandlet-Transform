from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

def _pad_amount(n: int, mult: int) -> int:
    return (mult - (n % mult)) % mult

def pad_to_multiple_3d(
    x: torch.Tensor,
    mult_d: int,
    mult_h: int,
    mult_w: int,
    mode: str = "replicate",
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    assert x.ndim == 5
    D, H, W = x.shape[-3:]
    pd = _pad_amount(D, mult_d)
    ph = _pad_amount(H, mult_h)
    pw = _pad_amount(W, mult_w)
    if pd == 0 and ph == 0 and pw == 0:
        return x, (D, H, W)
    xpad = F.pad(x, (0, pw, 0, ph, 0, pd), mode=mode)
    return xpad, (D, H, W)

def crop_to_shape_3d(x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    D, H, W = shape
    return x[..., :D, :H, :W]

def extract_blocks_3d(
    x: torch.Tensor,
    block_size: int,
    pad_mode: str = "replicate",
):
    assert x.ndim == 5
    x_pad, orig_shape = pad_to_multiple_3d(x, block_size, block_size, block_size, mode=pad_mode)
    B, C, Dp, Hp, Wp = x_pad.shape
    b = block_size
    nd, nh, nw = Dp // b, Hp // b, Wp // b
    blocks = x_pad.reshape(B, C, nd, b, nh, b, nw, b).permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
    blocks = blocks.reshape(B, C, nd * nh * nw, b, b, b)
    return blocks, orig_shape, (Dp, Hp, Wp), nd, nh, nw

def assemble_blocks_3d(
    blocks: torch.Tensor,
    padded_shape: Tuple[int, int, int],
    num_blocks_dhw: Tuple[int, int, int],
    block_size: int,
) -> torch.Tensor:
    B, C, N, b1, b2, b3 = blocks.shape
    assert b1 == b2 == b3 == block_size
    nd, nh, nw = num_blocks_dhw
    Dp, Hp, Wp = padded_shape
    out = blocks.reshape(B, C, nd, nh, nw, block_size, block_size, block_size).permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    out = out.reshape(B, C, Dp, Hp, Wp)
    return out
