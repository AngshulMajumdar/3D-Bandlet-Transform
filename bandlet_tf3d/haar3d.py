from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import torch
import torch.nn.functional as F

@dataclass
class HaarLevel3D:
    llh: torch.Tensor
    lhl: torch.Tensor
    lhh: torch.Tensor
    hll: torch.Tensor
    hlh: torch.Tensor
    hhl: torch.Tensor
    hhh: torch.Tensor

_SUBBANDS = ("lll","llh","lhl","lhh","hll","hlh","hhl","hhh")

def _haar_filters_3d(device, dtype):
    h0 = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], device=device, dtype=dtype)
    h1 = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], device=device, dtype=dtype)
    bank = []
    for a in (h0, h1):
        for b in (h0, h1):
            for c in (h0, h1):
                filt = a[:, None, None] * b[None, :, None] * c[None, None, :]
                bank.append(filt)
    return torch.stack(bank, dim=0)  # [8,2,2,2]

def _analysis_once(x: torch.Tensor):
    B, C, D, H, W = x.shape
    bank = _haar_filters_3d(x.device, x.dtype).unsqueeze(1)  # [8,1,2,2,2]
    weight = bank.repeat(C, 1, 1, 1, 1)  # [8C,1,2,2,2]
    y = F.conv3d(x, weight, stride=2, groups=C)  # [B,8C,D/2,H/2,W/2]
    y = y.reshape(B, C, 8, y.shape[-3], y.shape[-2], y.shape[-1])
    return {name: y[:, :, i] for i, name in enumerate(_SUBBANDS)}

def _synthesis_once(lll: torch.Tensor, level: HaarLevel3D):
    B, C, D, H, W = lll.shape
    comps = [lll, level.llh, level.lhl, level.lhh, level.hll, level.hlh, level.hhl, level.hhh]
    y = torch.stack(comps, dim=2).reshape(B, 8 * C, D, H, W)
    bank = _haar_filters_3d(lll.device, lll.dtype).unsqueeze(1)
    weight = bank.repeat(C, 1, 1, 1, 1)
    x = F.conv_transpose3d(y, weight, stride=2, groups=C)
    return x

def dwt3_haar(x: torch.Tensor, levels: int) -> Tuple[torch.Tensor, List[HaarLevel3D]]:
    cur = x
    coeffs: List[HaarLevel3D] = []
    for _ in range(levels):
        bands = _analysis_once(cur)
        coeffs.append(HaarLevel3D(
            llh=bands["llh"], lhl=bands["lhl"], lhh=bands["lhh"],
            hll=bands["hll"], hlh=bands["hlh"], hhl=bands["hhl"], hhh=bands["hhh"],
        ))
        cur = bands["lll"]
    return cur, coeffs

def idwt3_haar(approx: torch.Tensor, coeffs: List[HaarLevel3D]) -> torch.Tensor:
    cur = approx
    for level in reversed(coeffs):
        cur = _synthesis_once(cur, level)
    return cur
