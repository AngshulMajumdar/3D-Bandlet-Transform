from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple
import torch

@dataclass
class PackedPlaneSpec3D:
    block_size: int
    num_normals: int
    plane_count: int
    plane_h: int
    plane_w: int
    gather_idx: torch.Tensor          # [K, P, Hf, Wf]
    scatter_idx: torch.Tensor         # [K, P, Hf, Wf]
    valid_mask: torch.Tensor          # [K, P, Hf, Wf]
    coeff_mask: torch.Tensor          # [K, P, Hf, Wf]
    plane_area_mask: torch.Tensor     # [K, P, Hf, Wf]
    voxel_norm: torch.Tensor          # [b^3]
    tight_scale: float
    normal_vectors: torch.Tensor      # [K,3]
    plane_offsets: torch.Tensor       # [K,P]
    plane_lengths: torch.Tensor       # [K,P]

    @property
    def device(self) -> torch.device:
        return self.gather_idx.device

def _gcd3(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))

def _normalize_normal(n: Tuple[int, int, int]) -> Tuple[int, int, int]:
    a, b, c = n
    g = _gcd3(a, b, c)
    if g == 0:
        raise ValueError("Zero normal is invalid.")
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _cross_int(u: Tuple[int, int, int], v: Tuple[int, int, int]) -> Tuple[int, int, int]:
    ux, uy, uz = u
    vx, vy, vz = v
    return (uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

def _dot_int(u: Tuple[int, int, int], v: Tuple[int, int, int]) -> int:
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def _find_integer_plane_basis(n: Tuple[int, int, int]):
    candidates = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
    seed = None
    for s in candidates:
        cr = _cross_int(n, s)
        if cr != (0,0,0):
            seed = s
            break
    if seed is None:
        raise RuntimeError(f"Could not find seed for normal {n}")
    u = _cross_int(n, seed)
    gu = _gcd3(*u)
    u = (u[0]//gu, u[1]//gu, u[2]//gu)
    v = _cross_int(n, u)
    gv = _gcd3(*v)
    v = (v[0]//gv, v[1]//gv, v[2]//gv)
    return u, v

def _flatten_zyx(z: int, y: int, x: int, b: int) -> int:
    return z * b * b + y * b + x

def _enumerate_cube_coords(b: int):
    return [(z, y, x) for z in range(b) for y in range(b) for x in range(b)]

def _enumerate_planes_for_normal(block_size: int, normal: Tuple[int,int,int], plane_h: int, plane_w: int):
    b = block_size
    n = _normalize_normal(normal)
    u, v = _find_integer_plane_basis(n)
    coords = _enumerate_cube_coords(b)
    plane_dict: Dict[int, List[Tuple[int,int,int]]] = {}
    for r in coords:
        t = _dot_int(n, r)
        plane_dict.setdefault(t, []).append(r)

    planes_gather, planes_valid, plane_offsets, plane_lengths = [], [], [], []
    for t in sorted(plane_dict.keys()):
        voxels = plane_dict[t]
        ab_list = [(_dot_int(u, r), _dot_int(v, r), r) for r in voxels]
        # stable ordering
        ab_list = sorted(ab_list, key=lambda x: (x[0], x[1], x[2]))
        a_vals = [a for a,_,_ in ab_list]
        b_vals = [bb for _,bb,_ in ab_list]
        a_min, a_max = min(a_vals), max(a_vals)
        b_min, b_max = min(b_vals), max(b_vals)
        ph = a_max - a_min + 1
        pw = b_max - b_min + 1
        if ph > plane_h or pw > plane_w:
            raise ValueError(f"Plane too large for patch for normal={n}, offset={t}, need=({ph},{pw}), have=({plane_h},{plane_w})")
        g = torch.zeros((plane_h, plane_w), dtype=torch.long)
        m = torch.zeros((plane_h, plane_w), dtype=torch.bool)
        a_shift = (plane_h - ph) // 2 - a_min
        b_shift = (plane_w - pw) // 2 - b_min
        for a, bb, r in ab_list:
            i, j = a + a_shift, bb + b_shift
            z, y, x = r
            g[i, j] = _flatten_zyx(z, y, x, b)
            m[i, j] = True
        planes_gather.append(g)
        planes_valid.append(m)
        plane_offsets.append(t)
        plane_lengths.append(len(voxels))
    return planes_gather, planes_valid, plane_offsets, plane_lengths

_SPEC_CACHE = {}

def build_packed_plane_spec_3d(
    block_size: int,
    normals: Sequence[Tuple[int, int, int]],
    plane_h: int,
    plane_w: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PackedPlaneSpec3D:
    key = (block_size, tuple(tuple(n) for n in normals), plane_h, plane_w, str(device), str(dtype))
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    normals = tuple(_normalize_normal(n) for n in normals)
    K = len(normals)
    V = block_size ** 3

    gather_per_k, valid_per_k, offsets_per_k, lengths_per_k = [], [], [], []
    max_planes = 0
    for n in normals:
        g_list, m_list, t_list, l_list = _enumerate_planes_for_normal(block_size, n, plane_h, plane_w)
        gather_per_k.append(g_list)
        valid_per_k.append(m_list)
        offsets_per_k.append(t_list)
        lengths_per_k.append(l_list)
        max_planes = max(max_planes, len(g_list))

    P = max_planes
    Hf, Wf = plane_h, plane_w
    gather_idx = torch.zeros((K, P, Hf, Wf), dtype=torch.long)
    scatter_idx = torch.zeros((K, P, Hf, Wf), dtype=torch.long)
    valid_mask = torch.zeros((K, P, Hf, Wf), dtype=torch.bool)
    coeff_mask = torch.zeros((K, P, Hf, Wf), dtype=torch.bool)
    plane_area_mask = torch.zeros((K, P, Hf, Wf), dtype=torch.bool)
    plane_offsets = torch.zeros((K, P), dtype=torch.long)
    plane_lengths = torch.zeros((K, P), dtype=torch.long)

    for k in range(K):
        for p in range(len(gather_per_k[k])):
            gather_idx[k, p] = gather_per_k[k][p]
            scatter_idx[k, p] = gather_per_k[k][p]
            valid_mask[k, p] = valid_per_k[k][p]
            coeff_mask[k, p] = valid_per_k[k][p]
            plane_area_mask[k, p] = valid_per_k[k][p]
            plane_offsets[k, p] = offsets_per_k[k][p]
            plane_lengths[k, p] = lengths_per_k[k][p]

    voxel_norm = torch.zeros((V,), dtype=dtype)
    for k in range(K):
        for p in range(P):
            idx = gather_idx[k, p][valid_mask[k, p]]
            if idx.numel() > 0:
                voxel_norm.index_add_(0, idx, torch.ones_like(idx, dtype=dtype))
    voxel_norm = torch.clamp(voxel_norm, min=torch.tensor(1.0, dtype=dtype))
    tight_scale = float(1.0 / math.sqrt(K))
    spec = PackedPlaneSpec3D(
        block_size=block_size,
        num_normals=K,
        plane_count=P,
        plane_h=Hf,
        plane_w=Wf,
        gather_idx=gather_idx.to(device),
        scatter_idx=scatter_idx.to(device),
        valid_mask=valid_mask.to(device),
        coeff_mask=coeff_mask.to(device),
        plane_area_mask=plane_area_mask.to(device),
        voxel_norm=voxel_norm.to(device),
        tight_scale=tight_scale,
        normal_vectors=torch.tensor(normals, dtype=torch.long, device=device),
        plane_offsets=plane_offsets.to(device),
        plane_lengths=plane_lengths.to(device),
    )
    _SPEC_CACHE[key] = spec
    return spec
