import torch
from bandlet_tf3d import Bandlet3DConfig, BandletTransform3D

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Bandlet3DConfig(levels=2, block_size=4, plane_h=13, plane_w=13, device=device)
T = BandletTransform3D(cfg)

x = torch.randn(1, 1, 16, 16, 16, device=T.device, dtype=T.dtype)
enc = T.encode(x)
xrec = T.reconstruct(enc)
vec, meta = T.encode_packed(x)
xrec2 = T.decode_packed(vec, meta)

print("device:", T.device)
print("pr_rel_err:", float((x - xrec).norm() / x.norm()))
print("packed_pr_rel_err:", float((x - xrec2).norm() / x.norm()))
print("stats:", T.stats(enc))
