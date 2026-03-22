
import json
from pathlib import Path

import torch

from bandlet_tf3d import Bandlet3DConfig, BandletTransform3D
from bandlet_tf3d.directional_ops3d import analyze_blocks_planewise, synthesize_blocks_planewise
from bandlet_tf3d.directional_spec3d import build_packed_plane_spec_3d


def main() -> None:
    results = {}

    spec = build_packed_plane_spec_3d(
        block_size=4,
        normals=Bandlet3DConfig().normals,
        plane_h=13,
        plane_w=13,
        device="cpu",
        dtype=torch.float32,
    )
    x1 = torch.ones((1, 1, 1, 4, 4, 4), dtype=torch.float32)
    packed = analyze_blocks_planewise(x1, spec)
    rec1 = synthesize_blocks_planewise(packed, spec, block_size=4)
    results["plane_spec_constant_reconstruction_max_abs_err"] = float((rec1 - x1).abs().max().item())

    cfg = Bandlet3DConfig(levels=2, block_size=4, plane_h=13, plane_w=13, device="cpu", dtype="float32")
    transform = BandletTransform3D(cfg)
    x = torch.randn(1, 1, 16, 16, 16)
    enc = transform.encode(x)
    xrec = transform.reconstruct(enc)
    results["end_to_end_pr_rel_err"] = float((x - xrec).norm() / (x.norm() + 1e-12))

    vec, meta = transform.encode_packed(x)
    xrec2 = transform.decode_packed(vec, meta)
    results["packed_roundtrip_rel_err"] = float((x - xrec2).norm() / (x.norm() + 1e-12))

    xg = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    yg = transform.denoise(xg, tau=0.01)
    loss = yg.square().mean()
    loss.backward()
    results["autograd_smoke_passed"] = bool(xg.grad is not None)

    print(json.dumps(results, indent=2))
    Path("cpu_results.json").write_text(json.dumps(results, indent=2) + "
")


if __name__ == "__main__":
    main()
