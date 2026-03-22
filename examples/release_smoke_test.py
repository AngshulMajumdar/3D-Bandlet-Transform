
import torch
from bandlet_tf3d import Bandlet3DConfig, BandletTransform3D


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Bandlet3DConfig(levels=2, block_size=4, plane_h=13, plane_w=13, device=device)
    transform = BandletTransform3D(cfg)

    x = torch.randn(1, 1, 32, 32, 32, device=transform.device, dtype=transform.dtype)
    enc = transform.encode(x)
    xrec = transform.reconstruct(enc)
    vec, meta = transform.encode_packed(x)
    xrec2 = transform.decode_packed(vec, meta)

    pr = float((x - xrec).norm() / (x.norm() + 1e-12))
    pr_packed = float((x - xrec2).norm() / (x.norm() + 1e-12))

    print({
        "device": str(transform.device),
        "pr_rel_err": pr,
        "packed_pr_rel_err": pr_packed,
        "stats": transform.stats(enc),
    })


if __name__ == "__main__":
    main()
