import torch
from bandlet_tf3d import Bandlet3DConfig, BandletTransform3D
from bandlet_tf3d.directional_spec3d import build_packed_plane_spec_3d
from bandlet_tf3d.directional_ops3d import analyze_blocks_planewise, synthesize_blocks_planewise

def test_plane_spec_constant_reconstruction():
    spec = build_packed_plane_spec_3d(
        block_size=4,
        normals=Bandlet3DConfig().normals,
        plane_h=13,
        plane_w=13,
        device="cpu",
        dtype=torch.float32,
    )
    x = torch.ones((1, 1, 1, 4, 4, 4), dtype=torch.float32)
    packed = analyze_blocks_planewise(x, spec)
    rec = synthesize_blocks_planewise(packed, spec, block_size=4)
    err = (rec - x).abs().max().item()
    assert err < 1e-5

def test_end_to_end_pr():
    cfg = Bandlet3DConfig(levels=2, block_size=4, plane_h=13, plane_w=13, device="cpu", dtype="float32")
    T = BandletTransform3D(cfg)
    x = torch.randn(2, 1, 16, 16, 16)
    enc = T.encode(x)
    xrec = T.reconstruct(enc)
    rel = float((x - xrec).norm() / x.norm())
    assert rel < 1e-5

def test_pack_roundtrip():
    cfg = Bandlet3DConfig(levels=2, block_size=4, plane_h=13, plane_w=13, device="cpu", dtype="float32")
    T = BandletTransform3D(cfg)
    x = torch.randn(1, 1, 16, 16, 16)
    vec, meta = T.encode_packed(x)
    xrec = T.decode_packed(vec, meta)
    rel = float((x - xrec).norm() / x.norm())
    assert rel < 1e-5

def test_autograd_smoke():
    cfg = Bandlet3DConfig(levels=1, block_size=4, plane_h=13, plane_w=13, device="cpu", dtype="float32")
    T = BandletTransform3D(cfg)
    x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    y = T.denoise(x, tau=0.01)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None

def test_cuda_parity_if_available():
    if not torch.cuda.is_available():
        return
    cpu_cfg = Bandlet3DConfig(levels=1, block_size=4, plane_h=13, plane_w=13, device="cpu", dtype="float32")
    gpu_cfg = Bandlet3DConfig(levels=1, block_size=4, plane_h=13, plane_w=13, device="cuda", dtype="float32")
    Tc = BandletTransform3D(cpu_cfg)
    Tg = BandletTransform3D(gpu_cfg)
    x = torch.randn(1, 1, 8, 8, 8)
    yc = Tc.reconstruct(Tc.encode(x))
    yg = Tg.reconstruct(Tg.encode(x.cuda())).cpu()
    rel = float((yc - yg).norm() / yc.norm())
    assert rel < 1e-5
