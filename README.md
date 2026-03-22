# bandlet-tf3d

A GPU-native 3D directional tight-frame, bandlet-like transform for PyTorch.

`bandlet-tf3d` is a volumetric extension of a 2D bandlet-inspired API. It is designed as a practical, differentiable transform layer for volumetric data rather than a literal implementation of classical adaptive bandlets.

## Features

- multilevel 3D Haar backbone
- plane-based directional analysis inside cubic blocks
- orthonormal 2D DCT on oriented plane patches
- overlap-normalized synthesis with near-perfect reconstruction
- structured encoded representation or flat packed coefficients
- CPU/CUDA support with the same public API
- PyTorch `nn.Module` wrappers for analysis, synthesis, packed export, and denoising

## Installation

```bash
pip install -e .
```

## Quick start

```python
import torch
from bandlet_tf3d import Bandlet3DConfig, BandletTransform3D

cfg = Bandlet3DConfig(levels=2, block_size=4, device="cuda")
T = BandletTransform3D(cfg)

x = torch.randn(1, 1, 32, 32, 32, device=T.device, dtype=T.dtype)

enc = T.encode(x)
xrec = T.reconstruct(enc)

vec, meta = T.encode_packed(x)
xrec2 = T.decode_packed(vec, meta)

stats = T.stats(enc)
```

## Accepted input shapes

- `[D, H, W]`
- `[B, D, H, W]`
- `[B, C, D, H, W]`

All inputs are internally converted to `[B, C, D, H, W]`.

## Main API

- `Bandlet3DConfig`
- `BandletTransform3D`
- `Bandlet3DAnalysisLayer`
- `Bandlet3DSynthesisLayer`
- `Bandlet3DPackedLayer`
- `Bandlet3DDenoiseLayer`
- `PackedPlaneCoeffs3D`
- `EncodedSubband3D`
- `EncodedBandlet3D`

## Development

Run the test suite:

```bash
pytest -q
```

Run the CPU verification script:

```bash
python run_cpu_checks.py
```

## Notes

- This implementation is plane-based and fixed-geometry. It is not a geometric-flow or adaptive bandlet implementation.
- The default configuration is tuned for a dependable first release rather than maximal directional richness.
