
# Release notes

## v0.1.0

Initial public release of `bandlet-tf3d`.

### Included
- 3D Haar analysis/synthesis backbone
- plane-based directional encoding inside cubic blocks
- orthonormal 2D DCT plane transform
- packed coefficient export/import
- PyTorch layer wrappers
- CPU test suite and smoke scripts

### Verified
- CPU near-perfect reconstruction
- CUDA smoke tested externally on 16^3 and 32^3 volumes
