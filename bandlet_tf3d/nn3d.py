from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from .config3d import Bandlet3DConfig
from .transform3d import BandletTransform3D
from .types3d import EncodedBandlet3D

class Bandlet3DAnalysisLayer(nn.Module):
    def __init__(self, config: Bandlet3DConfig | None = None):
        super().__init__()
        self.transform = BandletTransform3D(config)
    def forward(self, x) -> EncodedBandlet3D:
        return self.transform.encode(x)

class Bandlet3DSynthesisLayer(nn.Module):
    def __init__(self, config: Bandlet3DConfig | None = None):
        super().__init__()
        self.transform = BandletTransform3D(config)
    def forward(self, enc: EncodedBandlet3D) -> torch.Tensor:
        return self.transform.reconstruct(enc)

class Bandlet3DPackedLayer(nn.Module):
    def __init__(self, config: Bandlet3DConfig | None = None):
        super().__init__()
        self.transform = BandletTransform3D(config)
    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.transform.encode_packed(x)

class Bandlet3DDenoiseLayer(nn.Module):
    def __init__(self, tau: float, config: Bandlet3DConfig | None = None):
        super().__init__()
        self.transform = BandletTransform3D(config)
        self.tau = float(tau)
    def forward(self, x) -> torch.Tensor:
        return self.transform.denoise(x, tau=self.tau)
