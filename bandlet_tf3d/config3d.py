
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

DEFAULT_NORMALS_13: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)

@dataclass(frozen=True)
class Bandlet3DConfig:
    """Configuration for the 3D bandlet-like transform.

    Parameters are intentionally conservative for a dependable first release.
    `plane_h` and `plane_w` default to 13 because that safely accommodates the
    embedded plane patches induced by the default 13-direction normal bank for
    `block_size=4`.
    """

    levels: int = 2
    block_size: int = 4
    normals: Tuple[Tuple[int, int, int], ...] = DEFAULT_NORMALS_13
    plane_transform: Literal["dct2"] = "dct2"
    plane_h: int = 13
    plane_w: int = 13
    device: str = "auto"
    dtype: str = "float32"
    pad_mode_image: str = "replicate"
    pad_mode_block: str = "replicate"
    keep_dc_on_threshold: bool = True
