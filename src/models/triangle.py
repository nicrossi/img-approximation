from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]           # [0,1]
RGBA  = Tuple[int, int, int, int]     # 0..255

@dataclass(slots=True)
class Triangle:
    p1: Point; p2: Point; p3: Point
    color: RGBA
