from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]           # [0,1]
RGBA  = Tuple[int, int, int, int]     # 0..255

@dataclass(slots=True)
class Triangle:
    p1: Point; p2: Point; p3: Point
    color: RGBA
    z_index: float = 0

    @staticmethod
    def triangle_to_dict(t: Triangle) -> dict:
        return {
            "p1": [t.p1[0], t.p1[1]],
            "p2": [t.p2[0], t.p2[1]],
            "p3": [t.p3[0], t.p3[1]],
            "color": [t.color[0], t.color[1], t.color[2], t.color[3]],
            "z_index": t.z_index
        }

    @staticmethod
    def clone(t: Triangle) -> Triangle:
        """Create a value-equal copy of the triangle (no shared object reference)."""
        return Triangle(t.p1, t.p2, t.p3, t.color, t.z_index)
