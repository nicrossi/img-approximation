from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.models.triangle import Triangle


@dataclass(slots=True)
class Individual:
    triangles: List[Triangle]  # índice = z-order
