from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.models.triangle import Triangle


@dataclass(slots=True)
class Individual:
    triangles: List[Triangle]  # índice = z-order

    @staticmethod
    def individual_to_dict(ind: Individual) -> dict:
        return {"triangles": [Triangle.triangle_to_dict(t) for t in ind.triangles]}