from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from src.models.triangle import Triangle
from src.models.individual import Individual
from src.strategies.mutation.MutationStrategy import MutationStrategy


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _clamp255(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


@dataclass
class GenMutation(MutationStrategy):
    """Gaussian jitter mutation for fine-tuning.

    - Adds small Gaussian noise to each vertex coordinate (normalized [0,1]).
    - Adds Gaussian noise to each RGBA channel (0..255), with clamping.
    - No in-place mutation; returns a new Individual.
    """
    point_sigma: float = 0.01   # small spatial jitter
    color_sigma: float = 5.0    # subtle color jitter (in intensity units)

    rng: random.Random = field(default_factory=random.Random)

    def _jitter_point(self, p: Tuple[float, float]) -> Tuple[float, float]:
        x = _clamp01(p[0] + self.rng.gauss(0.0, self.point_sigma))
        y = _clamp01(p[1] + self.rng.gauss(0.0, self.point_sigma))
        return (x, y)

    def _jitter_color(self, rgba: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        r = _clamp255(int(round(rgba[0] + self.rng.gauss(0.0, self.color_sigma))))
        g = _clamp255(int(round(rgba[1] + self.rng.gauss(0.0, self.color_sigma))))
        b = _clamp255(int(round(rgba[2] + self.rng.gauss(0.0, self.color_sigma))))
        a = _clamp255(int(round(rgba[3] + self.rng.gauss(0.0, self.color_sigma))))
        return (r, g, b, a)

    def mutate(self, ind: Individual) -> Individual:
        new_tris: List[Triangle] = []
        for t in ind.triangles:
            p1 = self._jitter_point(t.p1)
            p2 = self._jitter_point(t.p2)
            p3 = self._jitter_point(t.p3)
            color = self._jitter_color(t.color)
            z = _clamp01(t.z_index + self.rng.gauss(0.0, self.point_sigma))
            new_tris.append(Triangle(p1, p2, p3, color, z))
        return Individual(new_tris)

