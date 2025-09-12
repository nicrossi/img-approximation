from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Tuple, List
from src.models.triangle import Triangle
from src.models.individual import Individual
from src.strategies.mutation.MutationStrategy import MutationStrategy


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _clamp255(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


def _gauss_perturb(rng: random.Random, val: float, sigma: float) -> float:
    return val + rng.gauss(0.0, sigma)


@dataclass
class UniformMutation(MutationStrategy):
    # Position mutation per coordinate in normalized [0,1]
    point_rate: float = 0.1
    point_sigma: float = 0.05

    # Color mutation per channel 0..255
    color_rate: float = 0.05
    color_sigma: float = 15.0

    # Occasionally swap two triangles to change z-order
    swap_rate: float = 0.01

    rng: random.Random = field(default_factory=random.Random)

    def _mutate_point(self, x: float, y: float) -> Tuple[float, float]:
        rx = self.rng.random()
        ry = self.rng.random()
        if rx < self.point_rate:
            x = _clamp01(_gauss_perturb(self.rng, x, self.point_sigma))
        if ry < self.point_rate:
            y = _clamp01(_gauss_perturb(self.rng, y, self.point_sigma))
        return x, y

    def _mutate_color(self, r: int, g: int, b: int, a: int) -> Tuple[int, int, int, int]:
        r = _clamp255(int(round(r + self.rng.gauss(0.0, self.color_sigma))) if self.rng.random() < self.color_rate else r)
        g = _clamp255(int(round(g + self.rng.gauss(0.0, self.color_sigma))) if self.rng.random() < self.color_rate else g)
        b = _clamp255(int(round(b + self.rng.gauss(0.0, self.color_sigma))) if self.rng.random() < self.color_rate else b)
        a = _clamp255(int(round(a + self.rng.gauss(0.0, self.color_sigma))) if self.rng.random() < self.color_rate else a)
        return r, g, b, a

    def _maybe_swap(self, tris: List[Triangle]) -> None:
        if len(tris) < 2:
            return
        if self.rng.random() < self.swap_rate:
            i = self.rng.randrange(len(tris))
            j = self.rng.randrange(len(tris))
            if i != j:
                tris[i], tris[j] = tris[j], tris[i]

    def mutate(self, ind: Individual) -> Individual:
        # Deep-copy triangles and mutate copy
        new_tris: List[Triangle] = []
        for t in ind.triangles:
            x1, y1 = self._mutate_point(t.p1[0], t.p1[1])
            x2, y2 = self._mutate_point(t.p2[0], t.p2[1])
            x3, y3 = self._mutate_point(t.p3[0], t.p3[1])
            r, g, b, a = self._mutate_color(t.color[0], t.color[1], t.color[2], t.color[3])
            z = _clamp01(_gauss_perturb(self.rng, t.z_index, self.point_sigma) if self.rng.random() < self.point_rate else t.z_index)
            new_tris.append(Triangle((x1, y1), (x2, y2), (x3, y3), (r, g, b, a), z))

        # Possibly swap triangles to change order
        self._maybe_swap(new_tris)

        return Individual(new_tris)

