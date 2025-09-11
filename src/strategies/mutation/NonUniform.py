from __future__ import annotations
from dataclasses import dataclass, field
import random

from src.models.triangle import Triangle
from src.models.individual import Individual

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))

@dataclass
class NonUniform:
    b: float = 3.0
    p_mutate_vertices: float = 0.5
    p_vertex_component: float = 1.0
    p_color_component: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    progress: float = 0.0

    def set_progress(self, progress: float) -> None:
        self.progress = _clamp(progress, 0.0, 1.0)

    def set_generation(self, gen_idx: int, max_generations: int) -> None:
        self.set_progress(gen_idx / max_generations if max_generations > 0 else 0.0)

    def _delta_non_uniform(self, dist_to_bound: float) -> float:
        if dist_to_bound <= 0.0:
            return 0.0
        factor = (1.0 - _clamp(self.progress, 0.0, 1.0)) ** max(self.b, 1.0)
        return dist_to_bound * (1.0 - self.rng.random() ** factor)

    def _mutate_scalar_non_uniform(self, value: float, low: float, high: float) -> float:
        if low >= high:
            return value
        # Decide direction randomly, then compute distance to that boundary
        to_high = self.rng.random() < 0.5
        dist = (high - value) if to_high else (value - low)
        delta = self._delta_non_uniform(dist)
        new_val = value + delta if to_high else value - delta
        return _clamp(new_val, low, high)

    def _maybe_mutate_coord(self, coord: float, low: float, high: float, p: float) -> float:
        return _clamp(self._mutate_scalar_non_uniform(coord, low, high), low, high) if self.rng.random() < p else coord

    def mutate(self, ind: Individual) -> Individual:
        if not ind.triangles:
            return ind

        tris = list(ind.triangles)
        idx = self.rng.randrange(len(tris))
        t = tris[idx]

        if self.rng.random() < self.p_mutate_vertices:
            # Mutate vertex coordinates (each component independently with probability p_vertex_component)
            p1 = (
                self._maybe_mutate_coord(t.p1[0], 0.0, 1.0, self.p_vertex_component),
                self._maybe_mutate_coord(t.p1[1], 0.0, 1.0, self.p_vertex_component),
            )
            p2 = (
                self._maybe_mutate_coord(t.p2[0], 0.0, 1.0, self.p_vertex_component),
                self._maybe_mutate_coord(t.p2[1], 0.0, 1.0, self.p_vertex_component),
            )
            p3 = (
                self._maybe_mutate_coord(t.p3[0], 0.0, 1.0, self.p_vertex_component),
                self._maybe_mutate_coord(t.p3[1], 0.0, 1.0, self.p_vertex_component),
            )
            new_z = max(0, min(255, t.z_index + self.rng.randint(-5, 5)))
            tris[idx] = Triangle(p1, p2, p3, t.color, new_z)
        else:
            # Mutate color channels independently with probability p_color_component
            r, g, b, a = t.color
            def mut_channel(c: int) -> int:
                if self.rng.random() < self.p_color_component:
                    return int(round(self._mutate_scalar_non_uniform(float(c), 0.0, 255.0)))
                return c
            mutated_color = tuple(int(_clamp(mut_channel(c), 0, 255)) for c in (r, g, b, a))  # type: ignore[arg-type]
            tris[idx] = Triangle(t.p1, t.p2, t.p3, mutated_color, max(0, min(255, t.z_index + self.rng.randint(-5, 5))))  # type: ignore[arg-type]

        return Individual(tris)