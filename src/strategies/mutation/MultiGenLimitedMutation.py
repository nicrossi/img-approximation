from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from src.models.triangle import Triangle
from src.models.individual import Individual
from src.strategies.mutation.MutationStrategy import MutationStrategy


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _clamp255(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else v


@dataclass
class MultiGenLimitedMutation(MutationStrategy):
    """Multi-gene mutation that selects a random subset of triangles to mutate.
    
    Unlike GenMutation which mutates ALL triangles, this strategy:
    - Selects a random number of triangles to mutate (between min_genes and max_genes)
    - Applies intensive mutation to selected triangles only
    - Leaves other triangles completely unchanged
    
    This provides more controlled diversity by preserving most of the genome
    while making significant changes to a few genes.
    """
    min_genes: int = 1          # Minimum triangles to mutate
    max_genes: int = 5          # Maximum triangles to mutate
    point_sigma: float = 0.1    # Larger mutations than GenMutation
    color_sigma: float = 25.0   # Larger color changes
    
    rng: random.Random = field(default_factory=random.Random)

    def _mutate_triangle(self, triangle: Triangle) -> Triangle:
        """Apply intensive mutation to a single triangle."""
        # Mutate all three vertices with larger sigma
        x1 = _clamp01(triangle.p1[0] + self.rng.gauss(0.0, self.point_sigma))
        y1 = _clamp01(triangle.p1[1] + self.rng.gauss(0.0, self.point_sigma))
        x2 = _clamp01(triangle.p2[0] + self.rng.gauss(0.0, self.point_sigma))
        y2 = _clamp01(triangle.p2[1] + self.rng.gauss(0.0, self.point_sigma))
        x3 = _clamp01(triangle.p3[0] + self.rng.gauss(0.0, self.point_sigma))
        y3 = _clamp01(triangle.p3[1] + self.rng.gauss(0.0, self.point_sigma))
        
        # Mutate color with larger sigma
        r = _clamp255(int(round(triangle.color[0] + self.rng.gauss(0.0, self.color_sigma))))
        g = _clamp255(int(round(triangle.color[1] + self.rng.gauss(0.0, self.color_sigma))))
        b = _clamp255(int(round(triangle.color[2] + self.rng.gauss(0.0, self.color_sigma))))
        a = _clamp255(int(round(triangle.color[3] + self.rng.gauss(0.0, self.color_sigma))))
        
        return Triangle((x1, y1), (x2, y2), (x3, y3), (r, g, b, a))

    def mutate(self, ind: Individual) -> Individual:
        if not ind.triangles:
            return Individual([])
        
        # Determine how many triangles to mutate
        n_triangles = len(ind.triangles)
        max_mutate = min(self.max_genes, n_triangles)
        min_mutate = min(self.min_genes, max_mutate)
        n_mutate = self.rng.randint(min_mutate, max_mutate)
        
        # Select random triangles to mutate
        indices_to_mutate: Set[int] = set()
        while len(indices_to_mutate) < n_mutate:
            indices_to_mutate.add(self.rng.randrange(n_triangles))
        
        # Create new individual with mutations applied to selected triangles
        new_triangles: List[Triangle] = []
        for i, triangle in enumerate(ind.triangles):
            if i in indices_to_mutate:
                new_triangles.append(self._mutate_triangle(triangle))
            else:
                # Keep original triangle unchanged
                new_triangles.append(Triangle(triangle.p1, triangle.p2, triangle.p3, triangle.color))
        
        return Individual(new_triangles)