from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Tuple, List

from src.models.individual import Individual
from src.models.triangle import Triangle
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy


@dataclass
class AnnularCrossover(CrossoverStrategy):
    """Annular crossover:
    - Picks a random segment (ring) of triangles from parent1.
    - Inserts it into the child.
    - Fills the rest from parent2 (preserving order and allowing duplicates).
    """
    rng: random.Random = field(default_factory=random.Random)

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        n = len(p1.triangles)
        if n == 0:
            return p1, p2

        start = self.rng.randint(0, n - 1)
        length = self.rng.randint(1, n)

        def crossover_one(a: Individual, b: Individual) -> Individual:
            a_ring = [a.triangles[(start + i) % n] for i in range(length)]
            b_tail = [b.triangles[i] for i in range(n) if i < start or i >= (start + length) % n]
            new_triangles = b_tail[:start] + a_ring + b_tail[start:]
            return Individual(new_triangles[:n])  # Ensure length stays constant

        return crossover_one(p1, p2), crossover_one(p2, p1)