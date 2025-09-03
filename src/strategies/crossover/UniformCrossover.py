import random
from dataclasses import dataclass, field
from typing import Tuple

from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.models.individual import Individual


@dataclass
class UniformCrossover(CrossoverStrategy):
    """Uniform crossover at the triangle level.

    Each triangle (including its color and alpha) is treated as a block.
    With probability ``p`` a child's triangle comes from ``parent1`` and the
    corresponding triangle in the other child comes from ``parent2``; otherwise
    the triangles are swapped. This preserves whole triangle structures while
    still allowing good recombination of features.
    """

    p: float = 0.5
    rng: random.Random = field(default_factory=random.Random)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if len(parent1.triangles) != len(parent2.triangles):
            raise ValueError("Parents must have the same number of triangles")

        child1_tris = []
        child2_tris = []
        for t1, t2 in zip(parent1.triangles, parent2.triangles):
            if self.rng.random() < self.p:
                child1_tris.append(t1)
                child2_tris.append(t2)
            else:
                child1_tris.append(t2)
                child2_tris.append(t1)

        return Individual(child1_tris), Individual(child2_tris)