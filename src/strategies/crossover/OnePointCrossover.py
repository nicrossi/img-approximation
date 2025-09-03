import random
from dataclasses import dataclass, field
from typing import Tuple
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.models.individual import Individual


@dataclass
class OnePointCrossover(CrossoverStrategy):
    rng: random.Random = field(default_factory=random.Random)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if len(parent1.triangles) != len(parent2.triangles):
            raise ValueError("Parents must have the same number of triangles")
        n = len(parent1.triangles)
        if n < 2:
            return Individual(list(parent1.triangles)), Individual(list(parent2.triangles))
        point = self.rng.randrange(1, n)
        child1_tris = parent1.triangles[:point] + parent2.triangles[point:]
        child2_tris = parent2.triangles[:point] + parent1.triangles[point:]
        return Individual(child1_tris), Individual(child2_tris)