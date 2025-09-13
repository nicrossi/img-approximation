import random
from dataclasses import dataclass, field
from typing import Tuple
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.models.individual import Individual
from src.models.triangle import Triangle


@dataclass
class TwoPointCrossover(CrossoverStrategy):
    rng: random.Random = field(default_factory=random.Random)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if len(parent1.triangles) != len(parent2.triangles):
            raise ValueError("Parents must have the same number of triangles")
        n = len(parent1.triangles)
        if n < 3:
            return (
                Individual([Triangle.clone(t) for t in parent1.triangles]),
                Individual([Triangle.clone(t) for t in parent2.triangles]),
            )

        # Select two crossover points, ensuring point1 < point2
        point1 = self.rng.randrange(1, n - 1)
        point2 = self.rng.randrange(point1 + 1, n)
        
        # Create children by swapping middle segment and cloning triangles
        child1_tris = [Triangle.clone(t) for t in (
                parent1.triangles[:point1] + parent2.triangles[point1:point2] + parent1.triangles[point2:]
        )]
        child2_tris = [Triangle.clone(t) for t in (
                parent2.triangles[:point1] + parent1.triangles[point1:point2] + parent2.triangles[point2:]
        )]

        return Individual(child1_tris), Individual(child2_tris)