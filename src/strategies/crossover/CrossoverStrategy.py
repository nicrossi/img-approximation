from __future__ import annotations
from typing import Tuple
from src.models.individual import Individual

class CrossoverStrategy:
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        raise NotImplementedError
