import random
from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy


@dataclass
class RouletteSelection(SelectionStrategy):
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        return [self._weighted_choice(fitness) for _ in range(k)]

    def _weighted_choice(self, weights: Sequence[float]) -> int:
        total = sum(weights)
        if total <= 0:
            raise ValueError("fitness must be non-negative with positive sum")
        r = self.rng.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                return i
        return len(weights) - 1
