from dataclasses import dataclass
from typing import Sequence, List

from src.strategies.selection.SelectionStrategy import SelectionStrategy

@dataclass
class UniversalSelection(SelectionStrategy):
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        """Stochastic Universal Sampling selection strategy.

        Select ``k`` individuals proportionally to their fitness values using
        equally spaced pointers across the fitness wheel. This method ensures a
        lower variance than simple roulette selection.
        """
        total = sum(fitness)
        if total <= 0:
            raise ValueError("fitness must be non-negative with positive sum")

        step = total / k
        start = self.rng.random() * step
        points = [start + i * step for i in range(k)]

        selected: List[int] = []
        cumulative = 0.0
        idx = 0

        for p in points:
            while cumulative + fitness[idx] < p and idx < len(fitness) - 1:
                cumulative += fitness[idx]
                idx += 1
            selected.append(idx)

        return selected