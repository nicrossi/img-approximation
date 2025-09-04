from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy
from random import sample

@dataclass
class TournamentSelection(SelectionStrategy):
    tournament_size : int = 3  # tournament size

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        """Orders and selects individuals using tournament selection.

        Args:
            fitness (Sequence[float]): fitness values of the population
            k (int): number of individuals to select

        Returns:
            List[int]: Returns the indices of the selected individuals
        """
        selected = []
        for _ in range(k):
            indices = sample(range(len(fitness)), self.tournament_size)
            best = max(indices, key=lambda idx: fitness[idx])
            selected.append(best)
        return selected