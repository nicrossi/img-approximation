from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy

@dataclass
class ProbTournamentSelection(SelectionStrategy):

    threshold: float = 0.75 # probability of selecting the better individual

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        """Select individuals using probabilistic tournament selection.

        Args:
            fitness (Sequence[float]): fitness values of the population
            k (int): number of individuals to select

        Returns:
            List[int]: list of indices of the selected individuals
        """
        selected = []
        for _ in range(k):
            indices = self.rng.sample(range(len(fitness)), 2)
            indices.sort(key=lambda idx: fitness[idx], reverse=True)
            if self.rng.random() < self.threshold:
                selected.append(indices[0])  # Selects the best
            else:
                selected.append(indices[1])  # Selects the worst
        return selected
