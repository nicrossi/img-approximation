from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy
from src.strategies.selection.RouletteSelection import RouletteSelection

@dataclass
class RankingSelection(SelectionStrategy):
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        # Get sorted indices (highest fitness first)
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        # Assign ranks: highest fitness gets 1, lowest gets len(fitness)
        pseudo_fit = [0.0] * len(fitness)
        for rank, idx in enumerate(sorted_indices, start=1):
            pseudo_fit[idx] = (len(fitness)-rank)/len(fitness)  # Normalize ranks to [0, 1]
        return RouletteSelection().select(pseudo_fit, k)