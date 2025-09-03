from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy


@dataclass
class EliteSelection(SelectionStrategy):
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        return order[:k]
