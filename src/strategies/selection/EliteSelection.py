from dataclasses import dataclass
from typing import Sequence, List
from src.engine.selection import SelectionStrategy


@dataclass
class EliteSelection(SelectionStrategy):
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        return order[:k]
