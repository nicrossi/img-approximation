from dataclasses import dataclass
from typing import Sequence, List
from src.strategies.selection.SelectionStrategy import SelectionStrategy

@dataclass
class TournamentSelection(SelectionStrategy):

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        # TODO implement
        pass