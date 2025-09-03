from dataclasses import dataclass
from typing import Sequence, List
from src.engine.selection import SelectionStrategy

@dataclass
class ProbTournamentSelection(SelectionStrategy):

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        # TODO implement
        pass
