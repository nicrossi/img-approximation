from __future__ import annotations

from typing import Dict, List, Sequence, Type
from src.strategies.selection.BoltzmannSelection import BoltzmannSelection
from src.strategies.selection.EliteSelection import EliteSelection
from src.strategies.selection.ProbTournamentSelection import ProbTournamentSelection
from src.strategies.selection.RankingSelection import RankingSelection
from src.strategies.selection.RouletteSelection import RouletteSelection
from src.strategies.selection.TournamentSelection import TournamentSelection
from src.strategies.selection.UniversalSelection import UniversalSelection

_SELECTION_STRATEGIES: Dict[str, Type[SelectionStrategy]] = {
    "elite": EliteSelection,
    "roulette": RouletteSelection,
    "universal": UniversalSelection,
    "boltzmann": BoltzmannSelection,
    "tournament": TournamentSelection,
    "prob_tournament": ProbTournamentSelection,
    "ranking": RankingSelection,
}


def build_selection(name: str, params: Dict) -> SelectionStrategy:
    return _SELECTION_STRATEGIES[name](**params)

class SelectionStrategy:
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        raise NotImplementedError
