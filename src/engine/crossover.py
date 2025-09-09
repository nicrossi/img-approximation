from __future__ import annotations
from typing import Dict, Type

from src.strategies.crossover.UniformCrossover import UniformCrossover
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.strategies.crossover.OnePointCrossover import OnePointCrossover
from src.strategies.crossover.TwoPointCrossover import TwoPointCrossover
from src.strategies.crossover.AnnularCrossover import AnnularCrossover

_CROSSOVER_STRATEGIES: Dict[str, Type[CrossoverStrategy]] = {
        "one_point": OnePointCrossover,
        "two_point": TwoPointCrossover,
        "uniform": UniformCrossover,
        "annular": AnnularCrossover,
}

def build_crossover(name: str, params: Dict) -> CrossoverStrategy:
    return _CROSSOVER_STRATEGIES[name](**params)

