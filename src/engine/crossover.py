from __future__ import annotations
from typing import Dict, Type

from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.strategies.crossover.OnePointCrossover import OnePointCrossover

_CROSSOVER_STRATEGIES: Dict[str, Type[CrossoverStrategy]] = {
        "one_point": OnePointCrossover,
}

def build_crossover(name: str, params: Dict) -> CrossoverStrategy:
    return _CROSSOVER_STRATEGIES[name](**params)

