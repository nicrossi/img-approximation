from __future__ import annotations
from typing import Dict, Type

from src.strategies.fitness.FitnessStrategy import FitnessStrategy
from src.strategies.fitness.PixelMSEFitness import PixelMSEFitness

_FITNESS_STRATEGIES: Dict[str, type] = {
    "pixel_mse": PixelMSEFitness,
}

def build_fitness(name: str, params: dict) -> FitnessStrategy:
    return _FITNESS_STRATEGIES[name](**params)  # type: ignore[call-arg]
