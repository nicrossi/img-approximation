from dataclasses import dataclass
from typing import Sequence, List
from src.engine.selection import SelectionStrategy
from src.strategies.selection.RouletteSelection import RouletteSelection
import numpy as np


@dataclass
class BoltzmannSelection(SelectionStrategy):
    t_initial: float = 100.0
    t_final: float = 10.0
    decay: float = 0.99
    generation_count: int = 0

    def __post_init__(self):
        assert self.t_initial > self.t_final > 0, "Require t_initial > t_final > 0"
        assert 0 < self.decay, "Require 0 < decay"

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        """Select individuals using Boltzmann selection.

        Args:
            fitness (Sequence[float]): fitness values of the population
            k (int): number of individuals to select

        Returns:
            List[int]: Returns the indices of the selected individuals
        """
        t = self.t_final + (self.t_initial - self.t_final) * np.exp(-self.decay*self.generation_count)
        exp_fitness = np.exp(np.array(fitness) / t)
        if np.isinf(exp_fitness).any():
            raise OverflowError("Overflow in exp(fitness/t). Try increasing t_initial or t_final")
        pseudo_fit = exp_fitness / exp_fitness.mean()
        self.generation_count += 1
        return RouletteSelection().select(pseudo_fit, k)
