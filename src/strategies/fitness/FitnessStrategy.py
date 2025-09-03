from __future__ import annotations
from typing import Protocol
from src.models.individual import Individual

class FitnessStrategy(Protocol):
    def evaluate(self, ind: Individual) -> float:
        """Return the fitness score for an individual. Lower or higher is better depending on GAEngine.maximize."""
        ...

