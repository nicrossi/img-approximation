from __future__ import annotations
from typing import Protocol
from src.models.individual import Individual

class MutationStrategy(Protocol):
    def mutate(self, ind: Individual) -> Individual:
        """Return a mutated copy of the given individual (must not modify in place)."""
        ...

