from __future__ import annotations
from typing import Dict, Type

from src.strategies.mutation.MutationStrategy import MutationStrategy
from src.strategies.mutation.SimpleMutation import SimpleMutation

_MUTATION_STRATEGIES: Dict[str, Type[MutationStrategy]] = {
    "simple": SimpleMutation,
}

def build_mutation(name: str, params: Dict) -> MutationStrategy:
    return _MUTATION_STRATEGIES[name](**params)