from __future__ import annotations
from typing import Dict, Type

from src.strategies.mutation.GenMutation import GenMutation
from src.strategies.mutation.MutationStrategy import MutationStrategy
from src.strategies.mutation.SimpleMutation import SimpleMutation
from src.strategies.mutation.NonUniform import NonUniform

_MUTATION_STRATEGIES: Dict[str, Type[MutationStrategy]] = {
    "simple": SimpleMutation,
    "gen": GenMutation,
    "nonuniform": NonUniform
}

def build_mutation(name: str, params: Dict) -> MutationStrategy:
    return _MUTATION_STRATEGIES[name](**params)