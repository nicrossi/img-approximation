from __future__ import annotations
from typing import Dict, Type

from src.strategies.mutation.GenMutation import GenMutation
from src.strategies.mutation.MutationStrategy import MutationStrategy
from src.strategies.mutation.UniformMutation import UniformMutation
from src.strategies.mutation.MultiGenLimitedMutation import MultiGenLimitedMutation
from src.strategies.mutation.NonUniform import NonUniform

_MUTATION_STRATEGIES: Dict[str, Type[MutationStrategy]] = {
    "uniform": UniformMutation,
    "gen": GenMutation,
    "multigen": MultiGenLimitedMutation,
    "nonuniform": NonUniform
}

def build_mutation(name: str, params: Dict) -> MutationStrategy:
    return _MUTATION_STRATEGIES[name](**params)