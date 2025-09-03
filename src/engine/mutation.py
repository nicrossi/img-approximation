from typing import Dict, Type

from src.strategies.mutation import MutationStrategy, SimpleMutation

_MUTATION_STRATEGIES: Dict[str, Type[MutationStrategy]] = {
    "simple": SimpleMutation,
}

def build_mutation(name: str, params: Dict) -> MutationStrategy:
    return _MUTATION_STRATEGIES[name](**params)