from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple
from src.models.individual import Individual


FitnessFunc = Callable[[Individual], float]
SelectionFunc = Callable[[Sequence[Individual], Sequence[float], int], List[Individual]]
CrossoverFunc = Callable[[Individual, Individual], Tuple[Individual, Individual]]
MutationFunc = Callable[[Individual], Individual]


@dataclass
class GAEngine:
    """Simple genetic algorithm engine"""
    fitness_fn: FitnessFunc
    selection_fn: SelectionFunc
    crossover_fn: CrossoverFunc
    mutation_fn: MutationFunc
    pop_size: int
    generations: int
    elitism: int = 1
    maximize: bool = False
    rng: random.Random = field(default_factory=random.Random)

    def _evaluate(self, population: Sequence[Individual]) -> List[float]:
        """Return fitness scores for each individual in ``population``."""
        return [self.fitness_fn(ind) for ind in population]

    def run(self, population: Sequence[Individual]) -> Individual:
        """Run the genetic algorithm and return the best individual found"""
        if len(population) != self.pop_size:
            raise ValueError(
                f"Population size {len(population)} != expected {self.pop_size}")

        pop: List[Individual] = list(population)
        fitness = self._evaluate(pop)

        for _ in range(self.generations):
            ranked = list(zip(fitness, pop))
            ranked.sort(key=lambda x: x[0], reverse=self.maximize)
            elite = [ind for _, ind in ranked[: self.elitism]]
            new_pop: List[Individual] = elite

            while len(new_pop) < self.pop_size:
                parents = self.selection_fn(pop, fitness, 2)
                child1, child2 = self.crossover_fn(parents[0], parents[1])
                child1 = self.mutation_fn(child1)
                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    child2 = self.mutation_fn(child2)
                    new_pop.append(child2)

            pop = new_pop
            fitness = self._evaluate(pop)

        ranked = list(zip(fitness, pop))
        ranked.sort(key=lambda x: x[0], reverse=self.maximize)
        return ranked[0][1]
