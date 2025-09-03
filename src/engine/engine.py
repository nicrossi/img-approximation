from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Sequence
from src.models.individual import Individual
from src.strategies.selection.SelectionStrategy import SelectionStrategy
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.strategies.mutation.MutationStrategy import MutationStrategy
from src.strategies.fitness.FitnessStrategy import FitnessStrategy


@dataclass
class GAEngine:
    """Simple genetic algorithm engine"""
    fitness: FitnessStrategy
    selection: SelectionStrategy
    crossover: CrossoverStrategy
    mutation: MutationStrategy | None
    pop_size: int
    generations: int
    elitism: int = 1
    maximize: bool = False
    rng: random.Random = field(default_factory=random.Random)

    def _evaluate(self, population: Sequence[Individual]) -> List[float]:
        """Return fitness scores for each individual in ``population``."""
        return [self.fitness.evaluate(ind) for ind in population]

    def _selection_scores(self, fitness: Sequence[float]) -> List[float]:
        """Transform fitness into selection scores where higher is better and non-negative when possible.
        This lets selection strategies assume maximization without worrying about GAEngine.maximize.
        """
        if self.maximize:
            return list(fitness)
        if not fitness:
            return []
        m = max(fitness)
        scores = [m - f for f in fitness]
        # If all equal (scores all zero), fall back to uniform weights
        if all(s <= 0 for s in scores):
            return [1.0 for _ in fitness]
        return scores

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
                sel_scores = self._selection_scores(fitness)
                idxs = self.selection.select(sel_scores, 2)
                parents = [pop[idxs[0]], pop[idxs[1]]]
                child1, child2 = self.crossover.crossover(parents[0], parents[1])
                if self.mutation is not None:
                    child1 = self.mutation.mutate(child1)
                new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    if self.mutation is not None:
                        child2 = self.mutation.mutate(child2)
                    new_pop.append(child2)

            pop = new_pop
            fitness = self._evaluate(pop)

        ranked = list(zip(fitness, pop))
        ranked.sort(key=lambda x: x[0], reverse=self.maximize)
        return ranked[0][1]
