from __future__ import annotations
import random
from itertools import islice
from dataclasses import dataclass, field
from typing import List, Sequence
from src.models.individual import Individual
from src.strategies.selection.SelectionStrategy import SelectionStrategy
from src.strategies.crossover.CrossoverStrategy import CrossoverStrategy
from src.strategies.mutation.MutationStrategy import MutationStrategy
from src.strategies.fitness.FitnessStrategy import FitnessStrategy
from src.utils.metrics import GAMetrics
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import numpy as np

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
    # Generation gap (youth bias): fraction of population replaced by offspring each generation
    rho: float = 0.5

    def _evaluate(self, population: Sequence[Individual]) -> List[float]:
        """Return fitness scores for each individual in ``population``."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.fitness.evaluate, population))
        return results

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

    def run(self, population: Sequence[Individual]) -> tuple[Individual, GAMetrics]:
        """Run the genetic algorithm and return the best individual and its fitness value"""
        if len(population) != self.pop_size:
            raise ValueError(
                f"Population size {len(population)} != expected {self.pop_size}"
            )

        pop: List[Individual] = list(population)
        fitness = self._evaluate(pop)
        #Instantiate metrics storage
        metrics = GAMetrics()
        #Store metrics for initial population
        fitness_arr = np.array(fitness)
        metrics.max_fitnesses.append(fitness_arr.max())
        metrics.min_fitnesses.append(fitness_arr.min())
        metrics.mean_fitnesses.append(fitness_arr.mean())
        metrics.std_fitnesses.append(fitness_arr.std())

        for _ in range(self.generations):
            ranked = sorted(zip(fitness, pop), key=lambda t: t[0], reverse=self.maximize)
            elite = [ind for _, ind in ranked[: self.elitism]]

            max_children = max(0, self.pop_size - self.elitism)
            desired = int(round(self.rho * self.pop_size))
            num_children = min(max_children, max(0, desired))

            sel_scores = self._selection_scores(fitness)

            def make_child():
                i, j = self.selection.select(sel_scores, 2)
                p1, p2 = pop[i], pop[j]
                c1, c2 = self.crossover.crossover(p1, p2)
                if self.mutation is not None:
                    c1 = self.mutation.mutate(c1)
                    c2 = self.mutation.mutate(c2)
                return c1, c2

            with ThreadPoolExecutor() as executor:
                # Each call returns (c1, c2), so we need num_children//2 calls
                num_pairs = (num_children+1)//2
                child_pairs = list(executor.map(lambda _: make_child(), range(num_pairs)))
                # Flatten pairs to a single list
                children = list(islice(chain.from_iterable(child_pairs), num_children))

            survivors_needed = self.pop_size - self.elitism - len(children)
            survivors: List[Individual] = (
                [ind for _, ind in ranked[self.elitism : self.elitism + survivors_needed]]
                if survivors_needed > 0
                else []
            )

            new_pop: List[Individual] = [*elite, *children, *survivors]

            # If rounding or elitism caused underfill, top up with best individual
            if len(new_pop) < self.pop_size:
                fill = self.pop_size - len(new_pop)
                best_ind = elite[0] if elite else ranked[0][1]
                new_pop.extend([best_ind] * fill)

            pop = new_pop
            fitness = self._evaluate(pop)
            #Store metrics for current population
            fitness_arr = np.array(fitness)
            metrics.max_fitnesses.append(fitness_arr.max())
            metrics.min_fitnesses.append(fitness_arr.min())
            metrics.mean_fitnesses.append(fitness_arr.mean())
            metrics.std_fitnesses.append(fitness_arr.std())
   
        best_idx = int(fitness_arr.argmax()) if self.maximize else int(fitness_arr.argmin())
        return pop[best_idx], metrics
