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
from src.utils.diversity import population_diversity
from src.utils.metrics import GAMetrics
from concurrent.futures import ProcessPoolExecutor
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
    survivor_selection: SelectionStrategy | None = None
    # Generation gap (youth bias): fraction of population replaced by offspring each generation
    rho: float = 0.5
    max_workers: int | None = None # thread-pool size for fitness evaluation
    # Early stopping options
    early_stopping_patience: int = 0  # Number of stagnant generations before stopping
    error_threshold: float | None = None  # Stop if min_fitness drops below this (for minimization)


    def __post_init__(self) -> None:
        # Provide separate RNGs for strategies to avoid shared-state contention
        if hasattr(self.selection, "rng"):
            self.selection.rng = random.Random(self.rng.random())
        if hasattr(self.crossover, "rng"):
            self.crossover.rng = random.Random(self.rng.random())
        if self.mutation is not None and hasattr(self.mutation, "rng"):
            self.mutation.rng = random.Random(self.rng.random())
        if self.survivor_selection is not None and hasattr(self.survivor_selection, "rng"):
            self.survivor_selection.rng = random.Random(self.rng.random())

    def _evaluate(self, population: Sequence[Individual], executor: ProcessPoolExecutor) -> List[float]:
        """Return fitness scores for each individual in ``population``."""
        return list(executor.map(self.fitness.evaluate, population))

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
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            fitness = self._evaluate(pop, executor)
            #Instantiate metrics storage
            metrics = GAMetrics()
            #Store metrics for initial population
            fitness_arr = np.array(fitness)
            metrics.max_fitnesses.append(fitness_arr.max())
            metrics.min_fitnesses.append(fitness_arr.min())
            metrics.mean_fitnesses.append(fitness_arr.mean())
            metrics.std_fitnesses.append(fitness_arr.std())
            metrics.population_diversities.append(population_diversity(pop))

            #for early stopping if convergence tracking variables:
            best_fitness = fitness_arr.max() if self.maximize else fitness_arr.min()
            stagnant_epochs = 0

            for gen in range(self.generations):
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

                children: List[Individual] = []
                num_pairs = (num_children + 1) // 2
                for _ in range(num_pairs):
                    c1, c2 = make_child()
                    children.extend((c1, c2))
                children = list(islice(children, num_children))

                survivors_needed = self.pop_size - self.elitism - len(children)
                if survivors_needed > 0:
                    if self.survivor_selection is None:
                        survivors = [
                            ind
                            for _, ind in ranked[
                                          self.elitism : self.elitism + survivors_needed
                                          ]
                        ]
                    else:
                        candidates = ranked[self.elitism :]
                        cand_scores = self._selection_scores(
                            [fit for fit, _ in candidates]
                        )
                        idxs = self.survivor_selection.select(
                            cand_scores, survivors_needed
                        )
                        pool = [ind for _, ind in candidates]
                        survivors = [pool[i] for i in idxs]
                else:
                    survivors = []

                new_pop: List[Individual] = [*elite, *children, *survivors]

                # If rounding or elitism caused underfill, top up with best individual
                if len(new_pop) < self.pop_size:
                    fill = self.pop_size - len(new_pop)
                    best_ind = elite[0] if elite else ranked[0][1]
                    new_pop.extend([best_ind] * fill)

                pop = new_pop
                fitness = self._evaluate(pop, executor)
                #Store metrics for current population
                fitness_arr = np.array(fitness)
                max_f = float(fitness_arr.max())
                min_f = float(fitness_arr.min())

                # --- Early Stopping Logic ---
                current_best = max_f if self.maximize else min_f

                if not self.maximize and self.error_threshold is not None:
                    if min_f <= self.error_threshold:
                        print(
                            f"Early stopping at generation {gen + 1}: error_threshold reached ({min_f:.6f} ≤ {self.error_threshold})")
                        break

                if (self.maximize and current_best > best_fitness) or (
                        not self.maximize and current_best < best_fitness):
                    best_fitness = current_best
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1

                if self.early_stopping_patience > 0 and stagnant_epochs >= self.early_stopping_patience:
                    print(
                        f"Early stopping at generation {gen + 1}: no improvement for {self.early_stopping_patience} generations.")
                    break

                # --- Continue with metrics ---
                mean_f = float(fitness_arr.mean())
                std_f = float(fitness_arr.std())
                metrics.max_fitnesses.append(max_f)
                metrics.min_fitnesses.append(min_f)
                metrics.mean_fitnesses.append(mean_f)
                metrics.std_fitnesses.append(std_f)
                metrics.population_diversities.append(population_diversity(pop))
                print(f"Generation {gen+1}/{self.generations}: max={max_f:.6g} min={min_f:.6g} mean={mean_f:.6g} std={std_f:.6g}")

        best_idx = int(fitness_arr.argmax()) if self.maximize else int(fitness_arr.argmin())
        return pop[best_idx], metrics
