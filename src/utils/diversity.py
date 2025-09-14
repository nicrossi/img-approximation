from __future__ import annotations

from typing import Sequence

import numpy as np

from src.models.individual import Individual


def population_diversity(pop: Sequence[Individual]) -> float:
    """Compute mean pairwise Euclidean distance between individuals.

    The individual's genome is flattened into a numeric vector containing all
    triangle parameters. The result is the average distance across all unique
    pairs in the population. Returns ``0.0`` for populations with fewer than
    two individuals.

    To avoid any parameter family (e.g., coordinates vs. colors vs. z-index)
    dominating due to differing numeric ranges, each genome dimension is
    min-max normalized across the current population before distance is computed.
    """
    n = len(pop)
    if n < 2:
        return 0.0

    # Flatten individuals into arrays
    genomes: list[list[float]] = []
    for ind in pop:
        genes: list[float] = []
        for t in ind.triangles:
            genes.extend([
                t.p1[0], t.p1[1],
                t.p2[0], t.p2[1],
                t.p3[0], t.p3[1],
                float(t.color[0]), float(t.color[1]), float(t.color[2]), float(t.color[3]),
                t.z_index,
            ])
        genomes.append(genes)

    arr = np.array(genomes, dtype=float)

    # Normalize each feature (column) to [0, 1] across the population
    col_min = arr.min(axis=0)
    col_range = np.ptp(arr, axis=0)  # max - min
    # Avoid division by zero for constant columns
    safe_range = np.where(col_range > 0, col_range, 1.0)
    arr_norm = (arr - col_min) / safe_range
    # Pairwise distances on normalized genomes
    diff = arr_norm[:, None, :] - arr_norm[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    if iu[0].size == 0:
        return 0.0
    return float(dists[iu].mean())