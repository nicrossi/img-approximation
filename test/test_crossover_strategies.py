import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.triangle import Triangle
from src.models.individual import Individual
from src.strategies.crossover.OnePointCrossover import OnePointCrossover


def _make_triangle(idx: int) -> Triangle:
    return Triangle((0, 0), (0, 0), (0, 0), (idx, idx, idx, idx))


def test_one_point_crossover_exchanges_slices_and_preserves_parents():
    t0, t1, t2 = (_make_triangle(i) for i in range(3))
    t3, t4, t5 = (_make_triangle(i) for i in range(3, 6))
    p1 = Individual([t0, t1, t2])
    p2 = Individual([t3, t4, t5])
    original_p1 = list(p1.triangles)
    original_p2 = list(p2.triangles)

    xover = OnePointCrossover(rng=random.Random(1))  # crossover point = 1
    c1, c2 = xover.crossover(p1, p2)

    assert c1.triangles == [t0, t4, t5]
    assert c2.triangles == [t3, t1, t2]
    assert p1.triangles == original_p1
    assert p2.triangles == original_p2