import argparse
import json
import random
from typing import Tuple, List

import numpy as np
from PIL import Image
from pathlib import Path
from src.engine.PillowRenderer import PillowRenderer
from src.engine.crossover import build_crossover
from src.engine.engine import GAEngine
from src.engine.selection import build_selection
from src.engine.mutation import build_mutation
from src.models.individual import Individual
from src.models.triangle import Triangle
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile, args.overrides)
    out = Path(cfg.experiment.output_dir);
    out.mkdir(parents=True, exist_ok=True)

    target = np.array(Image.open(cfg.data.image_path).convert("RGBA").resize(cfg.data.canvas_size))
    renderer = PillowRenderer(*cfg.data.canvas_size) if cfg.renderer.backend == "pillow" else None # Other renderers TBD
    fit = None # Fitness function TBD
    select = build_selection(cfg.selection.name, cfg.selection.params)
    xover = build_crossover(cfg.crossover.name, cfg.crossover.params)
    mutate = build_mutation(cfg.mutation.name, cfg.mutation.params) if "mutation" in cfg else None

    eng = GAEngine(
        fitness_fn=fit,
        selection=select,
        crossover=xover,
        mutation=mutate,
        pop_size=cfg.ga.pop_size,
        generations=cfg.ga.generations,
        elitism=cfg.ga.elitism,
        maximize=cfg.ga.maximize,
        rng=random.Random(cfg.seed),
    )

    pop = _init_population(cfg.ga.pop_size, cfg.genome.num_triangles, cfg.data.canvas_size)
    best = eng.run(pop)
    (out / "best.json").write_text(json.dumps(best, default=lambda o: o.__dict__, indent=2))

if __name__ == "__main__": main()

def _init_population(pop_size: int, num_triangles: int, canvas_size: Tuple[int, int]) -> List[Individual]:
    """Create the initial population for the genetic algorithm.
    Each individual consists of ``num_triangles`` randomly generated

    Args:
        pop_size: Number of individuals in the population.
        num_triangles: Number of triangles per individual.
        canvas_size: Tuple with the canvas width and height.
    """
    width, height = canvas_size  # kept for future use / interface symmetry
    population: List[Individual] = []

    for _ in range(pop_size):
        triangles: List[Triangle] = []
        for _ in range(num_triangles):
            p1 = (random.random(), random.random())
            p2 = (random.random(), random.random())
            p3 = (random.random(), random.random())
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            triangles.append(Triangle(p1, p2, p3, color))

        population.append(Individual(triangles))

    return population