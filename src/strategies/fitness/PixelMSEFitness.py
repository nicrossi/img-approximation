from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.engine.PillowRenderer import PillowRenderer
from src.models.individual import Individual
from src.strategies.fitness.FitnessStrategy import FitnessStrategy


@dataclass
class PixelMSEFitness(FitnessStrategy):
    renderer: PillowRenderer
    target: np.ndarray  # shape (H, W, 4), dtype uint8

    def evaluate(self, ind: Individual) -> float:
        img = self.renderer.render(ind.triangles)
        # Compute MSE in float32 to avoid overflow and keep precision
        diff = img.astype(np.float32) - self.target.astype(np.float32)
        return float(np.mean(diff * diff))

