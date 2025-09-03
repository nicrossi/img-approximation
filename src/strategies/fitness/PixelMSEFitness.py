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
    alpha_reg_lambda: float = 1.0  # Regularization strength

    def evaluate(self, ind: Individual) -> float:
        # RGBA MSE on white background
        # Blend rendered image over white background using alpha
        img = self.renderer.render(ind.triangles)
        white_bg = np.ones_like(img, dtype=np.float32) * 255
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        blended = img[..., :3].astype(np.float32) * alpha + white_bg[..., :3] * (1 - alpha)
        target_rgb = self.target[..., :3].astype(np.float32)
        mse_rgb = np.mean((blended - target_rgb) ** 2)

        # Alpha regularization: promote transparency (lower alpha) in early layers
        # Normalize alpha to [0,1]
        alphas = np.array([t.color[3] / 255.0 for t in ind.triangles])
        alpha_reg = self.alpha_reg_lambda * np.mean(alphas)

        return float(mse_rgb + alpha_reg)
