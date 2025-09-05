import numpy as np
from src.engine.PillowRenderer import PillowRenderer
from src.strategies.fitness.PixelMSEFitness import PixelMSEFitness
from src.models.triangle import Triangle
from src.models.individual import Individual


def _triangles_basic():
    # Two simple triangles with solid colors
    return [
        Triangle(p1=(0.0, 0.0), p2=(1.0, 0.0), p3=(0.0, 1.0), color=(255, 0, 0, 255)),  # red
        Triangle(p1=(1.0, 0.0), p2=(1.0, 1.0), p3=(0.0, 1.0), color=(0, 255, 0, 255)),  # green
    ]


def test_pixel_mse_zero_on_identical_render():
    renderer = PillowRenderer(width=16, height=16)
    tris = _triangles_basic()
    ind = Individual(triangles=tris)
    target = renderer.render(ind.triangles)
    fit = PixelMSEFitness(renderer=renderer, target=target)

    score = fit.evaluate(ind)
    # The score should equal the alpha regularization term since MSE is zero
    expected_alpha_reg = fit.alpha_reg_lambda * np.mean([t.color[3] / 255.0 for t in tris])
    assert np.isclose(score, expected_alpha_reg, atol=1e-6)


def test_pixel_mse_positive_on_difference():
    renderer = PillowRenderer(width=16, height=16)
    tris = _triangles_basic()
    ind = Individual(triangles=tris)
    target = renderer.render(ind.triangles)
    fit = PixelMSEFitness(renderer=renderer, target=target)

    # Change one triangle color to force a difference
    changed = [
        Triangle(p1=tris[0].p1, p2=tris[0].p2, p3=tris[0].p3, color=(0, 0, 255, 255)),
        tris[1],
    ]
    ind2 = Individual(changed)

    score = fit.evaluate(ind2)
    # The score should be greater than the alpha regularization term
    expected_alpha_reg = fit.alpha_reg_lambda * np.mean([t.color[3] / 255.0 for t in changed])
    assert score > expected_alpha_reg
