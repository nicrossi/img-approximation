import os
import numpy as np
from PIL import Image
import pytest

from src.engine.PillowRenderer import PillowRenderer
from src.models.triangle import Triangle

def mse(img1, img2):
    """Compute Mean Squared Error (MSE) between two images."""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def test_render_argentina_flag():
    # reference image
    ref_path = os.path.join(os.path.dirname(__file__), '../assets/argentina-flag.png')
    ref_img = Image.open(ref_path).convert('RGBA')
    width, height = ref_img.size
    expected = np.array(ref_img)

    # create triangles approximating the flag (for demo, just one blue and one white)
    triangles = [
        Triangle(p1=(0,0), p2=(1,0), p3=(0,1), color=(116, 172, 223, 255)),  # blue triangle
        Triangle(p1=(1,0), p2=(1,1), p3=(0,1), color=(255,255,255,255)),     # white triangle
    ]

    renderer = PillowRenderer(width=width, height=height)
    rendered = renderer.render(triangles)

    # For a dummy example, just check if error is below a high threshold
    error = mse(rendered, expected)
    assert error < 4000, f"Rendered image too different from reference (MSE={error})"
