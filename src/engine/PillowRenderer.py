from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple
from PIL import Image, ImageDraw
from ..models.triangle import Triangle


@dataclass(slots=True)
class PillowRenderer:
    """Render a list of `Triangle` objects onto an RGBA canvas.

    This renderer uses Pillow's ``ImageDraw`` with alpha compositing to draw
    semi-transparent triangles. Triangle coordinates are in [0, 1]
    and scaled to the canvas resolution.
    """

    width: int
    height: int
    background: Tuple[int, int, int, int] = (255, 255, 255, 255)

    def render(self, triangles: Iterable[Triangle]) -> np.ndarray:
        """Render ``triangles`` and return the resulting image as ``numpy.ndarray``
        of shape ``(height, width, 4)`` with dtype ``uint8``.
        """
        canvas = Image.new("RGBA", (self.width, self.height), self.background)
        for tri in triangles:
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")
            pts = [
                (tri.p1[0] * self.width, tri.p1[1] * self.height),
                (tri.p2[0] * self.width, tri.p2[1] * self.height),
                (tri.p3[0] * self.width, tri.p3[1] * self.height),
            ]
            draw.polygon(pts, fill=tri.color)
            canvas = Image.alpha_composite(canvas, overlay)
        return np.asarray(canvas, dtype=np.uint8)