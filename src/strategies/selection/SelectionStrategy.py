from __future__ import annotations
from typing import List, Sequence

class SelectionStrategy:
    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        raise NotImplementedError
