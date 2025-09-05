from __future__ import annotations
from typing import List, Sequence
import random
from dataclasses import dataclass, field


@dataclass
class SelectionStrategy:
    rng: random.Random = field(default_factory=random.Random)

    def select(self, fitness: Sequence[float], k: int) -> List[int]:
        raise NotImplementedError