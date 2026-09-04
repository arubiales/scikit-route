"""Typing stub of the insertion construction kernel (``skroute.construction._insert``)."""

import numpy as np
from numpy.typing import NDArray

STRATEGIES: dict[str, int]

def insertion_tour(
    C: NDArray[np.float64],
    depot: int,
    strategy: str,
    order: NDArray[np.int64] | None = None,
    after: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]: ...
