"""Typing stub of the compiled BruteForce kernel (``skroute.exact._brute``)."""

import numpy as np
from numpy.typing import NDArray

def brute_force_search(
    C: NDArray[np.float64],
    T: NDArray[np.float64],
    tour: NDArray[np.int64],
    best: NDArray[np.int64],
    max_time: float,
    fixed_cost: float,
    split: int,
    halve: bool,
    dp: NDArray[np.float64],
    pred: NDArray[np.int64],
) -> tuple[float, int]: ...
