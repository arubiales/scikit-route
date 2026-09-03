"""Typing stub of the compiled HeldKarp kernel (``skroute.exact._hk``)."""

import numpy as np
from numpy.typing import NDArray

def held_karp_search(
    C: NDArray[np.float64],
    others: NDArray[np.int64],
    depot: int,
    out: NDArray[np.int64],
) -> float: ...
