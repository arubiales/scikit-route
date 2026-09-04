"""Typing stubs of the AntColony kernels (``skroute.metaheuristics._aco``).

Arrays must be C-contiguous: ``float64`` matrices, weights, uniforms and costs, ``int64`` tours,
positions and candidate lists, ``uint8`` flags.
"""

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]

def construct_tours(
    choice: Float64Array,
    cand: Int64Array,
    depot: int,
    u: Float64Array,
    tours: Int64Array,
    visited: UInt8Array,
    w: Float64Array,
) -> None: ...
def polish_and_evaluate(
    C: Float64Array,
    T: Float64Array,
    tours: Int64Array,
    max_time: float,
    fixed_cost: float,
    split: int,
    ls_mode: int,
    ls_moves: int,
    cand: Int64Array,
    tour: Int64Array,
    pos: Int64Array,
    dont_look: UInt8Array,
    scratch_tour: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    costs: Float64Array,
) -> None: ...
