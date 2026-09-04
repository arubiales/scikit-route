"""Typing stubs of the TabuSearch kernel (``skroute.metaheuristics._tabu``)."""

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
Int32Matrix = NDArray[np.int32]

def tabu_step(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    pos: Int64Array,
    cand: Int64Array,
    until: Int32Matrix,
    it: int,
    tenure: int,
    max_time: float,
    fixed_cost: float,
    split: int,
    fast_path: bool,
    symmetric: bool,
    or_opt: bool,
    scratch: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    best: Int64Array,
    state: Float64Array,
) -> bool: ...
