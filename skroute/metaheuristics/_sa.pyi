"""Typing stubs of the SimulatedAnnealing kernels (``skroute.metaheuristics._sa``)."""

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

def anneal_level(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    best: Int64Array,
    u: Float64Array,
    ri: Int64Array,
    rj: Int64Array,
    mv: Int64Array,
    temperature: float,
    max_time: float,
    fixed_cost: float,
    split: int,
    fast_path: bool,
    scratch: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    state: Float64Array,
) -> int: ...
def sample_deltas(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    ri: Int64Array,
    rj: Int64Array,
    mv: Int64Array,
    max_time: float,
    fixed_cost: float,
    split: int,
    fast_path: bool,
    scratch: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    out: Float64Array,
) -> None: ...
