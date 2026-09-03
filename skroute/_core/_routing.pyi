"""Typing stubs of the compiled core (``skroute._core._routing``).

Only the Python-visible surface is declared here: the ``cpdef`` functions of the frozen
contract (SPEC §3.5), the ``SplitRule`` enum and the ``*_py`` wrappers of the ``cdef inline``
primitives. Arrays must be C-contiguous: ``float64`` for matrices and costs, ``int64`` for
tours, positions and candidate lists, ``uint8`` for don't-look bits.
"""

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]

class SplitRule(IntEnum):
    SPLIT_GREEDY = 0
    SPLIT_OPTIMAL = 1

# ------------------------------------------------------------------ evaluation
def problem_cost_py(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    max_time: float,
    fixed_cost: float,
    split: int,
) -> float: ...
def trip_starts(
    T: Float64Array,
    tour: Int64Array,
    max_time: float,
    split: int,
    C: Float64Array,
    fixed_cost: float,
    out: Int64Array,
) -> int: ...
def trip_costs(C: Float64Array, tour: Int64Array, starts: Int64Array, out: Float64Array) -> None: ...
def trip_times(T: Float64Array, tour: Int64Array, starts: Int64Array, out: Float64Array) -> None: ...

# ------------------------------------------------------------------ moves
def double_bridge(tour: Int64Array, p1: int, p2: int, p3: int, out: Int64Array) -> None: ...
def rebuild_pos(tour: Int64Array, pos: Int64Array) -> None: ...

# ------------------------------------------------------------------ descents
def two_opt_descent(
    C: Float64Array,
    tour: Int64Array,
    pos: Int64Array,
    cand: Int64Array,
    dont_look: UInt8Array,
    first_improvement: bool,
    max_passes: int,
) -> float: ...
def or_opt_descent(
    C: Float64Array,
    tour: Int64Array,
    pos: Int64Array,
    cand: Int64Array,
    dont_look: UInt8Array,
    max_segment: int,
    allow_reverse: bool,
    max_passes: int,
) -> float: ...
def local_search_generic(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    pos: Int64Array,
    cand: Int64Array,
    max_time: float,
    fixed_cost: float,
    split: int,
    moves: int,
    max_segment: int,
    max_passes: int,
    scratch_tour: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
) -> float: ...

# ------------------------------------------------------------------ construction
def nearest_neighbour_tour(C: Float64Array, depot: int, out: Int64Array) -> None: ...

# ------------------------------------------------------------------ Python wrappers of the inline primitives
def tour_cost_py(C: Float64Array, tour: Int64Array) -> float: ...
def greedy_split_cost_py(
    C: Float64Array, T: Float64Array, tour: Int64Array, max_time: float, fixed_cost: float
) -> float: ...
def optimal_split_cost_py(
    C: Float64Array, T: Float64Array, tour: Int64Array, max_time: float, fixed_cost: float
) -> float: ...
def two_opt_delta_py(C: Float64Array, tour: Int64Array, i: int, j: int) -> float: ...
def two_opt_delta_asym_py(C: Float64Array, tour: Int64Array, i: int, j: int) -> float: ...
def or_opt_delta_py(
    C: Float64Array, tour: Int64Array, i: int, L: int, j: int, reverse: bool = False
) -> float: ...
def swap_delta_py(C: Float64Array, tour: Int64Array, i: int, j: int) -> float: ...
def reverse_segment_py(tour: Int64Array, i: int, j: int) -> None: ...
def reverse_segment_pos_py(tour: Int64Array, pos: Int64Array, i: int, j: int) -> None: ...
def swap_positions_py(tour: Int64Array, i: int, j: int) -> None: ...
def swap_positions_pos_py(tour: Int64Array, pos: Int64Array, i: int, j: int) -> None: ...
def move_segment_py(tour: Int64Array, i: int, L: int, j: int, reverse: bool = False) -> None: ...
def move_segment_pos_py(
    tour: Int64Array, pos: Int64Array, i: int, L: int, j: int, reverse: bool = False
) -> None: ...
