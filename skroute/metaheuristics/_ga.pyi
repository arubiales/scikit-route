"""Typing stubs of the Genetic kernels (``skroute.metaheuristics._ga``).

Arrays must be C-contiguous: ``float64`` matrices, costs and uniforms, ``int64`` chromosomes,
tours, positions, candidate lists and pre-drawn indices, ``uint8`` flags, ``uint64`` row hashes.
"""

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]
UInt64Array = NDArray[np.uint64]

def evaluate_population(
    C: Float64Array,
    T: Float64Array,
    pop: Int64Array,
    depot: int,
    max_time: float,
    fixed_cost: float,
    split: int,
    tour: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    out: Float64Array,
) -> None: ...
def ox(p1: Int64Array, p2: Int64Array, a: int, b: int, child: Int64Array, present: UInt8Array) -> None: ...
def pmx(
    p1: Int64Array,
    p2: Int64Array,
    a: int,
    b: int,
    child: Int64Array,
    present: UInt8Array,
    mapping: Int64Array,
) -> None: ...
def mutate(child: Int64Array, kind: int, i: int, j: int) -> None: ...
def polish_tour(
    C: Float64Array,
    T: Float64Array,
    tour: Int64Array,
    pos: Int64Array,
    cand: Int64Array,
    dont_look: UInt8Array,
    max_time: float,
    fixed_cost: float,
    split: int,
    ls_mode: int,
    ls_moves: int,
    scratch_tour: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
) -> float: ...
def ga_generation(
    C: Float64Array,
    T: Float64Array,
    max_time: float,
    fixed_cost: float,
    split: int,
    depot: int,
    pop: Int64Array,
    fit: Float64Array,
    new_pop: Int64Array,
    new_fit: Float64Array,
    elite_idx: Int64Array,
    tourn: Int64Array,
    u_cross: Float64Array,
    cuts: Int64Array,
    u_mut: Float64Array,
    mut: Int64Array,
    remut: Int64Array,
    p_crossover: float,
    p_mutation: float,
    crossover: int,
    mutation: int,
    ls_mode: int,
    ls_moves: int,
    cand: Int64Array,
    par1: Int64Array,
    par2: Int64Array,
    child: Int64Array,
    tour: Int64Array,
    pos: Int64Array,
    dont_look: UInt8Array,
    scratch_tour: Int64Array,
    dp: Float64Array,
    pred: Int64Array,
    present: UInt8Array,
    mapping: Int64Array,
    hashes: UInt64Array,
) -> int: ...
