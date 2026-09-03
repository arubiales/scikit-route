"""``ClarkeWright``: the parallel savings heuristic, budget-aware (SPEC §4.2)."""

from __future__ import annotations

import math
from itertools import pairwise
from numbers import Real

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["ClarkeWright"]

_CHUNK = 1 << 15  # savings pairs converted to Python ints at a time (bounds the Python-level memory)


def _closed_duration(T: np.ndarray, depot: int, path: list[int]) -> float:
    """Duration of the closed trip ``depot -> path -> depot`` with the greedy decoder's arithmetic.

    The legs are accumulated one by one in driving direction from the depot, exactly as the core
    decoder does (``t = T[d, p0]; t += T[a, b]; ...; t + T[p_last, d]``), so a trip accepted here is
    accepted by the decoder even when the budget equals its duration to the last ulp.
    """
    t = float(T[depot, path[0]])
    for a, b in pairwise(path):
        t += float(T[a, b])
    return t + float(T[path[-1], depot])


def _oriented(C: np.ndarray, depot: int, path: list[int]) -> list[int]:
    """The path with its endpoint nearer to the depot first (``C[d, .]``; ties: the lower index)."""
    first, last = path[0], path[-1]
    if C[depot, last] < C[depot, first] or (C[depot, last] == C[depot, first] and last < first):
        return path[::-1]
    return path


def savings_trips(
    C: np.ndarray,
    depot: int,
    shape: float = 1.0,
    T: np.ndarray | None = None,
    max_time: float = math.inf,
) -> list[list[int]]:
    """Parallel Clarke-Wright savings; returns the savings trips of SPEC §4.2 in index space.

    Every non-depot node starts as its own trip whose *creation index* is the node index; the
    savings ``s_ij = C[d, i] + C[j, d] - shape * C[i, j]`` (``i < j``) are visited in descending
    order (stable, then by ``(i, j)``) and two trips are merged at their endpoints ``i`` and ``j``
    when neither is interior, they belong to different trips and — under a budget — the merged
    closed trip fits ``max_time`` as it will be driven. A merged trip keeps the smaller creation
    index of its parts. Each trip is oriented so that its first node is the endpoint nearer to the
    depot (``C[d, .]``, ties: lower index); when only the reverse orientation fits the budget
    (possible with an asymmetric time matrix) the reverse is kept instead, so every trip returned
    fits the budget in the direction in which it is driven, summed with the decoder's arithmetic.

    Parameters
    ----------
    C : ndarray of shape (n, n), float64
        Symmetric cost matrix (the estimator enforces the symmetry).
    depot : int
        Depot index.
    shape : float, default 1.0
        Savings weight ``lambda``.
    T : ndarray of shape (n, n) or None
        Time matrix; required when ``max_time`` is finite.
    max_time : float, default inf
        Per-trip budget; ``inf`` means plain TSP (everything merges into one trip).

    Returns
    -------
    trips : list of list of int
        The savings trips by increasing creation index, each oriented as it will be driven; their
        concatenation after the depot is the giant tour of :func:`savings_tour`.
    """
    n = C.shape[0]
    d = int(depot)
    nodes = np.delete(np.arange(n, dtype=np.int64), d)
    m = nodes.size
    if m == 1:
        return [[int(nodes[0])]]
    budget = math.isfinite(max_time)
    if budget and T is None:
        raise ValueError("a finite max_time needs the time matrix T")

    # savings in lexicographic (i, j) order, then a stable descending sort -> ties by (i, j)
    ii, jj = np.triu_indices(m, k=1)
    nodes32 = nodes.astype(np.int32)
    i_all = nodes32[ii]
    j_all = nodes32[jj]
    del ii, jj
    s = C[d, i_all] + C[j_all, d] - shape * C[i_all, j_all]
    order = np.argsort(-s, kind="stable")
    del s

    trips: dict[int, list[int]] = {int(k): [int(k)] for k in nodes}  # creation index -> oriented path
    trip_of = list(range(n))  # node -> creation index of its trip
    deg = bytearray(n)  # 0 singleton, 1 endpoint, 2 interior
    deg_view = np.frombuffer(deg, dtype=np.uint8)  # live view: vectorised pre-filter of each chunk

    # the pairs are visited in savings order, one chunk of Python ints at a time; a pair with an
    # interior endpoint at the start of its chunk stays interior (degrees only grow), so dropping
    # those pairs before the conversion is exactly the per-pair test of the rule
    for start in range(0, order.size, _CHUNK):
        sel = order[start : start + _CHUNK]
        ic, jc = i_all[sel], j_all[sel]
        keep_pairs = (deg_view[ic] < 2) & (deg_view[jc] < 2)
        for a, b in zip(ic[keep_pairs].tolist(), jc[keep_pairs].tolist(), strict=True):
            if deg[a] == 2 or deg[b] == 2:
                continue
            ta, tb = trip_of[a], trip_of[b]
            if ta == tb:
                continue
            A, B = trips[ta], trips[tb]
            if A[-1] != a:  # a must close A ...
                A = A[::-1]
            if B[0] != b:  # ... and b must open B
                B = B[::-1]
            merged = _oriented(C, d, A + B)
            if budget:
                assert T is not None
                if _closed_duration(T, d, merged) > max_time:
                    merged = merged[::-1]  # asymmetric T: the other direction may still fit
                    if _closed_duration(T, d, merged) > max_time:
                        continue
            keep, drop = (ta, tb) if ta < tb else (tb, ta)
            for node in trips[drop]:
                trip_of[node] = keep
            trips[keep] = merged
            del trips[drop]
            deg[a] += 1
            deg[b] += 1

    return [trips[idx] for idx in sorted(trips)]


def savings_tour(
    C: np.ndarray,
    depot: int,
    shape: float = 1.0,
    T: np.ndarray | None = None,
    max_time: float = math.inf,
) -> np.ndarray:
    """The giant tour of SPEC §4.2: the depot followed by the :func:`savings_trips` in order.

    Returns
    -------
    tour : ndarray of shape (n,), int64
        Permutation of ``range(n)`` with ``depot`` first.
    """
    out = [int(depot)]
    for trip in savings_trips(C, depot, shape, T, max_time):
        out.extend(trip)
    return np.asarray(out, dtype=np.int64)


class ClarkeWright(BaseRouter):
    """Clarke-Wright parallel savings: merge the pairs of trips that save the most, budget included.

    Every customer starts as its own out-and-back trip; the *saving* of serving ``i`` and ``j``
    consecutively is ``s_ij = C[d, i] + C[j, d] - shape * C[i, j]``. Savings are visited in
    descending order and two trips are merged at their endpoints whenever the merged trip is still
    feasible. Without a budget everything merges into one tour (the greedy-edge heuristic); with
    ``max_time_work`` the merge is refused when the merged closed trip would not fit, so the
    search itself sees the multi-trip objective (``budget_aware=True``).

    Parameters
    ----------
    shape : float, default 1.0
        Savings weight ``lambda`` in ``s_ij = C[d, i] + C[j, d] - shape * C[i, j]`` (Gaskell 1967,
        Yellow 1970). ``1.0`` is the classical rule; larger values favour short inter-customer
        legs, smaller values favour customers far from the depot. Must be ``>= 0``.

    Attributes
    ----------
    tour_ : ndarray of shape (n,)
        The open giant tour in label space, depot first: the savings trips concatenated by
        increasing creation index, each oriented with its endpoint nearer to the depot first
        (unless only the other direction fits the budget, which needs an asymmetric time matrix).
    route_ : ndarray of shape (n + n_trips,)
        The route as driven: depot, trip 1, depot, trip 2, ..., depot.
    cost_ : float
        Objective of ``route_`` recomputed by the base class.
    n_trips_, trips_, trip_costs_, trip_times_, fit_time_, problem_, labels_, depot_, n_nodes_
        The other fitted attributes of :class:`~skroute.base.BaseRouter`.

    Notes
    -----
    Algorithm: ``n-1`` singleton trips with creation index = node index; the ``(n-1)(n-2)/2``
    savings sorted descending (stable, then by ``(i, j)``); for each pair, merge when neither node
    is interior to its trip, the trips differ and — under a budget — the merged closed trip's
    duration is ``<= max_time_work``; the merged trip keeps the smaller creation index. The giant
    tour is the concatenation of the trips by increasing creation index, each oriented so that
    its first node is the endpoint with the smaller ``C[d, .]`` (ties: the smaller node index).
    Complexity O(n² log n) time (the sort) and O(n²) memory (the savings, as numpy arrays; the
    Python-level state is O(n)).

    The feasibility of a merge is decided on the trip *as it will be driven*: the merged path is
    oriented first and its duration is accumulated leg by leg from the depot, with the arithmetic
    of the greedy decoder, so a merge accepted here is never undone by rounding. When the time
    matrix is asymmetric (the cost matrix must be symmetric) the two directions of a trip last
    differently; if only the reverse of the preferred orientation fits, the trip is kept reversed
    rather than refused. Every savings trip therefore fits the budget in driving direction.

    The base class then re-decodes the giant tour with the problem's split rule. When the time
    matrix satisfies the triangle inequality, the greedy decoder can only merge consecutive
    savings trips further, so ``n_trips_ <=`` the number of savings trips; when it violates it
    (road matrices occasionally do) the decoder may also split a trip whose prefix cannot return
    to the depot in time, so ``n_trips_`` may exceed the number of savings trips. In both cases
    the reported ``trips_`` can differ from the savings trips and every trip fits the budget.

    Supports: symmetric matrices only (an asymmetric ``X`` raises ``ValueError``); multi-trip
    objective inside the search. Deterministic.

    References
    ----------
    .. [1] G. Clarke and J. W. Wright, "Scheduling of vehicles from a central depot to a number
       of delivery points", Operations Research 12(4), 1964, 568-581.
    .. [2] T. J. Gaskell, "Bases for vehicle fleet scheduling", Operational Research Quarterly
       18(3), 1967, 281-295.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.construction import ClarkeWright
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> cw = ClarkeWright().fit(C)
    >>> cw.tour_.tolist(), cw.cost_, cw.n_trips_
    ([0, 1, 2, 3], 22.0, 1)

    Under a budget the merges that would not fit are refused (the example of the base class:
    4-hour trips, 3 EUR per extra trip):

    >>> cost = {
    ...     1: {1: 0, 2: 5, 3: 9, 4: 10},
    ...     2: {1: 5, 2: 0, 3: 4, 4: 8},
    ...     3: {1: 9, 2: 4, 3: 0, 4: 3},
    ...     4: {1: 10, 2: 8, 3: 3, 4: 0},
    ... }
    >>> hours = {
    ...     1: {1: 0, 2: 1, 3: 2, 4: 2},
    ...     2: {1: 1, 2: 0, 3: 1, 4: 2},
    ...     3: {1: 2, 2: 1, 3: 0, 4: 1},
    ...     4: {1: 2, 2: 2, 3: 1, 4: 0},
    ... }
    >>> cw = ClarkeWright().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
    >>> cw.route_.tolist(), cw.cost_, cw.n_trips_
    ([1, 2, 3, 1, 4, 1], 41.0, 2)
    >>> bool(np.all(cw.trip_times_ <= 4.0))
    True
    """

    _parameter_constraints: dict = {"shape": [Interval(Real, 0.0, None, closed="left")]}

    def __init__(self, shape: float = 1.0) -> None:
        self.shape = shape

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="construction", requires_symmetric=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return savings_tour(
            problem.cost, problem.depot, float(self.shape), T=problem.time, max_time=problem.max_time_work
        )
