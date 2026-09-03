"""``ClarkeWright``: the parallel savings heuristic, budget-aware (SPEC §4.2)."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["ClarkeWright"]


def _closed_duration(T: np.ndarray, depot: int, path: list[int]) -> float:
    """Duration of the closed trip depot -> path -> depot, summed in driving direction."""
    p = np.asarray(path, dtype=np.int64)
    inner = float(T[p[:-1], p[1:]].sum()) if p.size > 1 else 0.0
    return float(T[depot, p[0]]) + inner + float(T[p[-1], depot])


def savings_tour(
    C: np.ndarray,
    depot: int,
    shape: float = 1.0,
    T: np.ndarray | None = None,
    max_time: float = math.inf,
) -> np.ndarray:
    """Parallel Clarke-Wright savings; returns the giant tour of SPEC §4.2 in index space.

    Every non-depot node starts as its own trip whose *creation index* is the node index; the
    savings ``s_ij = C[d, i] + C[j, d] - shape * C[i, j]`` (``i < j``) are visited in descending
    order (stable, then by ``(i, j)``) and two trips are merged at their endpoints ``i`` and ``j``
    when neither is interior, they belong to different trips and — under a budget — the merged
    closed trip lasts at most ``max_time``. A merged trip keeps the smaller creation index of its
    parts. The giant tour concatenates the trips by increasing creation index, each oriented so
    that its first node is the endpoint nearer to the depot (``C[d, .]``, ties: lower index).

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
    tour : ndarray of shape (n,), int64
        Permutation of ``range(n)`` with ``depot`` first.
    """
    n = C.shape[0]
    d = int(depot)
    nodes = np.delete(np.arange(n, dtype=np.int64), d)
    m = nodes.size
    if m == 1:
        return np.array([d, int(nodes[0])], dtype=np.int64)
    budget = math.isfinite(max_time)
    sym_T = False  # additive path times are exact only when T is symmetric
    if budget:
        if T is None:
            raise ValueError("a finite max_time needs the time matrix T")
        sym_T = bool(np.array_equal(T, T.T))

    # savings in lexicographic (i, j) order, then a stable descending sort -> ties by (i, j)
    ii, jj = np.triu_indices(m, k=1)
    i_all = nodes[ii]
    j_all = nodes[jj]
    s = C[d, i_all] + C[j_all, d] - shape * C[i_all, j_all]
    order = np.argsort(-s, kind="stable")
    i_list = i_all[order].tolist()
    j_list = j_all[order].tolist()

    trips: dict[int, list[int]] = {int(k): [int(k)] for k in nodes}  # creation index -> open path
    trip_of = list(range(n))  # node -> creation index of its trip
    deg = [0] * n  # 0 singleton, 1 endpoint, 2 interior
    path_time: dict[int, float] = dict.fromkeys(trips, 0.0)  # open-path duration, head -> tail

    for a, b in zip(i_list, j_list, strict=True):
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
        merged = A + B
        if budget:
            assert T is not None
            if sym_T:
                dur = (
                    float(T[d, merged[0]])
                    + path_time[ta]
                    + float(T[a, b])
                    + path_time[tb]
                    + float(T[merged[-1], d])
                )
                pt = path_time[ta] + float(T[a, b]) + path_time[tb]
            else:  # reversing a path changes its duration: re-sum in driving direction
                dur = _closed_duration(T, d, merged)
                pt = dur - float(T[d, merged[0]]) - float(T[merged[-1], d])
            if dur > max_time:
                continue
            path_time[min(ta, tb)] = pt
        keep, drop = (ta, tb) if ta < tb else (tb, ta)
        for node in trips[drop]:
            trip_of[node] = keep
        trips[keep] = merged
        del trips[drop]
        if budget:
            path_time.pop(drop, None)
        deg[a] += 1
        deg[b] += 1

    out = [d]
    for idx in sorted(trips):
        p = trips[idx]
        first, last = p[0], p[-1]
        if C[d, last] < C[d, first] or (C[d, last] == C[d, first] and last < first):
            p = p[::-1]
        out.extend(p)
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
        increasing creation index, each oriented with its endpoint nearer to the depot first.
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
    Complexity O(n² log n) time (the sort) and O(n²) memory (the savings).

    The base class then re-decodes the giant tour with the problem's split rule: the greedy
    decoder may merge consecutive savings trips further (never breaking feasibility), so the
    reported ``trips_`` can differ from the savings trips. When the time matrix is asymmetric
    (the cost matrix must be symmetric) the feasibility of a merge is checked in build orientation.

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
