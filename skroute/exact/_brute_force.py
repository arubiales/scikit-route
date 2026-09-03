"""``BruteForce``: exhaustive enumeration of every giant tour (SPEC §4.1)."""

from __future__ import annotations

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval
from . import _brute

__all__ = ["BruteForce"]


class BruteForce(BaseRouter):
    """Exact solver by exhaustive enumeration of the ``(n - 1)!`` giant tours.

    Every permutation of the non-depot nodes is priced with the problem's own decoder, so
    the optimum is certified for the plain tour **and** for the multi-trip objective under
    the chosen split rule (``fit(..., split="greedy" | "optimal")``). It is the reference
    every other solver is tested against on small instances.

    Parameters
    ----------
    max_nodes : int, default 11
        Hard cap on the number of nodes (``fit`` raises ``ValueError`` above it). The work
        grows as ``(n - 1)!``: ``10!`` at n = 11 is 3.6 million tours, a few seconds at most;
        n = 12 is eleven times more. Raise it only if you accept the time.

    Attributes
    ----------
    is_optimal_ : bool
        Always ``True``: the search is exhaustive.

    Notes
    -----
    The permutations of positions ``1..n-1`` are visited in **lexicographic order** with
    Knuth's *next permutation* (TAOCP 7.2.1.2, Algorithm L), inside a ``nogil`` loop, and the
    first strictly better tour is kept, so a tie resolves to the lexicographically first
    optimum — the same tour ``itertools.permutations`` would find first. On a symmetric
    matrix without a budget the reversed tour costs the same, so permutations with
    ``tour[1] > tour[n-1]`` are skipped and the kept orientation is the lexicographically
    smaller one; under a budget every orientation is priced because the split depends on
    direction.

    Under ``split="greedy"`` the result is exact over greedy-decoded giant tours (a partition
    that closes a trip while the next node would still fit is not representable); under
    ``split="optimal"`` it is exact for the distance-constrained multi-trip problem.

    Complexity O((n - 1)! · n) time (halved when symmetric and unbudgeted), O(n) memory.

    Supports: symmetric and asymmetric matrices, multi-trip objective (both split rules);
    deterministic.

    References
    ----------
    .. [1] D. E. Knuth, *The Art of Computer Programming*, Vol. 4A, §7.2.1.2 "Generating all
       permutations", Algorithm L.

    Examples
    --------
    A plain TSP from numpy; the optimum is certified:

    >>> import numpy as np
    >>> from skroute import BruteForce
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> bf = BruteForce().fit(C)
    >>> bf.tour_.tolist(), bf.cost_, bf.is_optimal_
    ([0, 1, 2, 3], 22.0, True)

    The multi-trip objective from a dict-of-dicts (the depot is the first key): a 4-hour
    working day and a 3.0 charge per extra trip.

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
    >>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
    >>> bf.route_.tolist(), bf.cost_, bf.n_trips_
    ([1, 2, 3, 1, 4, 1], 41.0, 2)
    >>> bf.trip_times_.tolist()
    [4.0, 4.0]
    """

    _parameter_constraints = {"max_nodes": [Interval(int, 3, None, closed="left")]}

    def __init__(self, max_nodes: int = 11) -> None:
        self.max_nodes = max_nodes

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="exact", exact=True, budget_aware=True, max_nodes=self.max_nodes)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        n = problem.n
        others = np.delete(np.arange(n, dtype=np.int64), problem.depot)
        tour = np.concatenate(([problem.depot], others)).astype(np.int64)  # the first permutation
        best = tour.copy()
        halve = problem.symmetric and not problem.multi_trip
        optimal_split = problem.multi_trip and problem.split == "optimal"
        scratch = n if optimal_split else 0
        dp = np.empty(scratch, dtype=np.float64)
        pred = np.empty(scratch, dtype=np.int64)
        _brute.brute_force_search(
            problem.cost,
            problem.time_or_cost,
            tour,
            best,
            problem.max_time_work,
            problem.fixed_cost,
            problem.split_code,
            halve,
            dp,
            pred,
        )
        self.is_optimal_ = True
        return best
