"""``HeldKarp``: bitmask dynamic programming over subsets (SPEC §4.1, P1)."""

from __future__ import annotations

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval
from . import _hk

__all__ = ["HeldKarp"]


class HeldKarp(BaseRouter):
    """Exact TSP solver by the Held-Karp dynamic programme over subsets of nodes.

    ``dp[S][j]`` is the cheapest path from the depot through exactly the node set ``S``
    ending at ``j``; the optimum is the best ``dp[all][j] + C[j, depot]``. Every arc is read
    directionally, so asymmetric matrices are solved exactly. The plain tour is the only
    objective it certifies: under a budget it raises (D6) — use `BruteForce` for a
    certified multi-trip optimum, or a heuristic solver.

    Parameters
    ----------
    max_nodes : int, default 20
        Hard cap on the number of nodes (``fit`` raises ``ValueError`` above it). Time is
        O(2^(n-1) · n²) and memory ``2^(n-1) · (n-1)`` doubles plus as many ``int8`` parent
        bytes: about 90 MB (80 MB of doubles) and well under a second at n = 20; ~830 MB at
        n = 23 (738 MB of doubles plus a 92 MB parent table). Raise it only if you accept
        the memory; `MILP` solves these sizes instantly in any case.

    Attributes
    ----------
    is_optimal_ : bool
        Always ``True``: the programme is exhaustive over subsets.

    Notes
    -----
    Push formulation in a ``nogil`` loop over the subsets in increasing numeric order (every
    proper subset of ``S`` is numerically smaller, so ``dp[S]`` is final when expanded);
    ``malloc``ed tables freed before returning; the parent table is ``int8``. Ties are broken
    by node index at each expansion, not lexicographically over whole tours as
    `BruteForce` does — equal-cost optima may come out in a different order.

    Complexity O(2^(n-1) · n²) time, O(2^(n-1) · n) memory.

    Supports: symmetric and asymmetric matrices; plain TSP only (raises under a budget);
    deterministic.

    References
    ----------
    .. [1] M. Held and R. M. Karp, "A dynamic programming approach to sequencing problems",
       Journal of the Society for Industrial and Applied Mathematics 10 (1962) 196-210.
    .. [2] R. Bellman, "Dynamic programming treatment of the travelling salesman problem",
       Journal of the ACM 9 (1962) 61-63.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import HeldKarp
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> hk = HeldKarp().fit(C)
    >>> hk.cost_, hk.is_optimal_
    (22.0, True)
    >>> int(hk.route_[0]) == int(hk.route_[-1]) == 0 and sorted(hk.tour_.tolist()) == [0, 1, 2, 3]
    True

    An asymmetric matrix is read directionally (the reversed tour costs 25 here):

    >>> A = np.array([[0, 1, 9, 9], [9, 0, 1, 9], [9, 9, 0, 1], [1, 9, 9, 0]], dtype=float)
    >>> HeldKarp().fit(A).tour_.tolist()
    [0, 1, 2, 3]
    """

    _parameter_constraints = {"max_nodes": [Interval(int, 3, None, closed="left")]}

    def __init__(self, max_nodes: int = 20) -> None:
        self.max_nodes = max_nodes

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="exact", exact=True, budget_aware=False, max_nodes=self.max_nodes)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        others = np.delete(np.arange(problem.n, dtype=np.int64), problem.depot)
        out = np.empty(problem.n, dtype=np.int64)
        _hk.held_karp_search(problem.cost, others, problem.depot, out)
        self.is_optimal_ = True
        return out
