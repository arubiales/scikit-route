"""``NearestNeighbour``: the greedy walk from the depot, over the core kernel (SPEC §4.2)."""

from __future__ import annotations

import numpy as np

from .._core import _routing as core
from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem

__all__ = ["NearestNeighbour"]


class NearestNeighbour(BaseRouter):
    """Nearest-neighbour construction: from the depot, always move to the closest unvisited node.

    The simplest tour builder and the default warm start (``init="nearest_neighbour"``) of every
    iterative solver. Deterministic: ties are broken by the lowest node index.

    Attributes
    ----------
    tour_ : ndarray of shape (n,)
        The open giant tour in label space, depot first.
    route_ : ndarray of shape (n + n_trips,)
        The route as driven: depot, trip 1, depot, trip 2, ..., depot.
    cost_ : float
        Objective of ``route_`` recomputed by the base class (travel cost plus the fixed charge
        per extra trip when a budget was given).
    n_trips_, trips_, trip_costs_, trip_times_, fit_time_, problem_, labels_, depot_, n_nodes_
        The other fitted attributes of :class:`~skroute.base.BaseRouter`.

    Notes
    -----
    Algorithm: start at the depot; while unvisited nodes remain, append the unvisited node ``j``
    with the smallest ``C[current, j]`` (lowest index on ties); the tour closes back to the depot.
    The whole walk runs in the compiled core (:func:`~skroute._core._routing.nearest_neighbour_tour`),
    O(n²) time and O(n) extra memory. The result is typically 25 % (rounded Euclidean instances)
    to 45 % above the optimum: use it as a baseline or a warm start, not as an answer.

    Supports: symmetric and asymmetric matrices (every step reads ``C[current, j]`` directionally);
    multi-trip objective only through the decoder — the search itself ignores ``max_time_work`` and
    a ``UserWarning`` says so (the returned trips still fit the budget). Deterministic.

    References
    ----------
    .. [1] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in *Local Search in Combinatorial Optimization*, Wiley, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.construction import NearestNeighbour
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> nn = NearestNeighbour().fit(C)
    >>> nn.tour_.tolist(), nn.route_.tolist(), nn.cost_
    ([0, 1, 2, 3], [0, 1, 2, 3, 0], 22.0)

    On a national instance the gap to the published optimum is a stable fact (31.8 % on wi29):

    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> nn = NearestNeighbour().fit(wi.distance_matrix(), labels=wi.labels)
    >>> int(nn.route_[0]) == int(nn.route_[-1]) == 1 and nn.cost_ / wi.optimal_tour_length < 1.5
    True
    """

    _parameter_constraints: dict = {}

    def __init__(self) -> None:
        pass

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="construction", budget_aware=False)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        out = np.empty(problem.n, dtype=np.int64)
        core.nearest_neighbour_tour(problem.cost, problem.depot, out)
        return out
