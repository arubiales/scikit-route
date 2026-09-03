"""``Insertion``: farthest, cheapest and nearest insertion over the ``_insert`` kernel (SPEC §4.2)."""

from __future__ import annotations

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Options
from ._insert import insertion_tour

__all__ = ["Insertion"]


class Insertion(BaseRouter):
    """Insertion construction: grow the tour one node at a time, each at its cheapest position.

    Start with the depot and one seed node, then repeatedly pick an unrouted node by the
    ``strategy`` and insert it between the two consecutive tour nodes ``a -> b`` that minimise
    ``C[a, j] + C[j, b] - C[a, b]``. There are no ``FarthestInsertion``/``CheapestInsertion``
    classes: the strategy is a parameter.

    Parameters
    ----------
    strategy : {"farthest", "cheapest", "nearest"}, default "farthest"
        Which unrouted node is inserted next.

        - ``"farthest"``: the node whose distance to the partial tour is largest (seed: the node
          farthest from the depot). Fixes the global shape first; the best of the three on
          Euclidean instances.
        - ``"cheapest"``: the node with the smallest insertion cost over every tour edge (seed: the
          node nearest to the depot).
        - ``"nearest"``: the node whose distance to the partial tour is smallest (seed: the node
          nearest to the depot).

        The distance from the partial tour to a node ``j`` is ``min(C[i, j])`` over the routed
        nodes ``i``.

    Attributes
    ----------
    tour_ : ndarray of shape (n,)
        The open giant tour in label space, depot first.
    route_ : ndarray of shape (n + n_trips,)
        The route as driven: depot, trip 1, depot, trip 2, ..., depot.
    cost_ : float
        Objective of ``route_`` recomputed by the base class.
    n_trips_, trips_, trip_costs_, trip_times_, fit_time_, problem_, labels_, depot_, n_nodes_
        The other fitted attributes of :class:`~skroute.base.BaseRouter`.

    Notes
    -----
    Algorithm (compiled kernel, typed memoryviews, no Python in the loop): the partial tour is a
    linked list closed on the depot; ``min_dist[j]`` (distance from the tour to every unrouted
    node) is updated incrementally after each insertion; the cheapest strategy additionally caches
    each unrouted node's best edge and rescans only the nodes whose cached edge was split.
    Complexity O(n²) for farthest and nearest; O(n²) in practice for cheapest (O(n³) in the
    adversarial worst case, when every unrouted node keeps preferring the edge that was just
    split). Deterministic: selection ties go to the lowest node index, position ties to the first
    edge met walking the tour from the depot.

    Every insertion cost is evaluated with the arcs in driving direction, so the heuristic is
    exact on asymmetric matrices (ATSP): nothing is reversed, nothing assumes ``C == C.T``.

    Supports: symmetric and asymmetric matrices; multi-trip objective only through the decoder —
    the search itself ignores ``max_time_work`` and a ``UserWarning`` says so (the returned trips
    still fit the budget). Deterministic.

    References
    ----------
    .. [1] D. J. Rosenkrantz, R. E. Stearns and P. M. Lewis, "An analysis of several heuristics
       for the traveling salesman problem", SIAM Journal on Computing 6(3), 1977, 563-581.
    .. [2] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in *Local Search in Combinatorial Optimization*, Wiley, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.construction import Insertion
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> far = Insertion().fit(C)  # strategy="farthest"
    >>> far.tour_.tolist(), far.cost_
    ([0, 1, 2, 3], 22.0)
    >>> cheap = Insertion(strategy="cheapest").fit(C)
    >>> cheap.tour_.tolist(), cheap.cost_  # the same cycle, driven the other way round
    ([0, 3, 2, 1], 22.0)

    Farthest insertion stays within the 25 % tolerance of the test-suite on the national instances:

    >>> from skroute.datasets import load_tsp
    >>> dj = load_tsp("dj38")
    >>> est = Insertion().fit(dj.distance_matrix(), labels=dj.labels)
    >>> est.cost_ / dj.optimal_tour_length < 1.25
    True
    """

    _parameter_constraints: dict = {"strategy": [Options(str, {"farthest", "cheapest", "nearest"})]}

    def __init__(self, strategy: str = "farthest") -> None:
        self.strategy = strategy

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="construction", budget_aware=False)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return insertion_tour(problem.cost, problem.depot, self.strategy)
