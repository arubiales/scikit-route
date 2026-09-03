"""``NRBS``: Node Ranking Based on Stats, the faithful port of the 2020 heuristic (SPEC §4.2)."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["NRBS"]

_EXPONENTS = ("mean_priority", "std_priority", "mean_connection", "std_connection", "distance_weight")
_ZERO_DISTANCE = 1e-12  # floor of C[i, j] in the connection score (coincident points, SPEC §4.2)


def _pow(x: np.ndarray, e: float) -> np.ndarray:
    """``x ** e`` element-wise through libm ``pow``, exactly as Python's float ``**`` computes it.

    ``np.power`` with a *scalar* exponent special-cases 0.5 and 2.0 (``sqrt``/``square``), which
    differ from ``pow(x, 0.5)``/``pow(x, 2.0)`` by an ulp on some inputs; an array-shaped exponent
    takes the generic loop and reproduces the 2020 arithmetic bit for bit (verified on 3e5 samples).
    """
    return np.power(x, np.full(x.shape, e))


def row_stats(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean and population standard deviation of ``C``, the zero diagonal included.

    Sums run left to right (``cumsum``), squares and the square root go through ``pow`` — the
    operations of the 2020 Python loops — so the statistics are bit-identical to 1.0's.
    """
    n = C.shape[0]
    mean = np.cumsum(C, axis=1)[:, -1] / n
    var = np.cumsum(_pow(C - mean[:, None], 2.0), axis=1)[:, -1] / n
    return mean, _pow(var, 0.5)


def _find(parent: list[int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:  # path compression
        parent[x], x = root, parent[x]
    return root


def _connect(nb: list[list[int]], deg: list[int], parent: list[int], k: int, c: int) -> None:
    rk, rc = _find(parent, k), _find(parent, c)
    if rk != rc:
        parent[rk] = rc
    nb[k].append(c)
    nb[c].append(k)
    deg[k] += 1
    deg[c] += 1


def join_endpoints(C: np.ndarray, nb: list[list[int]], deg: list[int], parent: list[int]) -> None:
    """Close a degenerate partial graph (paths and isolated nodes) into one Hamiltonian cycle.

    Repeatedly takes the lowest-index node with degree < 2 and links it to the nearest other node
    with degree < 2 (``C[k, c]``, ties: lower index) that is not already its neighbour and does not
    close a cycle before every node is covered. Used only when the two NRBS passes leave the graph
    disconnected (degenerate ties); modifies ``nb``, ``deg`` and ``parent`` in place.
    """
    n = len(deg)
    n_edges = sum(deg) // 2
    while n_edges < n:
        ends = [v for v in range(n) if deg[v] < 2]
        k = ends[0]
        rk = _find(parent, k)
        best, chosen = np.inf, -1
        for c in ends[1:]:
            if c in nb[k] or (_find(parent, c) == rk and n_edges != n - 1):
                continue
            if C[k, c] < best:
                best, chosen = C[k, c], c
        if chosen < 0:  # pragma: no cover - impossible: another component always has an endpoint
            raise RuntimeError("NRBS could not close the tour (bug in the solver)")
        _connect(nb, deg, parent, k, chosen)
        n_edges += 1


def nrbs_tour(
    C: np.ndarray,
    depot: int,
    mean_priority: float = 1.0,
    std_priority: float = 1.0,
    mean_connection: float = 1.0,
    std_connection: float = 1.0,
    distance_weight: float = 1.0,
) -> np.ndarray:
    """The NRBS construction in index space; see :class:`NRBS` for the algorithm.

    Returns an int64 permutation of ``range(n)`` starting at ``depot`` and continuing towards the
    first neighbour the passes attached to the depot (the direction of the 2020 result).
    """
    n = C.shape[0]
    a, b, c, e, f = (
        float(v) for v in (mean_priority, std_priority, mean_connection, std_connection, distance_weight)
    )
    mean, std = row_stats(C)
    # priority mu_i^a * sigma_i^b, descending, ties by index (Python's stable reverse sort)
    prio = _pow(mean, a) * _pow(std, b)
    order = np.argsort(-prio, kind="stable")
    # connection score of candidate j for node i: mu_j^c * sigma_j^e / max(C[i, j], 1e-12)^f, in PRIORITY
    # space so that ties keep the priority order, as the 2020 dict comprehension did
    num = _pow(mean, c) * _pow(std, e)
    den = _pow(np.maximum(C, _ZERO_DISTANCE), f)
    S = num[None, :] / den
    Sp = S[np.ix_(order, order)]
    np.fill_diagonal(Sp, -np.inf)  # a node never connects to itself: sorted last, then dropped
    cand = order[np.argsort(-Sp, axis=1, kind="stable")[:, : n - 1]]
    order_list = order.tolist()
    cand_lists = cand.tolist()

    nb: list[list[int]] = [[] for _ in range(n)]  # neighbours in insertion order
    deg = [0] * n
    parent = list(range(n))  # union-find over the path components
    n_edges = 0
    for _ in range(2):
        for p in range(n):
            k = order_list[p]
            if deg[k] >= 2:
                continue
            nbk = nb[k]
            for cnd in cand_lists[p]:
                if deg[cnd] >= 2 or cnd in nbk:
                    continue
                if _find(parent, k) == _find(parent, cnd) and n_edges != n - 1:
                    continue  # would close a cycle before every node is covered
                _connect(nb, deg, parent, k, cnd)
                n_edges += 1
                break
    if n_edges < n:  # degenerate ties left paths open: close them greedily
        join_endpoints(C, nb, deg, parent)

    # walk the Hamiltonian cycle from the depot towards its first-attached neighbour
    d = int(depot)
    tour = np.empty(n, dtype=np.int64)
    tour[0] = d
    prev, cur = d, nb[d][0]
    for k in range(1, n):
        tour[k] = cur
        nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
        prev, cur = cur, nxt
    return tour


class NRBS(BaseRouter):
    """Node Ranking Based on Stats: rank nodes by the statistics of their cost row, then link them.

    The construction heuristic of scikit-route 1.0 (2020), ported faithfully. Every node gets a
    *priority* from the mean and standard deviation of its row of ``C`` and a ranked list of
    *connection candidates*; two passes over the priority order give each node two neighbours
    without closing a premature cycle, and the resulting Hamiltonian cycle is the tour.

    Parameters
    ----------
    mean_priority : float, default 1.0
        Exponent ``a`` of the row mean in the priority ``mu_i^a * sigma_i^b``. Larger values rank
        remote nodes (large mean distance) earlier.
    std_priority : float, default 1.0
        Exponent ``b`` of the row standard deviation in the priority.
    mean_connection : float, default 1.0
        Exponent ``c`` of the candidate's row mean in the connection score
        ``mu_j^c * sigma_j^e / C[i, j]^f``.
    std_connection : float, default 1.0
        Exponent ``e`` of the candidate's row standard deviation in the connection score.
    distance_weight : float, default 1.0
        Exponent ``f`` of the distance in the connection score; larger values make the score
        closer to plain nearest-neighbour linking.

    All five must be ``>= 0``; ints are accepted (1.0 rejected them and had no defaults; 1.0's
    misspelt ``distance_weigth`` is now ``distance_weight``).

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
    Algorithm (the 2020 heuristic, with its arithmetic): ``mu_i`` and ``sigma_i`` are the mean and
    the population standard deviation of the full row ``C[i, :]``, zero diagonal included. The
    priority ``mu_i^a * sigma_i^b`` is sorted descending (ties by node index). The connection
    score of candidate ``j`` for node ``i`` is ``mu_j^c * sigma_j^e / max(C[i, j], 1e-12)^f``,
    sorted descending per node (ties by priority order); the depot participates like any node and
    coincident points (``C[i, j] == 0``) get the maximum score and are linked first. Two passes over
    the priority order: for each node with fewer than two neighbours, link it to its highest-scoring
    candidate that has fewer than two neighbours, is not already its neighbour and does not close a
    cycle before all nodes are covered — cycle detection by union-find plus degree counters (the
    2020 code deep-copied the graph per candidate; the selection order is unchanged). The cycle is
    read from the depot towards the neighbour attached to it first. If degenerate ties leave paths
    open after the two passes, the remaining endpoints are joined greedily by nearest endpoint.
    Complexity O(n² log n) time and O(n²) memory.

    On Barcelona with all five exponents ``= 0.5`` the tour and its cost reproduce the 1.0 result
    pinned in ``tests/data/nrbs_barcelona_1_0.json``.

    Supports: symmetric and asymmetric matrices (the score reads ``C[i, j]`` directionally; the
    linked cycle is undirected, so ATSP results are heuristic); multi-trip objective only through
    the decoder — the search itself ignores ``max_time_work`` and a ``UserWarning`` says so (the
    returned trips still fit the budget). Deterministic.

    References
    ----------
    .. [1] A. Rubiales, scikit-route 1.0.0a2, ``skroute.heuristics.NRBS`` (2020).

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.construction import NRBS
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> est = NRBS().fit(C)
    >>> sorted(est.tour_.tolist()) == [0, 1, 2, 3] and int(est.route_[0]) == int(est.route_[-1]) == 0
    True
    >>> est.cost_
    22.0

    The 1.0 regression: Barcelona with every exponent at 0.5 (``depot=`` is a label):

    >>> from skroute.datasets import load_barcelona
    >>> bcn = load_barcelona()
    >>> est = NRBS(0.5, 0.5, 0.5, 0.5, 0.5).fit(bcn.cost, labels=bcn.labels, depot=bcn.depot)
    >>> int(est.route_[0]) == int(est.route_[-1]) == 10000007 and est.n_nodes_ == 19
    True
    """

    _parameter_constraints: dict = {name: [Interval(Real, 0.0, None, closed="left")] for name in _EXPONENTS}

    def __init__(
        self,
        mean_priority: float = 1.0,
        std_priority: float = 1.0,
        mean_connection: float = 1.0,
        std_connection: float = 1.0,
        distance_weight: float = 1.0,
    ) -> None:
        self.mean_priority = mean_priority
        self.std_priority = std_priority
        self.mean_connection = mean_connection
        self.std_connection = std_connection
        self.distance_weight = distance_weight

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="construction", budget_aware=False)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return nrbs_tour(
            problem.cost,
            problem.depot,
            self.mean_priority,
            self.std_priority,
            self.mean_connection,
            self.std_connection,
            self.distance_weight,
        )
