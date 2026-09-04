"""``NRBS``: Node Ranking Based on Stats, the faithful port of the 2020 heuristic (SPEC §4.2)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from numbers import Real

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["NRBS"]

_EXPONENTS = ("mean_priority", "std_priority", "mean_connection", "std_connection", "distance_weight")
_ZERO_DISTANCE = 1e-12  # floor of C[i, j] in the connection score (coincident points, SPEC §4.2)
_ROW_BLOCK = 256  # rows of C whose statistics are computed at once (bounds the n x n temporaries)
_CAND_CHUNK = 64  # candidates converted to Python ints at a time (the passes read a short prefix)


def _pow(x: np.ndarray, e: float) -> np.ndarray:
    """``x ** e`` element-wise through the generic ``pow`` loop, as Python's float ``**`` does.

    Used for the five user exponents (which are usually fractional). ``np.power`` with a *scalar*
    exponent special-cases 0.5 and 2.0; an array-shaped exponent takes the generic loop, so the
    same formula is applied whatever the exponent's value.
    """
    return np.power(x, np.full(x.shape, e))


def row_stats(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean and population standard deviation of ``C``, the zero diagonal included.

    Sums run left to right (``cumsum``) like the 2020 Python loops; the squares and the square
    root use ``np.square``/``np.sqrt``, which are correctly rounded IEEE operations and therefore
    identical on every platform (``pow(x, 2.0)``/``pow(x, 0.5)`` are not: numpy dispatches them to
    SIMD libraries that differ from libm by an ulp). The rows are processed in blocks, so the
    temporaries never hold a second copy of the matrix.
    """
    n = C.shape[0]
    mean = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    with np.errstate(over="ignore"):  # huge costs square to inf: a legitimate IEEE result (see NRBS)
        for start in range(0, n, _ROW_BLOCK):
            block = C[start : start + _ROW_BLOCK]
            mu = np.cumsum(block, axis=1)[:, -1] / n
            var = np.cumsum(np.square(block - mu[:, None]), axis=1)[:, -1] / n
            mean[start : start + _ROW_BLOCK] = mu
            std[start : start + _ROW_BLOCK] = np.sqrt(var)
    return mean, std


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


def _lazy_ints(idx: np.ndarray) -> Iterator[int]:
    """The entries of an index array as Python ints, converted a chunk at a time."""
    for start in range(0, idx.size, _CAND_CHUNK):
        yield from idx[start : start + _CAND_CHUNK].tolist()


def nrbs_tour(
    C: np.ndarray,
    depot: int,
    mean_priority: float = 1.0,
    std_priority: float = 1.0,
    mean_connection: float = 1.0,
    std_connection: float = 1.0,
    distance_weight: float = 1.0,
    on_edge: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """The NRBS construction in index space; see :class:`NRBS` for the algorithm.

    Returns an int64 permutation of ``range(n)`` starting at ``depot`` and continuing towards the
    first neighbour the passes attached to the depot (the direction of the 2020 result). Raises
    ``ValueError`` when a priority or a connection score is NaN (negative costs with a fractional
    exponent, or costs so large that the powers overflow to ``inf / inf``).

    ``on_edge(k, c)`` is called right after every connection ``k -- c`` (index space, ``k`` the node
    being served by the pass, ``c`` its chosen candidate; D31); ``None`` costs nothing. Below three
    nodes no connection is made and it is never called.
    """
    n = C.shape[0]
    d = int(depot)
    if n < 3:  # below the estimator's minimum: the only tour there is
        return np.array([d, *(k for k in range(n) if k != d)], dtype=np.int64)
    a, b, c, e, f = (
        float(v) for v in (mean_priority, std_priority, mean_connection, std_connection, distance_weight)
    )
    mean, std = row_stats(C)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        # priority mu_i^a * sigma_i^b, descending, ties by index (Python's stable reverse sort)
        prio = _pow(mean, a) * _pow(std, b)
        # numerator mu_j^c * sigma_j^e of the connection score of candidate j
        num = _pow(mean, c) * _pow(std, e)
    if np.isnan(prio).any() or np.isnan(num).any():
        raise ValueError(
            "NRBS needs well-defined node statistics: mu^a, sigma^b, mu^c or sigma^e is NaN for some "
            "node (negative costs with a fractional exponent); use costs >= 0 or integer exponents"
        )
    order = np.argsort(-prio, kind="stable")
    order_list = order.tolist()
    num_p = num[order]  # in PRIORITY space, so that score ties keep the priority order (2020 dict)

    def candidates(p: int) -> np.ndarray:
        """Positions (priority space) of the candidates of the node at position ``p``, best first."""
        k = order_list[p]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):  # inf / inf is checked below
            den = _pow(np.maximum(C[k], _ZERO_DISTANCE), f)
            scores = num_p / den[order]
        scores[p] = -np.inf  # a node never connects to itself (dropped explicitly below as well)
        if np.isnan(scores).any():
            raise ValueError(
                f"NRBS connection scores of node {k} overflow to NaN (inf / inf): the costs are too "
                "large for the exponents; scale the cost matrix down"
            )
        return np.argsort(-scores, kind="stable")

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
            for q in _lazy_ints(candidates(p)):
                if q == p:
                    continue
                cnd = order_list[q]
                if deg[cnd] >= 2 or cnd in nbk:
                    continue
                if _find(parent, k) == _find(parent, cnd) and n_edges != n - 1:
                    continue  # would close a cycle before every node is covered
                _connect(nb, deg, parent, k, cnd)
                n_edges += 1
                if on_edge is not None:
                    on_edge(k, cnd)
                break
    if n_edges < n:  # pragma: no cover - every visit of a node with degree < 2 adds an edge (Notes)
        raise RuntimeError("NRBS left the graph open after two passes (bug in the solver)")

    # walk the Hamiltonian cycle from the depot towards its first-attached neighbour
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

        All five exponents must be ``>= 0`` — a restriction new in 2.0: 1.0.0a2 validated only
        the type, so a negative exponent was accepted there and raises ``ValueError`` here. Ints
        are accepted (1.0 rejected them and had no defaults; 1.0's misspelt ``distance_weigth``
        is now ``distance_weight``).

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
    read from the depot towards the neighbour attached to it first.

    Two passes always close the cycle. Before the last edge the graph is a forest of paths, so a
    node with fewer than two neighbours is an endpoint (or isolated): if another path exists, any
    of its endpoints is an eligible candidate (different component, not a neighbour); if the graph
    is already one Hamiltonian path, its other endpoint is eligible because the cycle ban is lifted
    at ``n - 1`` edges. Every visit of a node with degree below two therefore adds an edge, and
    after the second pass every degree is two — one cycle through every node, whatever the ties.

    The scores must be well defined: negative costs with a fractional exponent (``mu^a`` of a
    negative mean) or costs so large that the powers overflow to ``inf / inf`` give NaN scores and
    raise ``ValueError``; infinite scores (coincident points with a large ``distance_weight``) are
    fine and sort first. Complexity O(n² log n) time; the candidate ranking is computed one node at
    a time, so the memory beyond the cost matrix is O(n).

    On Barcelona with all five exponents ``= 0.5`` the tour and its cost reproduce the 1.0 result
    pinned in ``tests/data/nrbs_barcelona_1_0.json``.

    Supports: symmetric and asymmetric matrices (the score reads ``C[i, j]`` directionally; the
    linked cycle is undirected, so ATSP results are heuristic); multi-trip objective only through
    the decoder — the search itself ignores ``max_time_work`` and a ``UserWarning`` says so (the
    returned trips still fit the budget). Deterministic.

    Callback events (D30, D31): ``"start"`` has no tour; then one ``"iteration"`` per connection
    the two passes add — ``n`` events indexed ``1 .. n`` for ``n >= 3`` (the cycle has ``n`` edges;
    none below three nodes) — each with ``tour=None``, ``cost=nan``, ``best_cost=nan``,
    ``extra["edges"]``, the edge set built so far as ``(label, label)`` pairs (the node served by
    the pass first, its chosen candidate second: the graph is undirected), and ``extra["n_edges"]``;
    ``"end"`` carries the tour read from the closed cycle. The trace is built inline and costs O(n)
    per event only when a callback is set; a callback returning ``True`` silences the remaining
    trace events (the passes go on: the result never depends on the callback).

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
        on_edge = None
        if self._callback is not None:
            lab = problem.labels.tolist()
            edges: list[tuple[object, object]] = []

            def on_edge(k: int, c: int) -> None:
                """D31: one event per connection, with the edge set so far (a fresh list per event)."""
                edges.append((lab[k], lab[c]))
                if self._stop_requested:  # the passes go on; only the trace is cut short
                    return
                self._emit(
                    "iteration",
                    len(edges),
                    None,
                    math.nan,
                    None,
                    math.nan,
                    edges=list(edges),
                    n_edges=len(edges),
                )

        return nrbs_tour(
            problem.cost,
            problem.depot,
            self.mean_priority,
            self.std_priority,
            self.mean_connection,
            self.std_connection,
            self.distance_weight,
            on_edge=on_edge,
        )
