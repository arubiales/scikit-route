"""``TwoOpt``: the 2-opt descent as an estimator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval, Options
from ._local_search import run_descent

__all__ = ["TwoOpt"]


class TwoOpt(BaseRouter):
    """2-opt descent: repeatedly reverse a tour segment while that shortens the tour.

    Starts from ``init`` and applies Bentley's neighbour-list 2-opt with don't-look bits:
    for every active node ``a`` and each of its two tour edges, the candidates ``c`` of ``a``
    are scanned in ascending order of ``C[a, c]`` while ``C[a, c]`` is below the edge being
    removed, and the reversal that creates the edge ``(a, c)`` is applied when it improves.

    Parameters
    ----------
    first_improvement : bool, default True
        Apply the first improving move found for a node (``True``) or scan the node's whole
        candidate neighbourhood and apply the best (``False``). Honoured only on the
        symmetric plain-TSP path; the generic path (asymmetric or multi-trip) is always
        first-improvement.
    init : {"nearest_neighbour"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the nearest-neighbour construction from the depot, or the ``tour_``/
        ``route_`` of another solver (labels; the depot may repeat).
    n_candidates : int or None, default 10
        Size of the candidate lists (the ``k`` nearest neighbours of every node); ``None`` scans
        the full neighbourhood (``n - 1`` nodes), a markedly better local optimum at small ``n``
        for a higher cost per pass.
    max_passes : int, default 50
        Maximum number of outer iterations (one sweep over the active nodes each).
    verbose : int, default 0
        0 is silent; 1 logs every ``max(1, max_passes // 10)`` iterations and the stop; 2 logs
        every iteration. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Cost after each outer iteration (non-increasing, never above the cost of ``init``).
    n_iter_ : int
        Outer iterations run.
    stop_reason_ : {"converged", "max_iter", "callback"}
        ``"converged"`` when a sweep that started with every node active changed nothing,
        ``"max_iter"`` after ``max_passes`` sweeps, ``"callback"`` when the ``callback`` of
        ``fit`` returned ``True``.

    Notes
    -----
    Supports: symmetric and asymmetric matrices, multi-trip objective; deterministic.

    Callback events (D30): ``"start"`` carries the ``init`` tour with ``extra["moves"]`` (the
    listed descents, ``["two_opt"]``); each ``"iteration"`` event's ``tour`` is the working tour
    (also the best), with ``extra["moves_applied"]`` equal to ``["two_opt"]`` when the sweep
    changed the tour (``[]`` otherwise) and ``extra["gain"]`` the sweep's cost change.

    One outer iteration = one call of the core's ``two_opt_descent`` with ``max_passes=1``
    (SPEC §4.3); the ``pos``, candidate and don't-look-bit buffers persist across iterations.
    On a symmetric plain TSP a move is priced in O(1) and one sweep costs O(n k) evaluations
    plus O(n) per applied reversal; asymmetric matrices and the multi-trip objective use the
    full-evaluation kernel (``local_search_generic`` with the 2-opt mask), O(n) per candidate
    move. Don't-look bits skip nodes scanned earlier, so the first sweep that changes nothing
    re-activates every node and the descent converges only when the following full sweep
    changes nothing either: a converged tour admits no improving 2-opt move among the
    candidate edges (every 2-opt move with ``n_candidates=None``). The result is never worse
    than ``init``.

    References
    ----------
    .. [1] G. A. Croes, "A method for solving traveling-salesman problems", Operations
       Research 6 (1958) 791-812.
    .. [2] J. L. Bentley, "Fast algorithms for geometric traveling salesman problems", ORSA
       Journal on Computing 4 (1992) 387-411.
    .. [3] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in *Local Search in Combinatorial Optimization*, Wiley, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import TwoOpt
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> C = wi.distance_matrix()
    >>> two = TwoOpt().fit(C, labels=wi.labels)
    >>> two.cost_ / wi.optimal_tour_length < 1.20  # the fast-tier tolerance
    True
    >>> two.stop_reason_, two.n_iter_ == len(two.history_), bool(np.all(np.diff(two.history_) <= 0))
    ('converged', True, True)
    >>> full = TwoOpt(n_candidates=None).fit(C, labels=wi.labels)  # full neighbourhood
    >>> full.cost_ <= two.cost_
    True
    >>> TwoOpt(first_improvement=False, max_passes=10)
    TwoOpt(first_improvement=False, max_passes=10)
    """

    _parameter_constraints: dict[str, Any] = {
        "first_improvement": ["boolean"],
        "init": [Options(str, {"nearest_neighbour"}), "array-like"],
        "n_candidates": [Interval(int, 1, None, closed="left"), None],
        "max_passes": [Interval(int, 1, None, closed="left")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        first_improvement: bool = True,
        init: Any = "nearest_neighbour",
        n_candidates: int | None = 10,
        max_passes: int = 50,
        verbose: int = 0,
    ) -> None:
        self.first_improvement = first_improvement
        self.init = init
        self.n_candidates = n_candidates
        self.max_passes = max_passes
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="local_search", iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return run_descent(self, problem, ("two_opt",), first_improvement=self.first_improvement)
