"""``OrOpt``: the Or-opt segment-relocation descent as an estimator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval, Options
from ._local_search import run_descent

__all__ = ["OrOpt"]


class OrOpt(BaseRouter):
    """Or-opt descent: relocate segments of one to ``max_segment`` consecutive nodes.

    Starts from ``init`` and, for every active node, tries to move each segment starting there
    (lengths ``1..max_segment``) next to one of the candidates of either segment end — after it
    or before it, forward or (on symmetric matrices) reversed — applying the first relocation
    that shortens the tour. Candidate lists and don't-look bits keep a sweep at O(n k).

    Parameters
    ----------
    max_segment : int, default 3
        Longest segment relocated (Or-opt proper uses 3).
    init : {"nearest_neighbour"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the nearest-neighbour construction from the depot, or the ``tour_``/
        ``route_`` of another solver (labels; the depot may repeat).
    n_candidates : int or None, default 10
        Size of the candidate lists (the ``k`` nearest neighbours of every node); ``None`` scans
        the full neighbourhood (``n - 1`` nodes).
    max_passes : int, default 50
        Maximum number of outer iterations (one sweep over the active nodes each).
    verbose : int, default 0
        0 is silent; 1 logs every ``max(1, max_passes // 10)`` iterations and the stop; 2 logs
        every iteration. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Cost after each outer iteration (non-increasing).
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
    listed descents, ``["or_opt"]``); each ``"iteration"`` event's ``tour`` is the working tour
    (also the best), with ``extra["moves_applied"]`` equal to ``["or_opt"]`` when the sweep
    changed the tour (``[]`` otherwise) and ``extra["gain"]`` the sweep's cost change.

    One outer iteration = one call of the core's ``or_opt_descent`` with ``max_passes=1``
    (SPEC §4.3); the ``pos``, candidate and don't-look-bit buffers persist across iterations.
    On a symmetric plain TSP both orientations of the moved segment are tried with O(1) deltas
    and a segment is re-inserted only next to a candidate closer than the edge removed at that
    end (Bentley's pruning, a heuristic: a relocation whose whole gain comes from closing the
    gap it leaves is not generated); the first sweep that changes nothing re-activates every
    node and the descent converges only when the following full sweep changes nothing either.
    Asymmetric matrices and the multi-trip objective use the full-evaluation kernel
    (``local_search_generic`` with the Or-opt mask, no reversal), O(n) per candidate move,
    which enumerates every forward relocation next to a candidate. Or-opt alone leaves
    crossing edges that 2-opt would remove: for a plain descent prefer
    `LocalSearch`, which alternates both.

    References
    ----------
    .. [1] I. Or, "Traveling salesman-type combinatorial problems and their relation to the
       logistics of regional blood banking", PhD thesis, Northwestern University, 1976.
    .. [2] J. L. Bentley, "Fast algorithms for geometric traveling salesman problems", ORSA
       Journal on Computing 4 (1992) 387-411.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import OrOpt
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> oo = OrOpt().fit(wi.distance_matrix(), labels=wi.labels)
    >>> oo.cost_ / wi.optimal_tour_length < 1.25  # the fast-tier tolerance
    True
    >>> oo.stop_reason_, oo.n_iter_ == len(oo.history_), bool(np.all(np.diff(oo.history_) <= 0))
    ('converged', True, True)
    >>> OrOpt(max_segment=2)
    OrOpt(max_segment=2)
    """

    _parameter_constraints: dict[str, Any] = {
        "max_segment": [Interval(int, 1, None, closed="left")],
        "init": [Options(str, {"nearest_neighbour"}), "array-like"],
        "n_candidates": [Interval(int, 1, None, closed="left"), None],
        "max_passes": [Interval(int, 1, None, closed="left")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        max_segment: int = 3,
        init: Any = "nearest_neighbour",
        n_candidates: int | None = 10,
        max_passes: int = 50,
        verbose: int = 0,
    ) -> None:
        self.max_segment = max_segment
        self.init = init
        self.n_candidates = n_candidates
        self.max_passes = max_passes
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="local_search", iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return run_descent(self, problem, ("or_opt",), max_segment=self.max_segment)
