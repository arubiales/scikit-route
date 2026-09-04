"""``TabuSearch``: best-admissible-move search over a 2-opt/Or-opt candidate neighbourhood with edge tabus."""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._init_tour import initial_tour
from ..utils._param_validation import Interval, Options
from . import _tabu

__all__ = ["TabuSearch"]

log = logging.getLogger("skroute")

_TENURE_CHUNK = 1024  # "auto" tenures are drawn lazily in chunks (stream-identical to one draw of n_iter)
_INT32_MAX = 2**31 - 1


class TabuSearch(BaseRouter):
    """Tabu search over the giant tour: 2-opt and Or-opt candidate moves, edge tabus, aspiration.

    Each iteration scans the candidate neighbourhood of the current tour, applies the best
    admissible move even when it worsens the tour, and forbids re-adding the edges that move
    removed for ``tenure`` iterations (a tabu move is still admissible when it beats the best
    tour so far). The best tour seen is kept in its own buffer and is what ``fit`` returns;
    ``history_`` records its cost per iteration.

    Parameters
    ----------
    n_iter : int >= 1, default 1000
        Maximum number of iterations (``"max_iter"``).
    tenure : int >= 1 or "auto", default "auto"
        Tabu tenure of a removed edge, in iterations: an edge removed at iteration ``k`` is tabu
        at exactly the iterations ``k + 1 .. k + tenure``. ``"auto"`` draws one tenure per
        iteration uniformly from ``[ceil(sqrt(n)), 2 * ceil(sqrt(n))]`` (both ends inclusive,
        pre-drawn from ``random_state``: the robust tabu search of Taillard); an int is a fixed
        tenure (a tenure of ``n_iter`` or more keeps every removed edge tabu until the end).
    n_candidates : int >= 1 or None, default 10
        Size of the candidate lists (the ``k`` nearest nodes of every node by cost);
        ``None`` uses the full neighbourhood (``n - 1``).
    patience : int >= 1 or None, default 200
        Iterations without improvement of the best-so-far before stopping (``"patience"``);
        ``None`` disables the rule.
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the greedy nearest-neighbour construction, a random permutation, or the
        ``tour_``/``route_`` of another solver (labels; the depot may repeat).
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds, checked once per iteration (``"time_limit"``). Breaks
        bit-exact reproducibility across machines; ``None`` disables.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the tenure draws (and of ``init="random"``). The same seed on the same machine
        gives bit-identical results; a passed ``Generator`` is advanced by the fit.
    verbose : int, default 0
        ``0`` is silent; ``1`` logs every ``max(1, n_iter // 10)`` iterations at INFO; ``2``
        logs every iteration. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far cost after each iteration (monotone non-increasing); ``history_[-1]`` equals
        ``cost_`` exactly.
    n_iter_ : int
        Iterations run.
    stop_reason_ : {"max_iter", "patience", "time_limit", "callback"}
        Why the search stopped (``"callback"``: the ``callback`` of ``fit`` returned ``True``).

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and
    the other fitted attributes shared by every solver.

    Notes
    -----
    **Neighbourhood.** For every node ``a`` and every candidate ``c`` of ``a`` (the ``k``
    nearest nodes by cost): the 2-opt reversals that create the edge ``{a, c}`` (``a`` before
    ``c`` in the tour, or after) and the no-reversal Or-opt relocations of the segments of
    length 1..3 starting or ending at ``a`` next to ``c`` (and vice versa) — the move family of
    the core's generic descent. On a symmetric plain TSP each move is priced with the O(1)
    deltas of the core (``two_opt_delta``, ``or_opt_delta``); under a budget (multi-trip) or
    with an asymmetric matrix every move is applied on a scratch copy and priced with the full
    objective, so the search sees the decoded trips and their fixed charges, and directional
    costs on ATSP. The reversal of the whole tour (an orientation flip on a symmetric plain
    TSP) is not a move.

    **Tabu attributes.** The edges a move removes (arcs when the matrix is asymmetric) are
    stored in an ``int32`` ``(n, n)`` matrix ``tabu_until``; a move is tabu when any edge it
    adds is still tabu, unless it improves on the best cost so far (aspiration). On an
    asymmetric matrix a 2-opt reversal removes every arc of the reversed span (the inner arcs
    change direction), so all of them are marked and a reversal is tabu when any arc it adds is
    tabu; on a symmetric matrix only the two boundary edges change. When every move is tabu the
    best move overall is applied, so the search never stalls. The practical ceiling of the matrix
    is about 5 000 nodes (100 MB).

    **Complexity.** O(n * n_candidates) O(1) delta evaluations per iteration on the symmetric
    plain path (about 14 moves per candidate pair); O(n^2 * n_candidates) on the generic path
    (O(n) per move), which is meant for instances of a few hundred nodes.

    **Callback events (D30).** ``"start"`` carries the ``init`` tour; every iteration emits one
    ``"iteration"`` whose ``tour`` is the walker (which may have moved uphill) and ``best_tour``
    the best buffer, with ``extra["tenure"]`` the tenure applied in that iteration.

    **Supports:** symmetric and asymmetric matrices, the multi-trip objective (both split
    rules); stochastic, iterative, budget-aware.

    References
    ----------
    .. [1] F. Glover, "Tabu search — part I", ORSA Journal on Computing 1 (1989) 190-206.
    .. [2] E. Taillard, "Robust taboo search for the quadratic assignment problem", Parallel
       Computing 17 (1991) 443-455.
    .. [3] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in Local Search in Combinatorial Optimization, Wiley, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import TabuSearch
    >>> from skroute.datasets import load_tsp
    >>> dj = load_tsp("dj38")
    >>> ts = TabuSearch(random_state=0).fit(dj.distance_matrix(), labels=dj.labels)
    >>> ts.cost_ / dj.optimal_tour_length < 1.08  # the fast-tier tolerance of SPEC §6
    True
    >>> int(ts.route_[0]) == int(ts.route_[-1]) == int(ts.depot_) == 1
    True
    >>> ts.n_iter_ == len(ts.history_) and ts.stop_reason_ in {"max_iter", "patience"}
    True
    >>> bool(np.all(np.diff(ts.history_) <= 0))  # best-so-far per iteration
    True

    A multi-trip instance: eight Spanish towns, a per-trip budget and a fixed charge per
    extra trip, two people. The search itself prices the decoded trips.

    >>> from skroute.datasets import load_alicante_murcia
    >>> d = load_alicante_murcia()
    >>> budget = 1.5 * float((d.time[0] + d.time[:, 0]).max())
    >>> ts = TabuSearch(random_state=0).fit(
    ...     d.cost,
    ...     time_matrix=d.time,
    ...     labels=d.labels,
    ...     depot=d.depot,
    ...     max_time_work=budget,
    ...     extra_cost=10.0,
    ...     people=2,
    ... )
    >>> bool(np.all(ts.trip_times_ <= budget)) and ts.n_trips_ == len(ts.trips_)
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "tenure": [Options(str, {"auto"}), Interval(Integral, 1, None, closed="left")],
        "n_candidates": [None, Interval(Integral, 1, None, closed="left")],
        "patience": [None, Interval(Integral, 1, None, closed="left")],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "time_limit": [None, Interval(Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_iter: int = 1000,
        tenure: int | str = "auto",
        n_candidates: int | None = 10,
        patience: int | None = 200,
        init: Any = "nearest_neighbour",
        time_limit: float | None = None,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.n_iter = n_iter
        self.tenure = tenure
        self.n_candidates = n_candidates
        self.patience = patience
        self.init = init
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None  # stochastic tag: the base class always hands a Generator
        started = perf_counter()
        n, n_iter = problem.n, int(self.n_iter)
        fast_path = problem.symmetric and not problem.multi_trip
        C, T = problem.cost, problem.time_or_cost
        max_time, fixed_cost, split = problem.max_time_work, problem.fixed_cost, problem.split_code
        optimal = problem.multi_trip and problem.split == "optimal"
        k = n - 1 if self.n_candidates is None else min(int(self.n_candidates), n - 1)
        cand = problem.neighbours(k)
        scratch = np.empty(n, dtype=np.int64)
        dp = np.empty(n if optimal else 0, dtype=np.float64)
        pred = np.empty(n if optimal else 0, dtype=np.int64)

        tour = np.ascontiguousarray(initial_tour(problem, self.init, rng), dtype=np.int64)
        pos = np.empty(n, dtype=np.int64)
        pos[tour] = np.arange(n, dtype=np.int64)
        best = tour.copy()  # separate buffer: the kernel copies into it, never aliases it
        cost0 = float(problem.evaluate(tour))
        state = np.array([cost0, cost0], dtype=np.float64)
        self._emit("start", 0, tour, cost0)  # D30

        # robust tabu search: one tenure per iteration, pre-drawn (D10) in chunks so that a huge
        # n_iter stopped by patience/time_limit costs no memory; an int is a fixed tenure, clamped
        # to n_iter (any longer tenure is "until the end" anyway) and to the kernel's int32.
        base = math.ceil(math.sqrt(n))
        fixed = None if self.tenure == "auto" else min(int(self.tenure), n_iter, _INT32_MAX)
        tenures = np.empty(0, dtype=np.int32)
        chunk_start = 0
        until = np.zeros((n, n), dtype=np.int32)  # int32 on purpose: 100 MB at the ~5 000-node ceiling
        every = max(1, n_iter // 10) if self.verbose == 1 else 1

        history: list[float] = []
        since = 0
        reason = "max_iter"
        done = 0
        for it in range(n_iter):
            if fixed is not None:
                tenure = fixed
            else:
                if it - chunk_start >= tenures.shape[0]:
                    chunk_start = it
                    size = min(_TENURE_CHUNK, n_iter - it)
                    tenures = rng.integers(base, 2 * base + 1, size=size).astype(np.int32)
                tenure = int(tenures[it - chunk_start])
            before = float(state[1])
            _tabu.tabu_step(
                C,
                T,
                tour,
                pos,
                cand,
                until,
                it,
                tenure,
                max_time,
                fixed_cost,
                split,
                fast_path,
                problem.symmetric,
                True,
                scratch,
                dp,
                pred,
                best,
                state,
            )
            history.append(float(state[1]))
            done = it + 1
            since = 0 if state[1] < before - 1e-9 * max(1.0, abs(before)) else since + 1
            if self.verbose and (self.verbose >= 2 or done % every == 0):
                log.info(
                    "TabuSearch iteration %d: current=%.6f best=%.6f tenure=%d",
                    done,
                    state[0],
                    state[1],
                    tenure,
                )
            if (
                self._callback is not None
            ):  # D30: the walker (which may have moved uphill) and the best buffer
                self._emit("iteration", done, tour, float(state[0]), best, float(state[1]), tenure=tenure)
            if self._stop_requested:
                reason = "callback"
                break
            if self.time_limit is not None and perf_counter() - started > self.time_limit:
                reason = "time_limit"
                break
            if self.patience is not None and since >= self.patience:
                reason = "patience"
                break
        if self.verbose:
            log.info("TabuSearch stopped by %s after %d iterations: best=%.6f", reason, done, state[1])
        self.history_ = np.asarray(history, dtype=np.float64)
        self.n_iter_ = done
        self.stop_reason_ = reason
        return best
