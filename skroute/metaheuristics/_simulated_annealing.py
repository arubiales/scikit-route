"""``SimulatedAnnealing``: Metropolis search over 2-opt, Or-opt and swap moves with geometric cooling."""

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
from . import _sa

__all__ = ["SimulatedAnnealing"]

log = logging.getLogger("skroute")

_MOVE_CODES = {"two_opt": 0, "or_opt": 1, "swap": 2}
_N_CALIBRATION = 1000  # proposals sampled on the initial tour for t0="auto"


def _normalise_moves(moves: Any) -> tuple[str, ...]:
    """The ``moves`` parameter as a validated, non-empty tuple of distinct names (a string is a 1-tuple)."""
    names = (moves,) if isinstance(moves, str) else tuple(moves)
    if not names:
        raise ValueError("moves must contain at least one of 'two_opt', 'or_opt', 'swap'")
    bad = [m for m in names if not (isinstance(m, str) and m in _MOVE_CODES)]
    if bad:
        raise ValueError(f"moves must be a tuple of names among 'two_opt', 'or_opt', 'swap'; got {bad!r}")
    if len(set(names)) != len(names):
        # every move type is proposed with the same probability: a repeated name would silently weight it
        raise ValueError(f"moves must not repeat a move name; got {names!r}")
    return names


class SimulatedAnnealing(BaseRouter):
    """Simulated annealing over the giant tour with 2-opt, Or-opt and swap proposals.

    One outer iteration is one temperature level of ``n_moves`` Metropolis proposals; the
    temperature then cools geometrically (``T *= alpha``) until it drops below ``t_min``.
    Every proposal is a random move drawn uniformly from ``moves``: a downhill move is always
    accepted, an uphill move with probability ``exp(-delta / T)``. The best tour seen is kept
    in its own buffer and is what ``fit`` returns; ``history_`` records its cost per level.

    Parameters
    ----------
    t0 : float > 0 or "auto", default "auto"
        Initial temperature (finite). ``"auto"`` prices 1000 random proposals on the initial
        tour and sets ``t0 = -median(uphill deltas) / ln(0.5)``, so the median uphill move is
        accepted with probability one half at the start (``t0 = 1.0`` if no proposal goes
        uphill).
    t_min : float > 0 or "auto", default "auto"
        Final temperature (finite); the search stops (``"converged"``) once ``T < t_min``.
        ``"auto"`` is ``1e-4 * t0``, i.e. about ``ln(1e4) / -ln(alpha)`` levels (1838 at the
        default ``alpha``); a ``t0`` so small that ``1e-4 * t0`` underflows to zero is rejected.
    alpha : float in (0, 1), default 0.995
        Geometric cooling factor applied after every level.
    n_moves : int >= 1 or None, default None
        Proposals per temperature level; ``None`` means ``10 * n``. An invalid random draw
        (see Notes) is a rejected proposal and counts towards ``n_moves``.
    moves : tuple of {"two_opt", "or_opt", "swap"}, default ("two_opt", "or_opt", "swap")
        The move types proposed, each with the same probability; a name must not repeat. A
        single name is accepted.
    patience : int >= 1 or None, default None
        Levels without improvement of the best-so-far before stopping (``"patience"``). The
        count starts only once the current cost has first fallen below the initial cost: with
        ``t0="auto"`` the hot phase is non-improving by design (the current cost first drops
        below a nearest-neighbour start after several hundred levels on the Waterloo
        instances), so a small patience applied from level 0 would return the start tour.
        ``None`` disables the rule.
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the greedy nearest-neighbour construction, a random permutation, or the
        ``tour_``/``route_`` of another solver (labels; the depot may repeat).
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds, checked once per level (``"time_limit"``). Breaks
        bit-exact reproducibility across machines; ``None`` disables.
    random_state : int, numpy.random.Generator or None, default None
        Seed of every random draw. The same seed on the same machine gives bit-identical
        results; a passed ``Generator`` is advanced by the fit.
    verbose : int, default 0
        ``0`` is silent; ``1`` logs every ``max(1, n_levels // 10)`` levels at INFO; ``2`` logs
        every level. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    t0_ : float
        The initial temperature actually used (the calibrated value under ``t0="auto"``).
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far cost after each temperature level (monotone non-increasing);
        ``history_[-1]`` equals ``cost_`` exactly.
    n_iter_ : int
        Temperature levels run.
    stop_reason_ : {"converged", "patience", "time_limit"}
        Why the search stopped.

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and
    the other fitted attributes shared by every solver.

    Notes
    -----
    **Algorithm.** Per level the randomness is pre-drawn in Python (``u ~ U[0, 1)``,
    ``ri, rj ~ U{1..n-1}``, ``mv ~ U{moves}``) and handed to a ``nogil`` kernel (D10). The
    draw -> move mapping is fixed: 2-opt and swap use ``i, j = min(ri, rj), max(ri, rj)`` and
    are invalid when ``i == j``; Or-opt moves the segment ``tour[i..i+L-1]`` with ``i = ri``,
    ``L = 1 + (rj % 3)`` after the node at position ``j = rj`` (never reversed) and is invalid
    when the segment leaves the tour (``i + L - 1 > n - 1``) or ``j in [i-1, i+L-1]``. An
    invalid draw is a rejected proposal: it consumes its ``u`` and changes nothing, so
    ``n_moves`` means "proposals per level" everywhere.

    **Two evaluation paths.** On a symmetric plain TSP every proposal is priced with the O(1)
    deltas of the core and applied in place; under a budget (multi-trip) or with an
    asymmetric matrix the move is applied on a scratch copy and priced with the full
    objective (O(n) per proposal, O(n L) under ``split="optimal"``), so the search sees the
    decoded trips and their fixed charges. The best tour is a separate buffer written only on
    strict improvement (``new < best - 1e-9 * max(1, |best|)``).

    **Complexity.** O(levels * n_moves) proposals; with the defaults about
    ``1838 * 10 n`` O(1) evaluations for a symmetric TSP, O(n) each otherwise.

    **Supports:** symmetric and asymmetric matrices, the multi-trip objective (both split
    rules); stochastic, iterative, budget-aware.

    References
    ----------
    .. [1] S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, "Optimization by simulated
       annealing", Science 220 (1983) 671-680.
    .. [2] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in Local Search in Combinatorial Optimization, Wiley, 1997.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import SimulatedAnnealing
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> sa = SimulatedAnnealing(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
    >>> sa.cost_ / wi.optimal_tour_length < 1.03  # the fast-tier tolerance of SPEC §6
    True
    >>> int(sa.route_[0]) == int(sa.route_[-1]) == int(sa.depot_) == 1
    True
    >>> sa.n_iter_ == len(sa.history_) and sa.stop_reason_ == "converged" and sa.t0_ > 0
    True
    >>> bool(np.all(np.diff(sa.history_) <= 0))  # best-so-far per level
    True

    A multi-trip instance: eight Spanish towns, a per-trip budget and a fixed charge per
    extra trip, two people. The search itself prices the decoded trips.

    >>> from skroute.datasets import load_alicante_murcia
    >>> d = load_alicante_murcia()
    >>> budget = 1.5 * float((d.time[0] + d.time[:, 0]).max())
    >>> sa = SimulatedAnnealing(random_state=0).fit(
    ...     d.cost,
    ...     time_matrix=d.time,
    ...     labels=d.labels,
    ...     depot=d.depot,
    ...     max_time_work=budget,
    ...     extra_cost=10.0,
    ...     people=2,
    ... )
    >>> bool(np.all(sa.trip_times_ <= budget)) and sa.n_trips_ == len(sa.trips_)
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "t0": [Options(str, {"auto"}), Interval(Real, 0.0, None, closed="neither")],
        "t_min": [Options(str, {"auto"}), Interval(Real, 0.0, None, closed="neither")],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
        "n_moves": [None, Interval(Integral, 1, None, closed="left")],
        "moves": [Options(str, set(_MOVE_CODES)), tuple, list],
        "patience": [None, Interval(Integral, 1, None, closed="left")],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "time_limit": [None, Interval(Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    t0_: float

    def __init__(
        self,
        t0: float | str = "auto",
        t_min: float | str = "auto",
        alpha: float = 0.995,
        n_moves: int | None = None,
        moves: tuple[str, ...] | str = ("two_opt", "or_opt", "swap"),
        patience: int | None = None,
        init: Any = "nearest_neighbour",
        time_limit: float | None = None,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.t0 = t0
        self.t_min = t_min
        self.alpha = alpha
        self.n_moves = n_moves
        self.moves = moves
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
        n = problem.n
        codes = np.array([_MOVE_CODES[m] for m in _normalise_moves(self.moves)], dtype=np.int64)
        n_moves = 10 * n if self.n_moves is None else int(self.n_moves)
        fast_path = problem.symmetric and not problem.multi_trip
        C, T = problem.cost, problem.time_or_cost
        max_time, fixed_cost, split = problem.max_time_work, problem.fixed_cost, problem.split_code
        optimal = problem.multi_trip and problem.split == "optimal"
        scratch = np.empty(n, dtype=np.int64)
        dp = np.empty(n if optimal else 0, dtype=np.float64)
        pred = np.empty(n if optimal else 0, dtype=np.int64)

        tour = np.ascontiguousarray(initial_tour(problem, self.init, rng), dtype=np.int64)
        best = tour.copy()  # separate buffer: the kernel copies into it, never aliases it
        initial_cost = float(problem.evaluate(tour))
        state = np.array([initial_cost, initial_cost], dtype=np.float64)

        # ---- temperature schedule
        if self.t0 == "auto":
            t0 = self._calibrate_t0(
                rng, C, T, tour, codes, max_time, fixed_cost, split, fast_path, scratch, dp, pred
            )
        else:
            t0 = float(self.t0)
        t_min = 1e-4 * t0 if self.t_min == "auto" else float(self.t_min)
        # the Interval constraints exclude 0 and NaN but not inf; an infinite t0 never cools below
        # t_min (inf * alpha == inf) and a subnormal t0 makes t_min="auto" underflow to 0.0
        if not (0.0 < t0 < math.inf):
            raise ValueError(f"t0 must be a positive finite temperature; got {t0!r}")
        if not (0.0 < t_min < math.inf):
            hint = " (t_min='auto' is 1e-4 * t0: pass a positive t_min or a larger t0)"
            msg = f"t_min must be a positive finite temperature; got {t_min!r}"
            raise ValueError(msg + hint if self.t_min == "auto" else msg)
        self.t0_ = t0
        # expected number of levels, for the verbose cadence only (>= 1: one level always runs);
        # log(t_min) - log(t0) instead of log(t_min / t0): the ratio may underflow, the logs cannot
        if t_min < t0:
            n_levels = max(1, math.ceil((math.log(t_min) - math.log(t0)) / math.log(self.alpha)))
        else:
            n_levels = 1
        every = max(1, n_levels // 10) if self.verbose == 1 else 1

        history: list[float] = []
        temperature = t0
        armed = False  # patience counts only once the current cost has fallen below the start
        since = 0
        reason = "converged"
        level = 0
        while True:
            u = rng.random(n_moves)
            ri = rng.integers(1, n, size=n_moves, dtype=np.int64)
            rj = rng.integers(1, n, size=n_moves, dtype=np.int64)
            mv = codes[rng.integers(0, codes.shape[0], size=n_moves)]
            before = state[1]
            accepted = _sa.anneal_level(
                C,
                T,
                tour,
                best,
                u,
                ri,
                rj,
                mv,
                temperature,
                max_time,
                fixed_cost,
                split,
                fast_path,
                scratch,
                dp,
                pred,
                state,
            )
            history.append(float(state[1]))
            level += 1
            if self.verbose and (self.verbose >= 2 or level % every == 0):
                log.info(
                    "SimulatedAnnealing level %d: T=%.6g current=%.6f best=%.6f accepted=%d/%d",
                    level,
                    temperature,
                    state[0],
                    state[1],
                    accepted,
                    n_moves,
                )
            temperature *= self.alpha
            if self.time_limit is not None and perf_counter() - started > self.time_limit:
                reason = "time_limit"
                break
            if temperature < t_min:
                reason = "converged"
                break
            if self.patience is not None:
                if not armed and state[0] < initial_cost - 1e-9 * max(1.0, abs(initial_cost)):
                    armed = True
                    since = 0
                elif armed:
                    since = 0 if state[1] < before - 1e-9 * max(1.0, abs(before)) else since + 1
                    if since >= self.patience:
                        reason = "patience"
                        break
        if self.verbose:
            log.info(
                "SimulatedAnnealing stopped by %s after %d levels: best=%.6f (t0=%.6g)",
                reason,
                level,
                state[1],
                t0,
            )
        self.history_ = np.asarray(history, dtype=np.float64)
        self.n_iter_ = len(history)
        self.stop_reason_ = reason
        return best

    @staticmethod
    def _calibrate_t0(
        rng: np.random.Generator,
        C: np.ndarray,
        T: np.ndarray,
        tour: np.ndarray,
        codes: np.ndarray,
        max_time: float,
        fixed_cost: float,
        split: int,
        fast_path: bool,
        scratch: np.ndarray,
        dp: np.ndarray,
        pred: np.ndarray,
    ) -> float:
        """``t0 = -median(uphill deltas) / ln(0.5)`` over 1000 random proposals; ``1.0`` if none is uphill."""
        n = tour.shape[0]
        ri = rng.integers(1, n, size=_N_CALIBRATION, dtype=np.int64)
        rj = rng.integers(1, n, size=_N_CALIBRATION, dtype=np.int64)
        mv = codes[rng.integers(0, codes.shape[0], size=_N_CALIBRATION)]
        deltas = np.empty(_N_CALIBRATION, dtype=np.float64)
        _sa.sample_deltas(
            C, T, tour, ri, rj, mv, max_time, fixed_cost, split, fast_path, scratch, dp, pred, deltas
        )
        uphill = deltas[deltas > 0.0]  # NaN (invalid draws) and downhill moves drop out here
        if uphill.size == 0:
            return 1.0
        return float(-np.median(uphill) / math.log(0.5))
