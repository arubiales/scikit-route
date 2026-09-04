"""``LocalSearch`` and the descent engine shared by every solver of :mod:`skroute.local_search`.

The engine (:class:`Descent`) owns the working tour and the buffers the core descents need
(``pos``, the candidate lists and one don't-look-bit array per move) and implements the
iteration accounting of SPEC §4.3: one outer iteration = one call of each listed descent kernel
with ``max_passes=1``; the buffers persist across calls. ``TwoOpt``, ``OrOpt`` and
``LocalSearch`` drive it through :func:`run_descent`; ``IteratedLocalSearch`` and any memetic
solver drive it directly (``load`` a tour, ``converge``, read ``tour``/``cost``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._core import _routing as core
from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils import initial_tour
from ..utils._param_validation import Interval, Options

__all__ = [
    "MOVES",
    "MOVE_TUPLES",
    "Descent",
    "LocalSearch",
    "changed_nodes",
    "normalise_moves",
    "run_descent",
]

log = logging.getLogger("skroute")

#: The move names a descent accepts (glossary §4.0); ``"swap"`` is reachable only through the core's mask.
MOVES: tuple[str, ...] = ("two_opt", "or_opt")
#: Every legal ``moves``/``local_search`` tuple: a non-empty subset of :data:`MOVES`, in either order.
MOVE_TUPLES: frozenset[tuple[str, ...]] = frozenset(
    {("two_opt",), ("or_opt",), ("two_opt", "or_opt"), ("or_opt", "two_opt")}
)
_MASK = {"two_opt": 1, "or_opt": 2}  # bit mask of local_search_generic (SPEC §3.5)


def normalise_moves(value: Any) -> tuple[str, ...]:
    """``None`` -> ``()``, a single move name -> a 1-tuple, a tuple -> itself (glossary §4.0)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def changed_nodes(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Nodes whose pair of tour neighbours differs between two tours of the same nodes (symmetric view).

    The pair is unordered, so reversing a segment touches only its two ends: exactly the nodes whose
    don't-look bits must be re-activated after another move (or a kick) changed the tour.
    """
    n = before.shape[0]

    def pairs(t: np.ndarray) -> np.ndarray:
        s, p = np.roll(t, -1), np.roll(t, 1)
        out = np.empty((n, 2), dtype=np.int64)
        out[t, 0] = np.minimum(s, p)
        out[t, 1] = np.maximum(s, p)
        return out

    return np.flatnonzero((pairs(before) != pairs(after)).any(axis=1))


class Descent:
    """Stateful driver of the §4.3 descents over one :class:`~skroute.problem.RoutingProblem`.

    Parameters
    ----------
    problem : RoutingProblem
        The instance; decides the path: symmetric plain TSP uses the O(1) kernels
        (``two_opt_descent``/``or_opt_descent`` with ``pos`` and don't-look bits), anything else
        (asymmetric matrix or multi-trip objective) uses ``local_search_generic`` with the move mask.
    moves : tuple of str
        Descents run in this order in every iteration; each a member of :data:`MOVES`.
    first_improvement : bool, default True
        2-opt applies the first improving move of a node (``True``) or the best one (``False``);
        honoured only on the symmetric plain path (the generic kernel is first-improvement).
    max_segment : int, default 3
        Longest Or-opt segment.
    n_candidates : int or None, default 10
        Size of the candidate lists; ``None`` = the full neighbourhood (``n - 1``).

    Attributes
    ----------
    tour : ndarray of shape (n,), int64
        The working tour (index space, depot first); modified in place by every descent.
    cost : float
        Objective of ``tour``, kept as ``cost += gain`` with the kernels' returned gains.
    fast : bool
        Whether the O(1) symmetric path is in use.
    """

    def __init__(
        self,
        problem: RoutingProblem,
        moves: tuple[str, ...],
        *,
        first_improvement: bool = True,
        max_segment: int = 3,
        n_candidates: int | None = 10,
    ) -> None:
        for m in moves:
            if m not in _MASK:
                raise ValueError(f"unknown move {m!r}; the descents accept {list(MOVES)}")
        self.problem = problem
        self.moves = tuple(moves)
        self.first_improvement = bool(first_improvement)
        self.max_segment = int(max_segment)
        n = problem.n
        k = n - 1 if n_candidates is None else int(n_candidates)
        self.cand = problem.neighbours(k)  # clamps k to n - 1, cached on the problem
        self.fast = problem.symmetric and not problem.multi_trip
        self.tour = np.empty(n, dtype=np.int64)
        self.pos = np.empty(n, dtype=np.int64)
        self.cost = 0.0
        self.bits: dict[str, np.ndarray] = {}
        self._snapshot: np.ndarray | None = None
        self._full = False  # every don't-look bit active at the start of the next iteration
        if self.fast:
            self.bits = {m: np.zeros(n, dtype=np.uint8) for m in self.moves}
            if len(self.moves) > 1:
                self._snapshot = np.empty(n, dtype=np.int64)
        else:
            self._scratch = np.empty(n, dtype=np.int64)
            if problem.split == "optimal":
                self._dp, self._pred = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.int64)
            else:
                self._dp, self._pred = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------ state
    def load(self, tour: np.ndarray, cost: float | None = None, *, activate: bool = True) -> None:
        """Make ``tour`` the working solution: copy it, rebuild ``pos``, set ``cost``.

        ``activate=True`` re-activates every don't-look bit (a fresh start); ``activate=False``
        keeps them, for a caller that knows which nodes changed and calls :meth:`activate` itself.
        """
        self.tour[:] = tour
        core.rebuild_pos(self.tour, self.pos)
        self.cost = float(self.problem.evaluate(self.tour)) if cost is None else float(cost)
        if activate:
            self.activate_all()
        else:
            self._full = False

    def activate(self, nodes: np.ndarray) -> None:
        """Re-activate the don't-look bits of ``nodes`` for every move (no-op on the generic path)."""
        for b in self.bits.values():
            b[nodes] = 0
        self._full = False

    def activate_all(self) -> None:
        """Re-activate every don't-look bit: the next iteration sweeps every node."""
        for b in self.bits.values():
            b[:] = 0
        self._full = True

    # ------------------------------------------------------------------ descents
    def _run(self, move: str) -> float:
        p = self.problem
        if self.fast:
            if move == "two_opt":
                return core.two_opt_descent(
                    p.cost, self.tour, self.pos, self.cand, self.bits[move], self.first_improvement, 1
                )
            return core.or_opt_descent(
                p.cost, self.tour, self.pos, self.cand, self.bits[move], self.max_segment, True, 1
            )
        return core.local_search_generic(
            p.cost,
            p.time_or_cost,
            self.tour,
            self.pos,
            self.cand,
            p.max_time_work,
            p.fixed_cost,
            p.split_code,
            _MASK[move],
            self.max_segment,
            1,
            self._scratch,
            self._dp,
            self._pred,
        )

    def iterate(self) -> list[float]:
        """One outer iteration: one call of each listed descent with ``max_passes=1``.

        Returns the gain (``cost_after - cost_before <= 0``) of each descent, in order; ``cost`` is
        updated. When a descent changes the tour, the nodes whose neighbours changed are re-activated
        for the other moves, so an alternating descent converges to a tour that is a local optimum
        for every listed move.
        """
        gains: list[float] = []
        for move in self.moves:
            if self._snapshot is not None:
                self._snapshot[:] = self.tour
            g = self._run(move)
            if g < 0.0 and self._snapshot is not None:
                touched = changed_nodes(self._snapshot, self.tour)
                for other, b in self.bits.items():
                    if other != move:
                        b[touched] = 0
            self.cost += g
            gains.append(g)
        return gains

    def step(self) -> tuple[list[float], bool]:
        """One outer iteration plus the convergence test: ``(gains, converged)``.

        A sweep over the *active* nodes that changes nothing is not yet a local optimum: a node whose
        bit was set earlier may have gained an improving move since (don't-look bits are a heuristic).
        So the first zero-gain iteration re-activates every bit and the descent is declared converged
        only when an iteration that started with every node active returns ``0.0`` for every listed
        move. The generic path has no bits (every pass is a full sweep) and converges at once.
        """
        full = self._full
        gains = self.iterate()
        if any(g < 0.0 for g in gains):
            self._full = False
            return gains, False
        if full or not self.fast:
            return gains, True
        self.activate_all()
        return gains, False

    def converge(self, max_iter: int | None = None) -> tuple[list[float], bool]:
        """Iterate (:meth:`step`) until convergence or ``max_iter`` iterations.

        Returns ``(history, converged)`` with ``history[k]`` the cost after iteration ``k``.
        """
        history: list[float] = []
        k = 0
        while max_iter is None or k < max_iter:
            _, done = self.step()
            history.append(self.cost)
            k += 1
            if done:
                return history, True
        return history, False


def run_descent(
    est: BaseRouter,
    problem: RoutingProblem,
    moves: tuple[str, ...],
    *,
    first_improvement: bool = True,
    max_segment: int = 3,
) -> np.ndarray:
    """``_solve`` of the three deterministic descents: sets ``history_``/``n_iter_``/``stop_reason_``."""
    name = type(est).__name__
    init: Any = est.init  # type: ignore[attr-defined]
    max_passes: int = est.max_passes  # type: ignore[attr-defined]
    verbose: int = est.verbose  # type: ignore[attr-defined]
    engine = Descent(
        problem,
        moves,
        first_improvement=first_improvement,
        max_segment=max_segment,
        n_candidates=est.n_candidates,  # type: ignore[attr-defined]
    )
    engine.load(initial_tour(problem, init, None))
    est._emit("start", 0, engine.tour, engine.cost, moves=list(moves))
    every = max(1, max_passes // 10)
    history: list[float] = []
    reason = "max_iter"
    for k in range(max_passes):
        gains, done = engine.step()
        history.append(engine.cost)
        if verbose and (verbose >= 2 or (k + 1) % every == 0):
            log.info(
                "%s iteration %d/%d: cost %.6g (gain %.6g)", name, k + 1, max_passes, engine.cost, sum(gains)
            )
        if est._callback is not None:
            # D30: the working tour is the best tour (a descent never goes uphill); ``moves_applied`` names
            # the listed descents that changed the tour in this iteration, ``gain`` their summed cost change
            est._emit(
                "iteration",
                k + 1,
                engine.tour,
                engine.cost,
                moves_applied=[m for m, g in zip(moves, gains, strict=True) if g < 0.0],
                gain=float(sum(gains)),
            )
        if est._stop_requested:
            reason = "callback"
            break
        if done:
            reason = "converged"
            break
    if verbose:
        log.info("%s: stopped (%s) after %d iterations, cost %.6g", name, reason, len(history), engine.cost)
    est.history_ = np.asarray(history, dtype=np.float64)
    est.n_iter_ = len(history)
    est.stop_reason_ = reason
    return engine.tour.copy()


class LocalSearch(BaseRouter):
    """Alternating 2-opt / Or-opt descent with candidate lists and don't-look bits.

    Starts from ``init`` and, in every outer iteration, runs one pass of each listed descent
    (``moves``, in order) with the ``pos``, candidate and don't-look-bit buffers persisting
    across passes. Nodes whose tour neighbours were changed by one move are re-activated for
    the other, so the search stops only at a tour that no listed move improves.

    Parameters
    ----------
    moves : tuple of {"two_opt", "or_opt"}, default ("two_opt", "or_opt")
        Descents run in each outer iteration, in this order. A single name is accepted as a
        string. ``"swap"`` is not available here (it exists only in the generic kernel's mask).
    first_improvement : bool, default True
        2-opt applies the first improving move found for a node (``True``) or scans the node's
        whole candidate neighbourhood and applies the best (``False``). Honoured only on the
        symmetric plain-TSP path; the generic path (asymmetric or multi-trip) is always
        first-improvement.
    max_segment : int, default 3
        Longest segment relocated by Or-opt (segments of ``1..max_segment`` nodes).
    init : {"nearest_neighbour"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the nearest-neighbour construction from the depot, or the ``tour_``/
        ``route_`` of another solver (labels; the depot may repeat).
    n_candidates : int or None, default 10
        Size of the candidate lists (the ``k`` nearest neighbours of every node); ``None`` scans
        the full neighbourhood (``n - 1`` nodes), which gives a markedly better local optimum
        at small ``n`` for a higher cost per pass.
    max_passes : int, default 50
        Maximum number of outer iterations.
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
        ``"converged"`` when an iteration that started with every node active returned a zero
        gain for every listed move, ``"max_iter"`` after ``max_passes`` iterations,
        ``"callback"`` when the ``callback`` of ``fit`` returned ``True``.

    Notes
    -----
    Supports: symmetric and asymmetric matrices, multi-trip objective; deterministic.

    Callback events (D30): ``"start"`` carries the ``init`` tour; each ``"iteration"`` event's
    ``tour`` is the working tour (also the best: a descent never goes uphill), with the ``extra``
    keys ``moves_applied`` (the listed moves whose descent changed the tour in that iteration)
    and ``gain`` (the iteration's total cost change, ``<= 0``).

    One outer iteration = one call of each listed descent kernel with ``max_passes=1``
    (SPEC §4.3). Symmetric plain TSP uses Bentley's neighbour-list descents with O(1) move
    deltas: one pass is O(n k) delta evaluations plus O(n) per applied reversal. Because
    don't-look bits skip nodes scanned earlier, the first iteration that changes nothing
    re-activates every node and convergence is declared only when the following full sweep
    changes nothing either, so a converged tour admits no improving 2-opt move among the
    candidate edges (Or-opt's pruning is a heuristic: it re-inserts a segment only next to a
    candidate closer than the edge it removes). Asymmetric matrices and the multi-trip
    objective use the full-evaluation kernel (O(n) per candidate move, Or-opt without
    reversal): fine up to a few thousand nodes.

    References
    ----------
    .. [1] G. A. Croes, "A method for solving traveling-salesman problems", Operations
       Research 6 (1958) 791-812.
    .. [2] I. Or, "Traveling salesman-type combinatorial problems and their relation to the
       logistics of regional blood banking", PhD thesis, Northwestern University, 1976.
    .. [3] J. L. Bentley, "Fast algorithms for geometric traveling salesman problems", ORSA
       Journal on Computing 4 (1992) 387-411.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import LocalSearch
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> ls = LocalSearch().fit(wi.distance_matrix(), labels=wi.labels)
    >>> ls.cost_ / wi.optimal_tour_length < 1.12  # the fast-tier tolerance
    True
    >>> ls.stop_reason_, ls.n_iter_ == len(ls.history_), bool(np.all(np.diff(ls.history_) <= 0))
    ('converged', True, True)
    >>> LocalSearch(moves=("or_opt",), n_candidates=None)
    LocalSearch(moves=('or_opt',), n_candidates=None)
    """

    _parameter_constraints: dict[str, Any] = {
        "moves": [Options(tuple, MOVE_TUPLES), Options(str, set(MOVES))],
        "first_improvement": ["boolean"],
        "max_segment": [Interval(int, 1, None, closed="left")],
        "init": [Options(str, {"nearest_neighbour"}), "array-like"],
        "n_candidates": [Interval(int, 1, None, closed="left"), None],
        "max_passes": [Interval(int, 1, None, closed="left")],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        moves: tuple[str, ...] | str = ("two_opt", "or_opt"),
        first_improvement: bool = True,
        max_segment: int = 3,
        init: Any = "nearest_neighbour",
        n_candidates: int | None = 10,
        max_passes: int = 50,
        verbose: int = 0,
    ) -> None:
        self.moves = moves
        self.first_improvement = first_improvement
        self.max_segment = max_segment
        self.init = init
        self.n_candidates = n_candidates
        self.max_passes = max_passes
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="local_search", iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        return run_descent(
            self,
            problem,
            normalise_moves(self.moves),
            first_improvement=self.first_improvement,
            max_segment=self.max_segment,
        )
