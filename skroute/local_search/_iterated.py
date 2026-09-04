"""``IteratedLocalSearch``: kick a local optimum, descend again, keep the best."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import Any

import numpy as np

from .._core import _routing as core
from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils import initial_tour
from ..utils._param_validation import Interval, Options
from ._local_search import MOVE_TUPLES, MOVES, Descent, changed_nodes, normalise_moves

__all__ = ["IteratedLocalSearch"]

log = logging.getLogger("skroute")


def _improves(new: float, ref: float) -> bool:
    """The improvement test of SPEC §4.0."""
    return new < ref - 1e-9 * max(1.0, abs(ref))


def _default_temperature(init_cost: float) -> float:
    """The Metropolis temperature of ``temperature=None``: 0.5 % of the init tour's cost scale.

    The scale is ``abs(init_cost)`` so a matrix with negative entries (legal: only finiteness is
    required) never yields a negative temperature, under which ``exp(-delta / T) > 1`` for every worse
    candidate and the rule silently accepts everything. A zero scale (an all-zero matrix, a zero-cost
    init tour) falls back to ``1.0``, the same fallback ``SimulatedAnnealing`` uses when no proposal
    goes uphill, so the rule is always defined.
    """
    scale = abs(init_cost)
    return 0.005 * scale if scale > 0.0 else 1.0


def _reversal_pairs(n: int, symmetric: bool) -> np.ndarray:
    """The segment reversals ``(i, j)`` a kick may draw below 8 nodes, as an ``(m, 2)`` int64 array.

    Every pair ``1 <= i < j <= n - 1``, except that on a symmetric matrix the reversal of the whole
    body ``(1, n - 1)`` is left out whenever another pair exists: it is the same closed tour driven
    backwards, changes no edge, and the descent that follows would find nothing while the iteration
    still counted towards ``patience``. At ``n = 3`` the swap ``(1, 2)`` is the only kick (and every
    tour is optimal anyway); on an asymmetric matrix the whole-body reversal is a genuine change.
    """
    pairs = [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]
    if symmetric and n > 3:
        pairs.remove((1, n - 1))
    return np.asarray(pairs, dtype=np.int64)


class IteratedLocalSearch(BaseRouter):
    """Iterated local search: double-bridge kicks over a 2-opt / Or-opt local optimum.

    Descends from ``init`` to a local optimum, then repeats: copy the incumbent, apply
    ``perturbation_strength`` kicks, descend to a new local optimum, accept it when better
    (or by a Metropolis rule) and remember the best tour seen. The kick is a double bridge
    (``A B C D -> A C B D``), which no 2-opt or Or-opt move can undo, so every iteration
    explores a genuinely new basin. This is the recommended default solver.

    Parameters
    ----------
    n_iter : int, default 1000
        Maximum number of outer iterations (kick + descent).
    patience : int or None, default 100
        Stop after this many iterations without improving the best tour; ``None`` disables.
    perturbation_strength : int, default 1
        Number of kicks applied per iteration.
    acceptance : {"better", "metropolis"}, default "better"
        ``"better"`` moves to the new local optimum only when it is strictly better;
        ``"metropolis"`` also accepts one that is not worse, and a worse one with probability
        ``exp(-(new - current) / temperature)``.
    temperature : float or None, default None
        Fixed temperature of the Metropolis rule; ``None`` uses 0.5 % of the absolute cost of the
        ``init`` tour, or ``1.0`` when that cost is zero (an all-zero matrix), so the rule is
        always defined. Ignored under ``acceptance="better"``.
    local_search : tuple of {"two_opt", "or_opt"}, str or None, default ("two_opt", "or_opt")
        Descents run in every iteration, in this order (a single name is accepted as a string);
        ``None`` disables the descent, leaving a random walk of kicks.
    n_candidates : int or None, default 10
        Size of the candidate lists of the descents; ``None`` = the full neighbourhood.
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        Starting tour: the nearest-neighbour construction from the depot, a random permutation,
        or the ``tour_``/``route_`` of another solver (labels; the depot may repeat).
    time_limit : float or None, default None
        Wall-clock budget in seconds, checked once per outer iteration; ``None`` disables.
        A run stopped by ``time_limit`` is not reproducible bit for bit.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the kicks (and of the Metropolis draws and ``init="random"``). Two fits with the
        same integer seed on the same machine are bit-identical. A ``Generator`` is used in
        place and advanced.
    verbose : int, default 0
        0 is silent; 1 logs every ``max(1, n_iter // 10)`` iterations and the stop; 2 logs every
        iteration. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Best cost seen after each outer iteration (non-increasing).
    n_iter_ : int
        Outer iterations run (the initial descent is not counted).
    stop_reason_ : {"max_iter", "patience", "time_limit", "callback"}
        Why the search stopped (``"callback"``: the ``callback`` of ``fit`` returned ``True``).

    Notes
    -----
    Supports: symmetric and asymmetric matrices, multi-trip objective; stochastic.

    Callback events (D30): ``"start"`` carries the ``init`` tour before the initial descent;
    each ``"iteration"`` event's ``tour`` is the candidate that iteration produced (kick +
    descent) and its ``best_tour`` the incumbent best, with the ``extra`` keys ``kick`` (the cut
    positions of every kick applied, as tuples), ``accepted`` (whether the candidate replaced
    the current tour) and ``current_cost``.

    The descent is the alternating scheme of `LocalSearch`, run to
    convergence after every kick. On a symmetric plain TSP it uses O(1) move deltas, candidate
    lists and don't-look bits: only the nodes a kick touched start active, so the re-descent from
    the kick's endpoints is cheap, but convergence is declared only when a sweep that started with
    every node active changed nothing, so every iteration costs at least one full O(n k) sweep
    per listed move plus the O(n) copies — a fraction of a millisecond per iteration at
    ``n = 1000`` with the default ``k = 10``. Asymmetric matrices and the multi-trip objective use
    the full-evaluation kernel, O(n) per candidate move, so that confirming sweep is O(n² k) and
    is repeated after every kick: one descent is fine up to a few thousand nodes, but at the
    default ``n_iter``/``patience`` the iterated search is comfortable up to a few hundred nodes
    on those problems; above that, lower ``n_iter`` or set ``time_limit``.

    For ``n >= 8`` the kick is a double bridge with cut positions ``1 <= p1 < p2 < p3 <= n - 1``
    drawn without replacement. Below 8 nodes it is the reversal of a random segment
    ``tour[i..j]`` with ``(i, j)`` uniform over the pairs ``1 <= i < j <= n - 1``; on a symmetric
    matrix the whole-body reversal ``(1, n - 1)`` is left out, because it is the same closed tour
    driven backwards and changes no edge (at ``n = 3`` the swap of positions 1 and 2 is the only
    kick). All randomness is drawn in Python from ``random_state`` before each iteration, so
    results are reproducible and independent of the platform's kernels.

    References
    ----------
    .. [1] O. Martin, S. W. Otto and E. W. Felten, "Large-step Markov chains for the traveling
       salesman problem", Complex Systems 5 (1991) 299-326.
    .. [2] H. R. Lourenço, O. C. Martin and T. Stützle, "Iterated local search: framework and
       applications", in *Handbook of Metaheuristics*, Springer, 2010.
    .. [3] D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in
       local optimization", in *Local Search in Combinatorial Optimization*, Wiley, 1997.

    Examples
    --------
    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603; labels are the file's 1-based ids
    >>> ils = IteratedLocalSearch(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
    >>> ils.cost_ / wi.optimal_tour_length < 1.03  # the fast-tier tolerance
    True
    >>> int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == 1
    True
    >>> ils.n_iter_ == len(ils.history_) and ils.stop_reason_ in {"patience", "max_iter"}
    True
    >>> again = IteratedLocalSearch(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
    >>> again.tour_.tolist() == ils.tour_.tolist()  # same seed, same machine: bit-identical
    True
    >>> IteratedLocalSearch(acceptance="metropolis", local_search=("two_opt",), random_state=3)
    IteratedLocalSearch(acceptance='metropolis', local_search=('two_opt',), random_state=3)
    """

    _parameter_constraints: dict[str, Any] = {
        "n_iter": [Interval(int, 1, None, closed="left")],
        "patience": [Interval(int, 1, None, closed="left"), None],
        "perturbation_strength": [Interval(int, 1, None, closed="left")],
        "acceptance": [Options(str, {"better", "metropolis"})],
        "temperature": [Interval(float, 0.0, None, closed="neither"), None],
        "local_search": [Options(tuple, MOVE_TUPLES), Options(str, set(MOVES)), None],
        "n_candidates": [Interval(int, 1, None, closed="left"), None],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "time_limit": [Interval(float, 0.0, None, closed="neither"), None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_iter: int = 1000,
        patience: int | None = 100,
        perturbation_strength: int = 1,
        acceptance: str = "better",
        temperature: float | None = None,
        local_search: tuple[str, ...] | str | None = ("two_opt", "or_opt"),
        n_candidates: int | None = 10,
        init: Any = "nearest_neighbour",
        time_limit: float | None = None,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.n_iter = n_iter
        self.patience = patience
        self.perturbation_strength = perturbation_strength
        self.acceptance = acceptance
        self.temperature = temperature
        self.local_search = local_search
        self.n_candidates = n_candidates
        self.init = init
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="local_search", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None  # stochastic tag: fit always hands a Generator
        t0 = perf_counter()
        n = problem.n
        moves = normalise_moves(self.local_search)
        engine = Descent(problem, moves, n_candidates=self.n_candidates) if moves else None

        # ---- initial local optimum
        cur = initial_tour(problem, self.init, rng)
        cur_cost = float(problem.evaluate(cur))
        self._emit("start", 0, cur, cur_cost)  # the init tour, before the initial descent (D30)
        temperature = _default_temperature(cur_cost) if self.temperature is None else float(self.temperature)
        if engine is not None:
            engine.load(cur, cur_cost)
            engine.converge()
            cur, cur_cost = engine.tour.copy(), engine.cost
        best, best_cost = cur.copy(), cur_cost

        # ---- kick / descend / accept
        double_bridge = n >= 8
        positions = np.arange(1, n)  # cut positions of the double bridge
        pairs = _reversal_pairs(n, problem.symmetric)  # the reversals drawn below 8 nodes
        kicked, tmp = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        metropolis = self.acceptance == "metropolis"
        every = max(1, self.n_iter // 10)
        history: list[float] = []
        since, reason = 0, "max_iter"
        for k in range(self.n_iter):
            # every random number of the iteration is drawn here, before any kernel runs (D10)
            if double_bridge:
                cuts = [
                    np.sort(rng.choice(positions, size=3, replace=False))
                    for _ in range(self.perturbation_strength)
                ]
            else:
                cuts = list(pairs[rng.integers(len(pairs), size=self.perturbation_strength)])
            u = float(rng.random()) if metropolis else 0.0

            kicked[:] = cur
            for c in cuts:
                if double_bridge:
                    core.double_bridge(kicked, int(c[0]), int(c[1]), int(c[2]), tmp)
                    kicked, tmp = tmp, kicked
                else:  # n < 8: reversal of tour[c0..c1], the swap (1, 2) at n = 3
                    kicked[c[0] : c[1] + 1] = kicked[c[0] : c[1] + 1][::-1]

            if engine is not None:
                if engine.fast:
                    # the incumbent is a local optimum (every bit set): re-activate only what the kick touched
                    touched = changed_nodes(cur, kicked)
                    engine.load(kicked, activate=False)
                    engine.activate(touched)
                else:
                    engine.load(kicked)
                engine.converge()
                new, new_cost = engine.tour, engine.cost
            else:
                new, new_cost = kicked, float(problem.evaluate(kicked))

            # The Metropolis rule is total: a candidate that is not worse is accepted outright (it
            # never reaches exp), and a worse one gives a non-positive exponent, which cannot
            # overflow — exp underflows to 0.0 and the draw is rejected.
            delta = new_cost - cur_cost
            accepted = _improves(new_cost, cur_cost) or (
                metropolis and (delta <= 0.0 or u < math.exp(-delta / temperature))
            )
            if accepted:
                cur[:] = new
                cur_cost = new_cost
            if _improves(new_cost, best_cost):
                best[:] = new
                best_cost = new_cost
                since = 0
            else:
                since += 1
            history.append(best_cost)
            if self.verbose and (self.verbose >= 2 or (k + 1) % every == 0):
                log.info(
                    "IteratedLocalSearch iteration %d/%d: best %.6g, current %.6g",
                    k + 1,
                    self.n_iter,
                    best_cost,
                    cur_cost,
                )
            if self._callback is not None:
                # D30: the tour of the event is the candidate this iteration produced (kick + descent),
                # so a viewer sees the search move even under acceptance="better", where the incumbent
                # is always the best tour; ``kick`` lists the cut positions of every kick applied
                self._emit(
                    "iteration",
                    k + 1,
                    new,
                    new_cost,
                    best,
                    best_cost,
                    kick=[tuple(int(x) for x in c) for c in cuts],
                    accepted=bool(accepted),
                    current_cost=cur_cost,
                )
            if self._stop_requested:
                reason = "callback"
                break
            if self.time_limit is not None and perf_counter() - t0 > self.time_limit:
                reason = "time_limit"
                break
            if self.patience is not None and since >= self.patience:
                reason = "patience"
                break
        if self.verbose:
            log.info(
                "IteratedLocalSearch: stopped (%s) after %d iterations, best %.6g",
                reason,
                len(history),
                best_cost,
            )
        self.history_ = np.asarray(history, dtype=np.float64)
        self.n_iter_ = len(history)
        self.stop_reason_ = reason
        return best
