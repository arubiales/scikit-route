"""``AntColony``: a MAX-MIN Ant System with candidate lists and a per-ant 2-opt polish (SPEC §4.4)."""

from __future__ import annotations

import logging
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._init_tour import initial_tour
from ..utils._param_validation import Interval
from ._aco import construct_tours, polish_and_evaluate
from ._genetic import improvement, normalise_local_search

__all__ = ["AntColony"]

log = logging.getLogger("skroute")

_LS_NONE, _LS_SYMMETRIC, _LS_GENERIC = 0, 1, 2
_GLOBAL_BEST_EVERY = 5  # the global-best ant deposits on every 5th iteration (MMAS schedule)


class AntColony(BaseRouter):
    """MAX-MIN Ant System (MMAS) over candidate lists, with an optional local-search polish per ant.

    Every iteration ``n_ants`` ants build a tour from the depot by a roulette wheel over the
    unvisited candidates of their current node; each tour is optionally polished by the
    listed descents and priced with the problem objective (plain, greedy-split or
    optimal-split cost), then the pheromone evaporates and the iteration-best ant — the
    global-best one every fifth iteration — deposits ``1 / cost`` on its arcs. The trail is
    kept inside ``[tau_min, tau_max]`` so the colony never stagnates completely.

    Parameters
    ----------
    n_ants : int >= 1 or None, default None
        Ants per iteration; ``None`` means ``min(n, 50)``.
    alpha : float >= 0, default 1.0
        Exponent of the pheromone in the transition weights ``tau ** alpha * (1 / C) ** beta``.
    beta : float >= 0, default 2.0
        Exponent of the heuristic desirability ``1 / C``.
    rho : float in (0, 1), default 0.02
        Evaporation rate: ``tau *= 1 - rho`` every iteration; also scales the trail bounds
        ``tau_max = 1 / (rho * L_best)`` and ``tau_min = tau_max / (2 n)``.
    n_iter : int >= 1, default 200
        Maximum number of iterations (outer iterations).
    n_candidates : int >= 1 or None, default 20
        Size of the nearest-neighbour candidate lists the ants choose from (all unvisited nodes
        once a list is exhausted); ``None`` uses the full neighbourhood.
    local_search : None, str or tuple of {"two_opt", "or_opt"}, default ("two_opt",)
        Polish applied to every ant's tour: the listed descents run to convergence with the
        same candidate lists (the full-evaluation generic descent under a budget or on an
        asymmetric matrix). A single string is a 1-tuple; ``None`` disables the polish.
    patience : int >= 1 or None, default 50
        Iterations without improvement of the best-so-far before stopping; ``None`` disables.
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds, checked once per iteration. Breaks bit-exact
        reproducibility (the stopping iteration depends on the machine).
    random_state : None, int or numpy.random.Generator, default None
        Seed of the generator; one uniform per construction step is pre-drawn per iteration (D10).
    verbose : int, default 0
        0 is silent; 1 logs every ``max(1, n_iter // 10)`` iterations; 2 logs every iteration.
        Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far objective after each iteration (non-increasing).
    n_iter_ : int
        Iterations actually run.
    stop_reason_ : {"max_iter", "patience", "time_limit"}
        Why the search stopped.
    pheromone_ : ndarray of shape (n, n), float64
        The trail matrix at the end of the search (index space, matrix row order).

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and the
    other fitted attributes shared by every solver.

    Notes
    -----
    The trail starts at ``1 / (rho * L_NN)`` with ``L_NN`` the objective of the
    nearest-neighbour tour; the bounds follow the best-so-far cost. Transition weights are
    computed once per iteration as a dense ``(n, n)`` matrix, so an iteration costs
    O(n^2 + n_ants * n * k) plus the polish (O(n k) per ant on a symmetric plain instance,
    more under a budget or on an asymmetric matrix where every move is re-evaluated in O(n)).
    Memory: two ``(n, n)`` float64 matrices besides the cost matrix.

    Supports: symmetric and asymmetric matrices (the trail is directional on the latter),
    multi-trip objective; stochastic, iterative, budget-aware.

    References
    ----------
    .. [1] T. Stützle and H. H. Hoos, "MAX-MIN Ant System", Future Generation Computer Systems
       16(8), 2000, 889-914.
    .. [2] M. Dorigo and T. Stützle, *Ant Colony Optimization*, MIT Press, 2004.

    Examples
    --------
    >>> from skroute import AntColony
    >>> from skroute.datasets import load_tsp
    >>> dj = load_tsp("dj38")  # Djibouti, optimum 6656
    >>> C = dj.distance_matrix()
    >>> aco = AntColony(random_state=0).fit(C, labels=dj.labels)
    >>> aco.cost_ / dj.optimal_tour_length < 1.08  # the fast-tier tolerance of §6
    True
    >>> int(aco.route_[0]) == int(aco.route_[-1]) == int(aco.depot_) == 1
    True
    >>> aco.n_iter_ == len(aco.history_) and aco.stop_reason_ in {"patience", "max_iter"}
    True
    >>> aco.pheromone_.shape == (38, 38)
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "n_ants": [Interval(Integral, 1, None, closed="left"), None],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "beta": [Interval(Real, 0.0, None, closed="left")],
        "rho": [Interval(Real, 0.0, 1.0, closed="neither")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "n_candidates": [Interval(Integral, 1, None, closed="left"), None],
        "local_search": [None, str, tuple, list],
        "patience": [Interval(Integral, 1, None, closed="left"), None],
        "time_limit": [Interval(Real, 0.0, None, closed="neither"), None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    history_: np.ndarray
    n_iter_: int
    stop_reason_: str
    pheromone_: np.ndarray

    def __init__(
        self,
        n_ants: int | None = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.02,
        n_iter: int = 200,
        n_candidates: int | None = 20,
        local_search: Any = ("two_opt",),
        patience: int | None = 50,
        time_limit: float | None = None,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.local_search = local_search
        self.patience = patience
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None
        t0 = perf_counter()
        n, depot = problem.n, problem.depot
        n_ants = min(n, 50) if self.n_ants is None else int(self.n_ants)
        _, ls_moves = normalise_local_search(self.local_search, "AntColony")
        if ls_moves == 0:
            ls_mode = _LS_NONE
        elif problem.symmetric and not problem.multi_trip:
            ls_mode = _LS_SYMMETRIC
        else:
            ls_mode = _LS_GENERIC
        k = n - 1 if self.n_candidates is None else min(int(self.n_candidates), n - 1)
        cand = problem.neighbours(k)
        C, T = problem.cost, problem.time_or_cost
        max_time, fixed, split = problem.max_time_work, problem.fixed_cost, problem.split_code
        alpha, beta, rho = float(self.alpha), float(self.beta), float(self.rho)

        # heuristic desirability (1 / C) ** beta; zero distances (coincident points) are floored so
        # they stay finite and are still overwhelmingly preferred
        off = C[~np.eye(n, dtype=bool)]
        positive = off[off > 0]
        floor = float(positive.min()) * 1e-3 if positive.size else 1.0
        heur = np.power(np.maximum(C, floor), -beta)
        np.fill_diagonal(heur, 0.0)

        # trail: 1 / (rho * L_NN), bounded by the best-so-far cost from the first iteration on
        nn = initial_tour(problem, "nearest_neighbour", None)
        l_nn = float(problem.evaluate(nn))
        tau = np.full((n, n), 1.0 / (rho * l_nn))
        np.fill_diagonal(tau, 0.0)
        choice = np.empty_like(tau)

        i64, f64, u8 = np.int64, np.float64, np.uint8
        tours = np.empty((n_ants, n), dtype=i64)
        costs = np.empty(n_ants, dtype=f64)
        tour, pos, scratch = np.empty(n, i64), np.empty(n, i64), np.empty(n, i64)
        dont_look, visited, w = np.zeros(n, u8), np.zeros(n, u8), np.empty(k, f64)
        dp, pred = np.empty(n, f64), np.empty(n, i64)

        best_cost, best = np.inf, nn
        every = max(1, int(self.n_iter) // 10) if self.verbose == 1 else 1
        history: list[float] = []
        since, reason = 0, "max_iter"
        for it in range(int(self.n_iter)):
            np.multiply(np.power(tau, alpha), heur, out=choice)
            u = rng.random((n_ants, n - 1))  # one uniform per construction step (D10)
            construct_tours(choice, cand, depot, u, tours, visited, w)
            polish_and_evaluate(
                C, T, tours, max_time, fixed, split, ls_mode, ls_moves, cand,
                tour, pos, dont_look, scratch, dp, pred, costs,
            )  # fmt: skip
            ib = int(np.argmin(costs))
            if improvement(float(costs[ib]), best_cost):
                best_cost, best, since = float(costs[ib]), tours[ib].copy(), 0
            else:
                since += 1
            history.append(best_cost)
            # pheromone update: evaporation, one deposit, trail bounds
            tau *= 1.0 - rho
            if (it + 1) % _GLOBAL_BEST_EVERY == 0:
                dep_tour, dep_cost = best, best_cost
            else:
                dep_tour, dep_cost = tours[ib], float(costs[ib])
            delta = 1.0 / dep_cost
            heads, tails = dep_tour, np.roll(dep_tour, -1)
            tau[heads, tails] += delta
            if problem.symmetric:
                tau[tails, heads] += delta
            tau_max = 1.0 / (rho * best_cost)
            np.clip(tau, tau_max / (2.0 * n), tau_max, out=tau)
            np.fill_diagonal(tau, 0.0)
            if self.verbose and it % every == 0:
                log.info(
                    "AntColony iteration %d: best %.6f, iteration best %.6f", it, best_cost, float(costs[ib])
                )
            if self.time_limit is not None and perf_counter() - t0 > self.time_limit:
                reason = "time_limit"
                break
            if self.patience is not None and since >= self.patience:
                reason = "patience"
                break
        if self.verbose:
            log.info("AntColony stopped after %d iterations (%s): best %.6f", len(history), reason, best_cost)
        self.history_ = np.asarray(history, dtype=f64)
        self.n_iter_ = len(history)
        self.stop_reason_ = reason
        self.pheromone_ = tau
        return np.ascontiguousarray(best, dtype=i64)
