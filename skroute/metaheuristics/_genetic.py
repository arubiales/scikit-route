"""``Genetic``: a generational genetic algorithm with real permutation crossovers (SPEC §4.4)."""

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
from ._ga import evaluate_population, ga_generation

__all__ = ["Genetic"]

log = logging.getLogger("skroute")

_MOVE_BITS = {"two_opt": 1, "or_opt": 2}
_CROSSOVERS = {"ox": 0, "pmx": 1}
_MUTATIONS = {"inversion": 0, "swap": 1, "insertion": 2}
_LS_NONE, _LS_SYMMETRIC, _LS_GENERIC = 0, 1, 2
_N_CANDIDATES = 10  # candidate-list size of the memetic descents (the §4.3 default)


def normalise_local_search(local_search: Any, name: str) -> tuple[tuple[str, ...], int]:
    """``(moves, mask)`` from a ``local_search`` parameter (glossary §4.0).

    ``None`` -> ``((), 0)``; a string is a 1-tuple; a tuple or list must be a subset of
    ``{"two_opt", "or_opt"}`` (duplicates are folded). ``mask`` is the core's move bit mask
    (1 = two_opt, 2 = or_opt).
    """
    if local_search is None:
        return (), 0
    moves = (local_search,) if isinstance(local_search, str) else tuple(local_search)
    bad = [mv for mv in moves if not isinstance(mv, str) or mv not in _MOVE_BITS]
    if bad or not moves:
        raise ValueError(
            f"The 'local_search' parameter of {name} must be None or a tuple of move names among "
            f"{{'or_opt', 'two_opt'}}. Got {local_search!r} instead."
        )
    ordered = tuple(mv for mv in ("two_opt", "or_opt") if mv in moves)
    mask = sum(_MOVE_BITS[mv] for mv in ordered)
    return ordered, mask


def improvement(new: float, best: float) -> bool:
    """The improvement test of SPEC §4.0: ``new < best - 1e-9 * max(1, |best|)``.

    An infinite ``best`` (no incumbent yet) is beaten by any finite ``new``: the literal formula would
    compare against ``inf - inf = nan`` and never accept.
    """
    if not math.isfinite(best):
        return new < best
    return new < best - 1e-9 * max(1.0, abs(best))


class Genetic(BaseRouter):
    """Genetic algorithm over giant tours with OX/PMX crossover, elitism and an optional memetic polish.

    A chromosome is the tour without its depot (``tour[1:]``), so every individual is a valid
    giant tour; its fitness is the problem objective (plain, greedy-split or optimal-split
    cost), which makes the multi-trip budget steer the population directly.

    Parameters
    ----------
    pop_size : int >= 2, default 100
        Individuals per generation. The initial population is one individual from ``init``
        plus ``pop_size - 1`` random permutations.
    n_generations : int >= 1, default 500
        Maximum number of generations (outer iterations).
    crossover : {"ox", "pmx"}, default "ox"
        Order crossover (a segment of the first parent, the rest in the second parent's order)
        or partially mapped crossover (segment copy plus mapping repair).
    p_crossover : float in [0, 1], default 0.9
        Probability that a child is produced by crossover rather than copied from its first parent.
    mutation : {"inversion", "swap", "insertion"}, default "inversion"
        Inversion reverses a random segment (a 2-opt move), swap exchanges two genes, insertion
        moves one gene to another position.
    p_mutation : float in [0, 1], default 0.2
        Probability that a child is mutated.
    tournament_size : int >= 1, default 3
        Contestants of every tournament selection (drawn with replacement); the fittest wins.
    n_elite : int >= 0, default 2
        Best parents copied unchanged into the next generation. Must be smaller than ``pop_size``.
    local_search : None, str or tuple of {"two_opt", "or_opt"}, default None
        Memetic polish applied to every child: the listed descents run to convergence
        (2-opt and Or-opt with 10-nearest-neighbour candidate lists; the full-evaluation
        generic descent under a budget or on an asymmetric matrix). A single string is a
        1-tuple. ``("two_opt",)`` is the "memetic" configuration of the benchmarks.
    patience : int >= 1 or None, default 100
        Generations without improvement of the best-so-far before stopping; ``None`` disables.
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        The first individual: a nearest-neighbour tour, a random permutation, or the ``tour_``
        / ``route_`` of another solver (warm start).
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds, checked once per generation. Breaks bit-exact
        reproducibility (the stopping generation depends on the machine).
    random_state : None, int or numpy.random.Generator, default None
        Seed of the generator that draws every random quantity (pre-drawn per generation, D10).
    verbose : int, default 0
        0 is silent; 1 logs every ``max(1, n_generations // 10)`` generations; 2 logs every
        generation. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far objective after each generation (non-increasing).
    n_iter_ : int
        Generations actually run.
    stop_reason_ : {"max_iter", "patience", "time_limit", "callback"}
        Why the search stopped (``"callback"``: the ``callback`` of ``fit`` returned ``True``).
    n_duplicates_ : int
        Children that duplicated an individual of their generation and were mutated once more.

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and the
    other fitted attributes shared by every solver.

    Notes
    -----
    Per generation: the ``n_elite`` fittest parents are copied; each of the other children
    picks two parents by tournament — the fitter winner is *parent 1*: it contributes the
    segment ``[a, b]`` of the crossover, the other contributes its order (OX) or positions
    (PMX), which lets a child inherit a block of the better solution —, is produced by
    crossover with probability ``p_crossover`` (else copied from parent 1), mutated with
    probability ``p_mutation``, mutated once
    more if it exactly duplicates an individual already in the new generation, optionally
    polished by ``local_search`` and evaluated with the problem objective in one kernel call.
    Every random quantity of a generation is drawn beforehand from ``random_state`` and handed
    to the ``nogil`` kernel as arrays, so two fits with the same seed are bit-identical.
    Complexity: O(pop_size * n) per generation without polish; with polish the 2-opt/Or-opt
    descents dominate (O(pop_size * n * k) per generation, k = 10 candidates, more under a
    budget or on an asymmetric matrix where every move is re-evaluated in O(n)).

    Callback events (D30): ``"start"`` carries the ``init`` individual as ``tour`` and the best of
    the initial population as ``best_tour``; every generation emits one ``"iteration"`` whose
    ``tour`` is the generation's best individual and ``best_tour`` the run's best, with the
    ``extra`` keys ``generation`` (generations completed), ``n_evaluations`` (objective
    evaluations so far), ``mean_cost`` (mean objective of the population) and ``n_duplicates``.

    Supports: symmetric and asymmetric matrices, multi-trip objective; stochastic, iterative,
    budget-aware. A truncation ("top X %") selection is planned for 2.1 (issue #37).

    References
    ----------
    .. [1] L. Davis, "Applying adaptive algorithms to epistatic domains", IJCAI 1985 (OX).
    .. [2] D. E. Goldberg and R. Lingle, "Alleles, loci, and the traveling salesman problem",
       ICGA 1985 (PMX).
    .. [3] P. Moscato, "On evolution, search, optimization, genetic algorithms and martial arts:
       towards memetic algorithms", Caltech report 826, 1989.

    Examples
    --------
    >>> from skroute import Genetic
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
    >>> C = wi.distance_matrix()
    >>> ga = Genetic(random_state=0).fit(C, labels=wi.labels)
    >>> ga.cost_ / wi.optimal_tour_length < 1.15  # the fast-tier tolerance of the plain GA
    True
    >>> int(ga.route_[0]) == int(ga.route_[-1]) == int(ga.depot_) == 1
    True
    >>> ga.n_iter_ == len(ga.history_) and ga.stop_reason_ in {"patience", "max_iter"}
    True
    >>> memetic = Genetic(local_search=("two_opt",), random_state=0).fit(C, labels=wi.labels)
    >>> memetic.cost_ / wi.optimal_tour_length < 1.05
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "pop_size": [Interval(Integral, 2, None, closed="left")],
        "n_generations": [Interval(Integral, 1, None, closed="left")],
        "crossover": [Options(str, set(_CROSSOVERS))],
        "p_crossover": [Interval(Real, 0.0, 1.0, closed="both")],
        "mutation": [Options(str, set(_MUTATIONS))],
        "p_mutation": [Interval(Real, 0.0, 1.0, closed="both")],
        "tournament_size": [Interval(Integral, 1, None, closed="left")],
        "n_elite": [Interval(Integral, 0, None, closed="left")],
        "local_search": [None, str, tuple, list],
        "patience": [Interval(Integral, 1, None, closed="left"), None],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "time_limit": [Interval(Real, 0.0, None, closed="neither"), None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    history_: np.ndarray
    n_iter_: int
    stop_reason_: str
    n_duplicates_: int

    def __init__(
        self,
        pop_size: int = 100,
        n_generations: int = 500,
        crossover: str = "ox",
        p_crossover: float = 0.9,
        mutation: str = "inversion",
        p_mutation: float = 0.2,
        tournament_size: int = 3,
        n_elite: int = 2,
        local_search: Any = None,
        patience: int | None = 100,
        init: Any = "nearest_neighbour",
        time_limit: float | None = None,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover = crossover
        self.p_crossover = p_crossover
        self.mutation = mutation
        self.p_mutation = p_mutation
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.local_search = local_search
        self.patience = patience
        self.init = init
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None
        t0 = perf_counter()
        n, m, depot = problem.n, problem.n - 1, problem.depot
        pop_size, n_elite = int(self.pop_size), int(self.n_elite)
        if n_elite >= pop_size:
            raise ValueError(
                f"The 'n_elite' parameter of Genetic must be smaller than pop_size={pop_size}. "
                f"Got {self.n_elite!r} instead."
            )
        _, ls_moves = normalise_local_search(self.local_search, "Genetic")
        if ls_moves == 0:
            ls_mode = _LS_NONE
        elif problem.symmetric and not problem.multi_trip:
            ls_mode = _LS_SYMMETRIC
        else:
            ls_mode = _LS_GENERIC
        # candidate lists of the memetic descents; the plain GA never reads them, so it skips the
        # transient (n, n) copy ``neighbours`` makes (an empty (n, 0) view satisfies the kernel signature)
        cand = (
            problem.neighbours(min(_N_CANDIDATES, n - 1))
            if ls_mode != _LS_NONE
            else np.empty((n, 0), np.int64)
        )
        C, T = problem.cost, problem.time_or_cost
        max_time, fixed, split = problem.max_time_work, problem.fixed_cost, problem.split_code
        cx, mut_kind = _CROSSOVERS[self.crossover], _MUTATIONS[self.mutation]
        p_cx, p_mut, k_t = float(self.p_crossover), float(self.p_mutation), int(self.tournament_size)

        # scratch buffers, allocated once (never shared across threads)
        i64, f64, u8 = np.int64, np.float64, np.uint8
        tour, pos, scratch = np.empty(n, i64), np.empty(n, i64), np.empty(n, i64)
        dont_look, present = np.zeros(n, u8), np.zeros(n, u8)
        dp, pred, mapping = np.empty(n, f64), np.empty(n, i64), np.empty(n, i64)
        par1, par2, child = np.empty(m, i64), np.empty(m, i64), np.empty(m, i64)
        hashes = np.empty(pop_size, np.uint64)

        # initial population: one individual from init, the rest random permutations
        pop = np.empty((pop_size, m), dtype=i64)
        pop[0] = initial_tour(problem, self.init, rng)[1:]
        others = np.delete(np.arange(n, dtype=i64), depot)
        if pop_size > 1:
            pop[1:] = rng.permuted(np.tile(others, (pop_size - 1, 1)), axis=1)
        fit = np.empty(pop_size, dtype=f64)
        evaluate_population(C, T, pop, depot, max_time, fixed, split, tour, dp, pred, fit)
        new_pop, new_fit = np.empty_like(pop), np.empty_like(fit)
        best_idx = int(np.argmin(fit))
        best_cost, best = float(fit[best_idx]), pop[best_idx].copy()
        n_children = pop_size - n_elite
        if self._callback is not None:  # D30: the init individual, and the best of the initial population
            self._emit(
                "start",
                0,
                np.concatenate(([depot], pop[0])),
                float(fit[0]),
                np.concatenate(([depot], best)),
                best_cost,
                generation=0,
                n_evaluations=pop_size,
            )

        every = max(1, int(self.n_generations) // 10) if self.verbose == 1 else 1
        history: list[float] = []
        since, n_dups, reason = 0, 0, "max_iter"
        for gen in range(int(self.n_generations)):
            # every random quantity of the generation, pre-drawn (D10)
            tourn = rng.integers(0, pop_size, size=(2, n_children, k_t), dtype=i64)
            u_cross = rng.random(n_children)
            cuts = np.sort(rng.integers(0, m, size=(n_children, 2), dtype=i64), axis=1)
            u_mut = rng.random(n_children)
            mut = rng.integers(0, m, size=(n_children, 2), dtype=i64)
            remut = rng.integers(0, m, size=(n_children, 2), dtype=i64)
            elite_idx = np.ascontiguousarray(np.argsort(fit, kind="stable")[:n_elite], dtype=i64)
            n_dups += ga_generation(
                C, T, max_time, fixed, split, depot, pop, fit, new_pop, new_fit, elite_idx,
                np.ascontiguousarray(tourn), u_cross, np.ascontiguousarray(cuts), u_mut, mut, remut,
                p_cx, p_mut, cx, mut_kind, ls_mode, ls_moves, cand,
                par1, par2, child, tour, pos, dont_look, scratch, dp, pred, present, mapping, hashes,
            )  # fmt: skip
            pop, new_pop, fit, new_fit = new_pop, pop, new_fit, fit
            k = int(np.argmin(fit))
            if improvement(float(fit[k]), best_cost):
                best_cost, best, since = float(fit[k]), pop[k].copy(), 0
            else:
                since += 1
            history.append(best_cost)
            if self.verbose and gen % every == 0:
                log.info("Genetic generation %d: best %.6f, mean %.6f", gen, best_cost, float(fit.mean()))
            if self._callback is not None:
                # D30: the best individual of this generation is the current tour, the run's elite the best
                self._emit(
                    "iteration",
                    gen + 1,
                    np.concatenate(([depot], pop[k])),
                    float(fit[k]),
                    np.concatenate(([depot], best)),
                    best_cost,
                    generation=gen + 1,
                    n_evaluations=pop_size + (gen + 1) * n_children,
                    mean_cost=float(fit.mean()),
                    n_duplicates=int(n_dups),
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
            log.info("Genetic stopped after %d generations (%s): best %.6f", len(history), reason, best_cost)
        self.history_ = np.asarray(history, dtype=f64)
        self.n_iter_ = len(history)
        self.stop_reason_ = reason
        self.n_duplicates_ = int(n_dups)
        return np.concatenate(([depot], best)).astype(i64)
