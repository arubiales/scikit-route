"""``EnsembleGenetic`` and ``EnsembleSimulatedAnnealing``: the 1.0 ensembles as explicit-parameter
wrappers over :class:`~skroute.ensemble.MultiStart` (SPEC §4.5, D17)."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags
from ..metaheuristics import Genetic, SimulatedAnnealing
from ..problem import RoutingProblem
from ..utils._param_validation import Interval
from ._multistart import _N_JOBS, MultiStart

__all__ = ["EnsembleGenetic", "EnsembleSimulatedAnnealing"]

_COPIED = ("history_", "n_iter_", "stop_reason_", "estimators_", "costs_", "best_index_", "best_estimator_")


def _inner_constraints(solver: type[BaseRouter]) -> dict[str, Any]:
    """The wrapped solver's constraints for the knobs the wrapper re-exposes (``random_state``/``verbose``
    are the wrapper's own and keep the same spelling)."""
    return dict(solver._parameter_constraints)


class _EnsembleBase(BaseRouter):
    """Shared ``_solve`` of the two wrappers: a ``MultiStart`` of the inner solver fed with the outer rng."""

    estimators_: list[BaseRouter]
    costs_: np.ndarray
    best_index_: int
    best_estimator_: BaseRouter

    def _inner(self) -> BaseRouter:  # pragma: no cover - overridden
        raise NotImplementedError

    def _n_restarts(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="ensemble", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None  # stochastic tag: the base class always hands a Generator
        ms = MultiStart(
            self._inner(),
            n_restarts=self._n_restarts(),
            n_jobs=self.n_jobs,  # type: ignore[attr-defined]
            prefer="threads",
            random_state=rng,  # the whole run consumes exactly the outer random_state
            verbose=self.verbose,  # type: ignore[attr-defined]
        ).fit(problem)
        for attr in _COPIED:
            setattr(self, attr, getattr(ms, attr))
        return problem.to_index_tour(ms.tour_)


class EnsembleGenetic(_EnsembleBase):
    """``n_genetics`` independent :class:`~skroute.metaheuristics.Genetic` runs in parallel; the best wins.

    .. deprecated:: 2.0
        Kept for 1.0 users (``skroute.metaheuristics.genetics.EnsembleGenetic``); it is a thin
        wrapper over ``MultiStart(Genetic(...), n_restarts=n_genetics)`` and will be removed in
        3.0. New code should use :class:`~skroute.ensemble.MultiStart` directly. No warning is
        emitted at runtime.

    Parameters
    ----------
    n_genetics : int >= 1, default 10
        Number of independent genetic runs (the ``n_restarts`` of ``MultiStart``).
    n_jobs : int or None, default None
        Workers for :class:`joblib.Parallel` (threads); ``None`` runs the runs one after another,
        ``-1`` uses every CPU. Never changes the result.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the run seeds. The same seed on the same machine gives bit-identical results
        whatever ``n_jobs``; a passed ``Generator`` is advanced by the fit.
    verbose : int, default 0
        ``0`` is silent; ``1`` logs the finished runs and a summary at INFO; ``2`` logs every
        run. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.
    pop_size : int >= 2, default 100
        Individuals per generation of every run.
    n_generations : int >= 1, default 500
        Maximum number of generations of every run.
    crossover : {"ox", "pmx"}, default "ox"
        Permutation crossover.
    p_crossover : float in [0, 1], default 0.9
        Probability that a child is produced by crossover.
    mutation : {"inversion", "swap", "insertion"}, default "inversion"
        Mutation operator.
    p_mutation : float in [0, 1], default 0.2
        Mutation probability per child.
    tournament_size : int >= 1, default 3
        Tournament size of the parent selection.
    n_elite : int >= 0, default 2
        Best parents copied unchanged into the next generation.
    local_search : None, str or tuple of {"two_opt", "or_opt"}, default None
        Memetic polish applied to every child (``None`` = plain genetic algorithm).
    patience : int >= 1 or None, default 100
        Generations without improvement before a run stops (``"patience"``).
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        Starting individual of every run.
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds of every run (``"time_limit"``). Breaks bit-exact
        reproducibility across machines.

    Attributes
    ----------
    estimators_ : list of Genetic
        The ``n_genetics`` fitted runs.
    costs_ : ndarray of shape (n_genetics,), float64
        ``cost_`` of every run.
    best_index_ : int
        Index of the winning run (the lowest index on a tie).
    best_estimator_ : Genetic
        ``estimators_[best_index_]``.
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far cost per generation of the winning run.
    n_iter_ : int
        Generations run by the winning run.
    stop_reason_ : {"max_iter", "patience", "time_limit"}
        Why the winning run stopped.

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and
    the other fitted attributes shared by every solver.

    Notes
    -----
    Every knob after ``verbose`` is keyword-only and takes the 2.0 default of
    :class:`~skroute.metaheuristics.Genetic` (1.0 defaulted to ``pop=400, gen=1000``). The
    parameters are explicit so that ``get_params``/``set_params``/``clone`` work without a
    nested estimator; ``EnsembleGenetic(n_genetics=k, random_state=s, **knobs)`` returns exactly
    what ``MultiStart(Genetic(**knobs), n_restarts=k, random_state=s)`` returns.

    **Supports:** symmetric and asymmetric matrices, the multi-trip objective (both split
    rules); stochastic, iterative, budget-aware.

    Examples
    --------
    >>> from skroute import EnsembleGenetic
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
    >>> eg = EnsembleGenetic(n_genetics=4, random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
    >>> eg.cost_ / wi.optimal_tour_length < 1.15  # the fast-tier tolerance of the plain GA
    True
    >>> len(eg.estimators_) == 4 and eg.cost_ == float(eg.costs_.min())
    True
    >>> int(eg.route_[0]) == int(eg.route_[-1]) == int(eg.depot_) == 1
    True
    >>> eg.n_iter_ == len(eg.history_) and eg.stop_reason_ in {"patience", "max_iter"}
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "n_genetics": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": _N_JOBS,
        **_inner_constraints(Genetic),
    }

    def __init__(
        self,
        n_genetics: int = 10,
        n_jobs: int | None = None,
        random_state: Any = None,
        verbose: int = 0,
        *,
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
    ) -> None:
        self.n_genetics = n_genetics
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
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

    def _inner(self) -> BaseRouter:
        return Genetic(
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            crossover=self.crossover,
            p_crossover=self.p_crossover,
            mutation=self.mutation,
            p_mutation=self.p_mutation,
            tournament_size=self.tournament_size,
            n_elite=self.n_elite,
            local_search=self.local_search,
            patience=self.patience,
            init=self.init,
            time_limit=self.time_limit,
        )

    def _n_restarts(self) -> int:
        return int(self.n_genetics)


class EnsembleSimulatedAnnealing(_EnsembleBase):
    """``n_simulateds`` independent :class:`~skroute.metaheuristics.SimulatedAnnealing` runs; the best wins.

    .. deprecated:: 2.0
        Kept for 1.0 users (``skroute.metaheuristics.simulated_annealing.EnsembleSimulatedAnnealing``);
        it is a thin wrapper over ``MultiStart(SimulatedAnnealing(...), n_restarts=n_simulateds)``
        and will be removed in 3.0. New code should use :class:`~skroute.ensemble.MultiStart`
        directly. No warning is emitted at runtime.

    Parameters
    ----------
    n_simulateds : int >= 1, default 10
        Number of independent annealing runs (the ``n_restarts`` of ``MultiStart``). 1.0
        defaulted to 20.
    n_jobs : int or None, default None
        Workers for :class:`joblib.Parallel` (threads); ``None`` runs the runs one after another,
        ``-1`` uses every CPU. Never changes the result.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the run seeds. The same seed on the same machine gives bit-identical results
        whatever ``n_jobs``; a passed ``Generator`` is advanced by the fit.
    verbose : int, default 0
        ``0`` is silent; ``1`` logs the finished runs and a summary at INFO; ``2`` logs every
        run. Records go to the ``skroute`` logger at INFO; enable them with
        ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.
    t0 : float > 0 or "auto", default "auto"
        Initial temperature of every run (``"auto"`` calibrates it on the initial tour).
    t_min : float > 0 or "auto", default "auto"
        Final temperature (``"auto"`` is ``1e-4 * t0``).
    alpha : float in (0, 1), default 0.995
        Geometric cooling factor.
    n_moves : int >= 1 or None, default None
        Proposals per temperature level; ``None`` means ``10 * n``.
    moves : tuple of {"two_opt", "or_opt", "swap"}, default ("two_opt", "or_opt", "swap")
        Move types proposed.
    patience : int >= 1 or None, default None
        Levels without improvement before a run stops, counted once the current cost has fallen
        below the initial cost (``"patience"``); ``None`` disables.
    init : {"nearest_neighbour", "random"} or array-like of labels, default "nearest_neighbour"
        Starting tour of every run.
    time_limit : float > 0 or None, default None
        Wall-clock budget in seconds of every run (``"time_limit"``). Breaks bit-exact
        reproducibility across machines.

    Attributes
    ----------
    estimators_ : list of SimulatedAnnealing
        The ``n_simulateds`` fitted runs.
    costs_ : ndarray of shape (n_simulateds,), float64
        ``cost_`` of every run.
    best_index_ : int
        Index of the winning run (the lowest index on a tie).
    best_estimator_ : SimulatedAnnealing
        ``estimators_[best_index_]``.
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far cost per temperature level of the winning run.
    n_iter_ : int
        Temperature levels run by the winning run.
    stop_reason_ : {"converged", "patience", "time_limit"}
        Why the winning run stopped.

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and
    the other fitted attributes shared by every solver.

    Notes
    -----
    Every knob after ``verbose`` is keyword-only and takes the 2.0 default of
    :class:`~skroute.metaheuristics.SimulatedAnnealing` (1.0 defaulted to ``temp=12.0,
    neighbours=250, delta=0.78, tol=1.29``). ``EnsembleSimulatedAnnealing(n_simulateds=k,
    random_state=s, **knobs)`` returns exactly what
    ``MultiStart(SimulatedAnnealing(**knobs), n_restarts=k, random_state=s)`` returns.

    **Supports:** symmetric and asymmetric matrices, the multi-trip objective (both split
    rules); stochastic, iterative, budget-aware.

    Examples
    --------
    >>> from skroute import EnsembleSimulatedAnnealing
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
    >>> es = EnsembleSimulatedAnnealing(n_simulateds=4, random_state=0)
    >>> es = es.fit(wi.distance_matrix(), labels=wi.labels)
    >>> es.cost_ / wi.optimal_tour_length < 1.03  # the fast-tier tolerance of SimulatedAnnealing
    True
    >>> len(es.estimators_) == 4 and es.cost_ == float(es.costs_.min())
    True
    >>> int(es.route_[0]) == int(es.route_[-1]) == int(es.depot_) == 1
    True
    >>> es.n_iter_ == len(es.history_) and es.stop_reason_ == "converged"
    True
    """

    _parameter_constraints: dict[str, Any] = {
        "n_simulateds": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": _N_JOBS,
        **_inner_constraints(SimulatedAnnealing),
    }

    def __init__(
        self,
        n_simulateds: int = 10,
        n_jobs: int | None = None,
        random_state: Any = None,
        verbose: int = 0,
        *,
        t0: float | str = "auto",
        t_min: float | str = "auto",
        alpha: float = 0.995,
        n_moves: int | None = None,
        moves: tuple[str, ...] | str = ("two_opt", "or_opt", "swap"),
        patience: int | None = None,
        init: Any = "nearest_neighbour",
        time_limit: float | None = None,
    ) -> None:
        self.n_simulateds = n_simulateds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.t0 = t0
        self.t_min = t_min
        self.alpha = alpha
        self.n_moves = n_moves
        self.moves = moves
        self.patience = patience
        self.init = init
        self.time_limit = time_limit

    def _inner(self) -> BaseRouter:
        return SimulatedAnnealing(
            t0=self.t0,
            t_min=self.t_min,
            alpha=self.alpha,
            n_moves=self.n_moves,
            moves=self.moves,
            patience=self.patience,
            init=self.init,
            time_limit=self.time_limit,
        )

    def _n_restarts(self) -> int:
        return int(self.n_simulateds)
