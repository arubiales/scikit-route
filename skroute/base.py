"""The estimator contract: ``RouterTags``, ``BaseRouter``, ``clone``, ``is_router``.

How a solver is written (the template method)
---------------------------------------------
Every solver subclasses [`BaseRouter`][skroute.base.BaseRouter] and follows four rules:

1. ``__init__`` stores every argument verbatim as an attribute of the same name and
   sets nothing else — that is what makes [`get_params`][skroute.base.BaseRouter.get_params],
   [`set_params`][skroute.base.BaseRouter.set_params], [`clone`][skroute.clone] and the
   print-changed-only ``repr`` work without any registration.
2. It declares ``_parameter_constraints`` (see ``skroute.utils._param_validation``);
   [`fit`][skroute.base.BaseRouter.fit] validates the hyper-parameters at fit time, never
   in ``__init__``.
3. It overrides ``_get_tags()`` and returns a [`RouterTags`][skroute.base.RouterTags] describing
   what it can do. The defaults are the *honest* ones (D28): a solver that forgets the
   override is advertised as budget-unaware (it warns under a budget), non-stochastic,
   non-iterative and non-exact.
4. It implements ``_solve(problem, rng) -> int64 array``: a permutation of ``range(problem.n)``
   with ``problem.depot`` at position 0, in **index space**. ``rng`` is a
   ``numpy.random.Generator`` when the solver is stochastic, else ``None``.

Duties inside ``_solve``: iterative solvers (``tags.iterative``) set ``self.history_``
(best-so-far cost after each outer iteration, monotone non-increasing), ``self.n_iter_``
(``== len(history_)``) and ``self.stop_reason_`` (one of ``"converged"``, ``"max_iter"``,
``"patience"``, ``"time_limit"``, ``"callback"`` — the subset each solver documents); exact
solvers (``tags.exact``) set ``self.is_optimal_``. Everything else — the label-space ``tour_``,
``route_``, ``trips_``, the recomputed ``cost_`` (D2), ``trip_costs_``, ``trip_times_``,
``fit_time_`` — is set by the base class, which also validates the returned tour and
raises ``RuntimeError`` on a solver bug. Assign ``history_`` as an array
(``np.asarray(history)``): ``fit`` converts a list, but the attribute is typed as an ndarray.

Progress callbacks (D30)
------------------------
``fit(..., callback=f)`` makes the solver report its progress: ``f`` receives one
[`RouteEvent`][skroute.base.RouteEvent] per event and may return ``True`` to ask the solver to
stop after the current outer iteration (``stop_reason_ = "callback"``). Inside ``_solve`` a
solver calls ``self._emit(stage, iteration, tour_idx, cost, best_tour_idx, best_cost, **extra)``
with **index-space** tours; the base class converts them to labels, builds the event and calls
the callback — and returns at once when no callback is set, so the fit path carries no overhead.
The duties: emit ``"start"`` once (iteration 0, with the initial tour when the search starts from
one), ``"iteration"`` once per outer iteration exactly where ``history_`` is appended (the tour
the solver is working on, the best-so-far, their costs, and solver-specific facts in ``extra``),
and after every iteration event break out of the loop with ``stop_reason_ = "callback"`` when
``self._stop_requested`` is set. **Never emit ``"end"``** (``_emit`` raises ``ValueError`` on it, as
on a second ``"start"``): the base class does, after the returned tour has been validated and
priced, so the end event carries exactly ``tour_`` and ``cost_``; a solver that emits no
``"start"`` gets a synthetic one (no tour) as well, so a construction or exact solver need not
call ``_emit`` at all. ``_callback``, ``_callback_state`` and ``_stop_requested`` live on the
instance for the duration of ``fit`` only: afterwards the class defaults stand in again, so a
fitted estimator carries no trace of the callback and a bare ``_solve`` never inherits a stale
stop request.
"""

from __future__ import annotations

import inspect
import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np

from .problem import RoutingProblem
from .utils._param_validation import validate_parameter_constraints
from .utils.validation import check_random_state

__all__ = ["BaseRouter", "RouteEvent", "RouterTags", "clone", "is_router"]

_STAGES = ("start", "iteration", "end")

log = logging.getLogger("skroute")

_FIT_KWARGS = ("depot", "coords", "labels", "max_time_work", "extra_cost", "people", "split")


@dataclass(frozen=True)
class RouterTags:
    """Capabilities of a solver, returned by ``BaseRouter._get_tags()``.

    Parameters
    ----------
    kind : {"exact", "construction", "local_search", "metaheuristic", "ensemble"}, default "metaheuristic"
        Family of the solver (D28); used by the capability table and the tolerance tests.
    exact : bool, default False
        Provably optimal for the objective it accepts; such solvers set ``is_optimal_``.
    stochastic : bool, default False
        Consumes ``random_state``; can be wrapped by ``MultiStart``.
    iterative : bool, default False
        Sets ``history_``, ``n_iter_`` and ``stop_reason_``.
    budget_aware : bool, default False
        The search itself sees the multi-trip objective. Solvers opt IN (D28): a solver
        that does not declare it warns under a budget and its result is still decoded
        and priced under the multi-trip objective (D6).
    requires_symmetric : bool, default False
        Raises on an asymmetric cost matrix (``ClarkeWright``).
    requires_coords : bool, default False
        Raises without ``coords=`` (``SOM``).
    max_nodes : int or None, default None
        Hard cap on the number of nodes (``BruteForce``, ``HeldKarp``, ``MILP``).

    Notes
    -----
    ``budget_aware=True`` is declared explicitly by: BruteForce, ClarkeWright, TwoOpt,
    OrOpt, LocalSearch, IteratedLocalSearch, SimulatedAnnealing, TabuSearch, Genetic,
    AntColony, EnsembleGenetic, EnsembleSimulatedAnnealing; MultiStart delegates to its
    estimator. Everything else warns under a budget.

    Examples
    --------
    >>> from skroute import RouterTags
    >>> RouterTags(kind="exact", exact=True, budget_aware=True, max_nodes=11).budget_aware
    True
    >>> RouterTags().budget_aware
    False
    """

    kind: str = (
        "metaheuristic"  # "exact" | "construction" | "local_search" | "metaheuristic" | "ensemble" (D28)
    )
    exact: bool = False  # provably optimal for the objective it accepts; sets is_optimal_
    stochastic: bool = False  # consumes random_state; MultiStart-able
    iterative: bool = False  # sets history_, n_iter_, stop_reason_
    budget_aware: bool = False  # the search itself sees the multi-trip objective; solvers opt IN (D28)
    requires_symmetric: bool = False  # raises on asymmetric X
    requires_coords: bool = False  # raises without coords=
    max_nodes: int | None = None  # hard cap on n (BruteForce, HeldKarp, MILP)


def _decode_trips(
    problem: RoutingProblem, tour: np.ndarray, starts: np.ndarray | None = None
) -> list[np.ndarray]:
    """Closed label trips ``[depot, ..., depot]`` of a valid index tour under the problem's split rule.

    The one decoder behind ``trips_``/``route_`` and ``RouteEvent.trips``/``RouteEvent.route``.
    """
    if starts is None:
        starts = problem.trip_starts(tour)
    lab = problem.labels
    d = lab[problem.depot : problem.depot + 1]  # 1-element array, keeps the label dtype
    return [np.concatenate((d, lab[tour[a:b]], d)) for a, b in pairwise(starts)]


def _join_trips(trips: list[np.ndarray]) -> np.ndarray:
    """The route as driven — depot, trip 1, depot, trip 2, ..., depot — from closed trips."""
    return np.concatenate([trips[0]] + [t[1:] for t in trips[1:]])


@dataclass(frozen=True, eq=False, repr=False)
class RouteEvent:
    """One progress report of a running solver, handed to the ``callback`` of ``fit`` (D30).

    Parameters
    ----------
    solver : str
        Class name of the solver that emitted the event (``"SimulatedAnnealing"``). Inside a
        ``MultiStart`` the inner solvers report under their own name, with ``extra["restart"]``.
    stage : {"start", "iteration", "end"}
        ``"start"`` once before the first outer iteration, ``"iteration"`` once per outer
        iteration (where ``history_`` grows), ``"end"`` once with the final result.
    iteration : int
        Outer iteration index: 0 at ``"start"``, ``1, 2, ...`` for the iterations, and the last
        iteration index at ``"end"`` — ``n_iter_`` for every iterative solver, the wrappers
        included (``MultiStart`` and the Ensembles emit no iterations of their own and report the
        winning restart's ``n_iter_``).
    cost : float
        Objective of ``tour`` as the solver knows it; ``nan`` when there is no tour yet.
    best_cost : float
        Objective of ``best_tour``; ``nan`` when there is none yet. Non-increasing over the
        events of one fit; at ``"end"`` it equals the recomputed ``cost_``.
    tour : ndarray of shape (n,) or None
        The solver's CURRENT solution as a label-space open giant tour, depot first — the tour it
        is working on (the annealing walker, the candidate of an iterated local search, the best
        individual of a generation...); ``None`` when the solver has no tour yet (``MILP`` before
        its first integral solution, a construction heuristic at ``"start"``).
    best_tour : ndarray of shape (n,) or None
        The best-so-far tour in the same format; at ``"end"`` exactly ``tour_``.
    problem : RoutingProblem
        The instance being solved (``est.problem_`` after the fit): labels, coordinates, budget.
    extra : dict
        Solver-specific facts: ``temperature`` (SimulatedAnnealing), ``tenure`` (TabuSearch),
        ``generation``/``n_evaluations`` (Genetic), ``kick`` (IteratedLocalSearch),
        ``moves_applied`` (the descents), ``radius``/``learning_rate``/``ring`` (SOM), ``n_ants``
        (AntColony), ``n_components``/``lower_bound`` (MILP), ``n_trips`` (ClarkeWright),
        ``n_edges`` (NRBS), ``restart`` (added by ``MultiStart`` to every forwarded event). Three
        keys are standardised by D31 so that viewers can draw the structure a solver is building:
        ``edges`` — a list of ``(label, label)`` tuples (the growing path or partial cycle of a
        construction heuristic, MILP's current LP support, the strongest pheromone trails of
        AntColony); ``edge_weights`` — floats in ``[0, 1]`` parallel to ``edges`` (pheromone
        strength, LP values); ``ring`` — an ``(m, 2)`` float array with the SOM neurons in the
        units of ``problem.coords``. Construction solvers emit one ``"iteration"`` event per step
        with ``tour=None``, ``cost=nan`` and ``edges``. Each solver's docstring lists its keys.

    Notes
    -----
    Events are frozen and compare by identity. The arrays are copies: a solver's buffers keep
    changing after the callback returns, the event does not. ``route`` and ``trips`` decode
    ``best_tour`` with the problem's own split rule (the one ``fit`` uses for ``route_`` and
    ``trips_``), so a multi-trip instance is drawn trip by trip.

    Examples
    --------
    A deterministic descent on four nodes: one event per stage, and the ``"end"`` event carries
    the fitted tour.

    >>> import numpy as np
    >>> from skroute import TwoOpt
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> events = []
    >>> est = TwoOpt().fit(C, callback=events.append)
    >>> [e.stage for e in events]
    ['start', 'iteration', 'end']
    >>> events[-1]
    RouteEvent(solver='TwoOpt', stage='end', iteration=1, best_cost=22)
    >>> events[-1].best_tour.tolist() == est.tour_.tolist(), events[-1].route.tolist()
    (True, [0, 1, 2, 3, 0])

    Returning ``True`` stops an iterative solver after the current outer iteration:

    >>> from skroute import SimulatedAnnealing
    >>> sa = SimulatedAnnealing(random_state=0).fit(C, callback=lambda e: e.stage == "iteration")
    >>> sa.n_iter_, sa.stop_reason_
    (1, 'callback')
    """

    solver: str
    stage: str
    iteration: int
    cost: float
    best_cost: float
    tour: np.ndarray | None
    best_tour: np.ndarray | None
    problem: RoutingProblem
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def trips(self) -> list[np.ndarray]:
        """``best_tour`` decoded into closed label trips ``[depot, ..., depot]``; ``[]`` without a tour."""
        if self.best_tour is None:
            return []
        return _decode_trips(self.problem, self.problem.to_index_tour(self.best_tour))

    @property
    def route(self) -> np.ndarray | None:
        """``best_tour`` as driven — depot, trip 1, depot, ..., depot — or ``None`` without a tour."""
        trips = self.trips
        return _join_trips(trips) if trips else None

    def __repr__(self) -> str:
        return (
            f"RouteEvent(solver={self.solver!r}, stage={self.stage!r}, "
            f"iteration={self.iteration}, best_cost={self.best_cost:.6g})"
        )


@dataclass
class _CallbackState:
    """Bookkeeping of one watched fit: the problem (for label conversion) and what was emitted so far."""

    problem: RoutingProblem
    started: bool = False
    last_iteration: int = 0


class BaseRouter:
    """Base class of every solver.

    Subclasses: store every ``__init__`` argument verbatim as an attribute of the same name and
    nothing else; declare ``_parameter_constraints``; override ``_get_tags()``; implement
    ``_solve(problem, rng) -> int64 array`` (a permutation of ``range(n)`` with ``problem.depot`` first).
    Iterative solvers set ``self.history_``, ``self.n_iter_``, ``self.stop_reason_`` inside ``_solve``;
    exact solvers set ``self.is_optimal_``. Everything else is set here.

    Attributes
    ----------
    problem_ : RoutingProblem
        The coerced instance (reusable: ``Other().fit(est.problem_)``).
    n_nodes_ : int
        Number of nodes.
    labels_ : ndarray of shape (n,), label dtype
        Labels in matrix row order.
    depot_ : scalar, label dtype
        The depot's label.
    tour_ : ndarray of shape (n,), label dtype
        Open giant tour, depot first — the warm-start format (``init=``).
    route_ : ndarray of shape (n + n_trips,), label dtype
        As driven: depot, trip 1, depot, trip 2, ..., depot.
    trips_ : list of ndarray
        One closed ``[depot, ..., depot]`` array per trip; ``len == 1`` for plain TSP.
    n_trips_ : int
        ``len(trips_)``.
    trip_costs_ : ndarray of shape (n_trips,), float64
        Travel cost of each closed trip (fixed charge excluded).
    trip_times_ : ndarray of shape (n_trips,), float64
        Only when a time matrix was given; each ``<= max_time_work + 1e-9``.
    cost_ : float
        ``trip_costs_.sum() + fixed_cost * (n_trips_ - 1)``, recomputed from the tour (D2).
    fit_time_ : float
        Seconds spent in ``_solve`` — including the time taken by the ``callback`` of ``fit``,
        which runs inside it (a live plot redrawn at every iteration inflates it; time a solver
        without a callback).
    history_ : ndarray of shape (n_iter_,), float64
        Iterative solvers only: best-so-far cost after each outer iteration.
    n_iter_ : int
        Iterative solvers only: outer iterations actually run.
    stop_reason_ : str
        Iterative solvers only: ``"converged" | "max_iter" | "patience" | "time_limit" |
        "callback"`` — the last one when the ``callback`` of ``fit`` returned ``True`` (D30).
    is_optimal_ : bool
        Exact solvers only.

    Notes
    -----
    ``fit`` never trusts a cost reported by the solver: the returned index tour is
    validated (a permutation with the depot first, else ``RuntimeError``) and
    ``cost_`` is recomputed with the problem's own decoder, so a route that does not
    match its cost is impossible by construction.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.base import BaseRouter, RouterTags
    >>> class Identity(BaseRouter):
    ...     '''Returns the nodes in matrix order.'''
    ...
    ...     def __init__(self, verbose=0):
    ...         self.verbose = verbose
    ...
    ...     def _get_tags(self):
    ...         return RouterTags(kind="construction")
    ...
    ...     def _solve(self, problem, rng):
    ...         return np.arange(problem.n)
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> est = Identity().fit(C)
    >>> est.tour_.tolist(), est.route_.tolist(), est.cost_, est.n_trips_
    ([0, 1, 2, 3], [0, 1, 2, 3, 0], 22.0, 1)
    >>> est
    Identity()
    >>> Identity(verbose=1)
    Identity(verbose=1)
    """

    _parameter_constraints: dict[str, Any] = {}

    # Fitted attributes (the table of SPEC §3.4), declared for type-checkers and the docs. They
    # exist on an instance only after ``fit``; ``_reset_fitted`` deletes them before a refit.
    problem_: RoutingProblem
    n_nodes_: int
    labels_: np.ndarray
    depot_: Any
    tour_: np.ndarray
    route_: np.ndarray
    trips_: list[np.ndarray]
    n_trips_: int
    trip_costs_: np.ndarray
    trip_times_: np.ndarray
    cost_: float
    fit_time_: float
    history_: np.ndarray
    n_iter_: int
    stop_reason_: str
    is_optimal_: bool

    # Callback plumbing (D30): ``fit`` sets these for its duration only and removes all three from the
    # instance afterwards (the class defaults keep a bare ``_solve`` call working and stop a fit's
    # stop request from leaking into the next one). ``_emit`` raises ``_stop_requested`` when the
    # callback returns True; every iterative solver honours it at its next outer-iteration boundary.
    _callback: Callable[[RouteEvent], Any] | None = None
    _callback_state: _CallbackState | None = None
    _stop_requested: bool = False

    # ---------- scikit-learn parameter protocol ----------
    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Sorted names of the ``__init__`` parameters (excluding ``self``, ``*args``, ``**kwargs``)."""
        sig = inspect.signature(cls.__init__)
        return sorted(
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind is not p.VAR_KEYWORD and p.kind is not p.VAR_POSITIONAL
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Parameters of this estimator.

        Parameters
        ----------
        deep : bool, default True
            If True, also return the parameters of nested estimators as ``<name>__<param>``.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{key}__{k}": v for k, v in value.get_params().items()})
            out[key] = value
        return out

    def set_params(self, **params: Any) -> BaseRouter:
        """Set the parameters of this estimator (``<name>__<param>`` reaches nested estimators).

        Returns
        -------
        self
        """
        if not params:
            return self
        valid = self._get_param_names()
        nested: dict[str, dict[str, Any]] = {}
        for full_key, value in params.items():
            key, delim, sub_key = full_key.partition("__")
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self!r}. Valid parameters are: {valid}."
                )
            if delim:
                nested.setdefault(key, {})[sub_key] = value
            else:
                setattr(self, key, value)
        for key, sub_params in nested.items():
            getattr(self, key).set_params(**sub_params)
        return self

    def __repr__(self) -> str:
        sig = inspect.signature(type(self).__init__).parameters
        parts = []
        for k, v in self.get_params(deep=False).items():
            default = sig[k].default
            same = (v is default) or (
                isinstance(v, type(default)) and not isinstance(v, np.ndarray) and _param_equal(v, default)
            )
            if not same:
                parts.append(f"{k}={v!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __eq__(
        self, other: object
    ) -> bool:  # equality of type and parameters; used by tests and clone checks
        if type(self) is not type(other):
            return False
        a, b = self.get_params(deep=False), other.get_params(deep=False)
        return all(_param_equal(a[k], b[k]) for k in a)  # dict == would raise on an ndarray init=

    __hash__ = None  # type: ignore[assignment]

    # ---------- capability protocol ----------
    def _get_tags(self) -> RouterTags:
        """Capabilities of the solver; override in every subclass (the defaults are the honest ones, D28)."""
        return RouterTags()

    # ---------- template method ----------
    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        """Solve ``problem`` and return an int64 index tour with the depot first (override)."""
        raise NotImplementedError

    def fit(
        self,
        X: Any,
        *,
        time_matrix: Any = None,
        depot: Any = None,
        coords: Any = None,
        labels: Any = None,
        max_time_work: float | None = None,
        extra_cost: float = 0.0,
        people: int = 1,
        split: str = "greedy",
        callback: Callable[[RouteEvent], Any] | None = None,
    ) -> BaseRouter:
        """Solve the instance and store the result in trailing-underscore attributes.

        Parameters
        ----------
        X : (n, n) array-like, DataFrame, dict-of-dicts or RoutingProblem
            Cost matrix (rows are origins). A ready [`RoutingProblem`][skroute.RoutingProblem] must be
            passed alone (``callback`` is not a problem argument and may accompany it).
        time_matrix : same kinds as X, optional, keyword-only
            Durations; required iff ``max_time_work`` is given.
        depot : label, optional
            Label of the depot. Default: the first node.
        coords : (n, 2) array-like, optional
            Node coordinates in row order (needed by ``SOM``).
        labels : sequence of n hashables, optional
            Labels for a plain ndarray ``X``.
        max_time_work : float > 0, optional
            Per-trip budget in the units of ``time_matrix``; ``None`` = plain TSP.
        extra_cost : float >= 0, default 0.0
            Fixed charge per trip beyond the first.
        people : int >= 1, default 1
            Multiplies ``extra_cost`` only.
        split : {"greedy", "optimal"}, default "greedy"
            Decoder of the giant tour into trips.
        callback : callable, optional
            Called with one [`RouteEvent`][skroute.base.RouteEvent] at ``"start"``, after every
            outer iteration and at ``"end"`` (D30). Return ``True`` to stop an iterative solver
            after the current outer iteration (``stop_reason_ = "callback"``); any other return
            value continues. The callback runs inside the timed search, so its cost counts in
            ``fit_time_``; an exception it raises propagates out of ``fit`` and leaves the
            estimator unfitted whatever the stage, the ``"end"`` event included.
            ``skroute.viz`` offers ready-made callbacks that draw the search live.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If ``callback`` is neither callable nor ``None``.
        """
        if callback is not None and not callable(callback):
            raise TypeError(
                "callback must be a callable taking one RouteEvent and returning None or a bool, "
                f"or None; got {type(callback).__name__}"
            )
        if isinstance(X, RoutingProblem):
            if (
                time_matrix is not None
                or any(v is not None for v in (depot, coords, labels, max_time_work))
                or extra_cost != 0.0
                or people != 1
                or split != "greedy"
            ):
                raise ValueError("X is a RoutingProblem: pass it alone, without other fit arguments")
            problem = X
        else:
            problem = RoutingProblem(
                X,
                time_matrix=time_matrix,
                depot=depot,
                coords=coords,
                labels=labels,
                max_time_work=max_time_work,
                extra_cost=extra_cost,
                people=people,
                split=split,
            )
        validate_parameter_constraints(
            self._parameter_constraints, self.get_params(deep=False), caller_name=type(self).__name__
        )
        tags = self._get_tags()
        name = type(self).__name__
        if tags.requires_symmetric and not problem.symmetric:
            raise ValueError(f"{name} requires a symmetric cost matrix")
        if tags.requires_coords and problem.coords is None:
            raise ValueError(f"{name} needs node coordinates: fit(X, coords=...)")
        if tags.max_nodes is not None and problem.n > tags.max_nodes:
            raise ValueError(
                f"{name} handles at most {tags.max_nodes} nodes, got {problem.n}; "
                "raise max_nodes only if you accept the time/memory cost"
            )
        if problem.multi_trip and not tags.budget_aware:
            if tags.exact:
                raise ValueError(
                    f"{name} optimises the plain tour and cannot certify a multi-trip optimum; "
                    "use BruteForce (n <= 11) or a heuristic solver"
                )
            warnings.warn(
                f"{name} ignores max_time_work during its search; the result is still "
                "split into trips and priced under the multi-trip objective",
                UserWarning,
                stacklevel=2,
            )
        rng = check_random_state(getattr(self, "random_state", None)) if tags.stochastic else None
        self._reset_fitted()
        # the callback lives on the instance for the duration of this fit only (D30)
        self._callback = callback
        self._callback_state = _CallbackState(problem) if callback is not None else None
        self._stop_requested = False
        try:
            t0 = perf_counter()
            tour = self._solve(problem, rng)
            fit_time = perf_counter() - t0
            raw = np.asarray(tour)
            if raw.dtype.kind not in "iu":  # a float tour would be silently truncated by the cast (D2)
                raise RuntimeError(
                    f"{name}._solve returned an invalid tour (bug in the solver): "
                    f"expected an integer array, got dtype {raw.dtype}"
                )
            tour = np.ascontiguousarray(raw, dtype=np.int64)
            if (
                tour.shape != (problem.n,)
                or tour[0] != problem.depot
                or not np.array_equal(np.sort(tour), np.arange(problem.n))
            ):
                raise RuntimeError(
                    f"{name}._solve returned an invalid tour (bug in the solver): "
                    "expected a permutation of range(n) starting at the depot index"
                )
            if tags.iterative:
                for attr in ("history_", "n_iter_", "stop_reason_"):
                    if not hasattr(self, attr):
                        raise RuntimeError(f"{name}._solve must set {attr} (bug in the solver)")
                self.history_ = np.asarray(self.history_, dtype=np.float64)
            if tags.exact and not hasattr(self, "is_optimal_"):
                raise RuntimeError(f"{name}._solve must set is_optimal_ (bug in the solver)")
            self._set_results(problem, tour, fit_time)
            if callback is not None:
                # the "end" event is the base class's: it carries the validated tour and the recomputed
                # cost_, so it can never disagree with the fitted attributes (D2); a solver that emitted
                # no "start" (construction, exact) gets a synthetic one first. The wrappers (MultiStart,
                # the Ensembles) emit no iteration events of their own and copy n_iter_ from the winning
                # restart, so an iterative solver's end event reports n_iter_ — which is the last
                # iteration index for every solver that emits its own iterations.
                state = self._callback_state
                assert state is not None
                last = int(self.n_iter_) if tags.iterative else state.last_iteration
                if not state.started:
                    self._dispatch("start", 0, None, math.nan)
                self._dispatch("end", last, tour, self.cost_)
        except BaseException:
            # a fit that raises never looks fitted — neither half-way (a solver's own attributes set before
            # its kernel or the callback raised) nor fully (the callback raised at the "end" event)
            self._reset_fitted()
            raise
        finally:
            self.__dict__.pop("_callback", None)
            self.__dict__.pop("_callback_state", None)
            self.__dict__.pop("_stop_requested", None)
        return self

    def _emit(
        self,
        stage: str,
        iteration: int,
        tour_idx: Any,
        cost: float | None,
        best_tour_idx: Any = None,
        best_cost: float | None = None,
        **extra: Any,
    ) -> None:
        """Report progress to the ``callback`` of the running ``fit`` (D30); a no-op without one.

        Parameters
        ----------
        stage : {"start", "iteration", "end"}
            Solvers emit ``"start"`` and ``"iteration"``; ``fit`` itself emits ``"end"``.
        iteration : int
            Outer iteration index (0 at ``"start"``).
        tour_idx : array-like of int or None
            The current tour in index space, depot first; ``None`` when there is no tour yet.
        cost : float or None
            Its objective as the solver knows it; ``None`` prices ``tour_idx`` with the problem's
            decoder (``nan`` when there is no tour).
        best_tour_idx : array-like of int or None, default None
            The best-so-far tour; ``None`` means "the current tour".
        best_cost : float or None, default None
            Its objective; ``None`` means ``cost`` when the best tour is the current one, else the
            problem's price of ``best_tour_idx``.
        **extra
            Solver-specific facts, stored verbatim in ``RouteEvent.extra``.

        Raises
        ------
        ValueError
            On a stage outside ``{"start", "iteration", "end"}``, on ``"end"`` (only ``fit`` emits
            it) and on a second ``"start"`` in the same fit — each a bug in the calling solver,
            reported at the offending call rather than through a malformed event trace.

        Notes
        -----
        Returns immediately when no callback is set, so a solver may call it unconditionally at
        every outer iteration; only work that exists solely to feed the event (building an extra,
        assembling a tour array) should sit behind ``if self._callback is not None``. The tours are
        converted with ``problem.to_label_tour`` (a copy). An ``"iteration"`` emitted before any
        ``"start"`` is preceded by a synthetic start without a tour. When the callback returns
        ``True`` (a Python or numpy bool — any other value is ignored) ``_stop_requested`` is set,
        and the solver honours it at the end of its outer iteration.
        """
        if self._callback is None:
            return
        state = self._callback_state
        assert state is not None  # set together with _callback by fit
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
        if stage == "end":
            raise ValueError(
                "solvers never emit 'end': fit does, after validating and pricing the returned tour"
            )
        if stage == "start":
            if state.started:
                raise ValueError("'start' is emitted once per fit; this solver emitted it twice")
        elif not state.started:
            self._dispatch("start", 0, None, math.nan)
        self._dispatch(stage, iteration, tour_idx, cost, best_tour_idx, best_cost, **extra)

    def _dispatch(
        self,
        stage: str,
        iteration: int,
        tour_idx: Any,
        cost: float | None,
        best_tour_idx: Any = None,
        best_cost: float | None = None,
        **extra: Any,
    ) -> None:
        """Build the ``RouteEvent`` of an already validated stage and hand it to the callback.

        The body of ``_emit`` without its stage guards; ``fit`` calls it directly for the synthetic
        ``"start"`` and the ``"end"`` event that only the base class may emit.
        """
        callback = self._callback
        state = self._callback_state
        assert callback is not None and state is not None  # set together by fit
        if stage == "start":
            state.started = True
        elif stage == "iteration":
            state.last_iteration = int(iteration)
        problem = state.problem
        tour = None if tour_idx is None else problem.to_label_tour(tour_idx)
        if cost is None:
            cost = math.nan if tour_idx is None else float(problem.evaluate(tour_idx))
        if best_tour_idx is None:
            best_tour = tour
            if best_cost is None:
                best_cost = cost
        else:
            best_tour = problem.to_label_tour(best_tour_idx)
            if best_cost is None:
                best_cost = float(problem.evaluate(best_tour_idx))
        event = RouteEvent(
            type(self).__name__,
            stage,
            int(iteration),
            float(cost),
            float(best_cost),
            tour,
            best_tour,
            problem,
            extra,
        )
        result = callback(event)
        if isinstance(result, (bool, np.bool_)) and bool(result):
            self._stop_requested = True

    def _reset_fitted(self) -> None:
        """Delete every fitted (trailing-underscore) attribute so a refit starts clean.

        Hyper-parameters are never fitted attributes, whatever their spelling: a knob stored as
        ``lambda_`` or ``class_`` (the usual way round a keyword) survives the reset.
        """
        params = set(self._get_param_names())
        for k in [k for k in vars(self) if k.endswith("_") and not k.startswith("_") and k not in params]:
            delattr(self, k)

    def _set_results(self, problem: RoutingProblem, tour: np.ndarray, fit_time: float) -> None:
        """Translate the validated index tour into the label-space fitted attributes."""
        starts = problem.trip_starts(tour)
        lab = problem.labels
        self.problem_ = problem
        self.n_nodes_ = problem.n
        self.labels_ = lab.copy()
        self.depot_ = lab[problem.depot]
        self.tour_ = lab[tour]
        self.trips_ = _decode_trips(problem, tour, starts)
        self.route_ = _join_trips(self.trips_)
        self.n_trips_ = len(self.trips_)
        self.trip_costs_ = problem.trip_costs(tour, starts)
        if problem.multi_trip:
            self.trip_times_ = problem.trip_times(tour, starts)
        self.cost_ = float(problem.evaluate(tour))  # D2: recomputed, never reported
        self.fit_time_ = float(fit_time)


def _param_equal(a: Any, b: Any) -> bool:
    """Equality of two parameter values that never raises.

    ndarrays compare element-wise; lists, tuples and dicts are compared item by item (so a
    list of warm-start tours works); anything whose ``==`` is ambiguous or unsupported falls
    back to identity.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_param_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_param_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except (
        ValueError,
        TypeError,
    ):  # e.g. a pandas object or another array-like with an ambiguous truth value
        return a is b


def clone(estimator: BaseRouter) -> BaseRouter:
    """New unfitted estimator with the same parameters (deep copies nothing; parameters are values).

    Parameters
    ----------
    estimator : BaseRouter
        The estimator to copy. Nested estimators (parameters with ``get_params``) are cloned recursively.

    Returns
    -------
    estimator : BaseRouter
        A fresh instance of the same class, ``== estimator`` but never fitted.

    Examples
    --------
    >>> from skroute import clone
    >>> from skroute.base import BaseRouter
    >>> class Identity(BaseRouter):
    ...     def __init__(self, verbose=0):
    ...         self.verbose = verbose
    ...
    ...     def _solve(self, problem, rng):
    ...         return problem.to_index_tour(problem.labels)
    >>> est = Identity(verbose=2).fit([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> copy = clone(est)
    >>> copy == est, copy is est, hasattr(copy, "cost_")
    (True, False, False)
    """
    params = estimator.get_params(deep=False)
    return type(estimator)(**{k: (clone(v) if hasattr(v, "get_params") else v) for k, v in params.items()})


def is_router(obj: Any) -> bool:
    """Whether ``obj`` is a scikit-route estimator (an instance of [`BaseRouter`][skroute.base.BaseRouter]).

    Examples
    --------
    >>> from skroute import is_router
    >>> from skroute.base import BaseRouter
    >>> is_router(BaseRouter()), is_router(BaseRouter), is_router(None)
    (True, False, False)
    """
    return isinstance(obj, BaseRouter)
