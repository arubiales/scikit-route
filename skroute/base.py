"""The estimator contract: :class:`RouterTags`, :class:`BaseRouter`, :func:`clone`, :func:`is_router`.

How a solver is written (the template method)
---------------------------------------------
Every solver subclasses :class:`BaseRouter` and follows four rules:

1. ``__init__`` stores every argument verbatim as an attribute of the same name and
   sets nothing else — that is what makes :meth:`~BaseRouter.get_params`,
   :meth:`~BaseRouter.set_params`, :func:`clone` and the print-changed-only ``repr``
   work without any registration.
2. It declares ``_parameter_constraints`` (see :mod:`skroute.utils._param_validation`);
   :meth:`~BaseRouter.fit` validates the hyper-parameters at fit time, never in ``__init__``.
3. It overrides :meth:`~BaseRouter._get_tags` and returns a :class:`RouterTags` describing
   what it can do. The defaults are the *honest* ones (D28): a solver that forgets the
   override is advertised as budget-unaware (it warns under a budget), non-stochastic,
   non-iterative and non-exact.
4. It implements ``_solve(problem, rng) -> int64 array``: a permutation of ``range(problem.n)``
   with ``problem.depot`` at position 0, in **index space**. ``rng`` is a
   :class:`numpy.random.Generator` when the solver is stochastic, else ``None``.

Duties inside ``_solve``: iterative solvers (``tags.iterative``) set ``self.history_``
(best-so-far cost after each outer iteration, monotone non-increasing), ``self.n_iter_``
(``== len(history_)``) and ``self.stop_reason_`` (one of ``"converged"``, ``"max_iter"``,
``"patience"``, ``"time_limit"`` — the subset each solver documents); exact solvers
(``tags.exact``) set ``self.is_optimal_``. Everything else — the label-space ``tour_``,
``route_``, ``trips_``, the recomputed ``cost_`` (D2), ``trip_costs_``, ``trip_times_``,
``fit_time_`` — is set by the base class, which also validates the returned tour and
raises ``RuntimeError`` on a solver bug. Assign ``history_`` as an array
(``np.asarray(history)``): ``fit`` converts a list, but the attribute is typed as an ndarray.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np

from .problem import RoutingProblem
from .utils._param_validation import validate_parameter_constraints
from .utils.validation import check_random_state

__all__ = ["BaseRouter", "RouterTags", "clone", "is_router"]

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
        Seconds spent in ``_solve``.
    history_ : ndarray of shape (n_iter_,), float64
        Iterative solvers only: best-so-far cost after each outer iteration.
    n_iter_ : int
        Iterative solvers only: outer iterations actually run.
    stop_reason_ : str
        Iterative solvers only: ``"converged" | "max_iter" | "patience" | "time_limit"``.
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
                isinstance(v, type(default)) and not isinstance(v, np.ndarray) and v == default
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
    ) -> BaseRouter:
        """Solve the instance and store the result in trailing-underscore attributes.

        Parameters
        ----------
        X : (n, n) array-like, DataFrame, dict-of-dicts or RoutingProblem
            Cost matrix (rows are origins). A ready :class:`RoutingProblem` must be passed alone.
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

        Returns
        -------
        self
        """
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
        t0 = perf_counter()
        tour = self._solve(problem, rng)
        fit_time = perf_counter() - t0
        tour = np.ascontiguousarray(tour, dtype=np.int64)
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
        return self

    def _reset_fitted(self) -> None:
        """Delete every fitted (trailing-underscore) attribute so a refit starts clean."""
        for k in [k for k in vars(self) if k.endswith("_") and not k.startswith("_")]:
            delattr(self, k)

    def _set_results(self, problem: RoutingProblem, tour: np.ndarray, fit_time: float) -> None:
        """Translate the validated index tour into the label-space fitted attributes."""
        starts = problem.trip_starts(tour)
        lab = problem.labels
        d = lab[problem.depot : problem.depot + 1]  # 1-element array, keeps the label dtype
        self.problem_ = problem
        self.n_nodes_ = problem.n
        self.labels_ = lab.copy()
        self.depot_ = lab[problem.depot]
        self.tour_ = lab[tour]
        self.trips_ = [np.concatenate((d, lab[tour[a:b]], d)) for a, b in pairwise(starts)]
        self.route_ = np.concatenate([self.trips_[0]] + [t[1:] for t in self.trips_[1:]])
        self.n_trips_ = len(self.trips_)
        self.trip_costs_ = problem.trip_costs(tour, starts)
        if problem.multi_trip:
            self.trip_times_ = problem.trip_times(tour, starts)
        self.cost_ = float(problem.evaluate(tour))  # D2: recomputed, never reported
        self.fit_time_ = float(fit_time)


def _param_equal(a: Any, b: Any) -> bool:
    """Equality of two parameter values; ndarrays compare element-wise."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return bool(a == b)


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
    """Whether ``obj`` is a scikit-route estimator (an instance of :class:`BaseRouter`).

    Examples
    --------
    >>> from skroute import is_router
    >>> from skroute.base import BaseRouter
    >>> is_router(BaseRouter()), is_router(BaseRouter), is_router(None)
    (True, False, False)
    """
    return isinstance(obj, BaseRouter)
