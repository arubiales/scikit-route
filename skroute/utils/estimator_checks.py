"""``check_router``: the structural test battery every solver must pass (SPEC §6, D27).

``check_router(estimator)`` takes an **unfitted instance**, builds its own small instances
(n = 6 symmetric Euclidean with coordinates, n = 6 asymmetric, n = 3 and n = 4 symmetric and
asymmetric, n = 12 and n = 40 for reproducibility, and the Alicante multi-trip bunch from
:mod:`skroute.datasets`) and runs the structural checks 1-11 and 13 of §6. Every sub-check is a
separately callable function ``fn(estimator)`` that raises ``AssertionError`` prefixed with its
check number; ``check_router.checks`` is the list of ``(name, fn)`` pairs, which
``tests/test_common.py`` exposes as parametrised tests over ``skroute.all_solvers()``. The
tolerance checks (12) are not here: they live in ``tests/test_common.py`` and are driven by
``tests/tolerances.py``.

A check that cannot run in the current environment (the datasets package is missing) raises
:class:`CheckSkipped`; the driver turns it into a ``UserWarning`` and the test-suite into a skip.
Nothing here imports from ``tests/``: the oracles are re-implemented with numpy only, and pandas
is looked up through :mod:`importlib` and skipped when absent.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import itertools
import logging
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from ..base import BaseRouter, RouterTags, _param_equal, clone
from ..exceptions import InfeasibleProblemError, NotFittedError
from ..problem import RoutingProblem
from .validation import check_is_fitted

__all__ = ["CheckSkipped", "check_router"]

_STOP_REASONS = {"converged", "max_iter", "patience", "time_limit"}


class CheckSkipped(Exception):
    """Raised by a sub-check that cannot run here (e.g. ``skroute.datasets`` is not installed)."""


# --------------------------------------------------------------------------- helpers
def _assert(cond: Any, number: int, message: str) -> None:
    if not cond:
        raise AssertionError(f"check {number}: {message}")


def _fresh(estimator: BaseRouter, **overrides: Any) -> BaseRouter:
    """A clone of ``estimator`` with ``random_state=0`` when accepted, plus ``overrides``."""
    est = clone(estimator)
    params = {}
    if "random_state" in est._get_param_names():
        params["random_state"] = 0
    params.update(overrides)
    return est.set_params(**params) if params else est


def _has_param(estimator: BaseRouter, name: str) -> bool:
    return name in estimator._get_param_names()


@functools.cache
def _euclid(n: int, seed: int, asymmetric: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(C, coords)``: the same generator as the test-suite's fixtures, numpy only."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    diff = xy[:, None, :] - xy[None, :, :]
    C = np.sqrt((diff**2).sum(axis=-1))
    if asymmetric:
        C = C * rng.uniform(0.7, 1.3, C.shape)
        np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C), xy


def _load_alicante() -> dict[str, Any]:
    """The Alicante multi-trip instance and its LABEL-space fit kwargs; ``CheckSkipped`` without datasets."""
    try:
        datasets = importlib.import_module("skroute.datasets")
        loader = datasets.load_alicante_murcia
    except (ImportError, AttributeError) as e:
        # AttributeError: before the datasets package lands, ``skroute/datasets/_data`` exists as a
        # bare namespace directory, so the import alone succeeds and proves nothing.
        raise CheckSkipped(
            "skroute.datasets is not available in this environment; "
            "the multi-trip check needs load_alicante_murcia"
        ) from e
    d = loader()
    budget = 1.5 * float((d.time[0, :] + d.time[:, 0]).max())
    kw = {"labels": d.labels, "depot": d.depot, "max_time_work": budget, "extra_cost": 10.0, "people": 2}
    return {"bunch": d, "kwargs": kw, "budget": budget}


def _tour_cost(C: np.ndarray, tour: list[int]) -> float:
    n = len(tour)
    return float(sum(C[tour[k], tour[(k + 1) % n]] for k in range(n)))


def _greedy_cost(C: np.ndarray, T: np.ndarray, tour: list[int], max_time: float, fixed: float) -> float:
    """Pure-Python greedy decoder of D1 (the oracle of ``tests/reference.py``, re-stated here)."""
    d = tour[0]
    t = 0.0
    cost = 0.0
    trips = 1
    for k in range(len(tour) - 1):
        a, b = tour[k], tour[k + 1]
        if t + T[a, b] + T[b, d] <= max_time:
            t += T[a, b]
            cost += C[a, b]
        else:
            cost += C[a, d] + C[d, b]
            t = T[d, b]
            trips += 1
    cost += C[tour[-1], d]
    return float(cost + (trips - 1) * fixed)


def _label_route_cost(
    est: BaseRouter, C: np.ndarray, T: np.ndarray | None, max_time: float, fixed: float
) -> float:
    """Objective of ``est.route_`` recomputed independently of the core (greedy split only)."""
    index = _label_index(est)
    d = index[est.depot_]
    body = [index[x] for x in est.route_.tolist() if index[x] != d]
    tour = [d, *body]
    if T is None or not math.isfinite(max_time):
        return _tour_cost(C, tour)
    return _greedy_cost(C, T, tour, max_time, fixed)


def _brute_force(C: np.ndarray) -> float:
    """Exhaustive plain-TSP optimum from depot 0 (n <= 6 in this module)."""
    n = C.shape[0]
    return min(_tour_cost(C, [0, *perm]) for perm in itertools.permutations(range(1, n)))


def _fit(est: BaseRouter, C: Any, **kw: Any) -> BaseRouter:
    """Fit while silencing the (legitimate) budget warning of budget-unaware solvers."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ignores max_time_work.*", category=UserWarning)
        return est.fit(C, **kw)


def _label_index(est: BaseRouter) -> dict[Any, int]:
    return {label: i for i, label in enumerate(est.labels_.tolist())}


# --------------------------------------------------------------------------- the checks
def check_init_and_params(estimator: BaseRouter) -> None:
    """1. ``__init__`` stores every parameter verbatim and nothing else; get/set_params, clone and
    ``eval(repr(est))`` round-trip."""
    params = estimator.get_params(deep=False)
    fresh = type(estimator)(**params)
    stored = vars(fresh)
    _assert(
        set(stored) == set(params),
        1,
        f"__init__ must store exactly its parameters as attributes; got {sorted(stored)} vs {sorted(params)}",
    )
    for k, v in params.items():
        _assert(stored[k] is v, 1, f"__init__ must store parameter {k!r} verbatim (same object)")
    # get_params/set_params round-trip
    other = clone(estimator)
    other.set_params(**params)
    for k, v in other.get_params(deep=False).items():
        _assert(_param_equal(v, params[k]), 1, f"set_params(**get_params()) changed parameter {k!r}")
    _assert(other.set_params() is other, 1, "set_params() must return self")
    try:
        other.set_params(this_parameter_does_not_exist=1)
    except ValueError as e:
        _assert(
            "Invalid parameter" in str(e), 1, "set_params with an unknown name must say 'Invalid parameter'"
        )
    else:
        raise AssertionError("check 1: set_params must raise ValueError on an unknown parameter")
    # deep get_params exposes nested estimators with '__'
    deep = estimator.get_params(deep=True)
    for k, v in params.items():
        if hasattr(v, "get_params") and not isinstance(v, type):
            for sub in v.get_params(deep=False):
                _assert(f"{k}__{sub}" in deep, 1, f"get_params(deep=True) must expose {k}__{sub}")
    # clone
    copy = clone(estimator)
    _assert(copy is not estimator, 1, "clone must return a new object")
    _assert(copy == estimator, 1, "clone must compare equal to the original (same type and parameters)")
    _assert(type(copy) is type(estimator), 1, "clone must preserve the class")
    # repr round-trip
    text = repr(estimator)
    _assert(
        text.startswith(type(estimator).__name__ + "("),
        1,
        f"repr must start with the class name, got {text!r}",
    )
    namespace: dict[str, Any] = {"np": np, "array": np.array, "inf": np.inf, "nan": np.nan, "int64": np.int64}
    namespace["float64"] = np.float64

    def _collect(est: Any) -> None:
        namespace[type(est).__name__] = type(est)
        for v in est.get_params(deep=False).values():
            if hasattr(v, "get_params") and not isinstance(v, type):
                _collect(v)

    _collect(estimator)
    try:
        rebuilt = eval(text, namespace)  # the text is our own repr
    except Exception as e:
        raise AssertionError(f"check 1: eval(repr(est)) failed for {text!r}: {e}") from None
    _assert(rebuilt == estimator, 1, f"eval(repr(est)) must equal est; repr was {text!r}")


def check_not_fitted(estimator: BaseRouter) -> None:
    """2. Before ``fit``: ``check_is_fitted`` raises ``NotFittedError`` and no trailing-underscore
    attribute exists."""
    fresh = clone(estimator)
    try:
        check_is_fitted(fresh)
    except NotFittedError as e:
        _assert("is not fitted yet" in str(e), 2, f"NotFittedError message is {str(e)!r}")
    else:
        raise AssertionError("check 2: check_is_fitted must raise NotFittedError before fit")
    fitted_attrs = [k for k in vars(fresh) if k.endswith("_") and not k.startswith("_")]
    _assert(
        not fitted_attrs, 2, f"no trailing-underscore attribute may exist before fit, found {fitted_attrs}"
    )


def _check_fitted_structure(est: BaseRouter, n: int, number: int) -> None:
    """Shared by checks 3 and 13: the fitted-attribute table of §3.4."""
    _assert(isinstance(est.problem_, RoutingProblem), number, "problem_ must be a RoutingProblem")
    _assert(isinstance(est.n_nodes_, int) and est.n_nodes_ == n, number, f"n_nodes_ must be the int {n}")
    labels = est.labels_
    _assert(
        isinstance(labels, np.ndarray) and labels.shape == (n,), number, "labels_ must be a 1-D array (n,)"
    )
    _assert(
        labels.dtype == np.int64 or labels.dtype == object,
        number,
        f"labels_ dtype must be int64/object, got {labels.dtype}",
    )
    tour, route = est.tour_, est.route_
    _assert(isinstance(tour, np.ndarray) and tour.shape == (n,), number, "tour_ must be a 1-D array (n,)")
    _assert(tour.dtype == labels.dtype, number, "tour_ must have the label dtype")
    _assert(route.dtype == labels.dtype, number, "route_ must have the label dtype")
    _assert(
        isinstance(est.trips_, list) and all(isinstance(t, np.ndarray) for t in est.trips_),
        number,
        "trips_ must be a list of arrays",
    )
    n_trips = est.n_trips_
    _assert(
        isinstance(n_trips, int) and n_trips == len(est.trips_), number, "n_trips_ must equal len(trips_)"
    )
    _assert(
        route.shape == (n + n_trips,), number, f"route_ must have shape (n + n_trips,), got {route.shape}"
    )
    depot = est.depot_
    _assert(depot == labels[est.problem_.depot], number, "depot_ must be the depot's label")
    _assert(route[0] == depot and route[-1] == depot, number, "route_ must start and end at depot_")
    _assert(tour[0] == depot, number, "tour_ must start at depot_")
    index = _label_index(est)
    d = index[depot]
    body = sorted(index[x] for x in route.tolist() if index[x] != d)
    _assert(
        body == [i for i in range(n) if i != d],
        number,
        "route_ minus the depot must visit every other label once",
    )
    _assert(
        sorted(index[x] for x in tour.tolist()) == list(range(n)),
        number,
        "tour_ must be a permutation of labels_",
    )
    _assert(
        route.tolist().count(depot) - 1 == n_trips, number, "n_trips_ must equal count(depot in route_) - 1"
    )
    for t in est.trips_:
        _assert(
            len(t) >= 3 and t[0] == depot and t[-1] == depot and depot not in t[1:-1].tolist(),
            number,
            "every trip must be closed [depot, ..., depot]",
        )
    rebuilt = np.concatenate([est.trips_[0]] + [t[1:] for t in est.trips_[1:]])
    _assert(np.array_equal(rebuilt, route), number, "route_ must be the concatenation of trips_")
    _assert(
        isinstance(est.trip_costs_, np.ndarray)
        and est.trip_costs_.dtype == np.float64
        and est.trip_costs_.shape == (n_trips,),
        number,
        "trip_costs_ must be float64 (n_trips,)",
    )
    _assert(isinstance(est.cost_, float) and math.isfinite(est.cost_), number, "cost_ must be a finite float")
    _assert(
        isinstance(est.fit_time_, float) and est.fit_time_ >= 0.0, number, "fit_time_ must be a float >= 0"
    )
    if est._get_tags().exact:
        _assert(isinstance(est.is_optimal_, bool), number, "exact solvers must set a bool is_optimal_")


def check_fit_results(estimator: BaseRouter) -> None:
    """3. ``fit`` returns self and the fitted attributes have the types, shapes and invariants of §3.4."""
    C, xy = _euclid(6, seed=6)
    est = _fresh(estimator)
    out = _fit(est, C, coords=xy)
    _assert(out is est, 3, "fit must return self")
    _check_fitted_structure(est, 6, 3)
    _assert(not hasattr(est, "trip_times_"), 3, "trip_times_ must not exist for a plain TSP")
    _assert(est.n_trips_ == 1 and len(est.trips_) == 1, 3, "a plain TSP has exactly one trip")
    if not estimator._get_tags().requires_symmetric:
        C_a, xy_a = _euclid(6, seed=6, asymmetric=True)
        est = _fit(_fresh(estimator), C_a, coords=xy_a)
        _check_fitted_structure(est, 6, 3)


def check_cost_recomputed(estimator: BaseRouter) -> None:
    """4. ``cost_`` equals an independent recomputation of ``route_`` and ``trip_costs_.sum() +
    fixed * (n_trips_ - 1)``."""
    C, xy = _euclid(6, seed=6)
    est = _fit(_fresh(estimator), C, coords=xy)
    ref = _label_route_cost(est, C, None, math.inf, 0.0)
    _assert(math.isclose(est.cost_, ref, rel_tol=1e-9), 4, f"cost_={est.cost_} but route_ costs {ref}")
    _assert(
        math.isclose(est.cost_, float(est.trip_costs_.sum()), rel_tol=1e-12),
        4,
        "cost_ must equal trip_costs_.sum() for a plain TSP",
    )
    tags = estimator._get_tags()
    if tags.exact and not tags.budget_aware:
        return  # raises under a budget by design (D6, check 7)
    ali = _load_alicante()
    d, kw = ali["bunch"], ali["kwargs"]
    est = _fit(_fresh(estimator), d.cost, time_matrix=d.time, coords=d.coords, **kw)
    fixed = kw["extra_cost"] * kw["people"]
    ref = _label_route_cost(est, d.cost, d.time, kw["max_time_work"], fixed)
    _assert(
        math.isclose(est.cost_, ref, rel_tol=1e-9), 4, f"multi-trip cost_={est.cost_} but route_ costs {ref}"
    )
    total = float(est.trip_costs_.sum()) + fixed * (est.n_trips_ - 1)
    _assert(
        math.isclose(est.cost_, total, rel_tol=1e-12),
        4,
        "cost_ must equal trip_costs_.sum() + fixed * (n_trips_ - 1)",
    )


def check_input_kinds(estimator: BaseRouter) -> None:
    """5. ndarray, DataFrame and dict-of-dicts give the same tour up to labels; string labels survive;
    ``depot=`` by label and ``labels=`` on an ndarray work."""
    C, xy = _euclid(6, seed=6)
    names = ["a", "b", "c", "d", "e", "f"]
    base = _fit(_fresh(estimator), C, coords=xy)
    as_idx = [int(x) for x in base.tour_.tolist()]
    _assert(
        base.labels_.dtype == np.int64 and base.labels_.tolist() == list(range(6)),
        5,
        "a plain ndarray must get int64 labels 0..n-1",
    )
    as_dict = {names[i]: {names[j]: float(C[i, j]) for j in range(6)} for i in range(6)}
    est = _fit(_fresh(estimator), as_dict, coords=xy)
    _assert(
        est.labels_.dtype == object and est.labels_.tolist() == names,
        5,
        "dict-of-dicts keys must become object labels",
    )
    _assert(
        [names.index(x) for x in est.tour_.tolist()] == as_idx,
        5,
        "dict-of-dicts input must give the same tour as the ndarray",
    )
    est = _fit(_fresh(estimator), C, labels=names, coords=xy)
    _assert(
        est.labels_.tolist() == names and [names.index(x) for x in est.tour_.tolist()] == as_idx,
        5,
        "labels= on an ndarray must give the same tour with the given labels",
    )
    _assert(all(isinstance(x, str) for x in est.route_.tolist()), 5, "string labels must survive in route_")
    try:
        pd = importlib.import_module("pandas")
    except ImportError:
        pd = None
    if pd is not None:
        frame = pd.DataFrame(C, index=names, columns=names)
        est = _fit(_fresh(estimator), frame, coords=xy)
        _assert(
            est.labels_.tolist() == names and [names.index(x) for x in est.tour_.tolist()] == as_idx,
            5,
            "a DataFrame must give the same tour as the ndarray, labelled by its index",
        )
    est = _fit(_fresh(estimator), C, labels=names, depot="c", coords=xy)
    _assert(
        est.depot_ == "c" and est.tour_[0] == "c" and est.route_[0] == "c" == est.route_[-1],
        5,
        "depot= by label must be honoured",
    )
    est = _fit(_fresh(estimator), C, depot=2, coords=xy)
    _assert(
        int(est.depot_) == 2 and int(est.tour_[0]) == 2,
        5,
        "depot= by position must be honoured on a plain ndarray",
    )


def check_invalid_inputs(estimator: BaseRouter) -> None:
    """6. Invalid inputs raise ``ValueError`` with the messages of §3.3; an infeasible node raises
    ``InfeasibleProblemError``."""
    C, xy = _euclid(6, seed=6)
    T = C / 10.0
    with_nan = C.copy()
    with_nan[0, 1] = np.nan
    cases: list[tuple[dict[str, Any], str]] = [
        ({"X": C[:, :5]}, "X must be a square 2-D matrix"),
        ({"X": with_nan}, "X contains NaN or infinite values"),
        ({"X": C, "depot": 99}, "depot 99 is not a label of X"),
        ({"X": C, "max_time_work": 5.0}, "max_time_work given but no time_matrix"),
        ({"X": C, "time_matrix": T}, "time_matrix given but no max_time_work"),
        ({"X": C, "extra_cost": 1.0}, "extra_cost, people and split have no effect without max_time_work"),
        ({"X": C, "time_matrix": T, "max_time_work": 0.0}, "max_time_work must be a finite number > 0"),
        ({"X": C, "time_matrix": T[:5, :5], "max_time_work": 5.0}, "time_matrix has shape"),
        (
            {"X": C, "time_matrix": T, "max_time_work": 5.0, "split": "both"},
            "split must be 'greedy' or 'optimal'",
        ),
        ({"X": C[:2, :2]}, "X must have at least 3 nodes"),
        ({"X": RoutingProblem(C), "depot": 0}, "X is a RoutingProblem: pass it alone"),
    ]
    for kw, message in cases:
        X = kw.pop("X")
        try:
            _fit(_fresh(estimator), X, **kw)
        except ValueError as e:
            _assert(message in str(e), 6, f"expected {message!r} in the error, got {str(e)!r}")
        except InfeasibleProblemError:  # a subclass of ValueError, but must not be what these raise
            raise AssertionError(f"check 6: {message!r} case raised InfeasibleProblemError") from None
        else:
            raise AssertionError(f"check 6: fit must raise ValueError ({message!r}) for {kw}")
    try:
        _fit(_fresh(estimator), C, time_matrix=T, max_time_work=float(T.max()) * 0.5, coords=xy)
    except InfeasibleProblemError as e:
        _assert("cannot be served in one trip" in str(e), 6, f"InfeasibleProblemError message is {str(e)!r}")
    else:
        raise AssertionError(
            "check 6: a node whose round trip exceeds the budget must raise InfeasibleProblemError"
        )


def _synthetic_multi_trip() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(C, T, coords, budget)``: a 6-node symmetric instance whose greedy decode needs several trips."""
    C, xy = _euclid(6, seed=6)
    T = C / 10.0
    budget = 1.5 * float((T[0, :] + T[:, 0]).max())
    return C, T, xy, budget


def check_tags_honoured(estimator: BaseRouter) -> None:
    """7. Tags are honoured: ``requires_symmetric``, ``requires_coords``, ``max_nodes``, and the D6 budget
    rule (exact + budget-unaware raises, non-exact budget-unaware warns, budget-aware is silent)."""
    tags = estimator._get_tags()
    _assert(isinstance(tags, RouterTags), 7, "_get_tags() must return a RouterTags")
    _assert(
        tags.kind in {"exact", "construction", "local_search", "metaheuristic", "ensemble"},
        7,
        f"RouterTags.kind must be one of the five kinds of D28, got {tags.kind!r}",
    )
    C, xy = _euclid(6, seed=6)
    if tags.requires_symmetric:
        C_a, xy_a = _euclid(6, seed=6, asymmetric=True)
        try:
            _fit(_fresh(estimator), C_a, coords=xy_a)
        except ValueError as e:
            _assert(
                "requires a symmetric cost matrix" in str(e), 7, f"asymmetric refusal message is {str(e)!r}"
            )
        else:
            raise AssertionError(
                "check 7: requires_symmetric solvers must raise ValueError on an asymmetric matrix"
            )
    if tags.requires_coords:
        try:
            _fit(_fresh(estimator), C)
        except ValueError as e:
            _assert("needs node coordinates" in str(e), 7, f"missing-coords message is {str(e)!r}")
        else:
            raise AssertionError("check 7: requires_coords solvers must raise ValueError without coords=")
    if tags.max_nodes is not None:
        _assert(isinstance(tags.max_nodes, int) and tags.max_nodes >= 3, 7, "max_nodes must be an int >= 3")
        n = tags.max_nodes + 1
        C_big, xy_big = _euclid(n, seed=n)
        try:
            _fit(_fresh(estimator), C_big, coords=xy_big)
        except ValueError as e:
            _assert("handles at most" in str(e), 7, f"max_nodes message is {str(e)!r}")
        else:
            raise AssertionError(
                f"check 7: max_nodes={tags.max_nodes} solvers must raise ValueError at n={n}"
            )
    C, T, xy, budget = _synthetic_multi_trip()
    kw: dict[str, Any] = {"time_matrix": T, "max_time_work": budget, "extra_cost": 1.0, "coords": xy}
    if tags.exact and not tags.budget_aware:
        try:
            _fresh(estimator).fit(C, **kw)
        except ValueError as e:
            _assert("cannot certify a multi-trip optimum" in str(e), 7, f"D6 message is {str(e)!r}")
        else:
            raise AssertionError(
                "check 7: exact budget-unaware solvers must raise ValueError under a budget (D6)"
            )
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = _fresh(estimator).fit(C, **kw)
    budget_warnings = [
        w for w in caught if issubclass(w.category, UserWarning) and "ignores max_time_work" in str(w.message)
    ]
    if tags.budget_aware:
        _assert(not budget_warnings, 7, "budget-aware solvers must not warn under a budget")
    else:
        _assert(
            len(budget_warnings) == 1,
            7,
            "budget-unaware solvers must warn exactly once (UserWarning) under a budget",
        )
    _assert(
        np.all(est.trip_times_ <= budget + 1e-9), 7, "every trip must fit the budget, return leg included"
    )


def check_multi_trip(estimator: BaseRouter) -> None:
    """8. Multi-trip on Alicante: trips fit the budget, ``trip_times_`` exists only with a time matrix,
    and the optimal decoder never prices the fitted tour above the greedy one."""
    tags = estimator._get_tags()
    if tags.exact and not tags.budget_aware:
        return  # raises under a budget by design (D6); covered by check 7
    ali = _load_alicante()
    d, kw, budget = ali["bunch"], ali["kwargs"], ali["budget"]
    n = d.cost.shape[0]
    est = _fit(_fresh(estimator), d.cost, time_matrix=d.time, coords=d.coords, **kw)
    _check_fitted_structure(est, n, 8)
    _assert(
        est.depot_ == d.depot and est.labels_.tolist() == list(d.labels.tolist()),
        8,
        "labels_/depot_ must be the bunch's labels and depot",
    )
    times = est.trip_times_
    _assert(
        isinstance(times, np.ndarray) and times.dtype == np.float64 and times.shape == (est.n_trips_,),
        8,
        "trip_times_ must be float64 (n_trips,)",
    )
    _assert(
        bool(np.all(times <= budget + 1e-9)),
        8,
        f"every trip must fit max_time_work={budget}, got {times.tolist()}",
    )
    index = _label_index(est)
    for k, trip in enumerate(est.trips_):
        idx = [index[x] for x in trip.tolist()]
        dur = float(sum(d.time[idx[i], idx[i + 1]] for i in range(len(idx) - 1)))
        _assert(
            math.isclose(dur, times[k], rel_tol=1e-9, abs_tol=1e-12),
            8,
            f"trip_times_[{k}] must be the closed trip's duration",
        )
    plain = _fit(_fresh(estimator), d.cost, labels=d.labels, depot=d.depot, coords=d.coords)
    _assert(not hasattr(plain, "trip_times_"), 8, "trip_times_ must be absent for a plain TSP")
    tour_idx = est.problem_.to_index_tour(est.tour_)
    _assert(
        math.isclose(est.problem_.evaluate(tour_idx), est.cost_, rel_tol=1e-12),
        8,
        "problem_.evaluate(tour_) must reproduce cost_",
    )
    p_opt = RoutingProblem(d.cost, time_matrix=d.time, split="optimal", **kw)
    _assert(
        p_opt.evaluate(tour_idx) <= est.cost_ + 1e-9,
        8,
        "the optimal split of the fitted tour must not cost more than the greedy split",
    )
    est_opt = _fit(_fresh(estimator), d.cost, time_matrix=d.time, coords=d.coords, split="optimal", **kw)
    _assert(
        bool(np.all(est_opt.trip_times_ <= budget + 1e-9)),
        8,
        "under split='optimal' every trip must fit the budget too",
    )
    _assert(est_opt.problem_.split == "optimal", 8, "problem_.split must record the requested decoder")


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def check_no_printing(estimator: BaseRouter) -> None:
    """9. Nothing is written to stdout/stderr; ``verbose=1`` emits at least one record on the ``skroute``
    logger for iterative solvers (D24)."""
    C, xy = _euclid(6, seed=6)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        _fit(_fresh(estimator), C, coords=xy)
    _assert(out.getvalue() == "", 9, f"fit must not print to stdout, got {out.getvalue()!r}")
    _assert(err.getvalue() == "", 9, f"fit must not print to stderr, got {err.getvalue()!r}")
    if not _has_param(estimator, "verbose"):
        return
    logger = logging.getLogger("skroute")
    handler = _RecordingHandler()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            _fit(_fresh(estimator, verbose=1), C, coords=xy)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    _assert(out.getvalue() == "" and err.getvalue() == "", 9, "verbose=1 must log, never print")
    if estimator._get_tags().iterative:
        _assert(handler.records, 9, "verbose=1 must emit at least one record on logging.getLogger('skroute')")
        _assert(
            all(r.name.startswith("skroute") for r in handler.records),
            9,
            "records must belong to the 'skroute' logger",
        )


def check_iterative_contract(estimator: BaseRouter) -> None:
    """10. Iterative solvers: ``history_`` is best-so-far and non-increasing, ``n_iter_ == len(history_)``,
    ``history_[-1] == cost_``, ``stop_reason_`` is legal, and ``time_limit=1e-6`` stops after one
    iteration."""
    tags = estimator._get_tags()
    if not tags.iterative:
        C, xy = _euclid(6, seed=6)
        est = _fit(_fresh(estimator), C, coords=xy)
        for attr in ("history_", "n_iter_", "stop_reason_"):
            _assert(not hasattr(est, attr), 10, f"non-iterative solvers must not set {attr}")
        return
    C, xy = _euclid(12, seed=12)
    est = _fit(_fresh(estimator), C, coords=xy)
    hist = est.history_
    _assert(
        isinstance(hist, np.ndarray) and hist.dtype == np.float64 and hist.ndim == 1,
        10,
        "history_ must be a 1-D float64 array",
    )
    _assert(
        isinstance(est.n_iter_, int) and est.n_iter_ == len(hist) >= 1,
        10,
        "n_iter_ must be the int len(history_) >= 1",
    )
    tol = 1e-9 * max(1.0, float(np.abs(hist).max()))
    _assert(bool(np.all(np.diff(hist) <= tol)), 10, "history_ must be best-so-far (non-increasing)")
    _assert(
        math.isclose(float(hist[-1]), est.cost_, rel_tol=1e-9),
        10,
        f"history_[-1]={hist[-1]} must equal cost_={est.cost_}",
    )
    _assert(
        est.stop_reason_ in _STOP_REASONS,
        10,
        f"stop_reason_ must be one of {sorted(_STOP_REASONS)}, got {est.stop_reason_!r}",
    )
    if not _has_param(estimator, "patience"):
        _assert(
            est.stop_reason_ != "patience",
            10,
            "a solver without a patience parameter cannot stop by 'patience'",
        )
    if not _has_param(estimator, "time_limit"):
        _assert(
            est.stop_reason_ != "time_limit",
            10,
            "a solver without a time_limit parameter cannot stop by 'time_limit'",
        )
        return
    est = _fit(_fresh(estimator, time_limit=1e-6), C, coords=xy)
    _assert(
        est.stop_reason_ == "time_limit",
        10,
        f"time_limit=1e-6 must stop with 'time_limit', got {est.stop_reason_!r}",
    )
    _assert(
        est.n_iter_ <= 1,
        10,
        f"time_limit=1e-6 must stop after at most one outer iteration, ran {est.n_iter_}",
    )
    _check_fitted_structure(est, 12, 10)


def check_stochastic_reproducibility(estimator: BaseRouter) -> None:
    """11. Same seed -> bit-identical results; seeds 0 and 1 differ deterministically; a passed
    ``Generator`` is advanced. Deterministic solvers: two fits are identical."""
    tags = estimator._get_tags()
    has_rs = _has_param(estimator, "random_state")
    _assert(
        tags.stochastic == has_rs,
        11,
        "RouterTags.stochastic must be True iff the solver has a random_state parameter",
    )
    if not tags.stochastic:
        C, xy = _euclid(6, seed=6)
        a = _fit(_fresh(estimator), C, coords=xy)
        b = _fit(_fresh(estimator), C, coords=xy)
        _assert(
            np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_,
            11,
            "a deterministic solver must give identical results on refit",
        )
        return
    C, xy = _euclid(12, seed=12)
    a = _fit(_fresh(estimator, random_state=0), C, coords=xy)
    b = _fit(_fresh(estimator, random_state=0), C, coords=xy)
    _assert(np.array_equal(a.tour_, b.tour_), 11, "random_state=0 twice must give array_equal tour_")
    _assert(a.cost_ == b.cost_, 11, "random_state=0 twice must give equal cost_")
    if tags.iterative:
        _assert(
            np.array_equal(a.history_, b.history_), 11, "random_state=0 twice must give array_equal history_"
        )
    c = _fit(_fresh(estimator, random_state=1), C, coords=xy)
    differs = False
    if tags.iterative:
        differs = a.n_iter_ != c.n_iter_ or not np.array_equal(a.history_, c.history_)
    if not differs:
        C40, xy40 = _euclid(40, seed=40)
        a40 = _fit(_fresh(estimator, random_state=0), C40, coords=xy40)
        c40 = _fit(_fresh(estimator, random_state=1), C40, coords=xy40)
        differs = not np.array_equal(a40.tour_, c40.tour_)
    if not differs:
        # Strong solvers (IteratedLocalSearch, AntColony) reach the same optimum with every seed
        # at n <= 40, so their trajectories coincide; escalate to instances where seeds can show.
        for n, asym in ((80, True), (150, False)):
            if asym and (tags.requires_symmetric or tags.requires_coords):
                continue
            Cn, xyn = _euclid(n, seed=n, asymmetric=asym)
            an = _fit(_fresh(estimator, random_state=0), Cn, coords=xyn)
            cn = _fit(_fresh(estimator, random_state=1), Cn, coords=xyn)
            differs = not np.array_equal(an.tour_, cn.tour_) or (
                tags.iterative and not np.array_equal(an.history_, cn.history_)
            )
            if differs:
                break
    _assert(
        differs,
        11,
        "seeds 0 and 1 must give different history_/n_iter_ (n=12) or a different tour_ "
        "(n=40, then n=80 asymmetric and n=150)",
    )
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    g = _fit(_fresh(estimator, random_state=rng), C, coords=xy)
    _assert(rng.bit_generator.state != before, 11, "a passed Generator must be advanced by fit")
    _assert(
        np.array_equal(g.tour_, a.tour_),
        11,
        "random_state=default_rng(0) must reproduce random_state=0 (D10)",
    )


def check_smallest_sizes(estimator: BaseRouter) -> None:
    """13. Every solver fits n = 3 and n = 4, symmetric and asymmetric; exact solvers equal brute force
    there."""
    tags = estimator._get_tags()
    for n in (3, 4):
        for asym in (False, True):
            if asym and tags.requires_symmetric:
                continue
            C, xy = _euclid(n, seed=n, asymmetric=asym)
            est = _fit(_fresh(estimator), C, coords=xy)
            _check_fitted_structure(est, n, 13)
            if tags.exact:
                opt = _brute_force(C)
                _assert(
                    math.isclose(est.cost_, opt, rel_tol=1e-9),
                    13,
                    f"exact solver must reach the optimum {opt} at n={n} "
                    f"({'asym' if asym else 'sym'}), got {est.cost_}",
                )
                _assert(
                    est.is_optimal_ is True, 13, "an exact solver that finished must set is_optimal_ = True"
                )


_CHECKS: list[tuple[str, Callable[[BaseRouter], None]]] = [
    ("1_init_and_params", check_init_and_params),
    ("2_not_fitted", check_not_fitted),
    ("3_fit_results", check_fit_results),
    ("4_cost_recomputed", check_cost_recomputed),
    ("5_input_kinds", check_input_kinds),
    ("6_invalid_inputs", check_invalid_inputs),
    ("7_tags_honoured", check_tags_honoured),
    ("8_multi_trip", check_multi_trip),
    ("9_no_printing", check_no_printing),
    ("10_iterative_contract", check_iterative_contract),
    ("11_stochastic_reproducibility", check_stochastic_reproducibility),
    ("13_smallest_sizes", check_smallest_sizes),
]


def check_router(estimator: BaseRouter) -> None:
    """Run the structural test battery of SPEC §6 on an unfitted solver instance.

    Parameters
    ----------
    estimator : BaseRouter
        An **unfitted** instance (so ``MultiStart(SimulatedAnnealing())`` can be checked too).
        The battery never fits it: every check works on clones with ``random_state=0``.

    Raises
    ------
    AssertionError
        Prefixed with the number of the failing check (``"check 3: ..."``).
    TypeError
        If ``estimator`` is not a :class:`~skroute.base.BaseRouter` instance.
    ValueError
        If ``estimator`` is already fitted.

    Warns
    -----
    UserWarning
        For every check that had to be skipped (``skroute.datasets`` unavailable).

    Notes
    -----
    The checks: 1 parameter protocol, 2 not fitted, 3 fitted attributes, 4 recomputed cost,
    5 input kinds, 6 invalid inputs, 7 tags, 8 multi-trip, 9 no printing, 10 iterative
    contract, 11 reproducibility, 13 smallest sizes. ``check_router.checks`` lists them as
    ``(name, fn)`` pairs; the tolerance checks (12) live in ``tests/test_common.py``.

    Examples
    --------
    >>> from skroute import check_router
    >>> from skroute.base import BaseRouter, RouterTags
    >>> import numpy as np
    >>> class Identity(BaseRouter):
    ...     def __init__(self, verbose=0):
    ...         self.verbose = verbose
    ...
    ...     def _get_tags(self):
    ...         return RouterTags(kind="construction")
    ...
    ...     def _solve(self, problem, rng):
    ...         return np.roll(np.arange(problem.n), -problem.depot)
    >>> check_router(Identity())  # doctest: +SKIP
    >>> [name for name, fn in check_router.checks][:3]
    ['1_init_and_params', '2_not_fitted', '3_fit_results']
    """
    if not isinstance(estimator, BaseRouter):
        raise TypeError(f"check_router takes an unfitted BaseRouter instance, got {type(estimator).__name__}")
    if hasattr(estimator, "cost_"):
        raise ValueError("check_router takes an UNFITTED instance; pass skroute.clone(est) or a new instance")
    for name, fn in _CHECKS:
        try:
            fn(estimator)
        except CheckSkipped as e:
            warnings.warn(f"check_router: {name} skipped: {e}", UserWarning, stacklevel=2)


check_router.checks = _CHECKS  # type: ignore[attr-defined]
