"""RoutingProblem (coercion, labels, kernels and every message of SPEC §3.3), the BaseRouter protocol (§3.4)
and the installed-copy guard (D16)."""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
import re
import warnings

import numpy as np
import pytest
import reference
from conftest import _euclid

import skroute
from skroute import RoutingProblem, clone, is_router
from skroute.base import BaseRouter, RouterTags, _param_equal
from skroute.exceptions import InfeasibleProblemError, NotFittedError
from skroute.utils import check_is_fitted, initial_tour
from skroute.utils._param_validation import Interval, Options
from skroute.utils.validation import coerce_labels, coerce_matrix

# worked example 3 of SPEC §3.4: tour 0-1-2-3 costs 22 as a TSP and 41 under a 4 h budget (3.0 per extra trip)
C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
H4 = np.array([[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]], dtype=float)
NAMES = ["d", "a", "b", "c"]


def _dict(M, keys):
    return {keys[i]: {keys[j]: float(M[i, j]) for j in range(len(keys))} for i in range(len(keys))}


# --------------------------------------------------------------------------- dummy routers (test-local)
class Identity(BaseRouter):
    """Deterministic, budget-unaware construction dummy: matrix order from the depot."""

    def __init__(self, verbose=0):
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n, dtype=np.int64), -problem.depot)


class RandomWalk(BaseRouter):
    """Stochastic, iterative, budget-aware dummy: the best of ``n_iter`` random tours after ``init``."""

    _parameter_constraints = {
        "n_iter": [Interval(int, 1, None, closed="left")],
        "alpha": [Interval(float, 0.0, 1.0, closed="neither")],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(self, n_iter=10, alpha=0.5, init="random", random_state=None, verbose=0):
        self.n_iter = n_iter
        self.alpha = alpha
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem, rng):
        best = initial_tour(problem, self.init, rng)
        best_cost = problem.evaluate(best)
        rest = np.delete(np.arange(problem.n, dtype=np.int64), problem.depot)
        history = []
        for _ in range(self.n_iter):
            cand = np.concatenate(([problem.depot], rng.permutation(rest)))
            c = problem.evaluate(cand)
            if c < best_cost:
                best, best_cost = cand, c
            history.append(best_cost)
        self.history_ = history  # a list on purpose: fit must convert it to a float64 array
        self.n_iter_ = len(history)
        self.stop_reason_ = "max_iter"
        return best


class Nested(BaseRouter):
    """Wrapper dummy with an estimator parameter: the ``__`` protocol and the RoutingProblem passthrough."""

    def __init__(self, estimator, n_restarts=2):
        self.estimator = estimator
        self.n_restarts = n_restarts

    def _get_tags(self):
        return dataclasses.replace(self.estimator._get_tags(), kind="ensemble")

    def _solve(self, problem, rng):
        fits = [
            clone(self.estimator).set_params(random_state=rng).fit(problem) for _ in range(self.n_restarts)
        ]
        best = min(fits, key=lambda e: e.cost_)
        self.history_, self.n_iter_, self.stop_reason_ = best.history_, best.n_iter_, best.stop_reason_
        return problem.to_index_tour(best.tour_)


def _tagged(**tags):
    """A fresh Identity subclass advertising the given tags."""

    class Tagged(Identity):
        def _get_tags(self):
            return RouterTags(kind="construction", **tags)

    return Tagged


# --------------------------------------------------------------------------- RoutingProblem: coercion
def test_ndarray_defaults():
    p = RoutingProblem(C4.tolist())
    assert p.n == 4 and p.depot == 0 and p.symmetric and not p.multi_trip
    assert p.cost.dtype == np.float64 and p.cost.flags["C_CONTIGUOUS"] and p.cost is not C4
    assert p.labels.dtype == np.int64 and p.labels.tolist() == [0, 1, 2, 3]
    assert p.time is None and p.max_time_work == math.inf and p.fixed_cost == 0.0 and p.split == "greedy"
    assert p.coords is None and p.depot_label == 0 and p.time_or_cost is p.cost
    assert repr(p) == "RoutingProblem(n=4, TSP, symmetric, depot=0)"
    assert (
        p.split_code == 0
        and RoutingProblem(C4, time_matrix=H4, max_time_work=4.0, split="optimal").split_code == 1
    )


def test_fortran_and_integer_input_are_coerced():
    arr, lab = coerce_matrix(np.asfortranarray(C4.astype(int)), "X")
    assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"] and lab is None
    np.testing.assert_array_equal(arr, C4)


def test_dataframe_input():
    pd = pytest.importorskip("pandas")
    p = RoutingProblem(pd.DataFrame(C4, index=NAMES, columns=NAMES), depot="a")
    assert p.labels.dtype == object and p.labels.tolist() == NAMES and p.depot == 1 and p.depot_label == "a"
    np.testing.assert_array_equal(p.cost, C4)
    assert p.cost.flags["C_CONTIGUOUS"]
    ids = [10, 11, 12, 13]
    q = RoutingProblem(pd.DataFrame(C4, index=ids, columns=ids), depot=12)
    assert q.labels.dtype == np.int64 and q.labels.tolist() == ids and q.depot == 2
    assert repr(p) == "RoutingProblem(n=4, TSP, symmetric, depot='a')"
    with pytest.raises(ValueError, match="X: index and columns must hold the same labels in the same order"):
        RoutingProblem(pd.DataFrame(C4, index=NAMES, columns=NAMES[::-1]))
    with pytest.raises(ValueError, match="labels= disagrees with the labels carried by X"):
        RoutingProblem(pd.DataFrame(C4, index=NAMES, columns=NAMES), labels=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="time_matrix labels differ from the labels of X"):
        RoutingProblem(
            C4, labels=NAMES, time_matrix=pd.DataFrame(H4, index=ids, columns=ids), max_time_work=4.0
        )


def test_dict_of_dicts_input():
    p = RoutingProblem(_dict(C4, NAMES))
    assert p.labels.dtype == object and p.labels.tolist() == NAMES and p.depot == 0 and p.depot_label == "d"
    np.testing.assert_array_equal(p.cost, C4)
    q = RoutingProblem(_dict(C4, [3, 1, 2, 0]), depot=0)
    assert q.labels.dtype == np.int64 and q.labels.tolist() == [3, 1, 2, 0] and q.depot == 3
    broken = _dict(C4, NAMES)
    del broken["a"]["c"]
    with pytest.raises(ValueError, match="X: dict-of-dicts is not square, missing key 'c'"):
        RoutingProblem(broken)
    with pytest.raises(ValueError, match="time_matrix labels differ from the labels of X"):
        RoutingProblem(_dict(C4, NAMES), time_matrix=_dict(H4, NAMES[::-1]), max_time_work=4.0)


def test_labels_and_depot_by_label_on_ndarray():
    p = RoutingProblem(C4, labels=NAMES, depot="c")
    assert p.labels.tolist() == NAMES and p.depot == 3 and p.depot_label == "c"
    assert p.index_of("a") == 1
    with pytest.raises(ValueError, match="'z' is not a label of X"):
        p.index_of("z")
    assert RoutingProblem(C4, depot=2).depot == 2  # a position for plain arrays
    with pytest.raises(ValueError, match="depot 99 is not a label of X"):
        RoutingProblem(C4, depot=99)
    with pytest.raises(ValueError, match=re.escape("depot [0] is not a label of X")):
        RoutingProblem(C4, depot=[0])  # unhashable: TypeError inside is turned into the same ValueError
    with pytest.raises(ValueError, match="labels must be 4 unique hashables"):
        RoutingProblem(C4, labels=["d", "a", "a", "c"])
    with pytest.raises(ValueError, match="labels must be 4 unique hashables"):
        RoutingProblem(C4, labels=["d", "a", "b"])


def test_time_matrix_is_keyword_only():
    with pytest.raises(TypeError):
        RoutingProblem(C4, H4)  # 1.0's positional order (route, time, cost) must not be accepted
    with pytest.raises(TypeError):
        RoutingProblem(C4, H4, max_time_work=4.0)
    with pytest.raises(TypeError):
        Identity().fit(C4, H4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"X": C4[:2, :2]}, "X must have at least 3 nodes, got 2"),
        ({"X": C4[:, :3]}, "X must be a square 2-D matrix, got shape (4, 3)"),
        ({"X": C4[0]}, "X must be a square 2-D matrix, got shape (4,)"),
        ({"X": np.where(np.eye(4) == 1, 0.0, np.nan)}, "X contains NaN or infinite values"),
        ({"X": np.where(np.eye(4) == 1, 0.0, np.inf)}, "X contains NaN or infinite values"),
        (
            {"X": C4, "time_matrix": H4},
            "time_matrix given but no max_time_work; pass max_time_work=<hours per trip>",
        ),
        ({"X": C4, "extra_cost": 1.0}, "extra_cost, people and split have no effect without max_time_work"),
        ({"X": C4, "people": 2}, "extra_cost, people and split have no effect without max_time_work"),
        ({"X": C4, "split": "optimal"}, "extra_cost, people and split have no effect without max_time_work"),
        (
            {"X": C4, "max_time_work": 4.0},
            "max_time_work given but no time_matrix; pass time_matrix=X to use the cost matrix as durations",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 0},
            "max_time_work must be a finite number > 0, got 0",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": -1.5},
            "max_time_work must be a finite number > 0, got -1.5",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": np.inf},
            "max_time_work must be a finite number > 0, got inf",
        ),
        (
            {"X": C4, "time_matrix": H4[:3, :3], "max_time_work": 4.0},
            "time_matrix has shape (3, 3), X has shape (4, 4)",
        ),
        ({"X": C4, "time_matrix": -H4, "max_time_work": 4.0}, "time_matrix contains negative durations"),
        (
            {"X": C4, "time_matrix": H4 * np.nan, "max_time_work": 4.0},
            "time_matrix contains NaN or infinite values",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 4.0, "extra_cost": -1.0},
            "extra_cost must be a finite number >= 0, got -1.0",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 4.0, "extra_cost": np.inf},
            "extra_cost must be a finite number >= 0, got inf",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 4.0, "people": 0},
            "people must be an integer >= 1, got 0",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 4.0, "people": 1.5},
            "people must be an integer >= 1, got 1.5",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 4.0, "people": True},
            "people must be an integer >= 1, got True",
        ),
        ({"X": C4, "split": "both"}, "split must be 'greedy' or 'optimal', got 'both'"),
        ({"X": C4, "coords": np.zeros((4, 3))}, "coords must have shape (4, 2), got (4, 3)"),
        # service time (D32)
        (
            {"X": C4, "service_time": 0.5},
            "service_time given but no max_time_work; pass max_time_work=<hours per trip>",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": -1.0},
            "service_time must be a finite number >= 0, got -1.0",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": np.nan},
            "service_time must be a finite number >= 0, got nan",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": np.inf},
            "service_time must be a finite number >= 0, got inf",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": [0.5, 0.5, 0.5]},
            "service_time must be a scalar or have shape (4,), got shape (3,)",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": np.zeros((4, 1))},
            "service_time must be a scalar or have shape (4,), got shape (4, 1)",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": [0.0, np.nan, 0.5, 0.5]},
            "service_time contains NaN or infinite values",
        ),
        (
            {"X": C4, "time_matrix": H4, "max_time_work": 9.0, "service_time": [0.0, -0.5, 0.5, 0.5]},
            "service_time contains negative durations",
        ),
    ],
)
def test_error_messages_of_spec(kwargs, message):
    kwargs = dict(kwargs)
    X = kwargs.pop("X")
    with pytest.raises(ValueError) as exc:
        RoutingProblem(X, **kwargs)
    assert str(exc.value) == message
    assert not isinstance(exc.value, InfeasibleProblemError)


def test_infeasible_node_raises_naming_labels():
    with pytest.raises(InfeasibleProblemError) as exc:
        RoutingProblem(C4, time_matrix=H4, max_time_work=3)
    assert (
        str(exc.value)
        == "nodes [2, 3] cannot be served in one trip: depot round trip exceeds max_time_work=3.0"
    )
    assert isinstance(exc.value, ValueError)
    with pytest.raises(
        InfeasibleProblemError, match=re.escape("nodes ['b', 'c'] cannot be served in one trip")
    ):
        RoutingProblem(C4, labels=NAMES, time_matrix=H4, max_time_work=3.0)
    with pytest.raises(InfeasibleProblemError, match=re.escape("nodes ['c'] cannot be served in one trip")):
        RoutingProblem(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=3.0)  # only 'c' is 4 h away
    RoutingProblem(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=4.0)  # every round trip fits


def test_multi_trip_problem_attributes():
    p = RoutingProblem(
        C4, time_matrix=H4, max_time_work=4, extra_cost=3, people=2, split="optimal", coords=np.zeros((4, 2))
    )
    assert (
        p.multi_trip
        and p.max_time_work == 4.0
        and p.extra_cost == 3.0
        and p.people == 2
        and p.fixed_cost == 6.0
    )
    assert p.time is not None and p.time.dtype == np.float64 and p.time_or_cost is p.time
    assert p.coords is not None and p.coords.shape == (4, 2) and p.coords.dtype == np.float64
    assert repr(p) == "RoutingProblem(n=4, multi-trip, symmetric, depot=0)"
    assert not RoutingProblem(_euclid(5, seed=5, asymmetric=True)[0]).symmetric


def test_service_time_effective_matrix_definition():
    s = np.array([0.25, 0.5, 1.0, 0.75])
    p = RoutingProblem(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=9.0, service_time=s)
    d = p.depot
    assert d == 1
    assert p.service_time.tolist() == s.tolist() and p.service_time.dtype == np.float64
    assert p.service_time is not s and p.service_time.flags["C_CONTIGUOUS"]
    assert np.array_equal(p.time, H4) and p.time is not p.time_or_cost  # time stays raw
    eff = p.time_or_cost
    assert eff.dtype == np.float64 and eff.flags["C_CONTIGUOUS"] and eff.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            if i == j or j == d:  # the diagonal is never read and stays raw; nothing on returning
                expected = H4[i, j]
            elif i == d:  # leaving the depot: the arrival service plus the depot's own, once per trip
                expected = H4[i, j] + s[j] + s[d]
            else:
                expected = H4[i, j] + s[j]
            assert eff[i, j] == expected, (i, j)
    # a scalar applies to every non-depot node and equals the explicit array
    q = RoutingProblem(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=9.0, service_time=0.5)
    assert q.service_time.tolist() == [0.5, 0.0, 0.5, 0.5]
    arr = RoutingProblem(
        C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=9.0, service_time=[0.5, 0.0, 0.5, 0.5]
    )
    assert np.array_equal(q.time_or_cost, arr.time_or_cost)
    assert q.evaluate([1, 0, 2, 3]) == arr.evaluate([1, 0, 2, 3])
    assert np.array_equal(
        q.time_or_cost,
        RoutingProblem(
            C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=9.0, service_time=np.float32(0.5)
        ).time_or_cost,
    )
    # without a service the effective matrix IS the time matrix (aliased: nothing to copy); zeros too
    plain = RoutingProblem(C4, time_matrix=H4, max_time_work=4.0)
    assert plain.time_or_cost is plain.time and plain.service_time.tolist() == [0.0] * 4
    zeros = RoutingProblem(C4, time_matrix=H4, max_time_work=4.0, service_time=0.0)
    assert zeros.time_or_cost is zeros.time and zeros.service_time.tolist() == [0.0] * 4
    tsp = RoutingProblem(C4)
    assert tsp.service_time.tolist() == [0.0] * 4 and tsp.time_or_cost is tsp.cost
    # the caller's array is copied: later writes do not leak into the problem
    s[0] = 99.0
    assert p.service_time[0] == 0.25
    assert repr(p) == "RoutingProblem(n=4, multi-trip, symmetric, depot='a')"


def test_infeasible_with_service_time_names_the_node_and_the_service():
    with pytest.raises(InfeasibleProblemError) as exc:
        RoutingProblem(C4, labels=NAMES, time_matrix=H4, max_time_work=4.0, service_time=0.5)
    assert str(exc.value) == (
        "nodes ['b', 'c'] cannot be served in one trip: depot round trip plus service time exceeds "
        "max_time_work=4.0 ('b': travel 4 + service 0.5, 'c': travel 4 + service 0.5)"
    )
    assert isinstance(exc.value, ValueError)
    # the depot's own service counts once, at departure: 'c' needs 2 + 0.5 + 0.2 + 2 = 4.7 h
    with pytest.raises(InfeasibleProblemError, match=re.escape("('c': travel 4 + service 0.7)")):
        RoutingProblem(C4, labels=NAMES, time_matrix=H4, max_time_work=4.5, service_time=[0.2, 0.0, 0.0, 0.5])
    RoutingProblem(C4, labels=NAMES, time_matrix=H4, max_time_work=4.75, service_time=[0.2, 0.0, 0.0, 0.5])
    # with a zero service the message of the spec is unchanged
    with pytest.raises(InfeasibleProblemError) as exc:
        RoutingProblem(C4, time_matrix=H4, max_time_work=3.0, service_time=0.0)
    assert (
        str(exc.value)
        == "nodes [2, 3] cannot be served in one trip: depot round trip exceeds max_time_work=3.0"
    )
    RoutingProblem(C4, time_matrix=H4, max_time_work=4.0, service_time=0.0)  # every round trip fits


def test_trip_times_include_the_services_and_match_the_reference_on_the_folded_matrix():
    p = RoutingProblem(C4, time_matrix=H4, max_time_work=5.0, extra_cost=3.0, service_time=0.5)
    tour = [0, 1, 2, 3]
    starts = p.trip_starts(tour)
    assert starts.tolist() == [1, 3, 4]
    assert p.trip_times(tour, starts).tolist() == [5.0, 4.5]  # 1.5 + 1.5 + 2 ; 2.5 + 2
    assert p.evaluate(tour) == 41.0
    folded = H4.copy()
    folded[:, 1:] += 0.5
    for split in ("greedy", "optimal"):
        q = RoutingProblem(
            C4, time_matrix=H4, max_time_work=5.0, extra_cost=3.0, service_time=0.5, split=split
        )
        for perm in itertools.permutations([1, 2, 3]):
            t = [0, *perm]
            assert q.evaluate(t) == pytest.approx(reference.problem_cost(C4, folded, t, 5.0, 3.0, split))
            assert q.trip_starts(t).tolist() == reference.trip_starts(C4, folded, t, 5.0, 3.0, split)
            st = q.trip_starts(t)
            assert np.all(q.trip_times(t, st) <= 5.0 + 1e-9)
            for k in range(len(st) - 1):
                trip = [0, *t[st[k] : st[k + 1]], 0]
                driving = sum(H4[a, b] for a, b in itertools.pairwise(trip))
                assert q.trip_times(t, st)[k] == pytest.approx(driving + 0.5 * (len(trip) - 2))


def test_coerce_labels_dtypes():
    assert coerce_labels([3, 1, 2], 3).dtype == np.int64
    assert coerce_labels([np.int32(3), 1, np.int64(2)], 3).dtype == np.int64
    assert coerce_labels(np.array([7, 8, 9]), 3).dtype == np.int64
    for seq in (["a", "b", "c"], [1, "b", 2.5], [True, False, 2], [1.0, 2.0, 3.0]):
        out = coerce_labels(seq, 3)
        assert out.dtype == object and out.shape == (3,) and out.tolist() == seq
    tuples = coerce_labels([(0, 0), (0, 1), (1, 0)], 3)
    assert tuples.dtype == object and tuples.shape == (3,) and tuples[1] == (0, 1)
    with pytest.raises(ValueError, match="labels must be 3 unique hashables"):
        coerce_labels([1, 1, 2], 3)
    with pytest.raises(ValueError, match="labels must be 3 unique hashables"):
        coerce_labels([1, 2], 3)
    with pytest.raises(TypeError):
        coerce_labels([[1], [2], [3]], 3)


# --------------------------------------------------------------------------- RoutingProblem: labels, kernels
def test_to_index_tour_accepts_open_closed_and_multi_trip_routes():
    p = RoutingProblem(C4, labels=NAMES, depot="d")
    for route in (
        ["d", "a", "b", "c"],
        ["d", "a", "b", "c", "d"],
        ["d", "a", "d", "b", "c", "d"],
        np.array(["a", "b", "d", "c"]),
    ):
        tour = p.to_index_tour(route)
        assert tour.dtype == np.int64 and tour.tolist() == [0, 1, 2, 3]
    assert p.to_index_tour(("d", "c", "b", "a")).tolist() == [0, 3, 2, 1]
    assert p.to_label_tour([0, 3, 2, 1]).tolist() == ["d", "c", "b", "a"]
    assert p.to_label_tour(np.array([0, 1, 2, 3], dtype=np.int32)).dtype == object
    for bad in (["d", "a", "b"], ["d", "a", "a", "c"], ["d", "a", "b", "c", "a"]):
        with pytest.raises(
            ValueError,
            match=re.escape("init tour must contain every label exactly once (the depot may repeat)"),
        ):
            p.to_index_tour(bad)
    with pytest.raises(ValueError, match="'z' is not a label of X"):
        p.to_index_tour(["d", "a", "b", "z"])
    q = RoutingProblem(C4, depot=2)
    assert q.to_index_tour([2, 0, 1, 3, 2]).tolist() == [2, 0, 1, 3]
    assert q.to_label_tour(q.to_index_tour([2, 3, 1, 0])).dtype == np.int64


@pytest.mark.parametrize("n,asym", [(5, False), (7, False), (6, True), (8, True)])
def test_evaluate_and_trips_match_reference_for_both_split_rules(n, asym):
    C, _ = _euclid(n, seed=n, asymmetric=asym)
    rng = np.random.default_rng(n)
    T = np.ascontiguousarray(C * rng.uniform(0.5, 1.5, C.shape))
    np.fill_diagonal(T, 0.0)
    depot = int(rng.integers(0, n))
    round_trip = float((T[depot] + T[:, depot]).max())
    budget = 1.4 * round_trip
    fixed = 7.0 * 2
    others = [i for i in range(n) if i != depot]
    plain = RoutingProblem(C, depot=depot)
    for _ in range(20):
        tour = np.array([depot, *rng.permutation(others)], dtype=np.int64)
        assert plain.evaluate(tour.tolist()) == pytest.approx(reference.tour_cost(C, tour), rel=1e-12)
        assert plain.trip_starts(tour).tolist() == [1, n]
        assert plain.trip_costs(tour, [1, n]).tolist() == pytest.approx(
            [reference.tour_cost(C, tour)], rel=1e-12
        )
        for split in ("greedy", "optimal"):
            p = RoutingProblem(
                C, time_matrix=T, depot=depot, max_time_work=budget, extra_cost=7.0, people=2, split=split
            )
            cost_ref, starts_ref = (
                reference.problem_cost(C, T, tour, budget, fixed, split),
                reference.trip_starts(C, T, tour, budget, fixed, split),
            )
            starts = p.trip_starts(tour.astype(np.int32))  # a non-int64 tour is coerced, never a buffer error
            assert p.evaluate(tour) == pytest.approx(cost_ref, rel=1e-9) and starts.tolist() == starts_ref
            assert starts.dtype == np.int64 and starts[0] == 1 and starts[-1] == n
            costs, times = p.trip_costs(tour, starts), p.trip_times(tour, starts.tolist())
            assert costs.dtype == times.dtype == np.float64 and costs.shape == times.shape == (
                len(starts) - 1,
            )
            assert costs.sum() + fixed * (len(starts) - 2) == pytest.approx(p.evaluate(tour), rel=1e-12)
            assert np.all(times <= budget + 1e-9)
        p_g = RoutingProblem(C, time_matrix=T, depot=depot, max_time_work=budget, extra_cost=7.0, people=2)
        p_o = RoutingProblem(
            C, time_matrix=T, depot=depot, max_time_work=budget, extra_cost=7.0, people=2, split="optimal"
        )
        assert p_o.evaluate(tour) <= p_g.evaluate(tour) + 1e-9  # D1: optimal is never worse for a given tour
    with pytest.raises(ValueError, match="trip_times needs a time matrix"):
        plain.trip_times(np.arange(n), [1, n])


def test_coerce_labels_returns_python_scalars_from_numpy_inputs():
    from skroute import NearestNeighbour, RoutingProblem

    labels = coerce_labels(np.array(list("abcde")), 5)  # '<U1' input
    assert labels.dtype == object and all(type(x) is str for x in labels)
    assert [type(x) for x in coerce_labels([np.str_("a"), np.int64(2)], 2)] == [str, int]
    C = np.array([[0, 2, 9], [1, 0, 6], [7, 3, 0]], dtype=float)
    problem = RoutingProblem(C, labels=np.array(list("abc")))
    assert all(type(x) is str for x in problem.labels.tolist())
    seen = []
    est = NearestNeighbour().fit(C, labels=np.array(list("abc")), callback=seen.append)
    assert all(type(x) is str for x in est.tour_.tolist()) and type(est.depot_) is str
    pairs = [pair for e in seen if e.stage == "iteration" for pair in e.extra["edges"]]
    assert pairs and all(type(a) is str and type(b) is str for a, b in pairs)


def test_worked_example_of_spec():
    p = RoutingProblem(C4, labels=NAMES, depot="d")
    assert p.evaluate([0, 1, 2, 3]) == 22.0
    q = RoutingProblem(C4, time_matrix=H4, max_time_work=4.0, extra_cost=3.0)
    assert q.evaluate([0, 1, 2, 3]) == 41.0 and q.trip_starts([0, 1, 2, 3]).tolist() == [1, 3, 4]
    assert q.trip_costs([0, 1, 2, 3], [1, 3, 4]).tolist() == [18.0, 20.0]
    assert q.trip_times([0, 1, 2, 3], [1, 3, 4]).tolist() == [4.0, 4.0]
    costs = [
        q.evaluate([0, *perm]) for perm in ([1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1])
    ]
    assert costs == [41.0, 54.0, 41.0, 54.0, 41.0, 41.0]
    assert reference.brute_force(C4, H4, max_time_work=4.0, extra_cost=3.0)[0] == 41.0


def test_neighbours_with_coincident_points_and_non_zero_diagonal():
    xy = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [10, 0], [0, 20], [30, 30]], dtype=float)
    C = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(
        C, 1e-3
    )  # finite, non-zero, smaller than any off-diagonal entry: a kernel reading it would rank i first
    p = RoutingProblem(C)
    nb = p.neighbours(3)
    assert nb.dtype == np.int64 and nb.shape == (7, 3) and nb.flags["C_CONTIGUOUS"]
    for i in range(7):
        expected = sorted((j for j in range(7) if j != i), key=lambda j: (C[i, j], j))[:3]
        assert nb[i].tolist() == expected, i
    assert nb[0].tolist() == [1, 2, 3] and nb[4].tolist() == [0, 1, 2]
    assert p.neighbours(3) is nb  # cached
    assert p.neighbours(100).shape == (7, 6)  # k is clipped to n - 1
    assert p.cost[0, 0] == 1e-3  # the problem's matrix is untouched (the inf goes to a copy)
    C_a, _ = _euclid(9, seed=9, asymmetric=True)
    nb_a = RoutingProblem(C_a).neighbours(4)
    for i in range(9):
        assert nb_a[i].tolist() == sorted((j for j in range(9) if j != i), key=lambda j: (C_a[i, j], j))[:4]


def test_runs_against_installed_copy():
    if os.environ.get("SKROUTE_EXPECT_WHEEL") == "1":
        assert "site-packages" in skroute.__file__, skroute.__file__


# --------------------------------------------------------------------------- BaseRouter: parameter protocol
def test_get_params_set_params_with_nesting():
    est = Nested(RandomWalk(alpha=0.3), n_restarts=3)
    assert RandomWalk._get_param_names() == ["alpha", "init", "n_iter", "random_state", "verbose"]
    deep = est.get_params()
    assert deep["n_restarts"] == 3 and deep["estimator"] is est.estimator and deep["estimator__alpha"] == 0.3
    assert set(est.get_params(deep=False)) == {"estimator", "n_restarts"}
    assert est.set_params(estimator__alpha=0.7, n_restarts=1) is est
    assert est.estimator.alpha == 0.7 and est.n_restarts == 1
    assert est.set_params() is est
    with pytest.raises(ValueError) as exc:
        RandomWalk().set_params(gamma=1)
    assert str(exc.value) == (
        "Invalid parameter 'gamma' for estimator RandomWalk(). "
        "Valid parameters are: ['alpha', 'init', 'n_iter', 'random_state', 'verbose']."
    )
    with pytest.raises(ValueError, match="Invalid parameter 'other'"):
        est.set_params(other__alpha=0.1)


def test_repr_prints_changed_parameters_only():
    assert repr(RandomWalk()) == "RandomWalk()"
    assert repr(RandomWalk(alpha=0.3, random_state=0)) == "RandomWalk(alpha=0.3, random_state=0)"
    assert repr(Nested(RandomWalk(n_iter=3))) == "Nested(estimator=RandomWalk(n_iter=3))"
    est = RandomWalk(init=np.array([0, 2, 1, 3]))
    text = repr(est)
    assert text == "RandomWalk(init=array([0, 2, 1, 3]))"
    rebuilt = eval(text, {"RandomWalk": RandomWalk, "array": np.array})
    assert rebuilt == est
    assert repr(Identity(verbose=1)) == "Identity(verbose=1)" and repr(Identity(verbose=0)) == "Identity()"


def test_eq_and_clone():
    a, b = RandomWalk(alpha=0.3), RandomWalk(alpha=0.3)
    assert a == b and a != RandomWalk(alpha=0.4) and a != Identity() and a != "RandomWalk"
    assert RandomWalk(init=np.array([0, 1, 2])) == RandomWalk(init=np.array([0, 1, 2]))
    assert RandomWalk(init=np.array([0, 1, 2])) != RandomWalk(init=np.array([0, 2, 1]))
    assert _param_equal([1, 2], np.array([1, 2])) and not _param_equal(None, 0)
    with pytest.raises(TypeError):
        hash(a)
    nested = Nested(RandomWalk(alpha=0.3, random_state=1))
    fitted = nested.fit(C4)
    copy = clone(fitted)
    assert copy == nested and copy is not nested and not hasattr(copy, "cost_")
    assert copy.estimator is not nested.estimator and copy.estimator == nested.estimator
    assert is_router(copy) and is_router(BaseRouter()) and not is_router(RandomWalk) and not is_router(None)


def test_tags_defaults_are_honest_and_frozen():
    tags = BaseRouter()._get_tags()
    assert tags == RouterTags() and tags.kind == "metaheuristic" and not tags.budget_aware and not tags.exact
    assert (
        not tags.stochastic
        and not tags.iterative
        and not tags.requires_symmetric
        and not tags.requires_coords
    )
    assert tags.max_nodes is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        tags.exact = True
    with pytest.raises(NotImplementedError):
        BaseRouter().fit(C4)


# --------------------------------------------------------------------------- BaseRouter: fit and results
def test_fit_returns_self_and_sets_the_attribute_table():
    est = Identity()
    assert est.fit(C4, labels=NAMES, depot="d") is est
    assert isinstance(est.problem_, RoutingProblem) and est.n_nodes_ == 4 and type(est.n_nodes_) is int
    assert (
        est.labels_.dtype == object
        and est.labels_.tolist() == NAMES
        and est.labels_ is not est.problem_.labels
    )
    assert est.depot_ == "d"
    assert est.tour_.dtype == object and est.tour_.shape == (4,) and est.tour_.tolist() == NAMES
    assert est.route_.dtype == object and est.route_.shape == (5,) and est.route_.tolist() == [*NAMES, "d"]
    assert isinstance(est.trips_, list) and len(est.trips_) == 1 and est.trips_[0].tolist() == [*NAMES, "d"]
    assert est.n_trips_ == 1 and type(est.n_trips_) is int
    assert est.trip_costs_.dtype == np.float64 and est.trip_costs_.tolist() == [22.0]
    assert type(est.cost_) is float and est.cost_ == 22.0 == reference.route_cost_from_labels(
        C4, est.route_, NAMES, "d"
    )
    assert type(est.fit_time_) is float and est.fit_time_ >= 0.0
    for absent in ("trip_times_", "history_", "n_iter_", "stop_reason_", "is_optimal_"):
        assert not hasattr(est, absent), absent  # class-level annotations must not create attributes
    plain = Identity().fit(C4, depot=2)
    assert plain.labels_.dtype == plain.tour_.dtype == np.int64 and isinstance(plain.depot_, np.integer)
    assert (
        int(plain.depot_) == 2
        and plain.tour_.tolist() == [2, 3, 0, 1]
        and plain.route_.tolist() == [2, 3, 0, 1, 2]
    )
    assert plain.cost_ == pytest.approx(reference.tour_cost(C4, [2, 3, 0, 1]))


def test_fit_multi_trip_attributes_and_budget_warning():
    est = Identity()
    with pytest.warns(UserWarning) as record:
        est.fit(C4, labels=NAMES, time_matrix=H4, max_time_work=4.0, extra_cost=3.0)
    assert [str(w.message) for w in record] == [
        "Identity ignores max_time_work during its search; the result is still split into trips "
        "and priced under the multi-trip objective"
    ]
    assert est.tour_.tolist() == NAMES and est.route_.tolist() == ["d", "a", "b", "d", "c", "d"]
    assert est.n_trips_ == 2 and est.route_.shape == (4 + 2,)
    assert [t.tolist() for t in est.trips_] == [["d", "a", "b", "d"], ["d", "c", "d"]]
    assert est.trip_costs_.tolist() == [18.0, 20.0] and est.trip_times_.tolist() == [4.0, 4.0]
    assert est.trip_times_.dtype == np.float64 and np.all(est.trip_times_ <= 4.0 + 1e-9)
    assert est.cost_ == 41.0 == est.trip_costs_.sum() + 3.0 * (est.n_trips_ - 1)
    assert est.problem_.multi_trip and est.problem_.split == "greedy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # budget-aware solvers are silent under a budget
        aware = RandomWalk(random_state=0).fit(
            C4, time_matrix=H4, max_time_work=4.0, extra_cost=3.0, split="optimal"
        )
    assert aware.cost_ == 41.0 and aware.n_trips_ == 2 and aware.problem_.split == "optimal"
    assert aware.history_.dtype == np.float64 and aware.history_[-1] == 41.0


def test_fit_with_service_time_equals_the_fit_on_the_folded_time_matrix():
    import inspect

    from skroute.base import _FIT_KWARGS

    params = list(inspect.signature(BaseRouter.fit).parameters)
    assert params == [
        "self",
        "X",
        "time_matrix",
        "depot",
        "coords",
        "labels",
        "max_time_work",
        "extra_cost",
        "people",
        "service_time",
        "split",
        "callback",
    ]
    assert _FIT_KWARGS == (
        "depot",
        "coords",
        "labels",
        "max_time_work",
        "extra_cost",
        "people",
        "service_time",
        "split",
    )
    kw = {"labels": NAMES, "time_matrix": H4, "max_time_work": 5.0, "extra_cost": 3.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = RandomWalk(random_state=0).fit(C4, service_time=0.5, **kw)
    assert est.problem_.service_time.tolist() == [0.0, 0.5, 0.5, 0.5]
    assert np.array_equal(est.problem_.time, H4)
    assert est.n_trips_ == 2 and bool(np.all(est.trip_times_ <= 5.0 + 1e-9)) and est.cost_ == 41.0
    assert isinstance(est.fit_time_, float)
    folded = H4.copy()
    folded[:, 1:] += 0.5
    ref = RandomWalk(random_state=0).fit(C4, **dict(kw, time_matrix=folded))
    assert est.tour_.tolist() == ref.tour_.tolist() and est.cost_ == ref.cost_
    assert [t.tolist() for t in est.trips_] == [t.tolist() for t in ref.trips_]
    assert est.trip_times_.tolist() == ref.trip_times_.tolist()
    # the same through a ready problem, which is passed alone
    problem = RoutingProblem(C4, service_time=0.5, **kw)
    again = RandomWalk(random_state=0).fit(problem)
    assert again.tour_.tolist() == est.tour_.tolist() and again.problem_ is problem
    with pytest.raises(ValueError, match="X is a RoutingProblem: pass it alone, without other fit arguments"):
        RandomWalk().fit(problem, service_time=0.5)
    with pytest.raises(ValueError, match="service_time given but no max_time_work"):
        RandomWalk().fit(C4, service_time=0.5)
    with pytest.raises(ValueError, match=re.escape("service_time must be a scalar or have shape (4,)")):
        RandomWalk().fit(C4, service_time=[1.0, 2.0], **kw)


def test_fit_without_service_time_is_unchanged(alicante, fast_instance):
    from skroute import SimulatedAnnealing, TwoOpt

    d, kw = alicante["bunch"], alicante["kwargs"]
    inst = fast_instance
    C = inst["C"]
    T = np.ascontiguousarray(C / 50.0)
    budget = 1.5 * float((T[0] + T[:, 0]).max())
    cases = [
        (TwoOpt(), d.cost, dict(kw, time_matrix=d.time)),
        (SimulatedAnnealing(random_state=0), d.cost, dict(kw, time_matrix=d.time, split="optimal")),
        (TwoOpt(), C, {"time_matrix": T, "max_time_work": budget, "extra_cost": 5.0}),
        (
            SimulatedAnnealing(random_state=0),
            C,
            {"time_matrix": T, "max_time_work": budget, "extra_cost": 5.0},
        ),
    ]
    for est, X, fit_kw in cases:
        n = X.shape[0]
        fits = [
            clone(est).fit(X, **fit_kw),
            clone(est).fit(X, service_time=None, **fit_kw),
            clone(est).fit(X, service_time=0.0, **fit_kw),
            clone(est).fit(X, service_time=np.zeros(n), **fit_kw),
        ]
        base = fits[0]
        assert base.problem_.time_or_cost is base.problem_.time
        for other in fits[1:]:
            assert other.problem_.time_or_cost is other.problem_.time
            assert other.problem_.service_time.tolist() == [0.0] * n
            assert other.tour_.tolist() == base.tour_.tolist() and other.cost_ == base.cost_
            assert [t.tolist() for t in other.trips_] == [t.tolist() for t in base.trips_]
            assert other.trip_times_.tolist() == base.trip_times_.tolist()


def test_fit_accepts_a_routing_problem_only_alone():
    problem = RoutingProblem(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=4.0)
    est = Identity()
    with pytest.warns(UserWarning, match="ignores max_time_work"):
        est.fit(problem)
    assert est.problem_ is problem and est.depot_ == "a" and est.route_[0] == "a" == est.route_[-1]
    for bad in (
        {"depot": "a"},
        {"labels": NAMES},
        {"coords": np.zeros((4, 2))},
        {"max_time_work": 4.0},
        {"time_matrix": H4},
    ):
        with pytest.raises(
            ValueError, match="X is a RoutingProblem: pass it alone, without other fit arguments"
        ):
            Identity().fit(problem, **bad)
    for bad in ({"extra_cost": 1.0}, {"people": 2}, {"split": "optimal"}):
        with pytest.raises(ValueError, match="X is a RoutingProblem: pass it alone"):
            Identity().fit(problem, **bad)
    nested = Nested(RandomWalk(), n_restarts=2).fit(RoutingProblem(C4))  # inner fits reuse the shared problem
    assert nested.problem_.n == 4 and nested.n_iter_ == 10 and nested.stop_reason_ == "max_iter"


@pytest.mark.parametrize(
    "bad_tour",
    [[0, 1, 2], [1, 0, 2, 3], [0, 1, 1, 3], [[0, 1], [2, 3]], [0, 1, 2, 4]],
    ids=["short", "depot-not-first", "repeated", "2-D", "out-of-range"],
)
def test_invalid_tour_from_solve_is_a_runtime_error(bad_tour):
    class Broken(Identity):
        def _solve(self, problem, rng):
            return np.array(bad_tour)

    with pytest.raises(RuntimeError) as exc:
        Broken().fit(C4)
    assert str(exc.value) == (
        "Broken._solve returned an invalid tour (bug in the solver): "
        "expected a permutation of range(n) starting at the depot index"
    )


def test_iterative_and_exact_duties_are_enforced():
    class Forgetful(Identity):
        def _get_tags(self):
            return RouterTags(kind="local_search", iterative=True)

    with pytest.raises(
        RuntimeError, match=re.escape("Forgetful._solve must set history_ (bug in the solver)")
    ):
        Forgetful().fit(C4)

    class Unsure(Identity):
        def _get_tags(self):
            return RouterTags(kind="exact", exact=True)

    with pytest.raises(
        RuntimeError, match=re.escape("Unsure._solve must set is_optimal_ (bug in the solver)")
    ):
        Unsure().fit(C4)
    est = RandomWalk(n_iter=7, random_state=0).fit(C4)
    assert (
        isinstance(est.history_, np.ndarray)
        and est.history_.dtype == np.float64
        and est.history_.shape == (7,)
    )
    assert est.n_iter_ == 7 and est.stop_reason_ == "max_iter" and np.all(np.diff(est.history_) <= 0)
    assert (
        est.history_[-1]
        == est.cost_
        == pytest.approx(reference.route_cost_from_labels(C4, est.route_, est.labels_, 0))
    )


def test_fit_callback_plumbing_lives_only_for_the_duration_of_fit():
    """D30: the callback is validated first, stored under ``_callback`` while ``_solve`` runs (with the
    problem for label conversion), and removed afterwards whether the fit succeeded or raised."""
    seen_inside = {}

    class Peek(Identity):
        def _solve(self, problem, rng):
            seen_inside["callback"] = self._callback
            seen_inside["problem"] = self._callback_state.problem if self._callback_state else None
            seen_inside["stop"] = self._stop_requested
            return super()._solve(problem, rng)

    recorder = []
    est = Peek().fit(C4, callback=recorder.append)
    assert seen_inside["callback"] == recorder.append and seen_inside["problem"] is est.problem_
    assert seen_inside["stop"] is False and est._stop_requested is False
    assert "_callback" not in vars(est) and "_callback_state" not in vars(est) and est._callback is None
    assert [e.stage for e in recorder] == ["start", "end"] and recorder[-1].best_cost == est.cost_ == 22.0
    with pytest.raises(TypeError, match="callback must be a callable"):
        Peek().fit(C4, callback="draw me")
    assert seen_inside["callback"] == recorder.append  # unchanged: the bad callback never reached _solve
    problem = RoutingProblem(C4, labels=NAMES)
    assert Peek().fit(problem, callback=lambda e: None).problem_ is problem  # a problem plus callback is fine

    class Boom(Identity):
        def _solve(self, problem, rng):
            raise RuntimeError("no tour today")

    boom = Boom()
    with pytest.raises(RuntimeError, match="no tour today"):
        boom.fit(C4, callback=lambda e: None)
    assert "_callback" not in vars(boom) and "_callback_state" not in vars(boom)


def test_end_event_is_emitted_by_the_base_class_with_the_recomputed_cost():
    """D30 / D2: the "end" event carries the validated tour and cost_, never a cost the solver reported."""
    events = []
    est = RandomWalk(n_iter=3, random_state=0).fit(C4, labels=NAMES, depot="d", callback=events.append)
    end = events[-1]
    assert end.stage == "end" and end.solver == "RandomWalk" and end.problem is est.problem_
    assert end.best_tour.tolist() == est.tour_.tolist() == end.tour.tolist()
    assert end.best_cost == est.cost_ == end.cost and end.route.tolist() == est.route_.tolist()
    assert (
        events[0].stage == "start" and events[0].iteration == 0
    )  # synthesised: RandomWalk emits nothing itself
    assert est.stop_reason_ == "max_iter"  # a solver that never looks at _stop_requested is left alone


def test_tags_are_honoured_at_fit():
    C_a, xy = _euclid(5, seed=5, asymmetric=True)
    with pytest.raises(ValueError) as exc:
        _tagged(requires_symmetric=True)().fit(C_a)
    assert str(exc.value) == "Tagged requires a symmetric cost matrix"
    _tagged(requires_symmetric=True)().fit(C4)
    with pytest.raises(ValueError) as exc:
        _tagged(requires_coords=True)().fit(C4)
    assert str(exc.value) == "Tagged needs node coordinates: fit(X, coords=...)"
    _tagged(requires_coords=True)().fit(C_a, coords=xy)
    with pytest.raises(ValueError) as exc:
        _tagged(max_nodes=3)().fit(C4)
    assert (
        str(exc.value)
        == "Tagged handles at most 3 nodes, got 4; raise max_nodes only if you accept the time/memory cost"
    )
    _tagged(max_nodes=4)().fit(C4)

    class Exact(Identity):
        def _get_tags(self):
            return RouterTags(kind="exact", exact=True)

        def _solve(self, problem, rng):
            self.is_optimal_ = True
            return super()._solve(problem, rng)

    with pytest.raises(ValueError) as exc:
        Exact().fit(C4, time_matrix=H4, max_time_work=4.0)
    assert str(exc.value) == (
        "Exact optimises the plain tour and cannot certify a multi-trip optimum; "
        "use BruteForce (n <= 11) or a heuristic solver"
    )
    assert Exact().fit(C4).is_optimal_ is True


def test_parameter_constraints_are_validated_at_fit_not_init():
    est = RandomWalk(alpha=1.5)  # no error here
    with pytest.raises(ValueError) as exc:
        est.fit(C4)
    assert (
        str(exc.value)
        == "The 'alpha' parameter of RandomWalk must be a float in the range (0.0, 1.0). Got 1.5 instead."
    )
    with pytest.raises(ValueError, match="The 'init' parameter of RandomWalk must be a str among"):
        RandomWalk(init="both").fit(C4)
    Identity(verbose="anything goes without constraints").fit(C4)


def test_random_state_handling():
    seen = {}

    class Recorder(Identity):
        def _get_tags(self):
            return RouterTags(kind="metaheuristic", stochastic="random_state" in seen)

        def _solve(self, problem, rng):
            seen["rng"] = rng
            return super()._solve(problem, rng)

    Recorder().fit(C4)
    assert seen["rng"] is None  # non-stochastic solvers get no generator
    C, _ = _euclid(12, seed=12)
    a = RandomWalk(n_iter=30, random_state=0).fit(C)
    b = RandomWalk(n_iter=30, random_state=0).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    g = RandomWalk(n_iter=30, random_state=rng).fit(C)
    assert rng.bit_generator.state != state and np.array_equal(
        g.tour_, a.tour_
    )  # advanced, and D10: same stream as int 0
    assert not np.array_equal(RandomWalk(n_iter=30, random_state=1).fit(C).history_, a.history_)

    class Unconstrained(RandomWalk):
        _parameter_constraints = {}

    with pytest.raises(
        TypeError, match=re.escape("random_state must be None, an int or a numpy.random.Generator")
    ):
        Unconstrained(random_state=1.5).fit(C4)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The 'random_state' parameter of RandomWalk must be None, an int or a numpy.random.Generator"
        ),
    ):
        RandomWalk(random_state=np.random.RandomState(0)).fit(C4)


def test_reset_fitted_and_check_is_fitted():
    est = RandomWalk(random_state=0)
    with pytest.raises(NotFittedError) as exc:
        check_is_fitted(est)
    assert str(exc.value) == "This RandomWalk instance is not fitted yet. Call 'fit' first."
    assert isinstance(exc.value, ValueError) and isinstance(exc.value, AttributeError)
    est.fit(C4, time_matrix=H4, max_time_work=4.0)
    check_is_fitted(est)
    assert hasattr(est, "trip_times_") and hasattr(est, "history_")
    est.stale_ = "left over"
    est._private = "kept"
    est._reset_fitted()
    assert (
        not hasattr(est, "cost_")
        and not hasattr(est, "stale_")
        and est._private == "kept"
        and est.random_state == 0
    )
    with pytest.raises(NotFittedError):
        check_is_fitted(est)
    est.fit(C4)  # a refit after a multi-trip fit leaves no trip_times_ behind
    assert not hasattr(est, "trip_times_") and est.n_trips_ == 1 and est.cost_ == 22.0


# --------------------------------------------------------------------------- regressions of the first review
def test_reset_fitted_spares_trailing_underscore_parameters():
    class Lam(Identity):
        """A knob spelt the sklearn way round a keyword: ``lambda_`` is a knob, not a fitted attribute."""

        def __init__(self, lambda_=0.5, verbose=0):
            self.lambda_ = lambda_
            self.verbose = verbose

    est = Lam(lambda_=0.9)
    assert est.fit(C4) is est and est.lambda_ == 0.9 and est.cost_ == 22.0
    est.fit(C4, depot=2)  # the refit resets cost_ and friends but keeps the knob
    assert est.lambda_ == 0.9 and est.get_params() == {"lambda_": 0.9, "verbose": 0}
    assert repr(est) == "Lam(lambda_=0.9)" and clone(est) == est
    est._reset_fitted()
    assert not hasattr(est, "cost_") and est.lambda_ == 0.9


def test_param_equal_handles_containers_of_arrays_and_never_raises():
    class Warm(Identity):
        def __init__(self, tours=None, verbose=0):
            self.tours = tours
            self.verbose = verbose

    a = Warm(tours=[np.array([0, 1, 2]), np.array([0, 2, 1])])
    assert a == Warm(tours=[np.array([0, 1, 2]), np.array([0, 2, 1])])
    assert a != Warm(tours=[np.array([0, 1, 2]), np.array([1, 2, 0])])
    assert a != Warm(tours=[np.array([0, 1, 2])])
    assert a != Warm(tours=(np.array([0, 1, 2]), np.array([0, 2, 1])))  # a tuple is not a list
    assert clone(a) == a and repr(a).startswith("Warm(tours=[array([0, 1, 2]), array([0, 2, 1])])")
    assert _param_equal({"a": np.array([1])}, {"a": np.array([1])})
    assert not _param_equal({"a": np.array([1])}, {"b": np.array([1])})
    assert not _param_equal({"a": np.array([1])}, {"a": np.array([2])})
    assert _param_equal((1, [np.zeros(2)]), (1, [np.zeros(2)])) and not _param_equal([1], [1, 2])

    class Ambiguous:
        def __eq__(self, other):
            raise ValueError("ambiguous")

    x = Ambiguous()
    assert _param_equal(x, x) and not _param_equal(x, Ambiguous())  # falls back to identity


@pytest.mark.parametrize(
    "bad_tour",
    [np.array([0, 1, 2, 3.7]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([False, True, True, True]), None],
    ids=["float-truncates", "float-integral", "bool", "none"],
)
def test_non_integer_tour_from_solve_is_a_runtime_error(bad_tour):
    class Broken(Identity):
        def _solve(self, problem, rng):
            return bad_tour

    with pytest.raises(
        RuntimeError, match=r"Broken\._solve returned an invalid tour .* expected an integer array"
    ):
        Broken().fit(C4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_matrix": H4, "max_time_work": "8"}, "max_time_work must be a finite number > 0, got '8'"),
        ({"time_matrix": H4, "max_time_work": True}, "max_time_work must be a finite number > 0, got True"),
        (
            {"time_matrix": H4, "max_time_work": np.array([4.0])},
            "max_time_work must be a finite number > 0, got array([4.])",
        ),
        ({"extra_cost": "1"}, "extra_cost, people and split have no effect"),  # D3 fires first, as before
        (
            {"time_matrix": H4, "max_time_work": 4.0, "extra_cost": "1"},
            "extra_cost must be a finite number >= 0, got '1'",
        ),
        (
            {"time_matrix": H4, "max_time_work": 4.0, "extra_cost": True},
            "extra_cost must be a finite number >= 0, got True",
        ),
        (
            {"time_matrix": H4, "max_time_work": 4.0, "extra_cost": None},
            "extra_cost must be a finite number >= 0, got None",
        ),
    ],
)
def test_scalar_knobs_of_the_wrong_type_get_the_spec_message(kwargs, message):
    with pytest.raises(ValueError) as exc:
        RoutingProblem(C4, **kwargs)
    assert str(exc.value).startswith(message) and not isinstance(exc.value, InfeasibleProblemError)
    p = RoutingProblem(
        C4, time_matrix=H4, max_time_work=np.float64(4), extra_cost=np.int64(3)
    )  # numpy scalars are fine
    assert p.max_time_work == 4.0 and p.extra_cost == 3.0


def _brute_neighbours(C, k):
    n = C.shape[0]
    return np.array(
        [sorted((j for j in range(n) if j != i), key=lambda j: (C[i, j], j))[:k] for i in range(n)]
    )


def test_neighbours_matches_brute_force_on_tie_heavy_matrices_for_every_k():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(3, 45))
        xy = rng.integers(0, 5, size=(n, 2)).astype(float)  # many coincident points and integer distances
        C = np.round(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)))
        if trial % 2:
            C = C * rng.integers(1, 3, size=C.shape)  # asymmetric
        np.fill_diagonal(C, rng.uniform(0, 1))
        p = RoutingProblem(C)
        for k in range(1, n):  # k = n - 1 included: the (k + 1)-th smallest is then the inf diagonal
            got = p.neighbours(k)
            assert got.dtype == np.int64 and got.shape == (n, k) and got.flags["C_CONTIGUOUS"]
            np.testing.assert_array_equal(got, _brute_neighbours(C, k), err_msg=f"n={n} k={k}")
    n = 700  # several row blocks, n not a multiple of the block size
    xy = np.random.default_rng(7).random((n, 2)) * 100
    C = np.ascontiguousarray(np.round(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))))
    np.testing.assert_array_equal(RoutingProblem(C).neighbours(10), _brute_neighbours(C, 10))


def test_neighbours_peak_memory_is_one_matrix_copy():
    import tracemalloc

    n = 1500
    xy = np.random.default_rng(1).random((n, 2)) * 100
    C = np.ascontiguousarray(np.round(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))))
    p = RoutingProblem(C)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        p.neighbours(10)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # the accepted transient is ONE (n, n) copy; a whole-matrix argpartition or an (n, n) mask doubles it
    assert peak < 1.5 * C.nbytes, f"peak {peak / 2**20:.1f} MiB for a {C.nbytes / 2**20:.1f} MiB matrix"
