"""The helpers of ``skroute.utils`` (validation, ``initial_tour``, ``Bunch``), ``skroute.metrics`` and the
top-level surface of ``skroute`` (lazy exports, ``all_solvers``, ``set_log_level``)."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
import reference
from conftest import _euclid

import skroute
from skroute import RoutingProblem, all_solvers
from skroute.base import BaseRouter, RouterTags
from skroute.metrics import route_cost, split_trips
from skroute.utils import Bunch, check_is_fitted, check_random_state, initial_tour
from skroute.utils.validation import coerce_matrix

C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
H4 = np.array([[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]], dtype=float)
NAMES = ["d", "a", "b", "c"]


class Identity(BaseRouter):
    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n, dtype=np.int64), -problem.depot)


# --------------------------------------------------------------------------- validation
def test_check_random_state():
    assert isinstance(check_random_state(None), np.random.Generator)
    assert (
        check_random_state(0).integers(0, 1000, 5).tolist()
        == np.random.default_rng(0).integers(0, 1000, 5).tolist()
    )
    assert (
        check_random_state(np.int64(3)).integers(0, 1000, 5).tolist()
        == check_random_state(3).integers(0, 1000, 5).tolist()
    )
    rng = np.random.default_rng(1)
    assert check_random_state(rng) is rng
    for bad in (True, 1.5, "0", np.random.RandomState(0), [0]):
        with pytest.raises(
            TypeError, match=re.escape("random_state must be None, an int or a numpy.random.Generator")
        ):
            check_random_state(bad)


def test_check_is_fitted_and_utils_exports():
    est = Identity()
    with pytest.raises(
        skroute.exceptions.NotFittedError,
        match=re.escape("This Identity instance is not fitted yet. Call 'fit' first."),
    ):
        check_is_fitted(est)
    check_is_fitted(est.fit(C4))
    assert skroute.utils.__all__ == ["Bunch", "check_is_fitted", "check_random_state", "initial_tour"]
    assert skroute.check_router is skroute.utils.estimator_checks.check_router


def test_coerce_matrix_messages():
    with pytest.raises(
        ValueError, match=re.escape("time_matrix must be a square 2-D matrix, got shape (3,)")
    ):
        coerce_matrix([1.0, 2.0, 3.0], "time_matrix")
    with pytest.raises(ValueError, match="time_matrix contains NaN or infinite values"):
        coerce_matrix([[0, np.inf], [1, 0]], "time_matrix")
    with pytest.raises(ValueError, match="X: dict-of-dicts is not square, missing key 'b'"):
        coerce_matrix({"a": {"a": 0}, "b": {"a": 1, "b": 0}}, "X")
    arr, lab = coerce_matrix({1: {1: 0, 2: 3}, 2: {1: 3, 2: 0}}, "X")
    assert arr.tolist() == [[0.0, 3.0], [3.0, 0.0]] and lab.dtype == np.int64 and lab.tolist() == [1, 2]
    with pytest.raises(ValueError):
        coerce_matrix([["a", "b"], ["c", "d"]], "X")  # numpy's own conversion error is acceptable (SPEC §3.3)


# --------------------------------------------------------------------------- initial_tour
def test_initial_tour_nearest_neighbour_ties_go_to_the_lowest_index():
    p = RoutingProblem(C4, depot=3)
    assert initial_tour(p, "nearest_neighbour", None).tolist() == [3, 2, 1, 0]
    C, _ = _euclid(15, seed=15, asymmetric=True)
    C[2, :] = 1.0  # from node 2 every unvisited node ties: the lowest index must win
    np.fill_diagonal(C, 0.0)
    for depot in (0, 2, 14):
        q = RoutingProblem(C, depot=depot)
        got = initial_tour(q, "nearest_neighbour", np.random.default_rng(0))
        expected, seen = [depot], {depot}
        while len(expected) < 15:
            cur = expected[-1]
            nxt = min((j for j in range(15) if j not in seen), key=lambda j: (C[cur, j], j))
            expected.append(nxt)
            seen.add(nxt)
        assert got.dtype == np.int64 and got.tolist() == expected


def test_initial_tour_random_and_arrays():
    p = RoutingProblem(C4, labels=NAMES, depot="c")
    t = initial_tour(p, "random", np.random.default_rng(0))
    assert t.dtype == np.int64 and t[0] == 3 and sorted(t.tolist()) == [0, 1, 2, 3]
    assert np.array_equal(t, initial_tour(p, "random", np.random.default_rng(0)))
    with pytest.raises(
        ValueError, match="init='random' needs a random generator: this solver is not stochastic"
    ):
        initial_tour(p, "random", None)
    with pytest.raises(ValueError, match="init must be 'nearest_neighbour', 'random' or an array of labels"):
        initial_tour(p, "greedy", None)
    for warm in (
        ["c", "d", "a", "b"],
        ["c", "d", "a", "b", "c"],
        np.array(["c", "d", "c", "a", "b", "c"], dtype=object),
    ):
        assert initial_tour(p, warm, None).tolist() == [3, 0, 1, 2]
    with pytest.raises(ValueError, match="init tour must contain every label exactly once"):
        initial_tour(p, ["c", "d", "a"], None)
    fitted = Identity().fit(C4, labels=NAMES, depot="c")
    assert initial_tour(p, fitted.route_, None).tolist() == [3, 0, 1, 2]
    assert initial_tour(p, fitted.tour_, None).tolist() == [3, 0, 1, 2]


# --------------------------------------------------------------------------- metrics
def test_route_cost_plain_and_multi_trip():
    assert route_cost(C4, [0, 1, 2, 3, 0]) == 22.0 == route_cost(C4, [0, 1, 2, 3])
    assert route_cost(C4, [2, 3, 0, 1], labels=[0, 1, 2, 3]) == reference.tour_cost(C4, [2, 3, 0, 1])
    assert route_cost(C4, ["b", "c", "d", "a", "b"], labels=NAMES, depot="b") == reference.tour_cost(
        C4, [2, 3, 0, 1]
    )
    assert route_cost(C4, [0, 1, 2, 0, 3, 0], time_matrix=H4, max_time_work=4.0, extra_cost=3.0) == 41.0
    assert (
        route_cost(C4, [0, 1, 2, 3], time_matrix=H4, max_time_work=4.0, extra_cost=3.0, split="optimal")
        == 41.0
    )
    assert route_cost(_euclid(4, seed=4)[0], [0, 1, 2, 3], depot=0) == reference.tour_cost(
        _euclid(4, seed=4)[0], [0, 1, 2, 3]
    )
    with pytest.raises(ValueError, match="depot must be the first label of route"):
        route_cost(C4, [0, 1, 2, 3], depot=1)
    with pytest.raises(ValueError, match="route must not be empty"):
        route_cost(C4, [])
    with pytest.raises(ValueError, match="max_time_work given but no time_matrix"):
        route_cost(C4, [0, 1, 2, 3], max_time_work=4.0)
    est = Identity()
    with pytest.warns(UserWarning):
        est.fit(C4, labels=NAMES, depot="a", time_matrix=H4, max_time_work=4.0, extra_cost=3.0, people=2)
    kw = {"labels": NAMES, "time_matrix": H4, "max_time_work": 4.0, "extra_cost": 3.0, "people": 2}
    assert route_cost(C4, est.route_, **kw) == est.cost_ == route_cost(C4, est.tour_, depot="a", **kw)


@pytest.mark.parametrize("n,asym", [(6, False), (7, True)])
def test_route_cost_matches_the_label_space_oracle(n, asym):
    C, _ = _euclid(n, seed=n, asymmetric=asym)
    rng = np.random.default_rng(n)
    T = np.ascontiguousarray(C * rng.uniform(0.6, 1.4, C.shape))
    np.fill_diagonal(T, 0.0)
    labels = [f"n{i}" for i in range(n)]
    budget = 1.3 * float((T[1] + T[:, 1]).max())
    for _ in range(10):
        tour = [1, *rng.permutation([i for i in range(n) if i != 1])]
        route = [labels[i] for i in tour] + [labels[1]]
        for split in ("greedy", "optimal"):
            got = route_cost(
                C,
                route,
                labels=labels,
                time_matrix=T,
                max_time_work=budget,
                extra_cost=2.5,
                people=3,
                split=split,
            )
            ref = reference.route_cost_from_labels(C, route, labels, "n1", T, budget, 7.5, split)
            assert got == pytest.approx(ref, rel=1e-9)
        assert route_cost(C, route, labels=labels) == pytest.approx(reference.tour_cost(C, tour), rel=1e-12)


def test_route_cost_with_service_time_matches_the_estimator_and_the_folded_matrix():
    kw = {"labels": NAMES, "time_matrix": H4, "max_time_work": 5.0, "extra_cost": 3.0}
    est = Identity()
    with pytest.warns(UserWarning):
        est.fit(C4, service_time=0.5, **kw)
    assert est.n_trips_ == 2 and est.trip_times_.tolist() == [5.0, 4.5]  # services included
    assert route_cost(C4, est.route_, service_time=0.5, **kw) == est.cost_ == 41.0
    assert route_cost(C4, est.tour_, service_time=[0.0, 0.5, 0.5, 0.5], **kw) == 41.0  # array == scalar
    assert route_cost(C4, est.tour_, **kw) == 22.0  # without the services the tour is one trip
    folded = H4.copy()
    folded[:, 1:] += 0.5  # the definition: the service is paid on arrival at every non-depot node
    assert route_cost(C4, est.route_, **dict(kw, time_matrix=folded)) == 41.0
    with pytest.raises(ValueError, match="service_time given but no max_time_work"):
        route_cost(C4, [0, 1, 2, 3], service_time=0.5)
    with pytest.raises(ValueError, match=re.escape("service_time must be a finite number >= 0, got -1.0")):
        route_cost(C4, [0, 1, 2, 3], time_matrix=H4, max_time_work=5.0, service_time=-1.0)


@pytest.mark.parametrize("n,asym", [(6, False), (7, True)])
def test_route_cost_with_service_time_matches_the_oracle_on_the_folded_matrix(n, asym):
    C, _ = _euclid(n, seed=n, asymmetric=asym)
    rng = np.random.default_rng(100 + n)
    T = np.ascontiguousarray(C * rng.uniform(0.6, 1.4, C.shape))
    np.fill_diagonal(T, 0.0)
    service = rng.uniform(0.0, 5.0, n)
    depot = 1
    labels = [f"n{i}" for i in range(n)]
    folded = T + service[None, :]  # T_eff of D32, written out by hand
    folded[:, depot] = T[:, depot]
    folded[depot, :] += service[depot]
    folded[depot, depot] = 0.0
    budget = 1.3 * float((folded[depot] + folded[:, depot]).max())
    for _ in range(10):
        tour = [depot, *rng.permutation([i for i in range(n) if i != depot])]
        route = [labels[i] for i in tour]
        for split in ("greedy", "optimal"):
            got = route_cost(
                C,
                route,
                labels=labels,
                time_matrix=T,
                max_time_work=budget,
                extra_cost=2.5,
                people=3,
                service_time=service,
                split=split,
            )
            ref = reference.route_cost_from_labels(C, route, labels, "n1", folded, budget, 7.5, split)
            assert got == pytest.approx(ref, rel=1e-9)


def test_split_trips():
    trips = split_trips([0, 1, 2, 0, 3, 0])
    assert [t.tolist() for t in trips] == [[0, 1, 2, 0], [0, 3, 0]] and all(
        t.dtype == np.int64 for t in trips
    )
    assert [t.tolist() for t in split_trips(["d", "a", "b"])] == [["d", "a", "b", "d"]]
    assert split_trips(["d", "a", "b"])[0].dtype == object
    assert [t.tolist() for t in split_trips([5, 1, 5, 5, 2, 3, 5], depot=5)] == [[5, 1, 5], [5, 2, 3, 5]]
    assert [t.tolist() for t in split_trips(np.array([0, 2, 1, 0]))] == [[0, 2, 1, 0]]
    assert split_trips([0]) == []
    with pytest.raises(ValueError, match="depot must be the first label of route"):
        split_trips([0, 1, 2], depot=1)
    with pytest.raises(ValueError, match="route must not be empty"):
        split_trips([])
    est = Identity()
    with pytest.warns(UserWarning):
        est.fit(C4, labels=NAMES, time_matrix=H4, max_time_work=4.0)
    assert [t.tolist() for t in split_trips(est.route_)] == [t.tolist() for t in est.trips_]


# --------------------------------------------------------------------------- Bunch
def test_bunch():
    b = Bunch(cost=np.eye(2), labels=[1, 2])
    assert b.labels == b["labels"] == [1, 2] and set(b) == {"cost", "labels"}
    b.depot = 1
    assert b["depot"] == 1 and "depot" in dir(b) and repr(b) == "Bunch(cost, depot, labels)"
    del b.depot
    assert "depot" not in b
    with pytest.raises(AttributeError):
        _ = b.missing
    with pytest.raises(AttributeError):
        del b.missing


# --------------------------------------------------------------------------- top-level surface
def test_lazy_exports_and_dir():
    assert skroute.RoutingProblem is RoutingProblem and skroute.BaseRouter is BaseRouter
    assert (
        skroute.RouterTags is RouterTags
        and skroute.clone is skroute.base.clone
        and skroute.is_router is skroute.base.is_router
    )
    assert "RoutingProblem" in vars(skroute)  # cached after the first access
    for name in skroute._EXPORTS:  # every registered name (solvers register as their packages land, D29)
        assert name in skroute.__all__ and name in dir(skroute)
    assert (
        "CheapestInsertion" not in skroute.__all__ and "FarthestInsertion" not in skroute.__all__
    )  # D18: no aliases
    assert {"__version__", "all_solvers", "set_log_level", "check_router"} <= set(skroute.__all__)
    from skroute._version import __version__ as version

    assert skroute.__version__ == version and re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", version)
    with pytest.raises(AttributeError, match="module 'skroute' has no attribute 'NoSuchSolver'"):
        _ = skroute.NoSuchSolver
    if "BruteForce" not in skroute._EXPORTS:  # while the exact package has not registered itself (D29)
        with pytest.raises(AttributeError, match="no attribute 'BruteForce'"):
            _ = skroute.BruteForce
    else:
        assert isinstance(skroute.BruteForce, type)


def test_all_solvers_from_a_monkeypatched_registry(monkeypatch):
    registry = {
        "RoutingProblem": "skroute.problem",  # not a solver module: never returned
        "TinyBruteForce": "conftest",
        "IdentityRouter": "conftest",
        "MultiStart": "conftest",  # excluded by name: needs an estimator (D27)
        "RandomDescentRouter": "conftest",
    }
    monkeypatch.setattr(skroute, "_EXPORTS", registry)
    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"conftest"}))
    got = all_solvers()
    assert [cls.__name__ for cls in got] == ["IdentityRouter", "RandomDescentRouter", "TinyBruteForce"]
    assert all(issubclass(cls, BaseRouter) for cls in got) and all(cls() is not None for cls in got)
    monkeypatch.setattr(skroute, "_EXPORTS", {"Ghost": "skroute.no_such_package"})
    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"skroute.no_such_package"}))
    with pytest.raises(ImportError):
        all_solvers()  # a missing registered module fails loudly, never a shorter roster
    monkeypatch.setattr(skroute, "_EXPORTS", {"Ghost": "conftest"})
    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"conftest"}))
    with pytest.raises(ImportError, match="conftest does not define 'Ghost'"):
        all_solvers()
    assert all_solvers.__module__ == "skroute" and skroute.all_solvers is all_solvers


def test_set_log_level_attaches_one_stream_handler():
    log = logging.getLogger("skroute")
    saved_level, saved_handlers = log.level, list(log.handlers)
    try:
        for h in saved_handlers:
            log.removeHandler(h)
        log.addHandler(logging.NullHandler())  # the state after `import skroute`
        skroute.set_log_level("INFO")
        assert log.level == logging.INFO
        streams = [h for h in log.handlers if isinstance(h, logging.StreamHandler)]
        assert len(streams) == 1 and streams[0].formatter._fmt == "%(name)s %(levelname)s %(message)s"
        skroute.set_log_level(logging.DEBUG)
        assert log.level == logging.DEBUG
        assert len([h for h in log.handlers if isinstance(h, logging.StreamHandler)]) == 1  # not added twice
        assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("skroute").handlers)
    finally:
        for h in list(log.handlers):
            log.removeHandler(h)
        for h in saved_handlers:
            log.addHandler(h)
        log.setLevel(saved_level)


# --------------------------------------------------------------------------- regressions of the first review
def test_surface_modules_resolve_after_a_bare_import():
    """SPEC §3.4: ``skroute.exceptions``, ``.metrics``, ``.utils``, ``.datasets``, ``.preprocessing`` (and
    ``.base``, ``.problem``) are part of the surface — in a FRESH interpreter, where nothing else imported
    them."""
    import os
    import subprocess
    import sys

    code = (
        "import skroute\n"
        "names = ['exceptions', 'metrics', 'utils', 'datasets', 'preprocessing', 'base', 'problem']\n"
        "assert all(name not in vars(skroute) for name in names), 'must be lazy'\n"
        "for name in names:\n"
        "    mod = getattr(skroute, name)\n"
        "    assert mod.__name__ == f'skroute.{name}', mod\n"
        "    assert name in dir(skroute) and name in vars(skroute)\n"
        "assert skroute.metrics.route_cost([[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 1, 2]) == 4.0\n"
        "try:\n"
        "    raise skroute.exceptions.InfeasibleProblemError('x')\n"
        "except skroute.exceptions.InfeasibleProblemError:\n"
        "    pass\n"
        "assert not hasattr(skroute, 'no_such_module')\n"
        "print('ok')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(skroute.__file__)))
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_solver_roster_fails_loudly_on_an_empty_registry(monkeypatch):
    """A solver package directory without its registry lines must not shrink the merge gate to nothing."""
    import conftest

    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"conftest"}))  # a package that exists
    monkeypatch.setattr(skroute, "_EXPORTS", {"RoutingProblem": "skroute.problem"})  # ... but no solver line
    with pytest.raises(pytest.UsageError, match="lists no solver"):
        conftest.solver_roster()
    monkeypatch.setattr(skroute, "_EXPORTS", {"IdentityRouter": "conftest"})
    assert conftest.solver_roster() == [conftest.IdentityRouter]
    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"skroute.no_such_package"}))  # nothing exists
    assert conftest.solver_roster() == list(conftest.DUMMY_SOLVERS)


def _unregistered_solvers():
    """Solver classes in a subpackage's ``__all__`` (SPEC §3.4) that the registry-driven ``all_solvers()``
    misses."""
    import importlib
    import importlib.util

    registered = {cls.__name__ for cls in all_solvers()}
    missing = []
    for module in sorted(skroute._SOLVER_MODULES):
        if importlib.util.find_spec(module) is None:
            continue
        mod = importlib.import_module(module)
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            is_solver = isinstance(obj, type) and issubclass(obj, BaseRouter)
            if is_solver and name not in skroute._NEEDS_ARGUMENTS and name not in registered:
                missing.append(f"{module}.{name}")
    return missing


def test_every_solver_in_a_subpackage_all_is_registered(monkeypatch):
    assert _unregistered_solvers() == []  # the real registry: every exported solver reaches the test battery
    monkeypatch.setattr(skroute, "_SOLVER_MODULES", frozenset({"conftest"}))
    monkeypatch.setattr(
        skroute, "_EXPORTS", {"IdentityRouter": "conftest"}
    )  # RandomDescentRouter etc. missing
    assert "conftest.RandomDescentRouter" in _unregistered_solvers() and "conftest.IdentityRouter" not in (
        _unregistered_solvers()
    )


def test_set_log_level_doctest_leaves_the_logger_as_it_found_it():
    import doctest

    log = logging.getLogger("skroute")
    before = (log.level, list(log.handlers))
    runner = doctest.DocTestRunner(verbose=False)
    for test in doctest.DocTestFinder().find(skroute.set_log_level, "set_log_level", globs={}):
        runner.run(test)
    assert runner.failures == 0 and runner.tries >= 4
    assert (log.level, list(log.handlers)) == before


def test_initial_tour_rejects_a_non_iterable_init_with_the_documented_message():
    p = RoutingProblem(C4, depot=3)
    for bad in (5, None, 2.5):
        with pytest.raises(
            ValueError, match="init must be 'nearest_neighbour', 'random' or an array of labels"
        ):
            initial_tour(p, bad, None)


def test_spine_docstrings_use_no_sphinx_roles():
    """mkdocstrings does not understand ``:class:`` and friends: they render as literal text and
    ``mkdocs build --strict`` cannot validate them. Cross-references use ``[`Name`][full.path]``."""
    import os
    import re

    pkg = os.path.dirname(os.path.abspath(skroute.__file__))
    spine = [
        "__init__.py",
        "base.py",
        "problem.py",
        "metrics.py",
        "exceptions.py",
        os.path.join("utils", "__init__.py"),
        os.path.join("utils", "validation.py"),
        os.path.join("utils", "_param_validation.py"),
        os.path.join("utils", "_init_tour.py"),
        os.path.join("utils", "_bunch.py"),
        os.path.join("utils", "estimator_checks.py"),
    ]
    role = re.compile(r":(?:class|func|meth|mod|attr|exc|obj|data):`")
    offenders = []
    for rel in spine:
        with open(os.path.join(pkg, rel), encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if role.search(line):
                    offenders.append(f"skroute/{rel}:{lineno}: {line.strip()}")
    assert offenders == []
