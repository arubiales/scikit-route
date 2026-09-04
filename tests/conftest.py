"""Fixtures of SPEC §6, the ``Solver`` parametrisation over ``skroute.all_solvers()`` (D27) and the
test-suite's dummy routers.

Dataset-backed fixtures import ``skroute.datasets`` lazily and skip when the package is absent
(wave A of D29: the datasets are written in parallel). ``import reference`` works because
``pyproject.toml`` puts ``tests/`` on ``sys.path`` (``pythonpath = ["tests"]``, D16).
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import logging
import math
import time

import numpy as np
import pytest
import reference  # tests/ is on sys.path via pythonpath = ["tests"] (D16)

import skroute
from skroute import RoutingProblem, all_solvers
from skroute.base import BaseRouter, RouterTags
from skroute.utils import initial_tour
from skroute.utils._param_validation import Interval, Options

OPTIMA = {"wi29": 27603, "dj38": 6656, "qa194": 9352, "uy734": 79114, "zi929": 95345, "lu980": 11340}

log = logging.getLogger("skroute")


def _euclid(n, seed, asymmetric=False):
    """(C, coords). Every instance keeps the coordinates that generated it (SOM needs them).

    Euclidean distances computed with numpy (``skroute.preprocessing.distance_matrix(xy)`` is the
    same quantity; numpy keeps the fixture independent of the preprocessing package).
    """
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    diff = xy[:, None, :] - xy[None, :, :]
    C = np.sqrt((diff**2).sum(axis=-1))
    if asymmetric:
        C = C * rng.uniform(0.7, 1.3, C.shape)
        np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C), xy


@pytest.fixture(
    scope="session",
    params=[(5, False), (7, False), (9, False), (6, True), (8, True)],
    ids=lambda p: f"n{p[0]}{'-asym' if p[1] else ''}",
)
def tiny_instance(request):
    n, asym = request.param
    C, xy = _euclid(n, seed=n, asymmetric=asym)
    return {"C": C, "coords": xy, "n": n, "asymmetric": asym, "optimum": reference.brute_force(C)[0]}


@pytest.fixture(scope="session")
def small_euclidean():  # n = 12, for reproducibility and label round-trip tests
    C, xy = _euclid(12, seed=12)
    return {"C": C, "coords": xy, "n": 12, "asymmetric": False}


@pytest.fixture(scope="session")
def medium_euclidean():  # n = 40, where seeds 0 and 1 give different tours (check 11)
    C, xy = _euclid(40, seed=40)
    return {"C": C, "coords": xy, "n": 40, "asymmetric": False}


def _datasets():
    """``skroute.datasets``, or ``pytest.skip``: the package is written in parallel (wave A of D29) and
    ``skroute/datasets/_data`` already exists as a bare namespace directory, so the import alone proves
    nothing."""
    try:
        mod = importlib.import_module("skroute.datasets")
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"skroute.datasets is not available in this tree ({e})")
    if not hasattr(mod, "load_tsp"):
        pytest.skip("skroute.datasets is not available in this tree (loaders not written yet)")
    return mod


@pytest.fixture(scope="session")
def alicante():  # multi-trip fixture, 8 nodes; the depot is row 0 == label d.depot
    datasets = _datasets()
    d = datasets.load_alicante_murcia()
    budget = 1.5 * float((d.time[0, :] + d.time[:, 0]).max())
    fit_kw = dict(
        labels=d.labels, depot=d.depot, max_time_work=budget, extra_cost=10.0, people=2
    )  # LABEL space
    ref_kw = dict(depot=0, max_time_work=budget, extra_cost=10.0, people=2)  # INDEX space
    opt = {s: reference.brute_force(d.cost, d.time, split=s, **ref_kw)[0] for s in ("greedy", "optimal")}
    return {"bunch": d, "kwargs": fit_kw, "ref_kwargs": ref_kw, "optimum": opt}


@pytest.fixture(scope="session")
def barcelona():
    datasets = _datasets()
    return datasets.load_barcelona()


@pytest.fixture(scope="session", params=["wi29", "dj38"])
def fast_instance(request):
    datasets = _datasets()
    b = datasets.load_tsp(request.param)
    return {
        "name": b.name,
        "C": b.distance_matrix(),
        "coords": b.coords,
        "labels": b.labels,
        "asymmetric": False,
        "optimum": OPTIMA[b.name],
    }


@pytest.fixture(scope="session", params=["qa194", "uy734", "zi929", "lu980"])
def slow_instance(request):
    datasets = _datasets()
    b = datasets.load_tsp(request.param)
    return {
        "name": b.name,
        "C": b.distance_matrix(),
        "coords": b.coords,
        "labels": b.labels,
        "asymmetric": False,
        "optimum": OPTIMA[b.name],
    }


# --------------------------------------------------------------------------- dummy routers
# Minimal solvers written against the BaseRouter contract only. They prove that the test battery
# itself works (tests/test_common.py, tests/test_base.py) and parametrise ``Solver`` while no real
# solver package exists in the tree (wave A of D29). Never part of skroute.


class IdentityRouter(BaseRouter):
    """Deterministic, budget-unaware construction dummy: nodes in matrix order from the depot."""

    def __init__(self, verbose=0):
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n, dtype=np.int64), -problem.depot)


class SymmetricIdentityRouter(IdentityRouter):
    """Identity dummy that refuses asymmetric matrices (the ClarkeWright tag)."""

    def _get_tags(self):
        return RouterTags(kind="construction", requires_symmetric=True)


class CoordsIdentityRouter(IdentityRouter):
    """Construction dummy that needs coordinates (the SOM tag): nodes by angle around the depot."""

    def _get_tags(self):
        return RouterTags(kind="construction", requires_coords=True)

    def _solve(self, problem, rng):
        xy, d = problem.coords, problem.depot
        angles = np.arctan2(xy[:, 1] - xy[d, 1], xy[:, 0] - xy[d, 0])
        order = np.argsort(angles, kind="stable")
        return np.concatenate(([d], order[order != d])).astype(np.int64)


class RandomDescentRouter(BaseRouter):
    """Stochastic, iterative, budget-aware dummy: a random segment-reversal descent from ``init``.

    Each outer iteration draws positions ``1 <= i < j <= n-1``, reverses ``tour[i..j]`` and keeps the result
    when the objective improves; ``history_`` is the best-so-far cost. Cheap, honest about every duty of the
    iterative contract — D30 included: ``"start"`` with the init tour, one ``"iteration"`` per outer
    iteration (the candidate, the best-so-far, ``extra["positions"]``) and a stop request honoured at the
    iteration boundary — and observably seed-dependent (seeds 0 and 1 diverge on n = 12 already, check 11).
    """

    _parameter_constraints = {
        "n_iter": [Interval(int, 1, None, closed="left")],
        "patience": [Interval(int, 1, None, closed="left"), None],
        "time_limit": [Interval(float, 0.0, None, closed="neither"), None],
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_iter=50,
        patience=None,
        time_limit=None,
        init="nearest_neighbour",
        random_state=None,
        verbose=0,
    ):
        self.n_iter = n_iter
        self.patience = patience
        self.time_limit = time_limit
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem, rng):
        t0 = time.perf_counter()
        best = initial_tour(problem, self.init, rng)
        best_cost = problem.evaluate(best)
        self._emit("start", 0, best, best_cost)  # D30
        positions = np.arange(1, problem.n)  # n >= 3, so there are at least two movable positions
        history, since, reason = [], 0, "max_iter"
        for k in range(self.n_iter):
            i, j = np.sort(rng.choice(positions, size=2, replace=False))
            cand = best.copy()
            cand[i : j + 1] = cand[i : j + 1][::-1]
            c = problem.evaluate(cand)
            if c < best_cost - 1e-9 * max(1.0, abs(best_cost)):
                best, best_cost, since = cand, c, 0
            else:
                since += 1
            history.append(best_cost)
            if self.verbose:
                log.info("RandomDescentRouter iteration %d: best %.6f", k, best_cost)
            self._emit("iteration", k + 1, cand, c, best, best_cost, positions=(int(i), int(j)))  # D30
            if self._stop_requested:
                reason = "callback"
                break
            if self.time_limit is not None and time.perf_counter() - t0 > self.time_limit:
                reason = "time_limit"
                break
            if self.patience is not None and since >= self.patience:
                reason = "patience"
                break
        self.history_, self.n_iter_, self.stop_reason_ = np.asarray(history), len(history), reason
        return best


class TinyBruteForce(BaseRouter):
    """Exact, budget-aware dummy with a node cap: exhaustive search over ``problem.evaluate``."""

    def __init__(self, max_nodes=8):
        self.max_nodes = max_nodes

    def _get_tags(self):
        return RouterTags(kind="exact", exact=True, budget_aware=True, max_nodes=self.max_nodes)

    def _solve(self, problem, rng):
        others = [i for i in range(problem.n) if i != problem.depot]
        best, best_cost = None, math.inf
        for perm in itertools.permutations(others):
            tour = np.array([problem.depot, *perm], dtype=np.int64)
            c = problem.evaluate(tour)
            if c < best_cost:
                best, best_cost = tour, c
        self.is_optimal_ = True
        return best


class PlainTinyBruteForce(TinyBruteForce):
    """Exact but budget-unaware dummy: must raise under a budget (D6)."""

    def _get_tags(self):
        return RouterTags(kind="exact", exact=True, budget_aware=False, max_nodes=self.max_nodes)


DUMMY_SOLVERS = [
    IdentityRouter,
    SymmetricIdentityRouter,
    CoordsIdentityRouter,
    RandomDescentRouter,
    TinyBruteForce,
    PlainTinyBruteForce,
]


def is_dummy(Solver):
    """True for the test-suite's dummy routers (they have no entry in tests/tolerances.py)."""
    return Solver in DUMMY_SOLVERS


def solver_roster():
    """``all_solvers()`` — or the dummies while NO solver package exists in the tree.

    All-or-nothing on purpose: as soon as one solver package directory is present, ``all_solvers()``
    is called unguarded and a missing or broken registered module fails the collection loudly (D27).
    An EMPTY roster fails too: a solver directory whose registry lines are not in
    ``skroute/__init__.py`` yet would otherwise parametrise ``Solver`` over nothing and pytest would
    SKIP every test of the merge gate — green with zero solvers checked.
    """
    modules = sorted(skroute._SOLVER_MODULES)
    present = [m for m in modules if importlib.util.find_spec(m) is not None]
    if not present:
        return list(DUMMY_SOLVERS)
    roster = all_solvers()
    if not roster:
        raise pytest.UsageError(
            f"solver package(s) {present} exist but the registry in skroute/__init__.py lists no solver: "
            "add their lines (D27/D29), or the Solver-parametrised tests are silently skipped"
        )
    return roster


def pytest_generate_tests(metafunc):
    if "Solver" in metafunc.fixturenames:
        metafunc.parametrize("Solver", solver_roster(), ids=lambda s: s.__name__)  # D27: no MultiStart here


def make(Solver, **overrides):
    """Instantiate with random_state=0 when accepted; used by every parametrised test."""
    params = {"random_state": 0} if "random_state" in Solver._get_param_names() else {}
    params.update(overrides)
    return Solver(**params)


def fit_kwargs(Solver, inst):
    """coords= for requires_coords solvers; skipped on asymmetric instances (no meaningful coordinates)."""
    if make(Solver)._get_tags().requires_coords:
        if inst.get("asymmetric"):
            pytest.skip(f"{Solver.__name__} needs coordinates; asymmetric instances have none")
        return {"coords": inst["coords"]}
    return {}


__all__ = [
    "DUMMY_SOLVERS",
    "OPTIMA",
    "CoordsIdentityRouter",
    "IdentityRouter",
    "PlainTinyBruteForce",
    "RandomDescentRouter",
    "RoutingProblem",
    "SymmetricIdentityRouter",
    "TinyBruteForce",
    "fit_kwargs",
    "is_dummy",
    "make",
    "solver_roster",
]
