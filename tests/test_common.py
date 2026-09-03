"""The merge gate of every solver PR: ``check_router`` over ``all_solvers()`` (checks 1-11 and 13) plus
the tolerance tests (12) driven by ``tests/tolerances.py``; and the proof that the battery itself works,
run against the test-suite's dummy routers (which also expose what the battery must reject)."""

from __future__ import annotations

import ctypes
import dataclasses
import os
import re
import sys

import numpy as np
import pytest
import reference
import tolerances
from conftest import DUMMY_SOLVERS, fit_kwargs, is_dummy, make

from skroute.base import BaseRouter, RouterTags, clone
from skroute.exceptions import InfeasibleProblemError
from skroute.utils import Bunch, estimator_checks
from skroute.utils.estimator_checks import CheckSkipped, check_router

CHECKS = check_router.checks
CHECK_IDS = [name for name, _ in CHECKS]
CHECK_FNS = [fn for _, fn in CHECKS]


def _run(check, estimator):
    try:
        check(estimator)
    except CheckSkipped as e:
        pytest.skip(str(e))


# --------------------------------------------------------------------------- checks 1-11, 13 over the roster
@pytest.mark.parametrize("check", CHECK_FNS, ids=CHECK_IDS)
def test_check_router(Solver, check, capfd):  # capfd: a printf in a .pyx bypasses sys.stdout (capsys)
    _run(check, make(Solver))
    out, err = capfd.readouterr()
    assert out == "" and err == "", "nothing may be printed while a solver runs (D24)"


# --------------------------------------------------------------------------- 12. tolerances
def _keys(Solver):
    if is_dummy(Solver):
        pytest.skip(f"{Solver.__name__} is a test-suite dummy: no tolerance entry by design")
    return tolerances.keys_for(
        Solver.__name__
    )  # KeyError("add a tolerance for <Name> in tests/tolerances.py")


def _fit(Solver, key, seed, C, **kw):
    est = make(Solver, **tolerances.params_for(key))
    if "random_state" in Solver._get_param_names():
        est.set_params(random_state=seed)
    return est.fit(C, **kw)


def test_tiny_tolerance(Solver, tiny_instance):
    keys = _keys(Solver)
    tags = make(Solver)._get_tags()
    if tiny_instance["asymmetric"] and tags.requires_symmetric:
        pytest.skip(f"{Solver.__name__} refuses asymmetric matrices")
    if tags.max_nodes is not None and tiny_instance["n"] > tags.max_nodes:
        pytest.skip(f"{Solver.__name__} is capped at {tags.max_nodes} nodes")
    kw = fit_kwargs(Solver, tiny_instance)
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    to_optimum = tags.exact or Solver.__name__ in tolerances.SEEDS_TO_OPTIMUM
    seeds = (0, 1, 2) if Solver.__name__ in tolerances.SEEDS_TO_OPTIMUM else (0,)
    for key in keys:
        tol = tolerances.TINY[key]
        for seed in seeds:
            est = _fit(Solver, key, seed, C, **kw)
            assert opt <= est.cost_ + 1e-9, f"{key}: below the brute-force optimum (evaluator bug)"
            assert est.cost_ == pytest.approx(
                reference.route_cost_from_labels(C, est.route_, est.labels_, est.depot_), rel=1e-9
            )
            if to_optimum or tol == 0.0:
                assert est.cost_ == pytest.approx(opt, rel=1e-9), f"{key} seed {seed}: {est.cost_} != {opt}"
            elif tol is not None:
                assert est.cost_ / opt - 1 <= tol, f"{key}: gap {est.cost_ / opt - 1:.4f} > {tol}"


def test_fast_tolerance(Solver, fast_instance):
    keys = _keys(Solver)
    tags = make(Solver)._get_tags()
    if tags.max_nodes is not None and fast_instance["C"].shape[0] > tags.max_nodes:
        pytest.skip(f"{Solver.__name__} is capped at {tags.max_nodes} nodes")
    kw = fit_kwargs(Solver, fast_instance)
    kw["labels"] = fast_instance["labels"]
    C, opt = fast_instance["C"], fast_instance["optimum"]
    active = [k for k in keys if tolerances.FAST[k] is not None]
    if not active:
        pytest.skip(f"{Solver.__name__}: not run on the fast tier")
    for key in active:
        tol = tolerances.FAST[key]
        est = _fit(Solver, key, 0, C, **kw)
        assert opt <= est.cost_ + 1e-9, f"{key}: below the published optimum (reader/rounding bug)"
        assert int(est.route_[0]) == int(est.route_[-1]) == int(fast_instance["labels"][0])
        if tol == 0.0:
            assert est.cost_ == pytest.approx(opt, rel=1e-9)
        else:
            assert est.cost_ / opt - 1 <= tol, (
                f"{key} on {fast_instance['name']}: gap {est.cost_ / opt - 1:.4f} > {tol}"
            )


def test_every_solver_has_a_tolerance_entry(Solver):
    if is_dummy(Solver):
        pytest.skip("dummy router")
    for key in tolerances.keys_for(Solver.__name__):
        assert key in tolerances.TINY and key in tolerances.FAST and key in tolerances.SLOW


def test_missing_tolerance_message():
    with pytest.raises(KeyError, match=re.escape("add a tolerance for NoSuchSolver in tests/tolerances.py")):
        tolerances.keys_for("NoSuchSolver")
    assert tolerances.keys_for("Genetic") == ["Genetic", "Genetic[memetic]"]
    assert tolerances.params_for("Insertion[cheapest]") == {"strategy": "cheapest"}
    assert tolerances.params_for("TwoOpt") == {}


# --------------------------------------------------------------------------- the battery on the dummies
@pytest.mark.parametrize("Dummy", DUMMY_SOLVERS, ids=lambda s: s.__name__)
@pytest.mark.parametrize("check", CHECK_FNS, ids=CHECK_IDS)
def test_battery_passes_on_dummies(Dummy, check):
    _run(check, make(Dummy))


def _synthetic_alicante():
    """An 8-node multi-trip bunch shaped like load_alicante_murcia(), for the multi-trip checks."""
    rng = np.random.default_rng(8)
    xy = rng.random((8, 2)) * 100
    C = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    T = C / 40.0
    labels = np.arange(10000002, 10000010, dtype=np.int64)
    d = Bunch(
        cost=np.ascontiguousarray(C), time=np.ascontiguousarray(T), coords=xy, labels=labels, depot=10000002
    )
    budget = 1.5 * float((T[0, :] + T[:, 0]).max())
    kw = {"labels": labels, "depot": 10000002, "max_time_work": budget, "extra_cost": 10.0, "people": 2}
    return {"bunch": d, "kwargs": kw, "budget": budget}


@pytest.mark.parametrize("Dummy", DUMMY_SOLVERS, ids=lambda s: s.__name__)
def test_multi_trip_checks_on_dummies_with_synthetic_data(Dummy, monkeypatch):
    monkeypatch.setattr(estimator_checks, "_load_alicante", _synthetic_alicante)
    estimator_checks.check_cost_recomputed(make(Dummy))
    estimator_checks.check_multi_trip(make(Dummy))


def test_check_router_driver_runs_and_reports_skips(monkeypatch, recwarn):
    monkeypatch.setattr(estimator_checks, "_load_alicante", _synthetic_alicante)
    from conftest import RandomDescentRouter

    check_router(RandomDescentRouter())
    assert not [w for w in recwarn if "skipped" in str(w.message)]

    def unavailable():
        raise CheckSkipped("no datasets here")

    monkeypatch.setattr(estimator_checks, "_load_alicante", unavailable)
    with pytest.warns(UserWarning, match="check_router: 4_cost_recomputed skipped: no datasets here"):
        check_router(RandomDescentRouter())


def test_check_router_rejects_wrong_input():
    from conftest import IdentityRouter

    with pytest.raises(TypeError, match="unfitted BaseRouter instance"):
        check_router(IdentityRouter)
    fitted = IdentityRouter().fit(np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float))
    with pytest.raises(ValueError, match="UNFITTED"):
        check_router(fitted)


# --------------------------------------------------------------------------- what the battery must reject
class _StoresExtraAttribute(BaseRouter):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.cache = {}

    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n), -problem.depot)


class _Prints(BaseRouter):
    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        print("solving")
        return np.roll(np.arange(problem.n), -problem.depot)


class _NonMonotoneHistory(BaseRouter):
    def __init__(self, random_state=None, verbose=0):
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)

    def _solve(self, problem, rng):
        tour = np.roll(np.arange(problem.n), -problem.depot)
        c = problem.evaluate(tour)
        self.history_, self.n_iter_, self.stop_reason_ = [c, c + 1.0, c], 3, "max_iter"
        return tour


class _IgnoresSeed(BaseRouter):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def _get_tags(self):
        return RouterTags(kind="metaheuristic", stochastic=True)

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n), -problem.depot)


class _NotReallyExact(BaseRouter):
    def _get_tags(self):
        return RouterTags(kind="exact", exact=True, budget_aware=True)

    def _solve(self, problem, rng):
        self.is_optimal_ = True
        return np.roll(np.arange(problem.n), -problem.depot)


class _IllegalStopReason(_NonMonotoneHistory):
    def _solve(self, problem, rng):
        tour = np.roll(np.arange(problem.n), -problem.depot)
        c = problem.evaluate(tour)
        self.history_, self.n_iter_, self.stop_reason_ = [c], 1, "patience"  # no patience parameter
        return tour


class _ForgotStochasticTag(BaseRouter):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n), -problem.depot)


class SimulatedAnnealing(_NonMonotoneHistory):
    """Test-local namesake of the real solver: the §3.4 table says SA never stops by 'max_iter'."""

    def __init__(self, patience=None, time_limit=None, random_state=None, verbose=0):
        self.patience = patience
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose

    def _solve(self, problem, rng):
        tour = np.roll(np.arange(problem.n), -problem.depot)
        self.history_, self.n_iter_, self.stop_reason_ = [problem.evaluate(tour)], 1, "max_iter"
        return tour


class _RaisesInfeasible(BaseRouter):
    """Raises InfeasibleProblemError for every input: check 6 must tell it apart from a plain ValueError."""

    def _get_tags(self):
        return RouterTags(kind="construction")

    def fit(self, X, **kw):
        raise InfeasibleProblemError("not the message of §3.3")


class _PrintsFromC(BaseRouter):
    """A ``printf`` in a kernel: bypasses ``sys.stdout``, sits in libc's buffer until flushed."""

    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        ctypes.CDLL(None).printf(b"solving\n")
        return np.roll(np.arange(problem.n), -problem.depot)


class _WritesToFd(BaseRouter):
    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        os.write(2, b"solving\n")
        return np.roll(np.arange(problem.n), -problem.depot)


class _MutatesInput(BaseRouter):
    """Writes into the aliased cost matrix: the battery's inputs are read-only, so it raises at the write."""

    def _get_tags(self):
        return RouterTags(kind="construction")

    def _solve(self, problem, rng):
        problem.cost[0, 1] = 0.0
        return np.roll(np.arange(problem.n), -problem.depot)


@pytest.mark.parametrize(
    ("Bad", "check", "match"),
    [
        (
            _StoresExtraAttribute,
            estimator_checks.check_init_and_params,
            "check 1: __init__ must store exactly",
        ),
        (_Prints, estimator_checks.check_no_printing, "check 9: fit must not print to stdout"),
        (
            _NonMonotoneHistory,
            estimator_checks.check_iterative_contract,
            "check 10: history_ must be best-so-far",
        ),
        (
            _IllegalStopReason,
            estimator_checks.check_iterative_contract,
            r"check 10: .*cannot stop by 'patience'",
        ),
        (
            _IgnoresSeed,
            estimator_checks.check_stochastic_reproducibility,
            "check 11: seeds 0 and 1 must give",
        ),
        (
            _NotReallyExact,
            estimator_checks.check_smallest_sizes,
            "check 13: exact solver must reach the optimum",
        ),
        (
            _ForgotStochasticTag,
            estimator_checks.check_stochastic_reproducibility,
            "check 11: RouterTags.stochastic",
        ),
        (
            SimulatedAnnealing,
            estimator_checks.check_iterative_contract,
            re.escape(
                "check 10: SimulatedAnnealing may only stop by ['converged', 'patience', 'time_limit']"
            ),
        ),
        (
            _RaisesInfeasible,
            estimator_checks.check_invalid_inputs,
            "check 6: .* case raised InfeasibleProblemError",
        ),
        (_WritesToFd, estimator_checks.check_no_printing, "check 9: fit must not print to stderr"),
    ],
    ids=[
        "extra-attribute",
        "prints",
        "non-monotone",
        "illegal-stop",
        "ignores-seed",
        "not-exact",
        "no-stochastic-tag",
        "undocumented-stop-reason",
        "infeasible-instead-of-value-error",
        "writes-to-fd",
    ],
)
def test_battery_rejects_violations(Bad, check, match):
    with pytest.raises(AssertionError, match=match):
        check(Bad())


@pytest.mark.skipif(sys.platform == "win32", reason="ctypes.CDLL(None) is POSIX-only")
def test_battery_catches_a_c_level_printf():
    with pytest.raises(AssertionError, match="check 9: fit must not print to stdout, got 'solving"):
        estimator_checks.check_no_printing(_PrintsFromC())


def test_battery_inputs_are_read_only_so_a_mutating_solver_explodes_at_the_write():
    C, xy = estimator_checks._euclid(6, seed=6)
    assert not C.flags.writeable and not xy.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        estimator_checks.check_fit_results(_MutatesInput())
    check_router(make(DUMMY_SOLVERS[0]))  # read-only inputs are fine for a solver that only reads them


# --------------------------------------------------------------------------- what the battery must accept
class _Wrapper(BaseRouter):
    """MultiStart-shaped wrapper (SPEC §4.5): tags copied from the inner estimator with kind='ensemble', the
    inner estimator fitted several times on the shared problem. Around a budget-unaware solver it warns
    once itself and once per inner fit: check 7 requires a UserWarning, not exactly one."""

    def __init__(self, estimator, n_restarts=3):
        self.estimator = estimator
        self.n_restarts = n_restarts

    def _get_tags(self):
        return dataclasses.replace(self.estimator._get_tags(), kind="ensemble")

    def _solve(self, problem, rng):
        fits = [clone(self.estimator).fit(problem) for _ in range(self.n_restarts)]
        return problem.to_index_tour(min(fits, key=lambda e: e.cost_).tour_)


def test_check_tags_honoured_accepts_a_wrapper_around_a_budget_unaware_solver():
    from conftest import IdentityRouter

    wrapper = _Wrapper(IdentityRouter())
    C, T, xy, budget = estimator_checks._synthetic_multi_trip()
    with pytest.warns(UserWarning) as record:
        clone(wrapper).fit(C, time_matrix=T, max_time_work=budget, extra_cost=1.0, coords=xy)
    assert len([w for w in record if "ignores max_time_work" in str(w.message)]) == 4  # 1 outer + 3 inner
    estimator_checks.check_tags_honoured(wrapper)
    estimator_checks.check_init_and_params(wrapper)


def test_check_not_fitted_accepts_trailing_underscore_parameters():
    class Lam(BaseRouter):
        def __init__(self, lambda_=0.5):
            self.lambda_ = lambda_

        def _get_tags(self):
            return RouterTags(kind="construction")

        def _solve(self, problem, rng):
            return np.roll(np.arange(problem.n), -problem.depot)

    estimator_checks.check_not_fitted(Lam(lambda_=0.9))
    estimator_checks.check_init_and_params(Lam(lambda_=0.9))
    estimator_checks.check_fit_results(Lam(lambda_=0.9))


def test_allowed_stop_reasons_follow_the_table_or_the_parameters():
    from conftest import RandomDescentRouter

    allowed = estimator_checks._allowed_stop_reasons
    assert allowed(RandomDescentRouter()) == {"converged", "max_iter", "patience", "time_limit"}
    assert allowed(_NonMonotoneHistory()) == {"converged", "max_iter"}  # no patience/time_limit parameters
    assert allowed(SimulatedAnnealing()) == {"converged", "patience", "time_limit"}  # the §3.4 table
    assert allowed(_Wrapper(SimulatedAnnealing())) == {
        "converged",
        "patience",
        "time_limit",
    }  # copies the inner's
