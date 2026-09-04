"""EVERY slow-tier gap test of the tolerance table (SPEC §6), driven by ``tests/tolerances.py`` and
parametrised over ``skroute.all_solvers()`` on the four slow Waterloo instances (qa194, uy734, zi929,
lu980). Deselected by default (``addopts = "-m 'not slow'"``); run with ``python -m pytest tests -m slow``.

For every key of a solver (``"Genetic"``/``"Genetic[memetic]"``, ``"Insertion[farthest]"``...) whose
``SLOW`` entry lists the instance: ``optimum <= cost_ + 1e-9`` first (a violation is a reader/rounding/
evaluator bug, never a good result), then ``cost_ / optimum - 1 <= tolerance`` (``0.0`` means equality at
``rel=1e-9``). The per-solver instance restrictions of the table (Genetic plain on qa194 only, NRBS and SOM
on qa194, MILP on qa194 with ``time_limit=150``) are the keys of the ``SLOW`` dicts; exact solvers capped
below n = 194 are skipped by ``max_nodes``. ``MultiStart(SimulatedAnnealing(), 4)`` is not in
``all_solvers()`` and has its own test here (``<= SimulatedAnnealing``'s entry, ``n_jobs=1 == n_jobs=2``).
"""

from __future__ import annotations

import numpy as np
import pytest
import tolerances
from conftest import fit_kwargs, is_dummy, make

from skroute.ensemble import MultiStart
from skroute.metaheuristics import SimulatedAnnealing

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

# parameters that differ from the defaults on the slow tier, as stated in the tolerance table
SLOW_OVERRIDES: dict[str, dict] = {"MILP": {"time_limit": 150.0}}


def _gap(cost: float, optimum: float) -> float:
    return cost / optimum - 1.0


def test_slow_tier_gap(Solver, slow_instance, record_property):
    if is_dummy(Solver):
        pytest.skip(f"{Solver.__name__} is a test-suite dummy: no tolerance entry by design")
    name, C, opt = slow_instance["name"], slow_instance["C"], slow_instance["optimum"]
    tags = make(Solver)._get_tags()
    if tags.max_nodes is not None and C.shape[0] > tags.max_nodes:
        pytest.skip(f"{Solver.__name__} is capped at {tags.max_nodes} nodes")
    active = [
        (key, tolerances.SLOW[key][name])
        for key in tolerances.keys_for(Solver.__name__)
        if tolerances.SLOW[key] is not None and name in tolerances.SLOW[key]
    ]
    if not active:
        pytest.skip(f"{Solver.__name__}: not run on {name} (SLOW entry)")
    kw = fit_kwargs(Solver, slow_instance)
    kw["labels"] = slow_instance["labels"]
    for key, tol in active:
        est = make(Solver, **tolerances.params_for(key), **SLOW_OVERRIDES.get(Solver.__name__, {}))
        est.fit(C, **kw)
        gap = _gap(est.cost_, opt)
        record_property(f"{key}@{name}", f"gap={gap:.5f} tol={tol} time={est.fit_time_:.1f}s")
        assert opt <= est.cost_ + 1e-9, f"{key} on {name}: cost {est.cost_} below the published optimum {opt}"
        assert int(est.route_[0]) == int(est.route_[-1]) == int(slow_instance["labels"][0])
        if tol == 0.0:
            assert est.cost_ == pytest.approx(opt, rel=1e-9), f"{key} on {name}: {est.cost_} != {opt}"
        else:
            assert gap <= tol, f"{key} on {name}: gap {gap:.4f} > {tol} (cost {est.cost_}, optimum {opt})"


def test_multistart_of_simulated_annealing(slow_instance, record_property):
    name, C, opt = slow_instance["name"], slow_instance["C"], slow_instance["optimum"]
    tol = tolerances.SLOW["SimulatedAnnealing"][name]
    one = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=1, random_state=0).fit(
        C, labels=slow_instance["labels"]
    )
    two = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=2, random_state=0).fit(
        C, labels=slow_instance["labels"]
    )
    gap = _gap(one.cost_, opt)
    record_property(f"MultiStart[SA,4]@{name}", f"gap={gap:.5f} tol={tol} time={one.fit_time_:.1f}s")
    assert opt <= one.cost_ + 1e-9
    assert gap <= tol, f"MultiStart(SA, 4) on {name}: gap {gap:.4f} > {tol}"
    assert np.array_equal(one.tour_, two.tour_) and np.array_equal(one.costs_, two.costs_)
    assert one.best_index_ == two.best_index_ and one.cost_ == two.cost_
    # never worse than the single run it contains at the same seed budget
    assert one.cost_ <= one.costs_.max()
