"""Acceptance tests of ``TabuSearch`` (SPEC §4.4): tiny and alicante optima at three seeds, the fast
tier, bit-exact reproducibility, the stop rules, the tabu bookkeeping of the kernel and the generic
(asymmetric / multi-trip) path. Slow-tier gaps live in tests/benchmarks."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import reference
from conftest import _euclid

from skroute import RoutingProblem
from skroute.metaheuristics import SimulatedAnnealing, TabuSearch, _tabu
from skroute.metrics import route_cost


# --------------------------------------------------------------------------- acceptance (SPEC §4.4)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reaches_optimum_on_tiny(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    ts = TabuSearch(random_state=seed).fit(C)
    assert ts.cost_ == pytest.approx(opt, rel=1e-9)
    assert ts.cost_ == pytest.approx(route_cost(C, ts.route_))
    assert ts.history_[-1] == pytest.approx(ts.cost_) and np.all(np.diff(ts.history_) <= 1e-12)
    assert ts.stop_reason_ == "patience" and ts.n_iter_ == len(ts.history_)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_trip_respects_budget_and_matches_reference(alicante, seed):
    d, kw = alicante["bunch"], alicante["kwargs"]  # kw carries labels= and the LABEL depot
    ts = TabuSearch(random_state=seed).fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(ts.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert ts.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert ts.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, ts.problem_.to_index_tour(ts.tour_), kw["max_time_work"], 10.0 * 2, "greedy"
        )
    )


def test_multi_trip_optimal_split_reaches_the_optimum(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    ts = TabuSearch(random_state=0).fit(d.cost, time_matrix=d.time, split="optimal", **kw)
    assert ts.cost_ == pytest.approx(alicante["optimum"]["optimal"], rel=1e-9)
    assert ts.problem_.split == "optimal" and np.all(ts.trip_times_ <= kw["max_time_work"] + 1e-9)


def test_fast_tier_gap(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    ts = TabuSearch(random_state=0).fit(C, labels=fast_instance["labels"])
    assert opt <= ts.cost_ + 1e-9
    assert ts.cost_ / opt - 1 <= 0.08


def test_same_seed_is_bit_identical(small_euclidean):
    a, b = (TabuSearch(random_state=7).fit(small_euclidean["C"]) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)


def test_seeds_differ_on_the_medium_instance(medium_euclidean):
    C = medium_euclidean["C"]
    a, b = TabuSearch(random_state=0).fit(C), TabuSearch(random_state=1).fit(C)
    assert not np.array_equal(a.tour_, b.tour_)
    assert a.n_iter_ != b.n_iter_ or not np.array_equal(a.history_, b.history_)


def test_generator_is_advanced_and_reproduces_the_int_seed(small_euclidean):
    C = small_euclidean["C"]
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    a = TabuSearch(random_state=rng).fit(C)
    assert rng.bit_generator.state != before
    b = TabuSearch(random_state=3).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_smallest_sizes_reach_the_optimum(n, asymmetric):
    C, _ = _euclid(n, seed=n, asymmetric=asymmetric)
    ts = TabuSearch(random_state=0).fit(C)
    assert ts.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    assert sorted(ts.tour_.tolist()) == list(range(n)) and int(ts.tour_[0]) == 0


def test_asymmetric_matrix_uses_directional_costs(tiny_instance):
    if not tiny_instance["asymmetric"]:
        pytest.skip("symmetric instance")
    C = tiny_instance["C"]
    ts = TabuSearch(random_state=0).fit(C)
    assert ts.cost_ == pytest.approx(reference.tour_cost(C, ts.problem_.to_index_tour(ts.tour_)))
    assert not ts.problem_.symmetric


# --------------------------------------------------------------------------- parameters and stop rules
def test_max_iter_and_history_length(small_euclidean):
    ts = TabuSearch(n_iter=5, patience=None, random_state=0).fit(small_euclidean["C"])
    assert ts.stop_reason_ == "max_iter" and ts.n_iter_ == 5 and len(ts.history_) == 5


def test_patience_stops_after_that_many_non_improving_iterations(small_euclidean):
    ts = TabuSearch(patience=20, random_state=0).fit(small_euclidean["C"])
    assert ts.stop_reason_ == "patience" and ts.n_iter_ < 1000
    hist = ts.history_
    assert np.all(hist[-20:] == hist[-21]) and hist[-21] < hist[-22]


def test_time_limit_stops_after_one_iteration(small_euclidean):
    ts = TabuSearch(time_limit=1e-6, random_state=0).fit(small_euclidean["C"])
    assert ts.stop_reason_ == "time_limit" and ts.n_iter_ == 1 and len(ts.history_) == 1


def test_fixed_tenure_consumes_no_randomness():
    C, _ = _euclid(9, seed=9)
    a, b = TabuSearch(tenure=5, random_state=0).fit(C), TabuSearch(tenure=5, random_state=1).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)
    assert a.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    with pytest.raises(ValueError, match="'tenure' parameter"):
        TabuSearch(tenure=0).fit(C)
    with pytest.raises(ValueError, match="'tenure' parameter"):
        TabuSearch(tenure="random").fit(C)


def test_full_neighbourhood_and_small_candidate_lists():
    C, _ = _euclid(9, seed=9)
    opt = reference.brute_force(C)[0]
    full = TabuSearch(n_candidates=None, random_state=0).fit(C)
    assert full.cost_ == pytest.approx(opt, rel=1e-9)
    narrow = TabuSearch(n_candidates=2, random_state=0).fit(C)
    assert sorted(narrow.tour_.tolist()) == list(range(9))
    with pytest.raises(ValueError, match="'n_candidates' parameter"):
        TabuSearch(n_candidates=0).fit(C)


def test_warm_start_from_another_solver(small_euclidean):
    C = small_euclidean["C"]
    sa = SimulatedAnnealing(random_state=0).fit(C)
    ts = TabuSearch(init=sa.tour_, random_state=0).fit(C)
    assert ts.cost_ <= sa.cost_ + 1e-9
    C9, _ = _euclid(9, seed=9)
    ts = TabuSearch(init="random", random_state=0).fit(C9)
    assert ts.cost_ == pytest.approx(reference.brute_force(C9)[0], rel=1e-9)
    with pytest.raises(ValueError, match="init"):
        TabuSearch(init="warm").fit(C)


def test_verbose_logs_to_the_skroute_logger(small_euclidean, caplog):
    with caplog.at_level(logging.INFO, logger="skroute"):
        TabuSearch(verbose=1, random_state=0).fit(small_euclidean["C"])
    records = [r for r in caplog.records if r.name == "skroute"]
    assert records and all("TabuSearch" in r.getMessage() for r in records)
    assert any("stopped by patience" in r.getMessage() for r in records)


# --------------------------------------------------------------------------- the kernel contract
def _setup(C, T=None, max_time=math.inf, fixed=0.0, split=0, k=10):
    n = C.shape[0]
    p = (
        RoutingProblem(C)
        if T is None
        else RoutingProblem(C, time_matrix=T, max_time_work=max_time, extra_cost=fixed)
    )
    tour = np.arange(n, dtype=np.int64)
    pos = np.arange(n, dtype=np.int64)
    cand = p.neighbours(min(k, n - 1))
    until = np.zeros((n, n), dtype=np.int32)
    scratch = np.empty(n, dtype=np.int64)
    dp, pred = np.empty(0), np.empty(0, dtype=np.int64)
    best = tour.copy()
    cost = p.evaluate(tour)
    state = np.array([cost, cost])
    return p, tour, pos, cand, until, scratch, dp, pred, best, state


def _edges(tour):
    n = len(tour)
    return {(int(tour[k]), int(tour[(k + 1) % n])) for k in range(n)}


def test_step_marks_exactly_the_removed_edges_and_keeps_pos_consistent():
    C, _ = _euclid(12, seed=12)
    _p, tour, pos, cand, until, scratch, dp, pred, best, state = _setup(C)
    before = tour.copy()
    applied = _tabu.tabu_step(
        C, C, tour, pos, cand, until, 0, 7, np.inf, 0.0, 0, True, True, True, scratch, dp, pred, best, state
    )
    assert applied and not np.array_equal(tour, before)
    assert np.array_equal(pos[tour], np.arange(12))
    removed = {frozenset(e) for e in _edges(before) - _edges(tour)}
    assert 2 <= len(removed) <= 3  # a 2-opt move removes two edges, an Or-opt move three
    marked = {frozenset((int(x), int(y))) for x, y in zip(*np.nonzero(until), strict=True)}
    assert marked == removed
    for x, y in zip(*np.nonzero(until), strict=True):
        assert until[x, y] == 7 and until[y, x] == 7  # both orientations on a symmetric matrix
    assert state[0] == pytest.approx(reference.tour_cost(C, tour))
    assert state[1] == pytest.approx(min(reference.tour_cost(C, tour), reference.tour_cost(C, before)))


def test_asymmetric_step_marks_arcs_only():
    C, _ = _euclid(10, seed=10, asymmetric=True)
    _p, tour, pos, cand, until, scratch, dp, pred, best, state = _setup(C)
    before = tour.copy()
    _tabu.tabu_step(
        C, C, tour, pos, cand, until, 0, 5, np.inf, 0.0, 0, False, False, True, scratch, dp, pred, best, state
    )
    removed = _edges(before) - _edges(tour)
    marked = {(int(x), int(y)) for x, y in zip(*np.nonzero(until), strict=True)}
    assert marked == removed and all(until[y, x] == 0 for x, y in removed if (y, x) not in removed)
    assert state[0] == pytest.approx(reference.tour_cost(C, tour))


def test_tabu_edges_are_not_re_added_without_aspiration():
    C, _ = _euclid(12, seed=12)
    _p, tour, pos, cand, until, scratch, dp, pred, best, state = _setup(C)
    # make every edge NOT in the tour tabu: every move then adds a tabu edge and only aspiration lets it pass
    n = 12
    until[:] = 10**6
    for a, b in _edges(tour):
        until[a, b] = until[b, a] = 0
    np.fill_diagonal(until, 0)
    state[1] = -1.0  # nothing can beat the best -> aspiration never fires
    applied = _tabu.tabu_step(
        C, C, tour, pos, cand, until, 0, 5, np.inf, 0.0, 0, True, True, True, scratch, dp, pred, best, state
    )
    # every move is tabu: the kernel falls back to the best move overall so the search never stalls
    assert applied and sorted(tour.tolist()) == list(range(n))
    # with aspiration possible (a large best cost), the chosen move is the best overall move as well
    _p, tour, pos, cand, until, scratch, dp, pred, best, state = _setup(C)
    reference_tour = tour.copy()
    _tabu.tabu_step(
        C,
        C,
        reference_tour,
        pos.copy(),
        cand,
        until.copy(),
        0,
        5,
        np.inf,
        0.0,
        0,
        True,
        True,
        True,
        scratch,
        dp,
        pred,
        best.copy(),
        state.copy(),
    )
    until[:] = 10**6
    for a, b in _edges(tour):
        until[a, b] = until[b, a] = 0
    np.fill_diagonal(until, 0)
    _tabu.tabu_step(
        C, C, tour, pos, cand, until, 0, 5, np.inf, 0.0, 0, True, True, True, scratch, dp, pred, best, state
    )
    assert np.array_equal(tour, reference_tour)


def test_step_applies_a_worsening_move_at_a_local_optimum():
    C, _ = _euclid(9, seed=9)
    opt_cost, opt_tour = reference.brute_force(C)
    _p, tour, pos, cand, until, scratch, dp, pred, best, state = _setup(C)
    tour[:] = opt_tour
    pos[tour] = np.arange(9)
    best[:] = tour
    state[:] = opt_cost
    applied = _tabu.tabu_step(
        C, C, tour, pos, cand, until, 0, 4, np.inf, 0.0, 0, True, True, True, scratch, dp, pred, best, state
    )
    assert applied and state[0] > opt_cost and state[1] == opt_cost
    assert np.array_equal(best, opt_tour)  # the best buffer is untouched by a worsening move


def test_generic_path_prices_the_multi_trip_objective(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    p = RoutingProblem(d.cost, time_matrix=d.time, **kw)
    n = p.n
    tour = np.arange(n, dtype=np.int64)
    pos = np.arange(n, dtype=np.int64)
    cand = p.neighbours(n - 1)
    until = np.zeros((n, n), dtype=np.int32)
    scratch = np.empty(n, dtype=np.int64)
    dp, pred = np.empty(0), np.empty(0, dtype=np.int64)
    best = tour.copy()
    cost = p.evaluate(tour)
    state = np.array([cost, cost])
    _tabu.tabu_step(
        d.cost,
        d.time,
        tour,
        pos,
        cand,
        until,
        0,
        3,
        p.max_time_work,
        p.fixed_cost,
        p.split_code,
        False,
        True,
        True,
        scratch,
        dp,
        pred,
        best,
        state,
    )
    assert state[0] == pytest.approx(p.evaluate(tour)) and state[1] == pytest.approx(p.evaluate(best))
    assert np.array_equal(pos[tour], np.arange(n))
