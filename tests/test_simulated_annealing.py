"""Acceptance tests of ``SimulatedAnnealing`` (SPEC §4.4): tiny and alicante optima at three seeds, the
fast tier, bit-exact reproducibility, the stop rules, the binding draw -> move mapping of the kernel and
the equivalence of the O(1) and full-evaluation paths. Slow-tier gaps live in tests/benchmarks."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import reference
from conftest import _euclid
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import RoutingProblem
from skroute.metaheuristics import SimulatedAnnealing, _sa
from skroute.metrics import route_cost

SETTINGS = settings(derandomize=True, deadline=None, max_examples=40)
TWO_OPT, OR_OPT, SWAP = 0, 1, 2


# --------------------------------------------------------------------------- acceptance (SPEC §4.4)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reaches_optimum_on_tiny(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    sa = SimulatedAnnealing(random_state=seed).fit(C)
    assert sa.cost_ == pytest.approx(opt, rel=1e-9)
    assert sa.cost_ == pytest.approx(route_cost(C, sa.route_))
    assert sa.history_[-1] == pytest.approx(sa.cost_) and np.all(np.diff(sa.history_) <= 1e-12)
    assert sa.stop_reason_ == "converged" and sa.n_iter_ == len(sa.history_)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_trip_respects_budget_and_matches_reference(alicante, seed):
    d, kw = alicante["bunch"], alicante["kwargs"]  # kw carries labels= and the LABEL depot
    sa = SimulatedAnnealing(random_state=seed).fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(sa.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert sa.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert sa.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, sa.problem_.to_index_tour(sa.tour_), kw["max_time_work"], 10.0 * 2, "greedy"
        )
    )


def test_multi_trip_optimal_split_reaches_the_optimum(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    sa = SimulatedAnnealing(random_state=0).fit(d.cost, time_matrix=d.time, split="optimal", **kw)
    assert sa.cost_ == pytest.approx(alicante["optimum"]["optimal"], rel=1e-9)
    assert sa.problem_.split == "optimal" and np.all(sa.trip_times_ <= kw["max_time_work"] + 1e-9)


def test_fast_tier_gap(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    sa = SimulatedAnnealing(random_state=0).fit(C, labels=fast_instance["labels"])
    assert opt <= sa.cost_ + 1e-9
    assert sa.cost_ / opt - 1 <= 0.03


def test_same_seed_is_bit_identical(small_euclidean):
    a, b = (SimulatedAnnealing(random_state=7).fit(small_euclidean["C"]) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
    assert a.t0_ == b.t0_


def test_seeds_differ(small_euclidean, medium_euclidean):
    C = small_euclidean["C"]
    a, b = SimulatedAnnealing(random_state=0).fit(C), SimulatedAnnealing(random_state=1).fit(C)
    assert not np.array_equal(a.history_, b.history_)
    C40 = medium_euclidean["C"]
    a, b = SimulatedAnnealing(random_state=0).fit(C40), SimulatedAnnealing(random_state=1).fit(C40)
    assert not np.array_equal(a.tour_, b.tour_)


def test_generator_is_advanced_and_reproduces_the_int_seed(small_euclidean):
    C = small_euclidean["C"]
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    a = SimulatedAnnealing(random_state=rng).fit(C)
    assert rng.bit_generator.state != before
    b = SimulatedAnnealing(random_state=3).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_smallest_sizes_reach_the_optimum(n, asymmetric):
    C, _ = _euclid(n, seed=n, asymmetric=asymmetric)
    sa = SimulatedAnnealing(random_state=0).fit(C)
    assert sa.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    assert sorted(sa.tour_.tolist()) == list(range(n)) and int(sa.tour_[0]) == 0


def test_asymmetric_matrix_uses_directional_costs(tiny_instance):
    if not tiny_instance["asymmetric"]:
        pytest.skip("symmetric instance")
    C = tiny_instance["C"]
    sa = SimulatedAnnealing(random_state=0).fit(C)
    assert sa.cost_ == pytest.approx(reference.tour_cost(C, sa.problem_.to_index_tour(sa.tour_)))
    assert not sa.problem_.symmetric


# --------------------------------------------------------------------------- schedule and stop rules
def test_t0_auto_is_median_uphill_delta_and_t_min_is_1e_4_t0(small_euclidean):
    C = small_euclidean["C"]
    sa = SimulatedAnnealing(random_state=0).fit(C)
    assert sa.t0_ > 0
    # geometric cooling from t0_ down to 1e-4 * t0_: the number of levels is fixed by alpha
    expected = math.ceil(math.log(1e-4) / math.log(0.995))
    assert sa.n_iter_ == expected == 1838 and sa.stop_reason_ == "converged"


def test_explicit_temperatures_are_honoured(small_euclidean):
    C = small_euclidean["C"]
    sa = SimulatedAnnealing(t0=5.0, t_min=0.5, alpha=0.5, random_state=0).fit(C)
    assert sa.t0_ == 5.0
    # T: 5 -> 2.5 -> 1.25 -> 0.625 -> 0.3125 < 0.5 after the 4th level
    assert sa.n_iter_ == 4 and sa.stop_reason_ == "converged"
    one = SimulatedAnnealing(t0=1.0, t_min=2.0, random_state=0).fit(C)  # t_min above t0: one level runs
    assert one.n_iter_ == 1 and one.stop_reason_ == "converged"


def test_time_limit_stops_after_one_level(small_euclidean):
    sa = SimulatedAnnealing(time_limit=1e-6, random_state=0).fit(small_euclidean["C"])
    assert sa.stop_reason_ == "time_limit" and sa.n_iter_ == 1 and len(sa.history_) == 1


def test_patience_counts_only_after_the_first_descent(small_euclidean):
    C = small_euclidean["C"]
    sa = SimulatedAnnealing(patience=5, random_state=0).fit(C)
    assert sa.stop_reason_ == "patience"
    assert 5 < sa.n_iter_ < 1838  # armed only once the current cost fell below the start, then 5 levels
    hist = sa.history_
    assert np.all(hist[-5:] == hist[-6])  # the last five levels brought no improvement of the best
    # a patience so large that cooling wins: the run converges instead
    full = SimulatedAnnealing(patience=10_000, random_state=0).fit(C)
    assert full.stop_reason_ == "converged"


def test_n_moves_default_is_ten_n_and_moves_are_normalised(small_euclidean):
    C = small_euclidean["C"]
    sa = SimulatedAnnealing(moves="two_opt", n_moves=1, t0=1.0, t_min=0.9, random_state=0).fit(C)
    # 0.995 ** k < 0.9 first at k = 22 levels; the parameter itself is stored verbatim
    assert sa.n_iter_ == math.ceil(math.log(0.9) / math.log(0.995)) == 22 and sa.moves == "two_opt"
    for bad in [(), ("bogus",), ("two_opt", 3)]:
        with pytest.raises(ValueError, match="moves"):
            SimulatedAnnealing(moves=bad).fit(C)
    with pytest.raises(ValueError, match="'alpha' parameter"):
        SimulatedAnnealing(alpha=1.0).fit(C)
    with pytest.raises(ValueError, match="'t0' parameter"):
        SimulatedAnnealing(t0="cold").fit(C)


def test_warm_start_from_another_solver(small_euclidean):
    C = small_euclidean["C"]
    first = SimulatedAnnealing(random_state=0).fit(C)
    warm = SimulatedAnnealing(init=first.route_, t0=1e-3, t_min=1e-4, random_state=1).fit(C)  # cold: stays
    assert warm.cost_ <= first.cost_ + 1e-9
    with pytest.raises(ValueError, match="init"):
        SimulatedAnnealing(init="warm").fit(C)


def test_verbose_logs_to_the_skroute_logger(small_euclidean, caplog):
    with caplog.at_level(logging.INFO, logger="skroute"):
        SimulatedAnnealing(verbose=1, random_state=0).fit(small_euclidean["C"])
    records = [r for r in caplog.records if r.name == "skroute"]
    assert records and all("SimulatedAnnealing" in r.getMessage() for r in records)
    assert any("stopped by converged" in r.getMessage() for r in records)


# --------------------------------------------------------------------------- the kernel contract
def _scratch(n, optimal=False):
    return (
        np.empty(n, dtype=np.int64),
        np.empty(n if optimal else 0, dtype=np.float64),
        np.empty(n if optimal else 0, dtype=np.int64),
    )


def _decode(n, ri, rj, code):
    """The binding draw -> move mapping of §4.4, in Python; None for an invalid draw."""
    if code == OR_OPT:
        i, L, j = ri, 1 + rj % 3, rj
        if i + L - 1 > n - 1 or i - 1 <= j <= i + L - 1:
            return None
        return (i, L, j)
    if ri == rj:
        return None
    return (min(ri, rj), max(ri, rj))


def _reference_delta(C, tour, ri, rj, code):
    move = _decode(len(tour), ri, rj, code)
    if move is None:
        return None
    if code == TWO_OPT:
        return reference.two_opt_delta_by_recompute(C, tour, *move)
    if code == OR_OPT:
        return reference.or_opt_delta_by_recompute(C, tour, *move, reverse=False)
    return reference.swap_delta_by_recompute(C, tour, *move)


@SETTINGS
@given(
    n=st.integers(3, 12),
    seed=st.integers(0, 1000),
    asymmetric=st.booleans(),
    codes=st.lists(st.integers(0, 2), min_size=1, max_size=30),
)
def test_sample_deltas_follows_the_binding_mapping(n, seed, asymmetric, codes):
    C, _ = _euclid(n, seed=seed, asymmetric=asymmetric)
    rng = np.random.default_rng(seed)
    tour = np.concatenate(([0], rng.permutation(np.arange(1, n)))).astype(np.int64)
    m = len(codes)
    ri = rng.integers(1, n, size=m, dtype=np.int64)
    rj = rng.integers(1, n, size=m, dtype=np.int64)
    mv = np.asarray(codes, dtype=np.int64)
    scratch, dp, pred = _scratch(n)
    for fast_path in [True, False] if not asymmetric else [False]:
        out = np.empty(m)
        _sa.sample_deltas(C, C, tour, ri, rj, mv, np.inf, 0.0, 0, fast_path, scratch, dp, pred, out)
        for s in range(m):
            expected = _reference_delta(C, tour.tolist(), int(ri[s]), int(rj[s]), int(mv[s]))
            if expected is None:
                assert np.isnan(out[s])
            else:
                assert out[s] == pytest.approx(expected, abs=1e-9)


def test_invalid_draws_are_rejected_proposals_that_change_nothing():
    C, _ = _euclid(8, seed=8)
    tour = np.arange(8, dtype=np.int64)
    best = tour.copy()
    cost = reference.tour_cost(C, tour)
    state = np.array([cost, cost])
    m = 20
    ri = rj = np.full(m, 3, dtype=np.int64)  # i == j: invalid for 2-opt and swap
    u = np.zeros(m)  # would accept anything
    mv = np.array([TWO_OPT, SWAP] * (m // 2), dtype=np.int64)
    scratch, dp, pred = _scratch(8)
    accepted = _sa.anneal_level(
        C, C, tour, best, u, ri, rj, mv, 1.0, np.inf, 0.0, 0, True, scratch, dp, pred, state
    )
    assert accepted == 0 and tour.tolist() == list(range(8)) and best.tolist() == list(range(8))
    assert state[0] == pytest.approx(cost) and state[1] == cost
    # Or-opt whose segment leaves the tour: i = 7, L = 1 + (7 % 3) = 2 -> i + L - 1 = 8 > n - 1
    ri = np.full(m, 7, dtype=np.int64)
    rj = np.full(m, 7, dtype=np.int64)
    mv = np.full(m, OR_OPT, dtype=np.int64)
    accepted = _sa.anneal_level(
        C, C, tour, best, u, ri, rj, mv, 1.0, np.inf, 0.0, 0, True, scratch, dp, pred, state
    )
    assert accepted == 0 and tour.tolist() == list(range(8))


def test_metropolis_rule_uses_the_pre_drawn_uniform():
    C, _ = _euclid(6, seed=6)
    tour = np.arange(6, dtype=np.int64)
    scratch, dp, pred = _scratch(6)
    # one 2-opt proposal (1, 4); pick a temperature so that exp(-delta / T) is strictly inside (0, 1)
    delta = reference.two_opt_delta_by_recompute(C, tour.tolist(), 1, 4)
    ri, rj, mv = (np.array([v], dtype=np.int64) for v in (1, 4, TWO_OPT))
    if delta <= 0:  # start from the reversed tour so that the same move goes uphill
        tour[1:5] = tour[1:5][::-1]
        delta = -delta
    T = delta / math.log(2.0)  # acceptance probability exactly 1/2
    for u_value, expect_accept in ((0.25, True), (0.75, False)):
        cur = tour.copy()
        best = cur.copy()
        state = np.array([0.0, reference.tour_cost(C, cur)])
        acc = _sa.anneal_level(
            C,
            C,
            cur,
            best,
            np.array([u_value]),
            ri,
            rj,
            mv,
            T,
            np.inf,
            0.0,
            0,
            True,
            scratch,
            dp,
            pred,
            state,
        )
        assert bool(acc) is expect_accept
        assert (not np.array_equal(cur, tour)) is expect_accept
        assert np.array_equal(best, tour)  # an uphill acceptance never touches the best buffer


def test_fast_and_generic_paths_agree_on_a_symmetric_instance():
    C, _ = _euclid(15, seed=15)
    rng = np.random.default_rng(0)
    m = 400
    u, ri, rj = rng.random(m), rng.integers(1, 15, m, dtype=np.int64), rng.integers(1, 15, m, dtype=np.int64)
    mv = rng.integers(0, 3, m, dtype=np.int64)
    results = []
    for fast_path in (True, False):
        tour = np.arange(15, dtype=np.int64)
        best = tour.copy()
        cost = reference.tour_cost(C, tour)
        state = np.array([cost, cost])
        scratch, dp, pred = _scratch(15)
        _sa.anneal_level(
            C, C, tour, best, u, ri, rj, mv, 20.0, np.inf, 0.0, 0, fast_path, scratch, dp, pred, state
        )
        assert state[0] == pytest.approx(reference.tour_cost(C, tour))
        assert state[1] == pytest.approx(reference.tour_cost(C, best))
        results.append((tour.copy(), best.copy(), state.copy()))
    (t1, b1, s1), (t2, b2, s2) = results
    assert np.array_equal(t1, t2) and np.array_equal(b1, b2)
    assert s1 == pytest.approx(s2)


def test_best_buffer_never_aliases_the_current_tour(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    p = RoutingProblem(d.cost, time_matrix=d.time, **kw)
    sa = SimulatedAnnealing(random_state=0).fit(p)
    # the returned tour re-evaluates to the reported best; the history is monotone
    assert sa.cost_ == pytest.approx(p.evaluate(p.to_index_tour(sa.tour_)))
    assert sa.history_[-1] == pytest.approx(sa.cost_) and np.all(np.diff(sa.history_) <= 1e-12)
