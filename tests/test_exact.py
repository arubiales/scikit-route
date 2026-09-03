"""Acceptance tests of ``skroute.exact`` (SPEC §4.1): BruteForce, HeldKarp and MILP.

Every exact solver must equal ``reference.brute_force`` on the tiny instances (symmetric and
asymmetric); BruteForce also on the Alicante multi-trip fixture under both split rules and with
the oracle's tie-breaking; HeldKarp and MILP raise under a budget (D6); MILP proves the published
optima of the fast tier and returns a valid tour when its time budget runs out. Slow-tier tests
(qa194) live in ``tests/benchmarks/test_waterloo.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import reference
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from skroute import RoutingProblem
from skroute.exact import MILP, BruteForce, HeldKarp, _brute, _hk

EXACT = [BruteForce, HeldKarp, MILP]
PLAIN_ONLY = [HeldKarp, MILP]  # certify the plain tour only: raise under a budget (D6)


def _euclid(n, seed, asymmetric=False):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    diff = xy[:, None, :] - xy[None, :, :]
    C = np.sqrt((diff**2).sum(axis=-1))
    if asymmetric:
        C = C * rng.uniform(0.7, 1.3, C.shape)
        np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C)


def _index_tour(est):
    return est.problem_.to_index_tour(est.tour_)


def _assert_valid(est, n, depot_label):
    assert est.route_[0] == depot_label == est.route_[-1] and est.tour_[0] == depot_label
    assert sorted(est.tour_.tolist()) == sorted(est.labels_.tolist()) and len(est.tour_) == n
    assert est.cost_ == pytest.approx(
        reference.route_cost_from_labels(est.problem_.cost, est.route_, est.labels_, est.depot_)
    )


# --------------------------------------------------------------------------- tiny tier: the oracle
@pytest.mark.parametrize("Exact", EXACT, ids=lambda s: s.__name__)
def test_equals_reference_on_tiny_instances(Exact, tiny_instance):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    est = Exact().fit(C)
    assert est.is_optimal_ is True
    assert est.cost_ == pytest.approx(opt, rel=1e-9)
    _assert_valid(est, tiny_instance["n"], 0)
    if Exact is BruteForce:  # ties: the lexicographically first optimum, exactly as itertools.permutations
        assert np.array_equal(_index_tour(est), reference.brute_force(C)[1])
    if Exact is MILP:
        assert est.lower_bound_ == pytest.approx(est.cost_, rel=1e-9) and est.gap_ == 0.0


@pytest.mark.parametrize("Exact", EXACT, ids=lambda s: s.__name__)
def test_depot_anywhere_in_the_matrix(Exact, tiny_instance):
    C, n = tiny_instance["C"], tiny_instance["n"]
    depot = n - 1
    est = Exact().fit(C, depot=depot)
    cost, tour = reference.brute_force(C, depot=depot)
    assert est.cost_ == pytest.approx(cost, rel=1e-9) and int(est.depot_) == depot
    if Exact is BruteForce:
        assert np.array_equal(_index_tour(est), tour)


def test_brute_force_tie_breaking_matches_itertools_on_integer_matrices():
    """Integer costs create many exact ties: the kept tour must be the one itertools finds first."""
    for seed in range(24):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 8))
        C = rng.integers(0, 6, (n, n)).astype(float)
        if seed % 2 == 0:
            C = np.triu(C, 1)
            C = C + C.T  # symmetric: exercises the halving and its orientation rule
        np.fill_diagonal(C, 0.0)
        depot = int(rng.integers(0, n))
        cost, tour = reference.brute_force(C, depot=depot)
        est = BruteForce().fit(C, depot=depot)
        assert est.cost_ == cost
        assert np.array_equal(_index_tour(est), tour), (seed, C, depot)


def test_brute_force_halving_keeps_the_bit_smaller_orientation():
    """On seed 7 the two orientations of the optimum differ in the last bit; the reference keeps the
    smaller float, which is the lexicographically LARGER tour, and so must the halved search."""
    C = _euclid(7, seed=7)
    cost, tour = reference.brute_force(C)
    est = BruteForce().fit(C)
    assert est.cost_ == cost and np.array_equal(_index_tour(est), tour)
    assert tour[1] > tour[-1]  # the reference's answer is the reversed orientation of the kept one


# --------------------------------------------------------------------------- multi-trip (BruteForce only)
def test_brute_force_alicante_matches_reference_under_both_splits(alicante):
    d, kw, ref_kw = alicante["bunch"], alicante["kwargs"], alicante["ref_kwargs"]
    costs = {}
    for split in ("greedy", "optimal"):
        est = BruteForce().fit(d.cost, time_matrix=d.time, split=split, **kw)
        cost, tour = reference.brute_force(d.cost, d.time, split=split, **ref_kw)
        assert est.is_optimal_ is True
        assert (
            est.cost_ == pytest.approx(cost, rel=1e-9) == pytest.approx(alicante["optimum"][split], rel=1e-9)
        )
        assert np.array_equal(_index_tour(est), tour)
        assert np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
        assert est.problem_.split == split and est.n_trips_ == len(est.trips_) >= 1
        costs[split] = est.cost_
    assert costs["optimal"] <= costs["greedy"] + 1e-9  # the optimal decoder is never worse (D1)


def test_brute_force_asymmetric_multi_trip_matches_reference():
    """No halving under a budget or on an asymmetric matrix: every orientation is priced."""
    rng = np.random.default_rng(3)
    C = _euclid(7, seed=3, asymmetric=True)
    T = C / 40.0 * rng.uniform(0.8, 1.2, C.shape)
    np.fill_diagonal(T, 0.0)
    budget = 1.5 * float((T[0, :] + T[:, 0]).max())
    for split in ("greedy", "optimal"):
        est = BruteForce().fit(C, time_matrix=T, max_time_work=budget, extra_cost=4.0, people=2, split=split)
        cost, tour = reference.brute_force(C, T, max_time_work=budget, extra_cost=4.0, people=2, split=split)
        assert est.cost_ == pytest.approx(cost, rel=1e-9)
        assert np.array_equal(_index_tour(est), tour)
        assert np.all(est.trip_times_ <= budget + 1e-9)


@pytest.mark.parametrize("Exact", PLAIN_ONLY, ids=lambda s: s.__name__)
def test_plain_exact_solvers_raise_under_a_budget(Exact, alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    with pytest.raises(ValueError, match="cannot certify a multi-trip optimum"):
        Exact().fit(d.cost, time_matrix=d.time, **kw)


# --------------------------------------------------------------------------- caps and knobs
@pytest.mark.parametrize(
    ("Exact", "n"), [(BruteForce, 12), (HeldKarp, 21), (MILP, 301)], ids=["BruteForce", "HeldKarp", "MILP"]
)
def test_max_nodes_raises_above_the_cap(Exact, n):
    C = _euclid(n, seed=n)
    with pytest.raises(ValueError, match=f"handles at most {n - 1} nodes, got {n}"):
        Exact().fit(C)


@pytest.mark.parametrize("Exact", EXACT, ids=lambda s: s.__name__)
def test_max_nodes_is_a_knob(Exact):
    C = _euclid(6, seed=6)
    with pytest.raises(ValueError, match="handles at most 5 nodes"):
        Exact(max_nodes=5).fit(C)
    assert Exact(max_nodes=6).fit(C).is_optimal_ is True
    with pytest.raises(
        ValueError, match=r"'max_nodes' parameter of \w+ must be an int in the range \[3, inf\)"
    ):
        Exact(max_nodes=2).fit(C)


def test_brute_force_raised_cap_solves_twelve_nodes():
    C = _euclid(12, seed=12)
    est = BruteForce(max_nodes=12).fit(C)
    assert est.cost_ == pytest.approx(HeldKarp().fit(C).cost_, rel=1e-9)


# --------------------------------------------------------------------------- HeldKarp
def test_held_karp_matches_brute_force_at_the_brute_force_cap():
    C = _euclid(11, seed=11)
    assert HeldKarp().fit(C).cost_ == pytest.approx(BruteForce().fit(C).cost_, rel=1e-9)
    A = _euclid(10, seed=10, asymmetric=True)
    assert HeldKarp().fit(A).cost_ == pytest.approx(BruteForce().fit(A).cost_, rel=1e-9)


def test_held_karp_is_direction_aware():
    A = np.array([[0, 1, 9, 9], [9, 0, 1, 9], [9, 9, 0, 1], [1, 9, 9, 0]], dtype=float)
    est = HeldKarp().fit(A)
    assert est.tour_.tolist() == [0, 1, 2, 3] and est.cost_ == 4.0


def test_held_karp_sixteen_nodes_matches_milp_quickly():
    C = _euclid(16, seed=16)
    hk = HeldKarp().fit(C)
    assert hk.cost_ == pytest.approx(MILP().fit(C).cost_, rel=1e-9)
    assert hk.fit_time_ < 5.0  # measured 0.015 s; generous for slow CI runners


def test_held_karp_kernel_rejects_bad_sizes():
    C = _euclid(4, seed=4)
    with pytest.raises(ValueError, match="between 3 and 41 nodes"):
        _hk.held_karp_search(C, np.array([1], dtype=np.int64), 0, np.empty(2, dtype=np.int64))
    with pytest.raises(ValueError, match="out must have length"):
        _hk.held_karp_search(C, np.array([1, 2, 3], dtype=np.int64), 0, np.empty(3, dtype=np.int64))


# --------------------------------------------------------------------------- BruteForce kernel
def test_brute_kernel_halving_and_buffer_restoration():
    n = 8
    C = _euclid(n, seed=n)
    tour = np.arange(n, dtype=np.int64)
    best = tour.copy()
    cost, evaluated = _brute.brute_force_search(
        C, C, tour, best, np.inf, 0.0, 0, True, np.empty(0), np.empty(0, np.int64)
    )
    full = math.factorial(n - 1)
    assert full // 2 <= evaluated < full  # the reversal is priced only near the incumbent
    assert tour.tolist() == list(range(n))  # Algorithm L leaves the buffer in ascending order
    assert cost == reference.brute_force(C)[0] and np.array_equal(best, reference.brute_force(C)[1])
    A = _euclid(n, seed=n, asymmetric=True)
    _, evaluated = _brute.brute_force_search(
        A, A, tour, best, np.inf, 0.0, 0, False, np.empty(0), np.empty(0, np.int64)
    )
    assert evaluated == full
    with pytest.raises(ValueError, match="at least 3 nodes"):
        _brute.brute_force_search(
            C, C, tour[:2].copy(), best[:2].copy(), np.inf, 0.0, 0, False, np.empty(0), np.empty(0, np.int64)
        )


# --------------------------------------------------------------------------- MILP
def test_milp_proves_the_fast_tier_optima(fast_instance):
    C, opt, labels = fast_instance["C"], fast_instance["optimum"], fast_instance["labels"]
    est = MILP().fit(C, labels=labels)
    assert est.cost_ == opt and est.is_optimal_ is True
    assert est.lower_bound_ == pytest.approx(est.cost_, rel=1e-9) and est.gap_ == 0.0
    assert int(est.route_[0]) == int(est.route_[-1]) == int(labels[0])
    assert isinstance(est.n_solves_, int) and est.n_solves_ >= 1
    assert isinstance(est.n_cuts_, int) and est.n_cuts_ >= 0


@pytest.mark.parametrize("time_limit", [1e-6, 1e-3])
def test_milp_time_out_returns_a_valid_tour(fast_instance, time_limit):
    C, opt, labels = fast_instance["C"], fast_instance["optimum"], fast_instance["labels"]
    est = MILP(time_limit=time_limit).fit(C, labels=labels)
    _assert_valid(est, C.shape[0], labels[0])
    assert est.is_optimal_ is False
    assert 0.0 <= est.gap_ < math.inf
    assert est.lower_bound_ <= est.cost_ + 1e-9 and opt <= est.cost_ + 1e-9
    assert est.gap_ == pytest.approx(max(0.0, (est.cost_ - est.lower_bound_) / est.cost_))


def test_milp_time_out_fallback_on_an_asymmetric_matrix():
    A = _euclid(30, seed=30, asymmetric=True)
    est = MILP(time_limit=1e-6).fit(A)
    _assert_valid(est, 30, 0)
    assert est.is_optimal_ is False and est.n_solves_ == 0 and est.n_cuts_ == 0 and est.gap_ >= 0.0


@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_milp_matches_held_karp_beyond_brute_force(asymmetric):
    C = _euclid(14, seed=14, asymmetric=asymmetric)
    a, b = MILP().fit(C), HeldKarp().fit(C)
    assert a.cost_ == pytest.approx(b.cost_, rel=1e-9) and a.is_optimal_
    assert a.n_solves_ >= 1 and a.n_cuts_ >= 0


def test_milp_without_time_limit():
    C = _euclid(12, seed=12)
    est = MILP(time_limit=None).fit(C)
    assert est.is_optimal_ and est.cost_ == pytest.approx(HeldKarp().fit(C).cost_, rel=1e-9)


def test_milp_relative_gap_relaxes_the_certificate():
    C = _euclid(20, seed=20)
    est = MILP(mip_rel_gap=0.5).fit(C)
    assert est.lower_bound_ <= est.cost_ + 1e-9
    assert est.is_optimal_ is (est.gap_ == 0.0)
    assert est.cost_ >= HeldKarp().fit(C).cost_ - 1e-9


def test_milp_labels_and_depot_by_label():
    C = _euclid(6, seed=6)
    names = ["a", "b", "c", "d", "e", "f"]
    est = MILP().fit(C, labels=names, depot="c")
    assert est.depot_ == "c" and est.route_[0] == "c" == est.route_[-1]
    assert est.cost_ == pytest.approx(reference.brute_force(C, depot=2)[0], rel=1e-9)


def test_milp_reuses_a_routing_problem():
    p = RoutingProblem(_euclid(9, seed=9, asymmetric=True), depot=4)
    a, b = MILP().fit(p), BruteForce().fit(p)
    assert a.cost_ == pytest.approx(b.cost_, rel=1e-9) and a.problem_ is p


# --------------------------------------------------------------------------- smallest sizes and determinism
@pytest.mark.parametrize("Exact", EXACT, ids=lambda s: s.__name__)
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_smallest_sizes(Exact, n, asymmetric):
    C = _euclid(n, seed=n, asymmetric=asymmetric)
    est = Exact().fit(C)
    cost, tour = reference.brute_force(C)
    assert est.cost_ == pytest.approx(cost, rel=1e-9) and est.is_optimal_ is True
    if Exact is BruteForce:
        assert np.array_equal(_index_tour(est), tour)


@pytest.mark.parametrize("Exact", EXACT, ids=lambda s: s.__name__)
def test_deterministic_refit(Exact, tiny_instance):
    a, b = (Exact().fit(tiny_instance["C"]) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_
    assert "random_state" not in Exact._get_param_names() and not Exact()._get_tags().stochastic


# --------------------------------------------------------------------------- property test
@settings(derandomize=True, deadline=None, max_examples=25)
@given(data=st.data())
def test_exact_solvers_agree_with_the_oracle_on_random_integer_instances(data):
    n = data.draw(st.integers(3, 7), label="n")
    asymmetric = data.draw(st.booleans(), label="asymmetric")
    depot = data.draw(st.integers(0, n - 1), label="depot")
    C = data.draw(hnp.arrays(np.int64, (n, n), elements=st.integers(0, 50)), label="C").astype(np.float64)
    if not asymmetric:
        C = np.triu(C, 1)
        C = C + C.T
    np.fill_diagonal(C, 0.0)
    C = np.ascontiguousarray(C)
    cost, tour = reference.brute_force(C, depot=depot)
    bf = BruteForce().fit(C, depot=depot)
    assert bf.cost_ == cost and np.array_equal(_index_tour(bf), tour)
    assert HeldKarp().fit(C, depot=depot).cost_ == cost
    est = MILP().fit(C, depot=depot)
    assert est.cost_ == cost and est.is_optimal_ is True
