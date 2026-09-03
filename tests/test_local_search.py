"""Acceptance tests of ``skroute.local_search`` (SPEC §4.3), in addition to ``check_router``.

Tiny instances against ``reference.brute_force``, the fast tier (wi29, dj38), the alicante multi-trip
fixture, asymmetric instances, the smallest sizes, bit-identical reproducibility of the ILS and the
local-optimality of the descents (checked move by move with the pure-Python oracles). Slow-tier gaps
live in ``tests/benchmarks/test_waterloo.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import reference
from hypothesis import given, settings
from hypothesis import strategies as st
from tolerances import FAST, TINY

from skroute import RoutingProblem
from skroute.local_search import IteratedLocalSearch, LocalSearch, OrOpt, TwoOpt
from skroute.local_search._iterated import _default_temperature, _reversal_pairs
from skroute.local_search._local_search import MOVE_TUPLES, Descent, changed_nodes, normalise_moves
from skroute.metrics import route_cost
from skroute.utils import initial_tour

DESCENTS = [TwoOpt, OrOpt, LocalSearch]
# tolerance numbers live in tests/tolerances.py only (SPEC §6), keyed by class name


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


def _no_reversal_improves(C, tour):
    """True when no 2-opt move (reversal of tour[i..j], 1 <= i < j <= n-1) improves the closed tour."""
    n = len(tour)
    cost = reference.tour_cost(C, tour)
    for i in range(1, n):
        for j in range(i + 1, n):
            if reference.tour_cost(C, reference.two_opt_apply(tour, i, j)) < cost - 1e-9 * max(1.0, cost):
                return False
    return True


def _no_or_opt_improves(C, tour, max_segment=3, reverse_too=True):
    """True when no Or-opt relocation (segments of 1..max_segment nodes) improves the closed tour."""
    n = len(tour)
    cost = reference.tour_cost(C, tour)
    for L in range(1, max_segment + 1):
        for i in range(1, n - L + 1):
            for j in range(n):
                if i - 1 <= j <= i + L - 1:
                    continue
                for rev in (False, True) if reverse_too else (False,):
                    moved = reference.or_opt_apply(tour, i, L, j, rev)
                    if reference.tour_cost(C, moved) < cost - 1e-9 * max(1.0, cost):
                        return False
    return True


def _iterative_contract(est, allowed):
    assert est.n_iter_ == len(est.history_) >= 1
    assert est.history_.dtype == np.float64
    assert np.all(np.diff(est.history_) <= 1e-9 * max(1.0, float(np.abs(est.history_).max())))
    assert est.history_[-1] == pytest.approx(est.cost_, rel=1e-9)
    assert est.stop_reason_ in allowed


# --------------------------------------------------------------------------- the three descents
@pytest.mark.parametrize("Cls", DESCENTS)
def test_descent_never_worse_than_init_and_within_tiny_tolerance(tiny_instance, Cls):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    problem = RoutingProblem(C)
    nn_cost = problem.evaluate(initial_tour(problem, "nearest_neighbour", None))
    est = Cls().fit(C)
    assert est.cost_ <= nn_cost + 1e-9, "a descent is never worse than its init"
    assert opt <= est.cost_ + 1e-9
    assert est.cost_ == pytest.approx(route_cost(C, est.route_), rel=1e-9)
    assert est.cost_ / opt - 1 <= TINY[Cls.__name__]
    _iterative_contract(est, {"converged", "max_iter"})


@pytest.mark.parametrize("Cls", DESCENTS)
def test_descent_fast_tier(fast_instance, Cls):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    est = Cls().fit(C, labels=fast_instance["labels"])
    assert opt <= est.cost_ + 1e-9
    assert est.cost_ / opt - 1 <= FAST[Cls.__name__], f"{Cls.__name__} on {fast_instance['name']}"
    assert int(est.route_[0]) == int(est.route_[-1]) == int(fast_instance["labels"][0])
    _iterative_contract(est, {"converged", "max_iter"})
    assert est.stop_reason_ == "converged"


@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_two_opt_converges_to_a_two_opt_optimal_tour(asymmetric):
    C = _euclid(14, seed=14, asymmetric=asymmetric)
    est = TwoOpt(n_candidates=None).fit(C)
    assert est.stop_reason_ == "converged"
    assert _no_reversal_improves(C, _index_tour(est).tolist())


@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_or_opt_converges_to_an_or_opt_optimal_tour(asymmetric):
    C = _euclid(14, seed=14, asymmetric=asymmetric)
    est = OrOpt(n_candidates=None).fit(C)
    assert est.stop_reason_ == "converged"
    if asymmetric:
        # the generic path enumerates every forward relocation next to a candidate: a true local optimum
        assert _no_or_opt_improves(C, _index_tour(est).tolist(), reverse_too=False)
    # the symmetric kernel prunes (Bentley): what is guaranteed is a fixed point of the pruned scan
    again = OrOpt(n_candidates=None, init=est.tour_).fit(C)
    assert (
        again.n_iter_ == 1 and again.stop_reason_ == "converged" and again.cost_ == pytest.approx(est.cost_)
    )


def test_local_search_alternation_reaches_a_joint_local_optimum(medium_euclidean):
    C = medium_euclidean["C"]
    est = LocalSearch(n_candidates=None).fit(C)
    tour = _index_tour(est).tolist()
    assert est.stop_reason_ == "converged"
    assert _no_reversal_improves(C, tour)  # 2-opt optimal for sure; Or-opt's pruned scan is a fixed point
    again = LocalSearch(n_candidates=None, init=est.tour_).fit(C)
    assert again.n_iter_ == 1 and again.stop_reason_ == "converged"
    # alternating both moves is at least as good as either descent alone from the same start
    assert (
        est.cost_ <= min(TwoOpt(n_candidates=None).fit(C).cost_, OrOpt(n_candidates=None).fit(C).cost_) + 1e-9
    )


def test_max_passes_bounds_the_iterations(fast_instance):
    C = fast_instance["C"]
    one = TwoOpt(max_passes=1).fit(C)
    assert one.n_iter_ == 1 and one.stop_reason_ == "max_iter"
    full = TwoOpt().fit(C)
    assert full.stop_reason_ == "converged" and full.cost_ <= one.cost_ + 1e-9
    assert full.history_[0] == pytest.approx(one.history_[0])  # the first pass is the same pass


@pytest.mark.parametrize("Cls", DESCENTS)
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_max_passes_run_is_an_exact_prefix_of_the_full_run(Cls, asymmetric):
    """Persistent buffers (SPEC §4.3) make ``max_passes=m`` the exact prefix of the converged run: the same
    ``history_`` bit for bit, ``"max_iter"`` before the convergence point and ``"converged"`` from it on."""
    C = _euclid(30 if asymmetric else 60, seed=6, asymmetric=asymmetric)
    full = Cls().fit(C)
    assert full.stop_reason_ == "converged" and 2 <= full.n_iter_ < 50
    for m in range(1, full.n_iter_ + 2):
        part = Cls(max_passes=m).fit(C)
        k = min(m, full.n_iter_)
        assert part.n_iter_ == k and np.array_equal(part.history_, full.history_[:k])
        assert part.stop_reason_ == ("converged" if m >= full.n_iter_ else "max_iter")
        assert part.cost_ == pytest.approx(full.history_[k - 1], rel=1e-9)


def test_first_improvement_false_scans_the_best_move(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    est = TwoOpt(first_improvement=False).fit(C)
    assert est.stop_reason_ == "converged" and est.cost_ / opt - 1 <= FAST["TwoOpt"]
    assert _no_reversal_improves(
        C, _index_tour(TwoOpt(first_improvement=False, n_candidates=None).fit(C)).tolist()
    )


def test_warm_start_from_a_local_optimum_converges_at_once(fast_instance):
    C, labels = fast_instance["C"], fast_instance["labels"]
    ls = LocalSearch().fit(C, labels=labels)
    for init in (ls.tour_, ls.route_):
        again = LocalSearch(init=init).fit(C, labels=labels)
        assert again.n_iter_ == 1 and again.stop_reason_ == "converged"
        assert again.cost_ == pytest.approx(ls.cost_) and np.array_equal(again.tour_, ls.tour_)


def test_full_neighbourhood_option(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    est = TwoOpt(n_candidates=None).fit(C)
    assert est.cost_ / opt - 1 <= FAST["TwoOpt"]
    assert est.problem_.neighbours(C.shape[0] - 1).shape == (C.shape[0], C.shape[0] - 1)


def test_moves_parameter_validation(small_euclidean):
    C = small_euclidean["C"]
    with pytest.raises(ValueError, match="'moves' parameter of LocalSearch"):
        LocalSearch(moves=("swap",)).fit(C)
    with pytest.raises(ValueError, match="'moves' parameter of LocalSearch"):
        LocalSearch(moves=["two_opt", "or_opt"]).fit(C)  # a list is not a tuple
    with pytest.raises(ValueError, match="'local_search' parameter of IteratedLocalSearch"):
        IteratedLocalSearch(local_search=("two_opt", "swap")).fit(C)
    with pytest.raises(ValueError, match="'init' parameter of TwoOpt"):
        TwoOpt(init="random").fit(C)  # the descents are deterministic
    with pytest.raises(ValueError, match="'max_segment' parameter of OrOpt"):
        OrOpt(max_segment=0).fit(C)
    single = LocalSearch(moves="or_opt").fit(C)  # a string is normalised to a 1-tuple
    assert single.cost_ == pytest.approx(OrOpt().fit(C).cost_)
    assert normalise_moves(None) == () and normalise_moves("two_opt") == ("two_opt",)


def test_move_order_is_honoured(fast_instance):
    C = fast_instance["C"]
    a = LocalSearch(moves=("two_opt", "or_opt"), max_passes=1).fit(C)
    b = LocalSearch(moves=("or_opt", "two_opt"), max_passes=1).fit(C)
    # one pass of each in a different order visits different tours; both are valid and improve the NN start
    problem = RoutingProblem(C)
    nn_cost = problem.evaluate(initial_tour(problem, "nearest_neighbour", None))
    assert a.cost_ < nn_cost and b.cost_ < nn_cost


# --------------------------------------------------------------------------- generic path (multi-trip, ATSP)
@pytest.mark.parametrize("Cls", [*DESCENTS, IteratedLocalSearch])
def test_multi_trip_alicante(alicante, Cls):
    d, kw, ref_kw = alicante["bunch"], alicante["kwargs"], alicante["ref_kwargs"]
    est = Cls(random_state=0) if Cls is IteratedLocalSearch else Cls()
    est.fit(d.cost, time_matrix=d.time, **kw)
    assert est.n_trips_ >= 2 and np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
    fixed = ref_kw["extra_cost"] * ref_kw["people"]
    expected = reference.problem_cost(d.cost, d.time, _index_tour(est), kw["max_time_work"], fixed, "greedy")
    assert est.cost_ == pytest.approx(expected, rel=1e-9)
    opt = alicante["optimum"]["greedy"]
    assert opt <= est.cost_ + 1e-9
    if Cls is IteratedLocalSearch:
        assert est.cost_ == pytest.approx(opt, rel=1e-9)
    else:  # the descents measure 3.8 % / 0 % / 3.6 %; the search is steered by the objective
        problem = est.problem_
        nn_cost = problem.evaluate(initial_tour(problem, "nearest_neighbour", None))
        assert est.cost_ <= nn_cost + 1e-9
        assert est.cost_ / opt - 1 <= 0.10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ils_reaches_the_alicante_optimum_under_both_split_rules(alicante, seed):
    d, kw = alicante["bunch"], alicante["kwargs"]
    for split in ("greedy", "optimal"):
        est = IteratedLocalSearch(random_state=seed).fit(d.cost, time_matrix=d.time, split=split, **kw)
        assert est.cost_ == pytest.approx(alicante["optimum"][split], rel=1e-9), f"seed {seed}, split {split}"
        assert np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
        _iterative_contract(est, {"max_iter", "patience", "time_limit"})


def test_descents_under_the_optimal_split(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    for Cls in DESCENTS:
        est = Cls().fit(d.cost, time_matrix=d.time, split="optimal", **kw)
        p_greedy = RoutingProblem(d.cost, time_matrix=d.time, **kw)
        assert est.cost_ <= p_greedy.evaluate(_index_tour(est)) + 1e-9  # optimal split never worse
        assert alicante["optimum"]["optimal"] <= est.cost_ + 1e-9
        _iterative_contract(est, {"converged", "max_iter"})


def test_asymmetric_descents_read_arcs_directionally():
    C = _euclid(10, seed=10, asymmetric=True)
    for Cls in DESCENTS:
        est = Cls().fit(C)
        tour = _index_tour(est)
        assert est.cost_ == pytest.approx(reference.tour_cost(C, tour), rel=1e-9)
        assert est.cost_ == pytest.approx(route_cost(C, est.route_), rel=1e-9)
        assert not est.problem_.symmetric


# --------------------------------------------------------------------------- IteratedLocalSearch
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ils_reaches_the_optimum_on_tiny(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    ils = IteratedLocalSearch(random_state=seed).fit(C)
    assert ils.cost_ == pytest.approx(opt, rel=1e-9)
    assert ils.cost_ == pytest.approx(route_cost(C, ils.route_), rel=1e-9)
    _iterative_contract(ils, {"max_iter", "patience", "time_limit"})


def test_ils_fast_tier(fast_instance):
    C, opt, labels = fast_instance["C"], fast_instance["optimum"], fast_instance["labels"]
    ils = IteratedLocalSearch(random_state=0).fit(C, labels=labels)
    assert opt <= ils.cost_ + 1e-9
    assert ils.cost_ / opt - 1 <= FAST["IteratedLocalSearch"]
    assert int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == int(labels[0])
    _iterative_contract(ils, {"max_iter", "patience", "time_limit"})


def test_ils_same_seed_is_bit_identical(medium_euclidean):
    C = medium_euclidean["C"]
    a, b = (IteratedLocalSearch(random_state=7).fit(C) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
    g = IteratedLocalSearch(random_state=np.random.default_rng(7)).fit(C)
    assert np.array_equal(g.tour_, a.tour_) and np.array_equal(g.history_, a.history_)


def test_ils_seed_changes_the_search_trajectory(medium_euclidean):
    C = medium_euclidean["C"]
    a = IteratedLocalSearch(random_state=0).fit(C)
    c = IteratedLocalSearch(random_state=1).fit(C)
    assert a.n_iter_ != c.n_iter_ or not np.array_equal(a.history_, c.history_)
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    IteratedLocalSearch(random_state=rng).fit(C)
    assert rng.bit_generator.state != before


def test_ils_stop_reasons(fast_instance):
    C = fast_instance["C"]
    by_patience = IteratedLocalSearch(random_state=0).fit(C)
    assert by_patience.stop_reason_ == "patience" and by_patience.n_iter_ <= 1000
    by_iter = IteratedLocalSearch(n_iter=5, patience=None, random_state=0).fit(C)
    assert by_iter.stop_reason_ == "max_iter" and by_iter.n_iter_ == 5
    by_time = IteratedLocalSearch(time_limit=1e-6, random_state=0).fit(C)
    assert by_time.stop_reason_ == "time_limit" and by_time.n_iter_ == 1
    assert by_time.history_[-1] == pytest.approx(by_time.cost_)


def test_ils_metropolis_acceptance(small_euclidean, fast_instance):
    C = small_euclidean["C"]
    target = IteratedLocalSearch(random_state=0).fit(C).cost_
    for temperature in (None, 5.0):
        met = IteratedLocalSearch(acceptance="metropolis", temperature=temperature, random_state=0).fit(C)
        _iterative_contract(met, {"max_iter", "patience", "time_limit"})
        assert met.cost_ == pytest.approx(target, rel=1e-9)  # n = 12: the optimum either way
    C, opt = fast_instance["C"], fast_instance["optimum"]
    met = IteratedLocalSearch(acceptance="metropolis", random_state=0).fit(C)
    assert opt <= met.cost_ + 1e-9 and met.cost_ / opt - 1 <= FAST["IteratedLocalSearch"]
    with pytest.raises(ValueError, match="'temperature' parameter"):
        IteratedLocalSearch(acceptance="metropolis", temperature=0.0).fit(C)


def _zero_ring(n):
    """0 = allowed transition, 1 = forbidden; the nearest-neighbour tour follows the zero ring at cost 0."""
    C = np.ones((n, n))
    np.fill_diagonal(C, 0.0)
    for i in range(n):
        C[i, (i + 1) % n] = C[(i + 1) % n, i] = 0.0
    return C


def test_ils_metropolis_rule_is_total_on_degenerate_costs(small_euclidean):
    """Legal inputs (finite is all RoutingProblem asks) used to crash the Metropolis rule: an init tour of
    cost 0 made the automatic temperature 0.0 (ZeroDivisionError), and a tiny explicit temperature
    overflowed ``exp`` on the reversed-orientation incumbent (delta ~ -1e-13, inside the improvement band)."""
    for C in (np.zeros((5, 5)), _zero_ring(6), _zero_ring(10)):
        for local_search in (("two_opt", "or_opt"), None):
            met = IteratedLocalSearch(
                acceptance="metropolis", local_search=local_search, n_iter=30, random_state=0
            ).fit(C)
            assert met.cost_ == 0.0
            _iterative_contract(met, {"max_iter", "patience", "time_limit"})
    C = small_euclidean["C"]
    for temperature in (1e-300, 1e-30, 1e300):
        met = IteratedLocalSearch(acceptance="metropolis", temperature=temperature, n_iter=50, random_state=0)
        met.fit(C)
        _iterative_contract(met, {"max_iter", "patience", "time_limit"})
        assert met.cost_ == pytest.approx(route_cost(C, met.route_), rel=1e-9)
    assert _default_temperature(0.0) == 1.0 and _default_temperature(200.0) == pytest.approx(1.0)


def test_ils_metropolis_default_temperature_is_positive_on_a_negative_matrix():
    """A negative temperature makes ``exp(-delta / T) > 1`` for every worse candidate (accept everything);
    the automatic temperature is 0.5 % of the *absolute* init cost, i.e. the run equals the explicit one."""
    N = _euclid(20, seed=20) - 60.0
    np.fill_diagonal(N, 0.0)
    problem = RoutingProblem(N)
    init_cost = float(problem.evaluate(initial_tour(problem, "nearest_neighbour", None)))
    assert init_cost < 0.0 and problem.symmetric
    temperature = _default_temperature(init_cost)
    assert temperature > 0.0 and temperature == pytest.approx(0.005 * abs(init_cost))
    automatic = IteratedLocalSearch(acceptance="metropolis", n_iter=30, random_state=0).fit(N)
    explicit = IteratedLocalSearch(
        acceptance="metropolis", temperature=0.005 * abs(init_cost), n_iter=30, random_state=0
    ).fit(N)
    assert np.array_equal(automatic.tour_, explicit.tour_)
    assert np.array_equal(automatic.history_, explicit.history_)
    assert automatic.cost_ == pytest.approx(route_cost(N, automatic.route_), rel=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_ils_kick_below_eight_nodes(n, asymmetric):
    C = _euclid(n, seed=n, asymmetric=asymmetric)
    opt = reference.brute_force(C)[0]
    for seed in (0, 1):
        ils = IteratedLocalSearch(random_state=seed).fit(C)
        assert ils.cost_ == pytest.approx(opt, rel=1e-9)
        assert int(ils.tour_[0]) == 0 and sorted(ils.tour_.tolist()) == list(range(n))
        _iterative_contract(ils, {"max_iter", "patience", "time_limit"})


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_ils_kick_below_eight_always_changes_an_edge_on_symmetric(n, monkeypatch):
    """Reversing the whole body ``(1, n - 1)`` of a symmetric tour is the same cycle driven backwards, so
    that kick changed nothing and the iteration only burnt patience (a third of them at n = 4). It is
    excluded on symmetric matrices and kept on asymmetric ones, where it is a genuine change."""
    import skroute.local_search._iterated as ils_module

    sym, asym = _reversal_pairs(n, True).tolist(), _reversal_pairs(n, False).tolist()
    assert len(asym) == (n - 1) * (n - 2) // 2 and [1, n - 1] in asym
    assert sym == [p for p in asym if p != [1, n - 1]]
    assert all(1 <= i < j <= n - 1 for i, j in asym)
    assert _reversal_pairs(3, True).tolist() == _reversal_pairs(3, False).tolist() == [[1, 2]]

    touched_sizes = []
    original = ils_module.changed_nodes

    def recording(before, after):
        touched = original(before, after)
        touched_sizes.append(int(touched.size))
        return touched

    monkeypatch.setattr(ils_module, "changed_nodes", recording)  # the ILS's own call, not the engine's
    C = _euclid(n, seed=n)
    ils = IteratedLocalSearch(n_iter=200, patience=None, random_state=0).fit(C)
    assert len(touched_sizes) == 200 and min(touched_sizes) > 0  # every kick changed at least one edge
    assert ils.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)


def test_ils_double_bridge_at_eight_nodes_and_more():
    for n in (8, 9, 11):
        C = _euclid(n, seed=n)
        opt = reference.brute_force(C)[0]
        ils = IteratedLocalSearch(random_state=0).fit(C)
        assert ils.cost_ == pytest.approx(opt, rel=1e-9)


def test_ils_options_perturbation_strength_and_no_local_search(small_euclidean):
    C = small_euclidean["C"]
    strong = IteratedLocalSearch(perturbation_strength=3, random_state=0).fit(C)
    walk = IteratedLocalSearch(local_search=None, n_iter=50, patience=None, random_state=0).fit(C)
    single = IteratedLocalSearch(local_search="two_opt", random_state=0).fit(C)
    for est in (strong, walk, single):
        _iterative_contract(est, {"max_iter", "patience", "time_limit"})
        assert est.cost_ == pytest.approx(route_cost(C, est.route_), rel=1e-9)
    assert walk.n_iter_ == 50 and walk.stop_reason_ == "max_iter"
    assert strong.cost_ <= LocalSearch().fit(C).cost_ + 1e-9


def test_ils_init_random_and_warm_start(small_euclidean):
    C = small_euclidean["C"]
    default = IteratedLocalSearch(random_state=0).fit(C)
    random_start = IteratedLocalSearch(init="random", random_state=0).fit(C)
    assert random_start.cost_ == pytest.approx(default.cost_, rel=1e-9)
    warm = IteratedLocalSearch(init=default.route_, random_state=0).fit(C)
    assert warm.cost_ == pytest.approx(default.cost_, rel=1e-9)
    with pytest.raises(ValueError, match="init"):
        IteratedLocalSearch(init=[0, 1, 2]).fit(C)


def test_verbose_logs_to_the_skroute_logger(small_euclidean, caplog):
    C = small_euclidean["C"]
    with caplog.at_level(logging.INFO, logger="skroute"):
        IteratedLocalSearch(n_iter=3, patience=None, verbose=2, random_state=0).fit(C)
        LocalSearch(verbose=2).fit(C)
    records = [r for r in caplog.records if r.name == "skroute"]
    assert sum("IteratedLocalSearch iteration" in r.getMessage() for r in records) == 3
    assert any("IteratedLocalSearch: stopped (max_iter)" in r.getMessage() for r in records)
    assert any(r.getMessage().startswith("LocalSearch: stopped (converged)") for r in records)


# --------------------------------------------------------------------------- the engine
def test_changed_nodes_marks_exactly_the_touched_endpoints():
    tour = np.arange(10, dtype=np.int64)
    reversed_ = reference.two_opt_apply(tour, 3, 6)
    assert changed_nodes(tour, np.asarray(reversed_)).tolist() == [2, 3, 6, 7]
    bridged = np.asarray(reference.double_bridge(tour, 2, 5, 8))
    assert changed_nodes(tour, bridged).tolist() == [1, 2, 4, 5, 7, 8]
    assert changed_nodes(tour, tour).size == 0


def test_descent_engine_bookkeeping_matches_full_evaluation(alicante, medium_euclidean):
    d, kw = alicante["bunch"], alicante["kwargs"]
    problem = RoutingProblem(d.cost, time_matrix=d.time, **kw)
    engine = Descent(problem, ("two_opt", "or_opt"))
    assert not engine.fast
    engine.load(initial_tour(problem, "nearest_neighbour", None))
    history, converged = engine.converge()
    assert converged and engine.cost == pytest.approx(problem.evaluate(engine.tour), rel=1e-12)
    assert history[-1] == engine.cost and np.all(np.diff(history) <= 0)
    fast = Descent(RoutingProblem(medium_euclidean["C"]), ("two_opt", "or_opt"), n_candidates=None)
    assert fast.fast
    fast.load(initial_tour(fast.problem, "nearest_neighbour", None))
    fast.converge()
    assert fast.cost == pytest.approx(fast.problem.evaluate(fast.tour), rel=1e-9)
    assert all(bool(np.all(b == 1)) for b in fast.bits.values())  # a converged descent has every bit set
    with pytest.raises(ValueError, match="unknown move"):
        Descent(fast.problem, ("swap",))


@pytest.mark.parametrize("moves", sorted(MOVE_TUPLES), ids="+".join)
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_descent_accounting_is_exact_step_by_step(moves, asymmetric):
    """SPEC §4.3: ``history_[k] == history_[k-1] + sum(gains)`` bit for bit — the kernels' returned gains are
    the ledger, and only a converged engine has every don't-look bit set."""
    C = _euclid(30 if asymmetric else 60, seed=5, asymmetric=asymmetric)
    problem = RoutingProblem(C)
    for n_candidates in (10, None):
        engine = Descent(problem, moves, n_candidates=n_candidates)
        engine.load(initial_tour(problem, "nearest_neighbour", None))
        assert engine.fast is not asymmetric
        ledger, done, steps = engine.cost, False, 0
        while not done:
            gains, done = engine.step()
            assert len(gains) == len(moves) and all(g <= 0.0 for g in gains)
            for g in gains:
                ledger += g
            assert ledger == engine.cost  # the same additions in the same order: exact
            steps += 1
        assert steps >= 2 and engine.cost == pytest.approx(problem.evaluate(engine.tour), rel=1e-9)
        assert all(bool(np.all(b == 1)) for b in engine.bits.values())


@settings(derandomize=True, deadline=None, max_examples=40)
@given(n=st.integers(4, 9), seed=st.integers(0, 10_000), asymmetric=st.booleans())
def test_descents_are_local_optima_on_random_instances(n, seed, asymmetric):
    C = _euclid(n, seed=seed, asymmetric=asymmetric)
    problem = RoutingProblem(C)
    nn_cost = problem.evaluate(initial_tour(problem, "nearest_neighbour", None))
    two = TwoOpt(n_candidates=None).fit(C)
    assert two.cost_ <= nn_cost + 1e-9 and np.all(np.diff(two.history_) <= 1e-9)
    assert _no_reversal_improves(C, _index_tour(two).tolist())
    ls = LocalSearch(n_candidates=None).fit(C)
    tour = _index_tour(ls).tolist()
    assert _no_reversal_improves(C, tour)
    if asymmetric:  # the generic path enumerates every forward Or-opt relocation as well
        assert _no_or_opt_improves(C, tour, reverse_too=False)
    assert ls.cost_ == pytest.approx(reference.tour_cost(C, tour), rel=1e-9)
    again = LocalSearch(n_candidates=None, init=ls.tour_).fit(C)
    assert again.n_iter_ == 1 and again.cost_ == pytest.approx(ls.cost_, rel=1e-9)
