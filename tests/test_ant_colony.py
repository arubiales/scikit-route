"""Acceptance tests of ``skroute.metaheuristics.AntColony`` (SPEC §4.4), beyond ``check_router``."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import reference
from conftest import _euclid

from skroute import RoutingProblem
from skroute.metaheuristics import AntColony, _aco
from skroute.metrics import route_cost
from skroute.utils import initial_tour


def _assert_consistent(est, C, **kw):
    assert est.cost_ == pytest.approx(route_cost(C, est.route_, **kw), rel=1e-9)
    assert est.history_[-1] == pytest.approx(est.cost_)
    assert est.n_iter_ == len(est.history_)
    assert np.all(np.diff(est.history_) <= 1e-9 * max(1.0, float(np.abs(est.history_).max())))
    assert est.stop_reason_ in {"max_iter", "patience", "time_limit"}


# --------------------------------------------------------------------------- optimality (tiny, alicante)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reaches_optimum_on_tiny(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    aco = AntColony(random_state=seed).fit(C)
    assert aco.cost_ == pytest.approx(opt, rel=1e-9)
    _assert_consistent(aco, C)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reaches_optimum_on_tiny_without_polish(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    aco = AntColony(random_state=seed, local_search=None).fit(C)
    assert aco.cost_ == pytest.approx(opt, rel=1e-9)
    _assert_consistent(aco, C)


@pytest.mark.parametrize("split", ["greedy", "optimal"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_trip_matches_brute_force(alicante, seed, split):
    d, kw = alicante["bunch"], alicante["kwargs"]  # kw carries labels= and the LABEL depot
    aco = AntColony(random_state=seed).fit(d.cost, time_matrix=d.time, split=split, **kw)
    assert np.all(aco.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert aco.cost_ == pytest.approx(alicante["optimum"][split], rel=1e-9)
    fixed = kw["extra_cost"] * kw["people"]
    assert aco.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, aco.problem_.to_index_tour(aco.tour_), kw["max_time_work"], fixed, split
        )
    )
    assert aco.n_trips_ == len(aco.trips_) >= 2


# --------------------------------------------------------------------------- fast tier
def test_fast_tier_gap(fast_instance):
    C, opt, labels = fast_instance["C"], fast_instance["optimum"], fast_instance["labels"]
    aco = AntColony(random_state=0).fit(C, labels=labels)
    assert opt <= aco.cost_ + 1e-9
    assert aco.cost_ / opt - 1 <= 0.08, f"gap {aco.cost_ / opt - 1:.4f} on {fast_instance['name']}"
    assert int(aco.route_[0]) == int(aco.route_[-1]) == int(labels[0])


# --------------------------------------------------------------------------- reproducibility
def test_same_seed_is_bit_identical(small_euclidean):
    C = small_euclidean["C"]
    a, b = (AntColony(random_state=7).fit(C) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
    assert np.array_equal(a.pheromone_, b.pheromone_) and a.n_iter_ == b.n_iter_


def test_seed_is_used(medium_euclidean):
    # At n = 12 every polished ant reaches the optimum from iteration 0 with any seed, so history_, n_iter_
    # and even pheromone_ are seed-independent there (the tour orientation is a coin flip); at n = 40 the
    # iteration of the first hit and the early trail differ between seeds.
    a40, c40 = (AntColony(random_state=s).fit(medium_euclidean["C"]) for s in (0, 1))
    assert a40.n_iter_ != c40.n_iter_ or not np.array_equal(a40.history_, c40.history_)
    assert not np.array_equal(a40.pheromone_, c40.pheromone_)


def test_generator_is_advanced_and_equals_int_seed(small_euclidean):
    C = small_euclidean["C"]
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    g = AntColony(random_state=rng).fit(C)
    assert rng.bit_generator.state != before
    assert np.array_equal(g.tour_, AntColony(random_state=3).fit(C).tour_)


# --------------------------------------------------------------------------- edge sizes and asymmetric path
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("local_search", [("two_opt",), None], ids=["polish", "no-polish"])
def test_smallest_sizes_reach_the_optimum(n, asym, local_search):
    C, _ = _euclid(n, seed=n, asymmetric=asym)
    aco = AntColony(random_state=0, local_search=local_search).fit(C)
    assert aco.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    assert sorted(aco.tour_.tolist()) == list(range(n)) and aco.tour_[0] == 0
    _assert_consistent(aco, C)


def _two_opt_local_optimum(C, tour, cost):
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if reference.tour_cost(C, reference.two_opt_apply(tour, i, j)) < cost - 1e-9 * max(1.0, cost):
                return False
    return True


def test_generic_path_on_asymmetric_instance():
    C, _ = _euclid(12, seed=12, asymmetric=True)
    aco = AntColony(random_state=0, n_candidates=None).fit(C)
    assert not aco.problem_.symmetric
    tour = aco.problem_.to_index_tour(aco.tour_)
    assert _two_opt_local_optimum(C, tour.tolist(), aco.cost_), (
        "the asymmetric polish must reach a 2-opt optimum"
    )
    assert not np.allclose(aco.pheromone_, aco.pheromone_.T), "the trail is directional on an ATSP"
    _assert_consistent(aco, C)


def test_generic_path_under_budget_with_or_opt(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    aco = AntColony(random_state=1, local_search=("two_opt", "or_opt")).fit(d.cost, time_matrix=d.time, **kw)
    assert aco.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert np.all(aco.trip_times_ <= kw["max_time_work"] + 1e-9)


# --------------------------------------------------------------------------- pheromone and parameters
def test_pheromone_is_bounded_and_symmetric(small_euclidean):
    C = small_euclidean["C"]
    aco = AntColony(random_state=0, rho=0.1).fit(C)
    tau, n = aco.pheromone_, C.shape[0]
    assert tau.shape == (n, n) and np.all(np.diag(tau) == 0.0)
    tau_max = 1.0 / (0.1 * aco.cost_)
    off = tau[~np.eye(n, dtype=bool)]
    assert np.all(off <= tau_max + 1e-12) and np.all(off >= tau_max / (2 * n) - 1e-12)
    assert np.allclose(tau, tau.T)
    best = aco.problem_.to_index_tour(aco.tour_)
    on_tour = tau[best, np.roll(best, -1)]
    assert on_tour.mean() > off.mean(), "the best tour's arcs must carry more pheromone than average"


def test_explicit_n_ants_and_full_candidate_lists(small_euclidean):
    C = small_euclidean["C"]
    aco = AntColony(n_ants=3, n_candidates=None, n_iter=20, patience=None, random_state=0).fit(C)
    assert aco.n_iter_ == 20 and aco.stop_reason_ == "max_iter"
    _assert_consistent(aco, C)


def test_local_search_string_is_normalised(small_euclidean):
    C = small_euclidean["C"]
    a = AntColony(random_state=0, local_search="two_opt").fit(C)
    b = AntColony(random_state=0, local_search=("two_opt",)).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)


@pytest.mark.parametrize("bad", ["both", ("two_opt", "swap"), (), ("three_opt",)])
def test_invalid_local_search_raises(small_euclidean, bad):
    with pytest.raises(ValueError, match="'local_search' parameter of AntColony"):
        AntColony(local_search=bad).fit(small_euclidean["C"])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"rho": 1.0}, "The 'rho' parameter of AntColony must be a float in the range"),
        ({"alpha": -1.0}, "The 'alpha' parameter of AntColony must be a float in the range"),
        ({"n_ants": 0}, "The 'n_ants' parameter of AntColony must be an int in the range"),
        ({"n_candidates": 0}, "The 'n_candidates' parameter of AntColony must be an int in the range"),
    ],
)
def test_parameter_constraints(small_euclidean, params, match):
    with pytest.raises(ValueError, match=match):
        AntColony(**params).fit(small_euclidean["C"])


# --------------------------------------------------------------------------- stop rules, history, logging
def test_stop_reasons(small_euclidean):
    C = small_euclidean["C"]
    aco = AntColony(n_iter=4, patience=None, random_state=0).fit(C)
    assert aco.n_iter_ == 4 and aco.stop_reason_ == "max_iter"
    aco = AntColony(patience=3, random_state=0).fit(C)
    assert aco.stop_reason_ == "patience" and aco.history_[-1] == pytest.approx(aco.history_[-4])
    aco = AntColony(time_limit=1e-6, random_state=0).fit(C)
    assert aco.stop_reason_ == "time_limit" and aco.n_iter_ == 1
    _assert_consistent(aco, C)


def test_history_is_best_so_far_without_polish(medium_euclidean):
    C = medium_euclidean["C"]
    aco = AntColony(local_search=None, n_iter=30, patience=None, random_state=0).fit(C)
    assert np.all(np.diff(aco.history_) <= 1e-12) and aco.history_[-1] == pytest.approx(aco.cost_)
    assert aco.history_[0] > aco.history_[-1], "the colony must learn on a 40-node instance"


def test_verbose_logs_to_skroute_logger(small_euclidean, caplog):
    with caplog.at_level(logging.INFO, logger="skroute"):
        AntColony(n_iter=20, patience=None, random_state=0, verbose=1).fit(small_euclidean["C"])
    records = [r for r in caplog.records if r.name == "skroute"]
    assert len(records) == 11  # iterations 0, 2, ..., 18 (every max(1, 20 // 10) = 2) plus the summary line
    with caplog.at_level(logging.INFO, logger="skroute"):
        caplog.clear()
        AntColony(n_iter=20, patience=None, random_state=0, verbose=2).fit(small_euclidean["C"])
    assert len([r for r in caplog.records if r.name == "skroute"]) == 21


def test_caller_data_untouched_and_labels(small_euclidean):
    C = small_euclidean["C"]
    before = C.copy()
    names = [f"c{i}" for i in range(C.shape[0])]
    aco = AntColony(random_state=0).fit(C, labels=names, depot="c4")
    assert aco.depot_ == "c4" and aco.tour_[0] == "c4" and sorted(aco.tour_.tolist()) == sorted(names)
    assert np.array_equal(C, before)


def test_coincident_points_are_handled():
    C, _ = _euclid(9, seed=9)
    C[3, :] = C[4, :]
    C[:, 3] = C[:, 4]
    C[3, 4] = C[4, 3] = 0.0  # nodes 3 and 4 coincide: a zero off-diagonal distance
    aco = AntColony(random_state=0).fit(C)
    assert np.isfinite(aco.cost_) and np.all(np.isfinite(aco.pheromone_))
    assert aco.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    tour = aco.problem_.to_index_tour(aco.tour_).tolist()
    assert abs(tour.index(3) - tour.index(4)) == 1, "coincident nodes must be visited consecutively"


def _zero_optimum_instances():
    """Legal inputs (square, finite) whose optimum is a zero-cost tour.

    Every ``1 / L`` of the trail (initial value, deposit, bounds) is at risk on them.
    """
    n, cycle = 5, [0, 2, 4, 1, 3]
    decoy = np.full((n, n), 5.0)  # the nearest-neighbour tour costs 10 (0 -> 1 is also free), the cycle 0
    np.fill_diagonal(decoy, 0.0)
    for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
        decoy[a, b] = 0.0
    decoy[0, 1] = 0.0
    return {
        "all-zero": np.zeros((5, 5)),
        "all-zero-n3": np.zeros((3, 3)),
        "zero-cycle-asym": decoy,
        "zero-cycle-sym": np.minimum(decoy, decoy.T),
        "zero-cycle-4": np.array([[0, 0, 5, 5], [5, 0, 0, 5], [5, 5, 0, 0], [0, 5, 5, 0]], dtype=float),
    }


@pytest.mark.parametrize("name", list(_zero_optimum_instances()))
@pytest.mark.parametrize("local_search", [("two_opt",), None], ids=["polish", "no-polish"])
def test_zero_cost_tours_are_returned_with_a_finite_trail(name, local_search):
    # regression: 1 / (rho * L_NN), 1 / L_deposit and 1 / (rho * L_best) raised ZeroDivisionError as soon
    # as a tour cost exactly 0 (all nodes coincident, or a zero-cost Hamiltonian cycle found by an ant after
    # a positive nearest-neighbour tour)
    C = _zero_optimum_instances()[name]
    aco = AntColony(random_state=0, local_search=local_search).fit(C)
    assert aco.cost_ == 0.0 == reference.brute_force(C)[0]
    assert np.all(np.isfinite(aco.pheromone_)) and np.all(aco.pheromone_ >= 0.0)
    assert np.all(aco.history_ == 0.0) and aco.stop_reason_ == "patience"
    _assert_consistent(aco, C)


def test_deposit_and_trail_bounds_use_the_problem_cost_under_a_budget(alicante):
    # One iteration, modelled exactly: tau0 = 1 / (rho * L_NN) evaporates once, the iteration-best ant
    # deposits 1 / cost on its arcs (both directions: alicante is symmetric) and the trail is clipped to
    # [tau_max / (2n), tau_max] with tau_max = 1 / (rho * L_best). Every L is the PROBLEM cost (fixed charges
    # and depot legs of the split included), not the plain giant-tour cost: the deposit steers the colony.
    d, kw = alicante["bunch"], alicante["kwargs"]
    rho = 0.05
    aco = AntColony(n_iter=1, patience=None, rho=rho, random_state=0).fit(d.cost, time_matrix=d.time, **kw)
    p, n = aco.problem_, d.cost.shape[0]
    assert p.symmetric and aco.n_trips_ >= 2
    best = p.to_index_tour(aco.tour_)
    plain = reference.tour_cost(d.cost, best)
    assert aco.cost_ > plain + 1.0, "under the budget the objective differs from the plain tour cost"
    l_nn = p.evaluate(initial_tour(p, "nearest_neighbour", None))
    expected = np.full((n, n), (1.0 - rho) / (rho * l_nn))
    heads, tails = best, np.roll(best, -1)
    expected[heads, tails] += 1.0 / aco.cost_
    expected[tails, heads] += 1.0 / aco.cost_
    tau_max = 1.0 / (rho * aco.cost_)
    expected = np.clip(expected, tau_max / (2.0 * n), tau_max)
    np.fill_diagonal(expected, 0.0)
    assert np.allclose(aco.pheromone_, expected, rtol=1e-12, atol=0.0)
    wrong = np.full((n, n), (1.0 - rho) / (rho * l_nn))
    wrong[heads, tails] += 1.0 / plain
    wrong[tails, heads] += 1.0 / plain
    np.fill_diagonal(wrong, 0.0)
    assert not np.allclose(aco.pheromone_, wrong, rtol=1e-6, atol=0.0), (
        "a 1 / plain-cost deposit is distinguishable"
    )


# --------------------------------------------------------------------------- kernels
def test_construct_tours_are_permutations_from_the_depot():
    C, _ = _euclid(15, seed=15)
    p = RoutingProblem(C, depot=6)
    n, n_ants, k = p.n, 7, 4
    choice = np.power(np.maximum(C, 1e-9), -2.0)
    np.fill_diagonal(choice, 0.0)
    u = np.random.default_rng(0).random((n_ants, n - 1))
    tours = np.empty((n_ants, n), dtype=np.int64)
    _aco.construct_tours(choice, p.neighbours(k), 6, u, tours, np.zeros(n, np.uint8), np.empty(k))
    for row in tours:
        assert row[0] == 6 and sorted(row.tolist()) == list(range(n))
    # a short candidate list forces the fallback to "all unvisited nodes" on the last steps
    tours2 = np.empty((n_ants, n), dtype=np.int64)
    _aco.construct_tours(choice, p.neighbours(1), 6, u, tours2, np.zeros(n, np.uint8), np.empty(1))
    for row in tours2:
        assert row[0] == 6 and sorted(row.tolist()) == list(range(n))
    # u == 0 everywhere picks the most attractive unvisited candidate at every step: the NN tour
    tours3 = np.empty((1, n), dtype=np.int64)
    _aco.construct_tours(
        choice, p.neighbours(n - 1), 6, np.zeros((1, n - 1)), tours3, np.zeros(n, np.uint8), np.empty(n - 1)
    )
    nn = np.empty(n, dtype=np.int64)
    from skroute._core import _routing as core

    core.nearest_neighbour_tour(C, 6, nn)
    assert tours3[0].tolist() == nn.tolist()


def test_polish_and_evaluate_matches_reference_costs():
    C, _ = _euclid(10, seed=10, asymmetric=True)
    p = RoutingProblem(C)
    n = p.n
    rng = np.random.default_rng(1)
    tours = np.stack([np.concatenate(([0], rng.permutation(np.arange(1, n)))) for _ in range(4)]).astype(
        np.int64
    )
    costs = np.empty(4)
    _aco.polish_and_evaluate(
        p.cost, p.time_or_cost, tours, np.inf, 0.0, 0, 0, 0, p.neighbours(n - 1),
        np.empty(n, np.int64), np.empty(n, np.int64), np.zeros(n, np.uint8), np.empty(n, np.int64),
        np.empty(n), np.empty(n, np.int64), costs,
    )  # fmt: skip
    assert costs.tolist() == pytest.approx([reference.tour_cost(C, row) for row in tours.tolist()])
    polished = tours.copy()
    _aco.polish_and_evaluate(
        p.cost, p.time_or_cost, polished, np.inf, 0.0, 0, 2, 1, p.neighbours(n - 1),
        np.empty(n, np.int64), np.empty(n, np.int64), np.zeros(n, np.uint8), np.empty(n, np.int64),
        np.empty(n), np.empty(n, np.int64), costs,
    )  # fmt: skip
    for row, c in zip(polished.tolist(), costs.tolist(), strict=True):
        assert row[0] == 0 and sorted(row) == list(range(n))
        assert c == pytest.approx(reference.tour_cost(C, row))
        assert _two_opt_local_optimum(C, row, c)


# --------------------------------------------------------------------------- D31: the trails a viewer draws
def test_iteration_events_carry_the_strongest_trails_scaled_to_the_bound(small_euclidean):
    C = small_euclidean["C"]
    labels = list("abcdefghijkl")
    events = []
    aco = AntColony(random_state=0, rho=0.1, n_iter=10, patience=None).fit(
        C, labels=labels, callback=events.append
    )
    iters = [e for e in events if e.stage == "iteration"]
    assert "edges" not in events[0].extra and "edges" not in events[-1].extra and len(iters) == 10
    index = {lab: i for i, lab in enumerate(labels)}
    for e in iters:
        edges, weights = e.extra["edges"], e.extra["edge_weights"]
        assert len(edges) == len(weights) == 36 == len(set(edges))  # min(3 n, n (n - 1) / 2) = min(36, 66)
        assert all(type(p) is tuple and index[p[0]] < index[p[1]] for p in edges)  # the upper triangle
        assert all(type(w) is float and 0.0 < w <= 1.0 for w in weights)
        assert weights == sorted(weights, reverse=True)  # strongest first
    last = iters[-1]
    tau, tau_max = aco.pheromone_, 1.0 / (0.1 * aco.history_[-1])  # the trail and bound of the last iteration
    reported = [tau[index[a], index[b]] for a, b in last.extra["edges"]]
    assert last.extra["edge_weights"] == pytest.approx([t / tau_max for t in reported], rel=1e-12)
    iu, ju = np.triu_indices(12, 1)
    assert sorted(reported, reverse=True) == pytest.approx(sorted(tau[iu, ju], reverse=True)[:36])
    assert (
        max(last.extra["edge_weights"]) <= 1.0
    )  # the bound itself is 1; a trail only reaches it asymptotically


def test_trails_are_arcs_on_an_asymmetric_matrix_and_capped_by_the_pool():
    C, _ = _euclid(7, seed=7, asymmetric=True)
    events = []
    AntColony(random_state=0, n_iter=3, patience=None).fit(C, callback=events.append)
    for e in events[1:-1]:
        edges = e.extra["edges"]
        assert len(edges) == 21 == 3 * 7 and len(set(edges)) == 21 and all(a != b for a, b in edges)
        assert all(type(a) is int and type(b) is int for a, b in edges)  # labels, as Python scalars
    events = []
    AntColony(random_state=0, n_iter=2).fit(_euclid(3, seed=3)[0], callback=events.append)
    assert all(len(e.extra["edges"]) == 3 for e in events[1:-1])  # n (n - 1) / 2 = 3 < 3 n: the whole pool


def test_strongest_trails_are_ordered_by_strength_then_position():
    from skroute.metaheuristics._ant_colony import _strongest_trails

    tau = np.array([[0.0, 2.0, 2.0, 8.0], [2.0, 0.0, 4.0, 2.0], [2.0, 4.0, 0.0, 2.0], [8.0, 2.0, 2.0, 0.0]])
    ii, jj = np.triu_indices(4, 1)
    labels = np.array(["a", "b", "c", "d"])
    edges, weights = _strongest_trails(tau, ii, jj, 4, 8.0, labels)
    assert edges == [("a", "d"), ("b", "c"), ("a", "b"), ("a", "c")] and weights == [1.0, 0.5, 0.25, 0.25]
    edges, weights = _strongest_trails(tau, ii, jj, 6, 8.0, labels)  # k == pool: everything, ties by position
    assert edges == [("a", "d"), ("b", "c"), ("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    assert _strongest_trails(tau * 2.0, ii, jj, 1, 8.0, labels) == ([("a", "d")], [1.0])  # never above 1
