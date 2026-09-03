"""Acceptance tests of ``skroute.metaheuristics.Genetic`` (SPEC §4.4), beyond ``check_router``."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import reference
from conftest import _euclid
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import RoutingProblem
from skroute.metaheuristics import Genetic, _ga
from skroute.metrics import route_cost

MEMETIC = {"local_search": ("two_opt",)}


def _assert_consistent(est, C, **kw):
    assert est.cost_ == pytest.approx(route_cost(C, est.route_, **kw), rel=1e-9)
    assert est.history_[-1] == pytest.approx(est.cost_)
    assert est.n_iter_ == len(est.history_)
    assert np.all(np.diff(est.history_) <= 1e-9 * max(1.0, float(np.abs(est.history_).max())))
    assert est.stop_reason_ in {"max_iter", "patience", "time_limit"}


# --------------------------------------------------------------------------- optimality (tiny, alicante)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("params", [{}, MEMETIC], ids=["plain", "memetic"])
def test_reaches_optimum_on_tiny(tiny_instance, seed, params):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    ga = Genetic(random_state=seed, **params).fit(C)
    assert ga.cost_ == pytest.approx(opt, rel=1e-9)
    _assert_consistent(ga, C)


@pytest.mark.parametrize("split", ["greedy", "optimal"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("params", [{}, MEMETIC], ids=["plain", "memetic"])
def test_multi_trip_matches_brute_force(alicante, seed, split, params):
    d, kw = alicante["bunch"], alicante["kwargs"]  # kw carries labels= and the LABEL depot
    ga = Genetic(random_state=seed, **params).fit(d.cost, time_matrix=d.time, split=split, **kw)
    assert np.all(ga.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert ga.cost_ == pytest.approx(alicante["optimum"][split], rel=1e-9)
    fixed = kw["extra_cost"] * kw["people"]
    assert ga.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, ga.problem_.to_index_tour(ga.tour_), kw["max_time_work"], fixed, split
        )
    )
    assert ga.n_trips_ == len(ga.trips_) >= 2


# --------------------------------------------------------------------------- fast tier
def test_fast_tier_gaps(fast_instance):
    C, opt, labels = fast_instance["C"], fast_instance["optimum"], fast_instance["labels"]
    plain = Genetic(random_state=0).fit(C, labels=labels)
    assert opt <= plain.cost_ + 1e-9
    assert plain.cost_ / opt - 1 <= 0.15, (
        f"plain GA gap {plain.cost_ / opt - 1:.4f} on {fast_instance['name']}"
    )
    memetic = Genetic(random_state=0, **MEMETIC).fit(C, labels=labels)
    assert memetic.cost_ / opt - 1 <= 0.05, f"memetic gap {memetic.cost_ / opt - 1:.4f}"
    assert int(memetic.route_[0]) == int(memetic.route_[-1]) == int(labels[0])


# --------------------------------------------------------------------------- reproducibility
def test_same_seed_is_bit_identical(small_euclidean):
    C = small_euclidean["C"]
    a, b = (Genetic(random_state=7).fit(C) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
    assert a.n_iter_ == b.n_iter_ and a.stop_reason_ == b.stop_reason_ and a.n_duplicates_ == b.n_duplicates_


def test_seeds_zero_and_one_differ(small_euclidean):
    C = small_euclidean["C"]
    a, c = Genetic(random_state=0).fit(C), Genetic(random_state=1).fit(C)
    assert a.n_iter_ != c.n_iter_ or not np.array_equal(a.history_, c.history_)


def test_generator_is_advanced_and_equals_int_seed(small_euclidean):
    C = small_euclidean["C"]
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    g = Genetic(random_state=rng).fit(C)
    assert rng.bit_generator.state != before
    assert np.array_equal(g.tour_, Genetic(random_state=3).fit(C).tour_)


# --------------------------------------------------------------------------- edge sizes and asymmetric path
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("params", [{}, MEMETIC], ids=["plain", "memetic"])
def test_smallest_sizes_reach_the_optimum(n, asym, params):
    C, _ = _euclid(n, seed=n, asymmetric=asym)
    ga = Genetic(random_state=0, **params).fit(C)
    assert ga.cost_ == pytest.approx(reference.brute_force(C)[0], rel=1e-9)
    assert sorted(ga.tour_.tolist()) == list(range(n)) and ga.tour_[0] == 0
    _assert_consistent(ga, C)


def _two_opt_local_optimum(C, tour, cost):
    """No segment reversal of the depot-anchored tour improves ``cost`` (full evaluation, ATSP-exact)."""
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if reference.tour_cost(C, reference.two_opt_apply(tour, i, j)) < cost - 1e-9 * max(1.0, cost):
                return False
    return True


def test_memetic_generic_path_on_asymmetric_instance():
    C, _ = _euclid(12, seed=12, asymmetric=True)
    ga = Genetic(random_state=0, **MEMETIC).fit(C)
    tour = ga.problem_.to_index_tour(ga.tour_)
    assert not ga.problem_.symmetric
    assert _two_opt_local_optimum(C, tour.tolist(), ga.cost_), (
        "the asymmetric polish must reach a 2-opt optimum"
    )
    _assert_consistent(ga, C)


def test_memetic_generic_path_under_budget(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    ga = Genetic(random_state=1, local_search=("two_opt", "or_opt")).fit(d.cost, time_matrix=d.time, **kw)
    assert ga.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert np.all(ga.trip_times_ <= kw["max_time_work"] + 1e-9)


# --------------------------------------------------------------------------- parameters
def test_local_search_string_is_normalised(small_euclidean):
    C = small_euclidean["C"]
    a = Genetic(random_state=0, local_search="two_opt").fit(C)
    b = Genetic(random_state=0, local_search=("two_opt",)).fit(C)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)


@pytest.mark.parametrize("bad", ["both", ("two_opt", "swap"), (), ("three_opt",)])
def test_invalid_local_search_raises(small_euclidean, bad):
    with pytest.raises(ValueError, match="'local_search' parameter of Genetic"):
        Genetic(local_search=bad).fit(small_euclidean["C"])


def test_n_elite_must_be_below_pop_size(small_euclidean):
    with pytest.raises(ValueError, match="'n_elite' parameter of Genetic must be smaller than pop_size"):
        Genetic(pop_size=4, n_elite=4).fit(small_euclidean["C"])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"crossover": "cx"}, "The 'crossover' parameter of Genetic must be a str among"),
        ({"mutation": "scramble"}, "The 'mutation' parameter of Genetic must be a str among"),
        ({"p_crossover": 1.5}, "The 'p_crossover' parameter of Genetic must be a float in the range"),
        ({"pop_size": 1}, "The 'pop_size' parameter of Genetic must be an int in the range"),
        ({"patience": 0}, "The 'patience' parameter of Genetic must be an int in the range"),
    ],
)
def test_parameter_constraints(small_euclidean, params, match):
    with pytest.raises(ValueError, match=match):
        Genetic(**params).fit(small_euclidean["C"])


@pytest.mark.parametrize("crossover", ["ox", "pmx"])
@pytest.mark.parametrize("mutation", ["inversion", "swap", "insertion"])
def test_every_operator_combination_runs(small_euclidean, crossover, mutation):
    C = small_euclidean["C"]
    ga = Genetic(crossover=crossover, mutation=mutation, n_generations=30, patience=None, random_state=0).fit(
        C
    )
    assert ga.n_iter_ == 30 and ga.stop_reason_ == "max_iter"
    _assert_consistent(ga, C)


# --------------------------------------------------------------------------- stop rules, history, logging
def test_stop_reasons(small_euclidean):
    C = small_euclidean["C"]
    ga = Genetic(n_generations=7, patience=None, random_state=0).fit(C)
    assert ga.n_iter_ == 7 and ga.stop_reason_ == "max_iter"
    ga = Genetic(patience=5, random_state=0).fit(C)
    assert ga.stop_reason_ == "patience" and ga.n_iter_ <= 500
    # the best-so-far did not improve during the last `patience` generations
    assert ga.history_[-1] == pytest.approx(ga.history_[-6])
    ga = Genetic(time_limit=1e-6, random_state=0).fit(C)
    assert ga.stop_reason_ == "time_limit" and ga.n_iter_ == 1
    _assert_consistent(ga, C)


def test_history_is_best_so_far_without_elitism(small_euclidean):
    C = small_euclidean["C"]
    ga = Genetic(n_elite=0, n_generations=40, patience=None, random_state=0).fit(C)
    assert np.all(np.diff(ga.history_) <= 1e-12)
    assert ga.history_[-1] == pytest.approx(ga.cost_)


def test_verbose_logs_to_skroute_logger(small_euclidean, caplog):
    with caplog.at_level(logging.INFO, logger="skroute"):
        Genetic(n_generations=30, patience=None, random_state=0, verbose=1).fit(small_euclidean["C"])
    records = [r for r in caplog.records if r.name == "skroute"]
    assert len(records) == 11  # generations 0, 3, ..., 27 (every max(1, 30 // 10) = 3) plus the summary line
    with caplog.at_level(logging.INFO, logger="skroute"):
        caplog.clear()
        Genetic(n_generations=30, patience=None, random_state=0, verbose=2).fit(small_euclidean["C"])
    assert len([r for r in caplog.records if r.name == "skroute"]) == 31


def test_warm_start_and_caller_data_untouched(small_euclidean):
    C = small_euclidean["C"]
    before = C.copy()
    first = Genetic(random_state=0).fit(C)
    tour_before = first.tour_.copy()
    # the init individual is kept by elitism: without crossover or mutation the result cannot be worse
    ga = Genetic(init=first.tour_, n_generations=3, p_crossover=0.0, p_mutation=0.0, random_state=1).fit(C)
    assert ga.cost_ <= first.cost_ + 1e-9
    route_start = Genetic(init=first.route_, n_generations=1, patience=None, random_state=1).fit(C)
    assert route_start.cost_ <= first.cost_ + 1e-9
    assert np.array_equal(C, before) and np.array_equal(first.tour_, tour_before)


def test_labels_and_depot_by_label(small_euclidean):
    C = small_euclidean["C"]
    names = [f"c{i}" for i in range(C.shape[0])]
    ga = Genetic(random_state=0).fit(C, labels=names, depot="c5")
    assert ga.depot_ == "c5" and ga.tour_[0] == "c5" and sorted(ga.tour_.tolist()) == sorted(names)


# --------------------------------------------------------------------------- kernels
def test_evaluate_population_matches_reference(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    rng = np.random.default_rng(0)
    n = d.cost.shape[0]
    pop = np.stack([rng.permutation(np.arange(1, n)) for _ in range(6)]).astype(np.int64)
    fixed = kw["extra_cost"] * kw["people"]
    for split_name, split_code in (("greedy", 0), ("optimal", 1)):
        out = np.empty(6)
        _ga.evaluate_population(
            d.cost, d.time, pop, 0, kw["max_time_work"], fixed, split_code,
            np.empty(n, np.int64), np.empty(n), np.empty(n, np.int64), out,
        )  # fmt: skip
        expected = [
            reference.problem_cost(d.cost, d.time, [0, *row], kw["max_time_work"], fixed, split_name)
            for row in pop.tolist()
        ]
        assert out.tolist() == pytest.approx(expected)
    out = np.empty(6)
    _ga.evaluate_population(
        d.cost, d.cost, pop, 0, np.inf, 0.0, 0, np.empty(n, np.int64), np.empty(0), np.empty(0, np.int64), out
    )
    assert out.tolist() == pytest.approx([reference.tour_cost(d.cost, [0, *row]) for row in pop.tolist()])


@settings(derandomize=True, deadline=None, max_examples=200)
@given(data=st.data())
def test_ox_and_pmx_match_the_reference(data):
    m = data.draw(st.integers(min_value=2, max_value=12))
    genes = list(range(1, m + 1))  # the depot (0) is never part of a chromosome
    p1 = np.asarray(data.draw(st.permutations(genes)), dtype=np.int64)
    p2 = np.asarray(data.draw(st.permutations(genes)), dtype=np.int64)
    a = data.draw(st.integers(min_value=0, max_value=m - 1))
    b = data.draw(st.integers(min_value=a, max_value=m - 1))
    child = np.empty(m, dtype=np.int64)
    present, mapping = np.zeros(m + 1, dtype=np.uint8), np.empty(m + 1, dtype=np.int64)
    _ga.ox(p1, p2, a, b, child, present)
    assert sorted(child.tolist()) == genes
    assert child.tolist() == reference.ox(p1.tolist(), p2.tolist(), a, b)
    _ga.pmx(p1, p2, a, b, child, present, mapping)
    assert sorted(child.tolist()) == genes
    assert child.tolist() == reference.pmx(p1.tolist(), p2.tolist(), a, b)


@settings(derandomize=True, deadline=None, max_examples=100)
@given(data=st.data())
def test_mutations_are_permutations(data):
    m = data.draw(st.integers(min_value=2, max_value=10))
    genes = list(range(1, m + 1))
    chrom = np.asarray(data.draw(st.permutations(genes)), dtype=np.int64)
    i = data.draw(st.integers(min_value=0, max_value=m - 1))
    j = data.draw(st.integers(min_value=0, max_value=m - 1))
    for kind in (0, 1, 2):
        c = chrom.copy()
        _ga.mutate(c, kind, i, j)
        assert sorted(c.tolist()) == genes
        if i == j:
            assert np.array_equal(c, chrom)
    c = chrom.copy()
    _ga.mutate(c, 0, i, j)
    lo, hi = min(i, j), max(i, j)
    assert (
        c.tolist() == reference.two_opt_apply([0, *chrom.tolist()], lo + 1, hi + 1)[1:] if lo < hi else True
    )
    c = chrom.copy()
    _ga.mutate(c, 1, i, j)
    assert c.tolist() == reference.swap_apply(chrom.tolist(), i, j)
    c = chrom.copy()
    _ga.mutate(c, 2, i, j)
    expected = chrom.tolist()
    expected.insert(j, expected.pop(i))
    assert c.tolist() == expected


def test_polish_tour_reaches_two_opt_optimum_on_symmetric_and_generic_paths():
    for asym in (False, True):
        C, _ = _euclid(10, seed=10, asymmetric=asym)
        p = RoutingProblem(C)
        n = p.n
        tour = np.arange(n, dtype=np.int64)
        cost = _ga.polish_tour(
            p.cost, p.time_or_cost, tour, np.empty(n, np.int64), p.neighbours(n - 1), np.zeros(n, np.uint8),
            np.inf, 0.0, 0, 2 if asym else 1, 1, np.empty(n, np.int64), np.empty(n), np.empty(n, np.int64),
        )  # fmt: skip
        assert cost == pytest.approx(reference.tour_cost(C, tour))
        assert sorted(tour.tolist()) == list(range(n)) and tour[0] == 0
        assert _two_opt_local_optimum(C, tour.tolist(), cost)
