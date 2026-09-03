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


def test_duplicates_are_counted_and_the_population_stays_valid(small_euclidean):
    # without crossover or mutation every child copies its tournament winner, so children duplicate the elites
    # and each other from the first generation on: the re-mutation path fires and is counted
    C = small_euclidean["C"]
    ga = Genetic(pop_size=8, n_elite=2, p_crossover=0.0, p_mutation=0.0, n_generations=5, patience=None,
                 random_state=0).fit(C)  # fmt: skip
    assert 0 < ga.n_duplicates_ <= 5 * (8 - 2)
    _assert_consistent(ga, C)
    assert Genetic(pop_size=8, n_elite=2, p_crossover=0.0, p_mutation=0.0, n_generations=5, patience=None,
                   random_state=0).fit(C).n_duplicates_ == ga.n_duplicates_  # fmt: skip


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


def _mutate_model(child, kind, i, j):
    """SPEC §4.4 mutations on a chromosome (list): inversion (a 2-opt move), swap, insertion.

    ``i == j`` is a no-op, like the kernel's.
    """
    child = list(child)
    if i == j:
        return child
    if kind == 0:
        lo, hi = min(i, j), max(i, j)
        return reference.two_opt_apply([0, *child], lo + 1, hi + 1)[1:]
    if kind == 1:
        return reference.swap_apply(child, i, j)
    child.insert(j, child.pop(i))
    return child


def _generation_model(pop, fit, elite_idx, tourn, u_cross, cuts, u_mut, mut, remut, p_cx, p_mut, cx, mk):
    """One SPEC §4.4 generation in pure Python over the SAME pre-drawn arrays as ``_ga.ga_generation``.

    Returns ``(rows, n_dups)``: the ``n_elite`` fittest parents unchanged, then one child per tournament
    pair -- the fitter winner is parent 1 (ties: the first) -- produced by OX/PMX when ``u_cross < p_cx``
    (else a copy of parent 1), mutated when ``u_mut < p_mut``, and mutated once more (counted) if it exactly
    duplicates a row already produced this generation.
    """

    def tournament(contestants):
        best = int(contestants[0])
        for idx in contestants[1:]:
            if fit[idx] < fit[best]:
                best = int(idx)
        return best

    rows = [pop[e].tolist() for e in elite_idx]
    n_dups = 0
    for c in range(len(u_cross)):
        p1, p2 = tournament(tourn[0, c]), tournament(tourn[1, c])
        if fit[p2] < fit[p1]:
            p1, p2 = p2, p1
        if u_cross[c] < p_cx:
            cross = reference.ox if cx == 0 else reference.pmx
            child = cross(pop[p1].tolist(), pop[p2].tolist(), int(cuts[c, 0]), int(cuts[c, 1]))
        else:
            child = pop[p1].tolist()
        if u_mut[c] < p_mut:
            child = _mutate_model(child, mk, int(mut[c, 0]), int(mut[c, 1]))
        if child in rows:
            child = _mutate_model(child, mk, int(remut[c, 0]), int(remut[c, 1]))
            n_dups += 1
        rows.append(child)
    return rows, n_dups


def _scratch(n, m, pop_size):
    i64, u8 = np.int64, np.uint8
    return [
        np.empty(m, i64), np.empty(m, i64), np.empty(m, i64),  # par1, par2, child
        np.empty(n, i64), np.empty(n, i64), np.zeros(n, u8), np.empty(n, i64),  # tour, pos, dont_look, scr.
        np.empty(n), np.empty(n, i64), np.zeros(n, u8), np.empty(n, i64),  # dp, pred, present, mapping
        np.empty(pop_size, np.uint64),  # hashes
    ]  # fmt: skip


@pytest.mark.parametrize("cx", [0, 1], ids=["ox", "pmx"])
@pytest.mark.parametrize("mk", [0, 1, 2], ids=["inversion", "swap", "insertion"])
def test_ga_generation_matches_the_python_model(cx, mk):
    # elitism, tournament, crossover, mutation, duplicate re-mutation and the objective of every row,
    # checked gene for gene against the model over random sizes, depots, symmetric/asymmetric matrices and
    # budgets (plain, greedy split, optimal split); the parents' arrays must come back untouched
    for trial in range(40):
        rng = np.random.default_rng(1000 * cx + 100 * mk + trial)
        n = int(rng.integers(3, 10))
        m = n - 1
        pop_size = int(rng.integers(2, 9))
        n_elite = int(rng.integers(0, pop_size))
        depot = int(rng.integers(0, n))
        C, _ = _euclid(n, seed=trial, asymmetric=bool(rng.integers(0, 2)))
        if rng.integers(0, 2):
            T = C * rng.uniform(0.5, 2.0, C.shape)
            np.fill_diagonal(T, 0.0)
            T = np.ascontiguousarray(T)
            max_time = float((T[depot] + T[:, depot]).max()) * float(rng.uniform(1.0, 3.0))
            fixed, split = float(rng.uniform(0.0, 20.0)), int(rng.integers(0, 2))
        else:
            T, max_time, fixed, split = C, np.inf, 0.0, 0
        split_name = "greedy" if split == 0 else "optimal"
        others = np.delete(np.arange(n), depot)
        pop = np.stack([rng.permutation(others) for _ in range(pop_size)]).astype(np.int64)
        fit = np.array(
            [reference.problem_cost(C, T, [depot, *r], max_time, fixed, split_name) for r in pop.tolist()]
        )
        elite_idx = np.ascontiguousarray(np.argsort(fit, kind="stable")[:n_elite], dtype=np.int64)
        n_children, k_t = pop_size - n_elite, int(rng.integers(1, 6))
        p_cx, p_mut = float(rng.random()), float(rng.random())
        tourn = rng.integers(0, pop_size, size=(2, n_children, k_t), dtype=np.int64)
        u_cross, u_mut = rng.random(n_children), rng.random(n_children)
        cuts = np.ascontiguousarray(np.sort(rng.integers(0, m, size=(n_children, 2), dtype=np.int64), axis=1))
        mut = rng.integers(0, m, size=(n_children, 2), dtype=np.int64)
        remut = rng.integers(0, m, size=(n_children, 2), dtype=np.int64)
        pop_before, fit_before = pop.copy(), fit.copy()
        new_pop, new_fit = np.empty_like(pop), np.empty_like(fit)
        n_dups = _ga.ga_generation(
            C, T, max_time, fixed, split, depot, pop, fit, new_pop, new_fit, elite_idx, tourn, u_cross, cuts,
            u_mut, mut, remut, p_cx, p_mut, cx, mk, 0, 0, np.empty((n, 0), np.int64),
            *_scratch(n, m, pop_size),
        )  # fmt: skip
        rows, n_dups_model = _generation_model(
            pop, fit, elite_idx, tourn, u_cross, cuts, u_mut, mut, remut, p_cx, p_mut, cx, mk
        )
        assert np.array_equal(pop, pop_before) and np.array_equal(fit, fit_before)
        assert new_pop.tolist() == rows, f"trial {trial}: generation differs from the model"
        assert n_dups == n_dups_model
        assert np.array_equal(new_pop[:n_elite], pop[elite_idx]) and np.array_equal(
            new_fit[:n_elite], fit[elite_idx]
        )
        for row, f in zip(new_pop.tolist(), new_fit.tolist(), strict=True):
            assert sorted(row) == others.tolist()
            assert f == pytest.approx(
                reference.problem_cost(C, T, [depot, *row], max_time, fixed, split_name)
            )


@pytest.mark.parametrize("n_elite", [0, 1, 3])
def test_ga_generation_re_mutates_every_duplicate_once(n_elite):
    # a converged population without crossover or mutation: every child copies the same chromosome, so each
    # one duplicates an earlier row (the first child is new when there is no elite), is mutated once at its
    # remut draw and counted; the elites are copied unchanged with their fitness
    C, _ = _euclid(7, seed=7)
    n, m, pop_size = 7, 6, 5
    chrom = np.array([3, 1, 6, 2, 5, 4], dtype=np.int64)
    pop = np.tile(chrom, (pop_size, 1))
    fit = np.full(pop_size, reference.tour_cost(C, [0, *chrom.tolist()]))
    elite_idx = np.arange(n_elite, dtype=np.int64)
    n_children = pop_size - n_elite
    rng = np.random.default_rng(0)
    tourn = rng.integers(0, pop_size, size=(2, n_children, 3), dtype=np.int64)
    remut = np.array([[0, 5], [1, 4], [2, 3], [0, 2], [4, 5]][:n_children], dtype=np.int64)  # i != j: visible
    new_pop, new_fit = np.empty_like(pop), np.empty_like(fit)
    n_dups = _ga.ga_generation(
        C, C, np.inf, 0.0, 0, 0, pop, fit, new_pop, new_fit, elite_idx, tourn, np.ones(n_children),
        np.zeros((n_children, 2), np.int64), np.ones(n_children), np.zeros((n_children, 2), np.int64), remut,
        0.0, 0.0, 0, 1, 0, 0, np.empty((n, 0), np.int64), *_scratch(n, m, pop_size),
    )  # fmt: skip
    assert n_dups == (n_children if n_elite else n_children - 1)
    assert np.array_equal(new_pop[:n_elite], pop[:n_elite]) and np.array_equal(
        new_fit[:n_elite], fit[:n_elite]
    )
    first_new = n_elite if n_elite else 1
    if not n_elite:
        assert new_pop[0].tolist() == chrom.tolist(), (
            "the first child of an elite-less generation is not a duplicate"
        )
    for c in range(first_new - n_elite, n_children):
        expected = reference.swap_apply(chrom.tolist(), int(remut[c, 0]), int(remut[c, 1]))
        assert new_pop[n_elite + c].tolist() == expected, (
            "a duplicate is re-mutated exactly once, at its remut draw"
        )
        assert new_fit[n_elite + c] == pytest.approx(reference.tour_cost(C, [0, *expected]))


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
