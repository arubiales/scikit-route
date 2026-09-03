"""Tests of the compiled core ``skroute._core._routing`` against the pure-Python oracles.

Every kernel of SPEC §3.5 is checked property-wise (hypothesis, ``derandomize=True``) against
``tests/reference.py`` on instances with ``n in 3..12``, symmetric and asymmetric finite
matrices, random depot-first permutations and budgets in ``[max round trip, 3 x max round
trip]`` (§6), plus unit tests for the ``n = 3`` edges, the ``SplitRule`` constants, argument
validation and the ``.pyi`` stub surface.
"""

from __future__ import annotations

import ast
import doctest
import enum
import itertools
import math
import pathlib

import numpy as np
import pytest
import reference
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute._core import _routing as core

GREEDY = int(core.SplitRule.SPLIT_GREEDY)
OPTIMAL = int(core.SplitRule.SPLIT_OPTIMAL)
TWO_OPT, OR_OPT, SWAP = 1, 2, 4
SETTINGS = settings(derandomize=True, deadline=None, max_examples=200)
FEW = settings(derandomize=True, deadline=None, max_examples=60)


# ----------------------------------------------------------------------------- helpers
def neighbours(C: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours per row, diagonal excluded, ties by index (RoutingProblem.neighbours)."""
    M = np.array(C, dtype=float, copy=True)
    np.fill_diagonal(M, np.inf)
    order = np.argsort(M, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(order, dtype=np.int64)


def is_permutation(tour: np.ndarray, n: int) -> bool:
    return sorted(int(v) for v in tour) == list(range(n))


def pos_consistent(tour: np.ndarray, pos: np.ndarray) -> bool:
    return bool(np.array_equal(pos[tour], np.arange(len(tour))))


def scratch(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.empty(n, dtype=np.int64), np.empty(n, dtype=np.float64), np.empty(n, dtype=np.int64)


def or_opt_moves(n: int, max_L: int = 3) -> list[tuple[int, int, int]]:
    """Every valid (i, L, j) of SPEC §3.5 for a tour of n nodes."""
    moves: list[tuple[int, int, int]] = []
    for L in range(1, max_L + 1):
        for i in range(1, n - L + 1):
            moves.extend((i, L, j) for j in range(n) if not (i - 1 <= j <= i + L - 1))
    return moves


def nearest_neighbour_reference(C: np.ndarray, depot: int) -> list[int]:
    """Greedy nearest neighbour from the depot, ties by lowest index (the §3.5 rule)."""
    n = C.shape[0]
    tour = [depot]
    seen = {depot}
    while len(tour) < n:
        cur = tour[-1]
        _, nxt = min((float(C[cur, j]), j) for j in range(n) if j not in seen)
        tour.append(nxt)
        seen.add(nxt)
    return tour


def descent_buffers(inst: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tour copy, consistent pos, zeroed don't-look bits) for a descent on ``inst``."""
    t = inst["tour"].copy()
    pos = np.empty(inst["n"], dtype=np.int64)
    core.rebuild_pos(t, pos)
    return t, pos, np.zeros(inst["n"], dtype=np.uint8)


def generic_moves_cost(C: np.ndarray, T, tour, max_time: float, fixed_cost: float, name: str, max_L: int):
    """Objective of every move ``local_search_generic`` enumerates with full candidate lists:
    every 2-opt reversal, every forward Or-opt insertion up to ``max_L`` and every swap."""
    n = len(tour)
    for i, j in two_opt_pairs(n):
        yield reference.problem_cost(C, T, reference.two_opt_apply(tour, i, j), max_time, fixed_cost, name)
        yield reference.problem_cost(C, T, reference.swap_apply(tour, i, j), max_time, fixed_cost, name)
    for i, L, j in or_opt_moves(n, max_L):
        yield reference.problem_cost(C, T, reference.or_opt_apply(tour, i, L, j), max_time, fixed_cost, name)


def two_opt_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(1, n - 1) for j in range(i + 1, n)]


# ----------------------------------------------------------------------------- strategies
@st.composite
def instances(draw, symmetric=None, with_time=False, min_n=3, max_n=12):
    """Instance dict: C (and T, max_time, fixed_cost), a depot-first random tour, n, symmetric."""
    n = draw(st.integers(min_n, max_n))
    seed = draw(st.integers(0, 2**31 - 1))
    sym = draw(st.booleans()) if symmetric is None else symmetric
    rng = np.random.default_rng(seed)
    C = rng.uniform(1.0, 100.0, (n, n))
    if sym:
        C = (C + C.T) / 2.0
    np.fill_diagonal(C, 0.0)
    depot = draw(st.integers(0, n - 1))
    others = rng.permutation([v for v in range(n) if v != depot])
    tour = np.concatenate(([depot], others)).astype(np.int64)
    inst = {"C": np.ascontiguousarray(C), "tour": tour, "n": n, "symmetric": sym, "depot": depot}
    if with_time:
        T = rng.uniform(0.5, 10.0, (n, n))
        if sym:
            T = (T + T.T) / 2.0
        np.fill_diagonal(T, 0.0)
        round_trip = max(T[depot, v] + T[v, depot] for v in range(n) if v != depot)
        inst["T"] = np.ascontiguousarray(T)
        inst["max_time"] = round_trip * draw(st.floats(1.0, 3.0))
        inst["fixed_cost"] = draw(st.floats(0.0, 50.0))
    return inst


@st.composite
def instance_and_two_opt(draw, symmetric=None):
    inst = draw(instances(symmetric=symmetric))
    i, j = draw(st.sampled_from(two_opt_pairs(inst["n"])))
    return inst, i, j


@st.composite
def instance_and_or_opt(draw, symmetric=None):
    inst = draw(instances(symmetric=symmetric))
    i, L, j = draw(st.sampled_from(or_opt_moves(inst["n"])))
    return inst, i, L, j, draw(st.booleans())


@st.composite
def instance_and_candidates(draw, symmetric=None, with_time=False):
    inst = draw(instances(symmetric=symmetric, with_time=with_time))
    k = draw(st.integers(1, inst["n"] - 1))
    inst["cand"] = neighbours(inst["C"], k)
    return inst


# ----------------------------------------------------------------------------- evaluation
@SETTINGS
@given(instances())
def test_tour_cost_matches_reference(inst):
    C, tour = inst["C"], inst["tour"]
    expected = reference.tour_cost(C, tour)
    assert core.tour_cost_py(C, tour) == pytest.approx(expected, rel=1e-12)
    # plain TSP through the dispatcher, whichever split rule is passed
    assert core.problem_cost_py(C, C, tour, math.inf, 7.0, GREEDY) == pytest.approx(expected, rel=1e-12)
    assert core.problem_cost_py(C, C, tour, math.inf, 7.0, OPTIMAL) == pytest.approx(expected, rel=1e-12)


@SETTINGS
@given(instances(with_time=True))
def test_split_costs_match_reference_and_optimal_is_never_worse(inst):
    C, T, tour, mt, fc = inst["C"], inst["T"], inst["tour"], inst["max_time"], inst["fixed_cost"]
    greedy = core.greedy_split_cost_py(C, T, tour, mt, fc)
    optimal = core.optimal_split_cost_py(C, T, tour, mt, fc)
    assert greedy == pytest.approx(reference.greedy_split(C, T, tour, mt, fc)[0], rel=1e-12)
    assert optimal == pytest.approx(reference.optimal_split(C, T, tour, mt, fc)[0], rel=1e-12)
    assert optimal <= greedy + 1e-9 * max(1.0, greedy)
    # the dispatcher agrees with both
    assert core.problem_cost_py(C, T, tour, mt, fc, GREEDY) == pytest.approx(greedy, rel=1e-12)
    assert core.problem_cost_py(C, T, tour, mt, fc, OPTIMAL) == pytest.approx(optimal, rel=1e-12)
    assert core.problem_cost_py(C, T, tour, mt, fc, GREEDY) == pytest.approx(
        reference.problem_cost(C, T, tour, mt, fc, "greedy"), rel=1e-12
    )
    assert core.problem_cost_py(C, T, tour, mt, fc, OPTIMAL) == pytest.approx(
        reference.problem_cost(C, T, tour, mt, fc, "optimal"), rel=1e-12
    )


@SETTINGS
@given(instances(with_time=True), st.sampled_from([GREEDY, OPTIMAL]))
def test_trip_starts_fit_the_budget_and_cover_the_tour(inst, split):
    C, T, tour, n = inst["C"], inst["T"], inst["tour"], inst["n"]
    mt, fc = inst["max_time"], inst["fixed_cost"]
    out = np.full(n + 1, -1, dtype=np.int64)
    k = core.trip_starts(T, tour, mt, split, C, fc, out)
    starts = out[: k + 1]
    assert k >= 1 and starts[0] == 1 and starts[-1] == n
    assert np.all(np.diff(starts) >= 1), "trips are non-empty and cover positions 1..n-1"
    name = "greedy" if split == GREEDY else "optimal"
    assert starts.tolist() == reference.trip_starts(C, T, tour, mt, fc, name)
    times = np.empty(k)
    core.trip_times(T, tour, starts, times)
    assert np.all(times <= mt + 1e-9 * max(1.0, mt))
    costs = np.empty(k)
    core.trip_costs(C, tour, starts, costs)
    total = core.problem_cost_py(C, T, tour, mt, fc, split)
    assert costs.sum() + fc * (k - 1) == pytest.approx(total, rel=1e-9)
    # per-trip values against a direct computation
    d = int(tour[0])
    for t in range(k):
        trip = [d, *tour[starts[t] : starts[t + 1]].tolist(), d]
        assert costs[t] == pytest.approx(sum(C[a, b] for a, b in itertools.pairwise(trip)), rel=1e-12)
        assert times[t] == pytest.approx(sum(T[a, b] for a, b in itertools.pairwise(trip)), rel=1e-12)


@SETTINGS
@given(instances())
def test_trip_starts_plain_tsp_is_one_trip(inst):
    C, tour, n = inst["C"], inst["tour"], inst["n"]
    out = np.empty(n + 1, dtype=np.int64)
    assert core.trip_starts(C, tour, math.inf, OPTIMAL, C, 3.0, out) == 1
    assert out[:2].tolist() == [1, n]
    costs = np.empty(1)
    core.trip_costs(C, tour, out[:2], costs)
    assert costs[0] == core.tour_cost_py(C, tour), "same summation order, bit-identical"


# ----------------------------------------------------------------------------- move deltas
@SETTINGS
@given(instance_and_two_opt(symmetric=True))
def test_two_opt_delta_is_exact_on_symmetric(args):
    inst, i, j = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.two_opt_delta_by_recompute(C, tour, i, j)
    assert core.two_opt_delta_py(C, tour, i, j) == pytest.approx(expected, abs=1e-9)
    assert core.two_opt_delta_asym_py(C, tour, i, j) == pytest.approx(expected, abs=1e-9)


@SETTINGS
@given(instance_and_two_opt(symmetric=False))
def test_two_opt_delta_asym_is_exact_on_asymmetric(args):
    inst, i, j = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.two_opt_delta_by_recompute(C, tour, i, j)
    assert core.two_opt_delta_asym_py(C, tour, i, j) == pytest.approx(expected, abs=1e-9)


@SETTINGS
@given(instance_and_or_opt(symmetric=False))
def test_or_opt_delta_forward_is_exact_on_asymmetric(args):
    inst, i, L, j, _ = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.or_opt_delta_by_recompute(C, tour, i, L, j, reverse=False)
    assert core.or_opt_delta_py(C, tour, i, L, j, False) == pytest.approx(expected, abs=1e-9)


@SETTINGS
@given(instance_and_or_opt(symmetric=True))
def test_or_opt_delta_both_orientations_are_exact_on_symmetric(args):
    inst, i, L, j, reverse = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.or_opt_delta_by_recompute(C, tour, i, L, j, reverse=reverse)
    assert core.or_opt_delta_py(C, tour, i, L, j, reverse) == pytest.approx(expected, abs=1e-9)


@SETTINGS
@given(instance_and_two_opt(symmetric=False))
def test_swap_delta_is_exact_on_asymmetric(args):
    inst, i, j = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.swap_delta_by_recompute(C, tour, i, j)
    assert core.swap_delta_py(C, tour, i, j) == pytest.approx(expected, abs=1e-9)


# ----------------------------------------------------------------------------- applying moves
@SETTINGS
@given(instance_and_two_opt())
def test_reverse_segment_matches_reference_and_is_an_involution(args):
    inst, i, j = args
    tour, n = inst["tour"], inst["n"]
    t = tour.copy()
    core.reverse_segment_py(t, i, j)
    assert t.tolist() == reference.two_opt_apply(tour, i, j)
    core.reverse_segment_py(t, i, j)
    assert np.array_equal(t, tour)
    t, pos = tour.copy(), np.empty(n, dtype=np.int64)
    core.rebuild_pos(t, pos)
    assert pos_consistent(t, pos)
    core.reverse_segment_pos_py(t, pos, i, j)
    assert t.tolist() == reference.two_opt_apply(tour, i, j) and pos_consistent(t, pos)
    core.reverse_segment_pos_py(t, pos, i, j)
    assert np.array_equal(t, tour) and pos_consistent(t, pos)


@SETTINGS
@given(instance_and_two_opt())
def test_swap_positions_matches_reference_and_is_an_involution(args):
    inst, i, j = args
    tour, n = inst["tour"], inst["n"]
    t = tour.copy()
    core.swap_positions_py(t, i, j)
    assert t.tolist() == reference.swap_apply(tour, i, j)
    core.swap_positions_py(t, i, j)
    assert np.array_equal(t, tour)
    t, pos = tour.copy(), np.empty(n, dtype=np.int64)
    core.rebuild_pos(t, pos)
    core.swap_positions_pos_py(t, pos, i, j)
    assert t.tolist() == reference.swap_apply(tour, i, j) and pos_consistent(t, pos)
    core.swap_positions_pos_py(t, pos, i, j)
    assert np.array_equal(t, tour) and pos_consistent(t, pos)


@SETTINGS
@given(instance_and_or_opt())
def test_move_segment_matches_reference_and_has_an_inverse(args):
    inst, i, L, j, reverse = args
    tour, n = inst["tour"], inst["n"]
    expected = reference.or_opt_apply(tour, i, L, j, reverse)
    t = tour.copy()
    core.move_segment_py(t, i, L, j, reverse)
    assert t.tolist() == expected
    # inverse: move the segment back behind its original predecessor, same orientation flag
    segment = tour[i : i + L].tolist()
    head = segment[-1] if reverse else segment[0]
    i2 = t.tolist().index(head)
    j2 = t.tolist().index(int(tour[i - 1]))
    core.move_segment_py(t, i2, L, j2, reverse)
    assert np.array_equal(t, tour)
    t, pos = tour.copy(), np.empty(n, dtype=np.int64)
    core.rebuild_pos(t, pos)
    core.move_segment_pos_py(t, pos, i, L, j, reverse)
    assert t.tolist() == expected and pos_consistent(t, pos)
    core.move_segment_pos_py(t, pos, i2, L, j2, reverse)
    assert np.array_equal(t, tour) and pos_consistent(t, pos)


@SETTINGS
@given(instances(min_n=4), st.data())
def test_double_bridge_is_a_permutation_with_the_depot_fixed(inst, data):
    tour, n = inst["tour"], inst["n"]
    p1 = data.draw(st.integers(1, n - 3))
    p2 = data.draw(st.integers(p1 + 1, n - 2))
    p3 = data.draw(st.integers(p2 + 1, n - 1))
    out = np.full(n, -1, dtype=np.int64)
    core.double_bridge(tour, p1, p2, p3, out)
    assert out.tolist() == reference.double_bridge(tour, p1, p2, p3)
    assert is_permutation(out, n) and out[0] == tour[0]
    assert np.array_equal(out[:p1], tour[:p1]) and np.array_equal(out[n - (n - p3) :], tour[p3:])


# ----------------------------------------------------------------------------- descents (symmetric, O(1))
@SETTINGS
@given(instance_and_candidates(symmetric=True), st.booleans())
def test_two_opt_descent_never_worsens_and_its_gain_is_the_cost_difference(inst, first_improvement):
    C, tour, n, cand = inst["C"], inst["tour"], inst["n"], inst["cand"]
    t, pos, dlb = descent_buffers(inst)
    before = reference.tour_cost(C, t)
    gain = core.two_opt_descent(C, t, pos, cand, dlb, first_improvement, 1000)
    after = reference.tour_cost(C, t)
    assert gain <= 0.0
    assert is_permutation(t, n) and t[0] == tour[0] and pos_consistent(t, pos)
    assert after <= before + 1e-9 * max(1.0, before)
    assert before + gain == pytest.approx(after, rel=1e-9, abs=1e-9)
    # the one-pass-at-a-time protocol of LocalSearch (persistent buffers) accounts identically
    t, pos, dlb = descent_buffers(inst)
    total = 0.0
    for _ in range(10 * n):
        g = core.two_opt_descent(C, t, pos, cand, dlb, first_improvement, 1)
        total += g
        if g == 0.0:
            break
    assert before + total == pytest.approx(reference.tour_cost(C, t), rel=1e-9, abs=1e-9)
    assert pos_consistent(t, pos)


@SETTINGS
@given(instance_and_candidates(symmetric=True), st.integers(1, 3), st.booleans())
def test_or_opt_descent_never_worsens_and_its_gain_is_the_cost_difference(inst, max_segment, allow_reverse):
    C, tour, n, cand = inst["C"], inst["tour"], inst["n"], inst["cand"]
    t, pos, dlb = descent_buffers(inst)
    before = reference.tour_cost(C, t)
    gain = core.or_opt_descent(C, t, pos, cand, dlb, max_segment, allow_reverse, 1000)
    after = reference.tour_cost(C, t)
    assert gain <= 0.0
    assert is_permutation(t, n) and t[0] == tour[0] and pos_consistent(t, pos)
    assert after <= before + 1e-9 * max(1.0, before)
    assert before + gain == pytest.approx(after, rel=1e-9, abs=1e-9)
    t, pos, dlb = descent_buffers(inst)
    total = 0.0
    for _ in range(10 * n):
        g = core.or_opt_descent(C, t, pos, cand, dlb, max_segment, allow_reverse, 1)
        total += g
        if g == 0.0:
            break
    assert before + total == pytest.approx(reference.tour_cost(C, t), rel=1e-9, abs=1e-9)
    assert pos_consistent(t, pos)


@SETTINGS
@given(instance_and_candidates(symmetric=True))
def test_descents_with_max_passes_zero_change_nothing(inst):
    C, tour, cand = inst["C"], inst["tour"], inst["cand"]
    t, pos, dlb = descent_buffers(inst)
    assert core.two_opt_descent(C, t, pos, cand, dlb, True, 0) == 0.0
    assert core.or_opt_descent(C, t, pos, cand, dlb, 3, True, 0) == 0.0
    scratch_tour, dp, pred = scratch(inst["n"])
    assert (
        core.local_search_generic(C, C, t, pos, cand, math.inf, 0.0, GREEDY, 7, 3, 0, scratch_tour, dp, pred)
        == 0.0
    )
    assert np.array_equal(t, tour) and pos_consistent(t, pos) and not dlb.any()


@FEW
@given(instances(symmetric=True, min_n=5, max_n=9))
def test_two_opt_descent_with_full_lists_reaches_a_two_opt_local_optimum(inst):
    """Best-improvement 2-opt over the complete neighbourhood ends where no reversal improves."""
    C, n = inst["C"], inst["n"]
    inst["cand"] = neighbours(C, n - 1)
    t, pos, dlb = descent_buffers(inst)
    core.two_opt_descent(C, t, pos, inst["cand"], dlb, False, 10_000)
    cur = reference.tour_cost(C, t)
    for i, j in two_opt_pairs(n):
        assert reference.tour_cost(C, reference.two_opt_apply(t, i, j)) >= cur - 1e-9 * max(1.0, cur) - 1e-12


# ----------------------------------------------------------------------------- generic descent
@SETTINGS
@given(
    instance_and_candidates(with_time=True),
    st.integers(1, 7),
    st.integers(1, 3),
    st.sampled_from([GREEDY, OPTIMAL]),
)
def test_local_search_generic_under_a_budget_never_worsens_and_reports_its_gain(
    inst, moves, max_segment, split
):
    C, T, tour, n, cand = inst["C"], inst["T"], inst["tour"], inst["n"], inst["cand"]
    mt, fc = inst["max_time"], inst["fixed_cost"]
    name = "greedy" if split == GREEDY else "optimal"
    t, pos, _ = descent_buffers(inst)
    scratch_tour, dp, pred = scratch(n)
    before = reference.problem_cost(C, T, t, mt, fc, name)
    gain = core.local_search_generic(
        C, T, t, pos, cand, mt, fc, split, moves, max_segment, 1000, scratch_tour, dp, pred
    )
    after = reference.problem_cost(C, T, t, mt, fc, name)
    assert gain <= 0.0
    assert is_permutation(t, n) and t[0] == tour[0] and pos_consistent(t, pos)
    assert after <= before + 1e-9 * max(1.0, before)
    assert before + gain == pytest.approx(after, rel=1e-9, abs=1e-9)
    out = np.empty(n + 1, dtype=np.int64)
    k = core.trip_starts(T, t, mt, split, C, fc, out)
    times = np.empty(k)
    core.trip_times(T, t, out[: k + 1], times)
    assert np.all(times <= mt + 1e-9 * max(1.0, mt))


@SETTINGS
@given(instance_and_candidates(symmetric=False), st.integers(1, 7), st.integers(1, 3))
def test_local_search_generic_on_asymmetric_plain_tsp_never_worsens_and_reports_its_gain(
    inst, moves, max_segment
):
    C, tour, n, cand = inst["C"], inst["tour"], inst["n"], inst["cand"]
    t, pos, _ = descent_buffers(inst)
    scratch_tour = np.empty(n, dtype=np.int64)
    dp, pred = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)  # zero-length: plain TSP
    before = reference.tour_cost(C, t)
    gain = core.local_search_generic(
        C, C, t, pos, cand, math.inf, 0.0, GREEDY, moves, max_segment, 1000, scratch_tour, dp, pred
    )
    after = reference.tour_cost(C, t)
    assert gain <= 0.0
    assert is_permutation(t, n) and t[0] == tour[0] and pos_consistent(t, pos)
    assert after <= before + 1e-9 * max(1.0, before)
    assert before + gain == pytest.approx(after, rel=1e-9, abs=1e-9)


@FEW
@given(instances(min_n=4, max_n=7), st.integers(1, 3))
def test_local_search_generic_with_full_lists_and_every_move_reaches_a_local_optimum(inst, max_segment):
    """At termination no enumerated move (2-opt, forward Or-opt, swap) improves the plain objective."""
    C, n = inst["C"], inst["n"]
    cand = neighbours(C, n - 1)
    t, pos, _ = descent_buffers(inst)
    scratch_tour, dp, pred = scratch(n)
    core.local_search_generic(
        C, C, t, pos, cand, math.inf, 0.0, GREEDY, 7, max_segment, 10_000, scratch_tour, dp, pred
    )
    cur = reference.tour_cost(C, t)
    floor = cur - 1e-9 * max(1.0, cur) - 1e-12
    assert all(c >= floor for c in generic_moves_cost(C, None, t, math.inf, 0.0, "greedy", max_segment))


@FEW
@given(instances(with_time=True, min_n=4, max_n=7), st.sampled_from([GREEDY, OPTIMAL]))
def test_local_search_generic_with_full_lists_reaches_a_local_optimum_under_a_budget(inst, split):
    C, T, n, mt, fc = inst["C"], inst["T"], inst["n"], inst["max_time"], inst["fixed_cost"]
    name = "greedy" if split == GREEDY else "optimal"
    cand = neighbours(C, n - 1)
    t, pos, _ = descent_buffers(inst)
    scratch_tour, dp, pred = scratch(n)
    core.local_search_generic(C, T, t, pos, cand, mt, fc, split, 7, 3, 10_000, scratch_tour, dp, pred)
    cur = reference.problem_cost(C, T, t, mt, fc, name)
    floor = cur - 1e-9 * max(1.0, cur) - 1e-12
    assert all(c >= floor for c in generic_moves_cost(C, T, t, mt, fc, name, 3))


# ----------------------------------------------------------------------------- construction
@SETTINGS
@given(instances(), st.booleans())
def test_nearest_neighbour_tour_matches_the_python_rule(inst, integer_costs):
    C, n, depot = inst["C"], inst["n"], inst["depot"]
    if integer_costs:  # many ties: exercises "ties by lowest index"
        C = np.ascontiguousarray(np.floor(C / 25.0))
    out = np.full(n, -1, dtype=np.int64)
    core.nearest_neighbour_tour(C, depot, out)
    assert out.tolist() == nearest_neighbour_reference(C, depot)
    assert is_permutation(out, n) and out[0] == depot


# ----------------------------------------------------------------------------- n = 3 edges and constants
C3 = np.array([[0.0, 1.0, 5.0], [2.0, 0.0, 3.0], [4.0, 6.0, 0.0]])  # asymmetric
T3 = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
TOUR3 = np.array([0, 1, 2], dtype=np.int64)


def test_n3_costs_and_deltas():
    assert core.tour_cost_py(C3, TOUR3) == 1.0 + 3.0 + 4.0
    assert core.problem_cost_py(C3, T3, TOUR3, math.inf, 9.0, GREEDY) == 8.0
    # budget 2.5: no leg can be chained (t + 1 + 1 > 2.5 after the first), two trips of one customer
    assert core.problem_cost_py(C3, T3, TOUR3, 2.5, 9.0, GREEDY) == (1.0 + 2.0) + (5.0 + 4.0) + 9.0
    assert core.problem_cost_py(C3, T3, TOUR3, 2.5, 9.0, OPTIMAL) == (1.0 + 2.0) + (5.0 + 4.0) + 9.0
    # budget 3: chaining fits exactly (<=), one trip
    assert core.problem_cost_py(C3, T3, TOUR3, 3.0, 9.0, GREEDY) == 8.0
    assert core.problem_cost_py(C3, T3, TOUR3, 3.0, 9.0, OPTIMAL) == 8.0
    reversed_cost = reference.tour_cost(C3, [0, 2, 1])
    assert core.two_opt_delta_asym_py(C3, TOUR3, 1, 2) == pytest.approx(reversed_cost - 8.0)
    assert core.swap_delta_py(C3, TOUR3, 1, 2) == pytest.approx(reversed_cost - 8.0)
    assert core.or_opt_delta_py(C3, TOUR3, 1, 1, 2) == pytest.approx(reversed_cost - 8.0)
    assert core.or_opt_delta_py(C3, TOUR3, 2, 1, 0) == pytest.approx(reversed_cost - 8.0)
    assert or_opt_moves(3) == [(1, 1, 2), (2, 1, 0)]
    sym = np.ascontiguousarray((C3 + C3.T) / 2.0)
    assert core.two_opt_delta_py(sym, TOUR3, 1, 2) == pytest.approx(0.0)  # orientation flip only


def test_n3_trips_moves_and_descents():
    out = np.empty(4, dtype=np.int64)
    assert core.trip_starts(T3, TOUR3, 2.5, GREEDY, C3, 9.0, out) == 2 and out[:3].tolist() == [1, 2, 3]
    assert core.trip_starts(T3, TOUR3, 2.5, OPTIMAL, C3, 9.0, out) == 2 and out[:3].tolist() == [1, 2, 3]
    assert core.trip_starts(T3, TOUR3, 3.0, OPTIMAL, C3, 9.0, out) == 1 and out[:2].tolist() == [1, 3]
    t = TOUR3.copy()
    core.move_segment_py(t, 1, 1, 2)
    assert t.tolist() == [0, 2, 1]
    core.move_segment_py(t, 2, 1, 0)
    assert t.tolist() == [0, 1, 2]
    nn = np.empty(3, dtype=np.int64)
    core.nearest_neighbour_tour(C3, 2, nn)
    assert nn.tolist() == [2, 0, 1]
    # symmetric descents at n = 3 cannot improve (the only move flips the orientation)
    sym = np.ascontiguousarray((C3 + C3.T) / 2.0)
    cand = neighbours(sym, 2)
    t, pos, dlb = np.array([0, 1, 2], dtype=np.int64), np.arange(3, dtype=np.int64), np.zeros(3, np.uint8)
    assert core.two_opt_descent(sym, t, pos, cand, dlb, True, 100) == 0.0
    assert core.or_opt_descent(sym, t, pos, cand, dlb, 3, True, 100) == 0.0
    assert t.tolist() == [0, 1, 2] and pos_consistent(t, pos)
    # the generic path on the asymmetric C3 flips to the cheaper orientation
    scratch_tour, dp, pred = scratch(3)
    t, pos = TOUR3.copy(), np.arange(3, dtype=np.int64)
    gain = core.local_search_generic(
        C3, C3, t, pos, neighbours(C3, 2), math.inf, 0.0, GREEDY, 7, 3, 100, scratch_tour, dp, pred
    )
    best = min(reference.tour_cost(C3, [0, 1, 2]), reference.tour_cost(C3, [0, 2, 1]))
    assert 8.0 + gain == pytest.approx(best) and reference.tour_cost(C3, t) == pytest.approx(best)
    assert pos_consistent(t, pos)


def test_double_bridge_smallest_case():
    out = np.empty(4, dtype=np.int64)
    core.double_bridge(np.arange(4, dtype=np.int64), 1, 2, 3, out)
    assert out.tolist() == [0, 2, 1, 3] == reference.double_bridge([0, 1, 2, 3], 1, 2, 3)


def test_split_rule_is_an_int_enum_and_not_module_constants():
    assert issubclass(core.SplitRule, enum.IntEnum)
    assert core.SplitRule.SPLIT_GREEDY == 0 and core.SplitRule.SPLIT_OPTIMAL == 1
    assert int(core.SplitRule.SPLIT_GREEDY) == 0 and int(core.SplitRule.SPLIT_OPTIMAL) == 1
    assert [m.name for m in core.SplitRule] == ["SPLIT_GREEDY", "SPLIT_OPTIMAL"]
    assert not hasattr(core, "SPLIT_GREEDY") and not hasattr(core, "SPLIT_OPTIMAL")  # SPEC §3.5


# ----------------------------------------------------------------------------- validation (Python surface)
def test_wrappers_reject_positions_outside_their_domain():
    tour = np.arange(6, dtype=np.int64)
    C = np.ascontiguousarray(np.arange(36, dtype=np.float64).reshape(6, 6))
    for i, j in [(0, 3), (3, 3), (4, 2), (2, 6)]:
        with pytest.raises(ValueError, match="1 <= i < j <= n - 1"):
            core.two_opt_delta_py(C, tour, i, j)
        with pytest.raises(ValueError):
            core.swap_delta_py(C, tour, i, j)
        with pytest.raises(ValueError):
            core.reverse_segment_py(tour.copy(), i, j)
        with pytest.raises(ValueError):
            core.swap_positions_py(tour.copy(), i, j)
    for i, L, j in [(0, 1, 3), (4, 3, 1), (1, 2, 2), (1, 2, 0), (2, 1, 6), (2, 1, -1), (1, 0, 3)]:
        with pytest.raises(ValueError, match="Or-opt"):
            core.or_opt_delta_py(C, tour, i, L, j)
        with pytest.raises(ValueError, match="Or-opt"):
            core.move_segment_py(tour.copy(), i, L, j)
    with pytest.raises(ValueError, match="pos must have"):
        core.reverse_segment_pos_py(tour.copy(), np.arange(5, dtype=np.int64), 1, 3)


def test_python_entry_points_validate_shapes_and_dtypes():
    tour = np.arange(4, dtype=np.int64)
    C = np.ones((4, 4))
    with pytest.raises(ValueError, match=r"C must be an \(4, 4\) matrix"):
        core.problem_cost_py(np.ones((3, 3)), C, tour, math.inf, 0.0, GREEDY)
    with pytest.raises(ValueError, match=r"T must be an \(4, 4\) matrix"):
        core.problem_cost_py(C, np.ones((4, 5)), tour, 3.0, 0.0, GREEDY)
    with pytest.raises(ValueError, match="out must have length n \\+ 1"):
        core.trip_starts(C, tour, 3.0, GREEDY, C, 0.0, np.empty(4, dtype=np.int64))
    with pytest.raises(ValueError, match="at least the depot"):
        core.tour_cost_py(C, np.empty(0, dtype=np.int64))
    # typed memoryviews refuse other dtypes and non-contiguous layouts
    with pytest.raises((ValueError, TypeError)):
        core.tour_cost_py(C, tour.astype(np.int32))
    with pytest.raises((ValueError, TypeError)):
        core.tour_cost_py(C.astype(np.float32), tour)
    with pytest.raises((ValueError, TypeError)):
        core.tour_cost_py(np.asfortranarray(np.arange(16.0).reshape(4, 4)), tour)
    # an infeasible optimal split (a customer's round trip exceeds the budget) is reported, not silent
    T = np.full((4, 4), 5.0)
    np.fill_diagonal(T, 0.0)
    assert core.problem_cost_py(C, T, tour, 6.0, 0.0, OPTIMAL) == math.inf
    with pytest.raises(ValueError, match="no feasible optimal split"):
        core.trip_starts(T, tour, 6.0, OPTIMAL, C, 0.0, np.empty(5, dtype=np.int64))


def test_stub_declares_exactly_the_python_surface():
    """The .pyi lists every public name of the compiled module (cpdef functions, wrappers, enum), no more."""
    stub = pathlib.Path(core.__file__).with_name("_routing.pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    declared = {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)}
    public = {name for name in dir(core) if not name.startswith("_")}
    assert declared == public
    assert set(core.__all__) == public
    enum_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SplitRule")
    assert [b.id for b in enum_cls.bases] == ["IntEnum"]  # type: ignore[attr-defined]


def test_docstring_examples_of_the_compiled_module_run():
    """``pytest --doctest-modules`` collects ``.py`` files only, so the examples embedded in the
    extension's docstrings are executed here."""
    result = doctest.testmod(core, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
