"""Tests of the compiled core ``skroute._core._routing`` against the pure-Python oracles.

Every kernel of SPEC §3.5 is checked property-wise (hypothesis, ``derandomize=True``) against
``tests/reference.py`` on instances with ``n in 3..12``, symmetric and asymmetric finite
matrices of several kinds (random, Euclidean, lattice points with coincidences and exact ties,
small integers with zero edges, all-zero, scaled by ``1e12`` and by ``1e-9``), random
depot-first permutations and budgets in ``[max round trip, 3 x max round trip]`` (§6), plus
unit tests for the ``n = 3`` edges, the ``SplitRule`` constants, argument validation and the
``.pyi`` stub surface. Two properties are checked beyond the reference: no kernel reads the
diagonal (§3.1) and the descents' don't-look bookkeeping and neighbourhoods are the documented
ones.
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
from hypothesis import assume, given, settings
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


def edges(tour) -> set[frozenset[int]]:
    """The undirected edges of a closed tour."""
    t = [int(v) for v in tour]
    return {frozenset((t[k], t[(k + 1) % len(t)])) for k in range(len(t))}


def delta_tol(C: np.ndarray, tour: np.ndarray) -> float:
    """Absolute tolerance for an O(1) delta against a recompute-by-difference: the rounding of the two
    full sums scales with the tour cost (1e12-scaled instances lose ~1e-3 in absolute terms)."""
    return 1e-9 * max(1.0, abs(reference.tour_cost(C, tour)))


# ----------------------------------------------------------------------------- strategies
# Matrix kinds: the bundled Waterloo data has coincident points (lu980: 346 duplicate rows) and
# integer ties, so zero edges and exact ties are normal input; the scaled kinds cover the extremes.
KINDS = ("uniform", "euclidean", "lattice", "integer", "zero", "huge", "tiny")


def make_matrix(rng: np.random.Generator, n: int, symmetric: bool, kind: str) -> np.ndarray:
    """Finite (n, n) float64 matrix of the given kind with a zero diagonal; symmetric when asked."""
    if kind in ("euclidean", "lattice"):
        xy = rng.random((n, 2)) * 100.0 if kind == "euclidean" else rng.integers(0, 3, (n, 2)).astype(float)
        M = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        if not symmetric:
            M = M * rng.uniform(0.7, 1.3, (n, n))
    else:
        if kind == "integer":
            M = rng.integers(0, 6, (n, n)).astype(float)
        elif kind == "zero":
            M = np.zeros((n, n))
        else:
            M = rng.uniform(1.0, 100.0, (n, n))
            if kind == "huge":
                M = M * 1e12
            elif kind == "tiny":
                M = M * 1e-9
        if symmetric:
            M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return np.ascontiguousarray(M)


@st.composite
def instances(draw, symmetric=None, with_time=False, min_n=3, max_n=12, kinds=KINDS):
    """Instance dict: C (and T, max_time, fixed_cost), a depot-first random tour, n, symmetric, kind."""
    n = draw(st.integers(min_n, max_n))
    seed = draw(st.integers(0, 2**31 - 1))
    sym = draw(st.booleans()) if symmetric is None else symmetric
    kind = draw(st.sampled_from(kinds))
    rng = np.random.default_rng(seed)
    C = make_matrix(rng, n, sym, kind)
    depot = draw(st.integers(0, n - 1))
    others = rng.permutation([v for v in range(n) if v != depot])
    tour = np.concatenate(([depot], others)).astype(np.int64)
    inst = {"C": C, "tour": tour, "n": n, "symmetric": sym, "depot": depot, "kind": kind}
    if with_time:
        T = make_matrix(rng, n, sym, kind)
        round_trip = max(T[depot, v] + T[v, depot] for v in range(n) if v != depot)
        inst["T"] = T
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
def instance_and_candidates(draw, symmetric=None, with_time=False, min_n=3):
    inst = draw(instances(symmetric=symmetric, with_time=with_time, min_n=min_n))
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
@given(instances(with_time=True), st.data())
def test_problem_with_service_time_feeds_the_kernels_the_folded_matrix(inst, data):
    """D32: ``RoutingProblem(service_time=s)`` prices, splits and times a tour exactly like the kernels on
    ``T_eff[i, j] = T[i, j] + s[j]`` (``j != depot``), ``T_eff[i, depot] = T[i, depot]``,
    ``T_eff[depot, j] += s[depot]`` — built here by hand."""
    from skroute import RoutingProblem

    C, T, tour, n = inst["C"], inst["T"], inst["tour"], inst["n"]
    mt, fc, d = inst["max_time"], inst["fixed_cost"], inst["depot"]
    assume(mt > 0.0)  # the all-zero kind gives a zero budget, which RoutingProblem rejects
    round_trip = max(T[d, v] + T[v, d] for v in range(n) if v != d)
    # services as fractions of the slack, so every round trip still fits with its service and the depot's
    slack = (mt - round_trip) / 2.0
    fractions = data.draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))
    service = np.array(fractions) * slack
    folded = T + service[None, :]
    folded[:, d] = T[:, d]
    folded[d, :] += service[d]
    np.fill_diagonal(folded, np.diagonal(T))
    folded = np.ascontiguousarray(folded)
    for split, name in ((GREEDY, "greedy"), (OPTIMAL, "optimal")):
        p = RoutingProblem(
            C, time_matrix=T, depot=d, max_time_work=mt, extra_cost=fc, service_time=service, split=name
        )
        assert np.array_equal(p.time_or_cost, folded) and np.array_equal(p.time, T)
        assert p.time_or_cost.flags["C_CONTIGUOUS"] and p.time_or_cost.dtype == np.float64
        assert p.service_time.tolist() == service.tolist()
        assert p.evaluate(tour) == core.problem_cost_py(C, folded, tour, mt, fc, split)
        out = np.empty(n + 1, dtype=np.int64)
        k = core.trip_starts(folded, tour, mt, split, C, fc, out)
        starts = p.trip_starts(tour)
        assert starts.tolist() == out[: k + 1].tolist()
        times = np.empty(k)
        core.trip_times(folded, tour, starts, times)
        assert p.trip_times(tour, starts).tolist() == times.tolist()
        assert np.all(times <= mt + 1e-9 * max(1.0, mt))
        # each trip's duration is its driving time plus the services of the nodes it visits (and the depot's)
        for t in range(k):
            trip = [d, *tour[starts[t] : starts[t + 1]].tolist(), d]
            driving = sum(T[a, b] for a, b in itertools.pairwise(trip))
            assert times[t] == pytest.approx(
                driving + service[trip[1:-1]].sum() + service[d], rel=1e-9, abs=1e-12 * max(1.0, mt)
            )


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
    assert core.two_opt_delta_py(C, tour, i, j) == pytest.approx(expected, abs=delta_tol(C, tour))
    assert core.two_opt_delta_asym_py(C, tour, i, j) == pytest.approx(expected, abs=delta_tol(C, tour))


@SETTINGS
@given(instance_and_two_opt(symmetric=False))
def test_two_opt_delta_asym_is_exact_on_asymmetric(args):
    inst, i, j = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.two_opt_delta_by_recompute(C, tour, i, j)
    assert core.two_opt_delta_asym_py(C, tour, i, j) == pytest.approx(expected, abs=delta_tol(C, tour))


@SETTINGS
@given(instance_and_or_opt(symmetric=False))
def test_or_opt_delta_forward_is_exact_on_asymmetric(args):
    inst, i, L, j, _ = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.or_opt_delta_by_recompute(C, tour, i, L, j, reverse=False)
    assert core.or_opt_delta_py(C, tour, i, L, j, False) == pytest.approx(expected, abs=delta_tol(C, tour))


@SETTINGS
@given(instance_and_or_opt(symmetric=True))
def test_or_opt_delta_both_orientations_are_exact_on_symmetric(args):
    inst, i, L, j, reverse = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.or_opt_delta_by_recompute(C, tour, i, L, j, reverse=reverse)
    assert core.or_opt_delta_py(C, tour, i, L, j, reverse) == pytest.approx(expected, abs=delta_tol(C, tour))


@SETTINGS
@given(instance_and_two_opt(symmetric=False))
def test_swap_delta_is_exact_on_asymmetric(args):
    inst, i, j = args
    C, tour = inst["C"], inst["tour"]
    expected = reference.swap_delta_by_recompute(C, tour, i, j)
    assert core.swap_delta_py(C, tour, i, j) == pytest.approx(expected, abs=delta_tol(C, tour))


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
@given(instances(symmetric=True, min_n=5, max_n=9), st.booleans())
def test_two_opt_descent_with_full_lists_and_cleared_bits_reaches_a_two_opt_local_optimum(inst, first):
    """With the complete neighbourhood and the don't-look bits cleared before every pass, the descent
    ends where no reversal improves: an improving reversal has a new edge shorter than the removed edge
    at one of its endpoints, and that endpoint's scan finds it. (A single call with persistent bits is
    only 2-opt-optimal up to the don't-look-bit approximation — see the kernel's Notes.)"""
    C, n = inst["C"], inst["n"]
    inst["cand"] = neighbours(C, n - 1)
    t, pos, dlb = descent_buffers(inst)
    for _ in range(10_000):
        dlb[:] = 0
        if core.two_opt_descent(C, t, pos, inst["cand"], dlb, first, 1) == 0.0:
            break
    cur = reference.tour_cost(C, t)
    for i, j in two_opt_pairs(n):
        assert reference.tour_cost(C, reference.two_opt_apply(t, i, j)) >= cur - 1e-9 * max(1.0, cur) - 1e-12


def assert_only_gap_closing_or_opt_moves_remain(C: np.ndarray, tour: np.ndarray, allow_reverse: bool) -> None:
    """The documented Or-opt neighbourhood: a move is found from any of its six endpoints except the two
    whose gap closes (p, q). So with full lists and cleared bits, an Or-opt move still improving at
    termination has none of its new edges at the segment ends or at the anchor shorter than the edge
    removed there — otherwise one of the four scans (segment start, segment end, anchor after, anchor
    before) would have applied it."""
    n = len(tour)
    cand = neighbours(C, n - 1)
    t = tour.copy()
    pos = np.empty(n, dtype=np.int64)
    core.rebuild_pos(t, pos)
    dlb = np.zeros(n, dtype=np.uint8)
    for _ in range(10_000):
        dlb[:] = 0
        if core.or_opt_descent(C, t, pos, cand, dlb, 3, allow_reverse, 1) == 0.0:
            break
    cur = reference.tour_cost(C, t)
    floor = cur - 1e-9 * max(1.0, cur) - 1e-12
    for i, L, j in or_opt_moves(n):
        for reverse in (False, True) if (allow_reverse and L > 1) else (False,):
            if reference.tour_cost(C, reference.or_opt_apply(t, i, L, j, reverse)) >= floor:
                continue
            p, s0, sL, q = t[i - 1], t[i], t[i + L - 1], t[(i + L) % n]
            c, d = t[j], t[(j + 1) % n]
            x, y = (sL, s0) if reverse else (s0, sL)  # new edges (c, x) and (y, d)
            removed_x, removed_y = (C[sL, q], C[p, s0]) if reverse else (C[p, s0], C[sL, q])
            assert C[c, x] >= C[c, d] and C[c, x] >= removed_x, (t.tolist(), i, L, j, reverse)
            assert C[y, d] >= C[c, d] and C[y, d] >= removed_y, (t.tolist(), i, L, j, reverse)


@SETTINGS
@given(instances(symmetric=True, min_n=5, max_n=10, kinds=("uniform", "euclidean")), st.booleans())
def test_or_opt_descent_with_full_lists_and_cleared_bits_misses_only_gap_closing_moves(inst, allow_reverse):
    assert_only_gap_closing_or_opt_moves_remain(inst["C"], inst["tour"], allow_reverse)


def test_or_opt_descent_finds_the_move_seen_only_from_the_anchor():
    """Regression: a segment-end scan alone cannot see this move (n = 10 Euclidean instance of the
    review): at the tour it converged to, node 5 belonged right after the depot — its new edge
    (depot, 5) is shorter than the depot's removed edge, but not shorter than the edges removed at 5.
    Only a scan from the anchor (the depot's own candidate list) finds it."""
    xy = np.array(
        [
            [45.501748315239986, 5.677436239291678],
            [99.53616594289969, 88.86993067083327],
            [91.6323934973403, 24.657553007363873],
            [39.411025472496796, 22.717950091335336],
            [12.49063821776657, 3.302392466573567],
            [50.3336447986407, 12.313365606638971],
            [17.630437136812716, 86.04756804576209],
            [48.42427686339313, 18.370352102024935],
            [66.98645598173123, 26.58648776948195],
            [52.693720005005716, 28.295286786234442],
        ]
    )
    C = np.ascontiguousarray(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)))
    start = np.array([0, 9, 4, 5, 3, 8, 2, 6, 7, 1], dtype=np.int64)
    assert_only_gap_closing_or_opt_moves_remain(C, start, allow_reverse=False)
    assert_only_gap_closing_or_opt_moves_remain(C, start, allow_reverse=True)


@SETTINGS
@given(instance_and_candidates(symmetric=True, min_n=4), st.integers(1, 3), st.booleans())
def test_descents_reset_the_bits_of_every_endpoint_of_an_applied_move(inst, max_segment, allow_reverse):
    """Don't-look bookkeeping (.pxd): after an applied move every node that lost a tour edge — the four
    endpoints of a reversal, the six of a segment move — has its bit active again. Only node n - 1 is
    active, so it is the only node processed in the single pass (a node reset by one of its moves is
    never revisited in that pass); its own bit is set when it is done. Nodes outside the net-removed
    edges may be active too (an edge removed by one move and restored by a later one)."""
    C, tour, n, cand = inst["C"], inst["tour"], inst["n"], inst["cand"]
    a = n - 1
    for kernel in ("two_opt", "or_opt"):
        t, pos, dlb = descent_buffers(inst)
        dlb[:] = 1
        dlb[a] = 0
        if kernel == "two_opt":
            gain = core.two_opt_descent(C, t, pos, cand, dlb, True, 1)
        else:
            gain = core.or_opt_descent(C, t, pos, cand, dlb, max_segment, allow_reverse, 1)
        assert dlb[a] == 1
        if gain == 0.0:
            assert np.array_equal(t, tour) and dlb.all()
            continue
        lost = set().union(*(edges(tour) - edges(t)))
        assert lost, "an applied move removes at least one edge"
        assert all(dlb[v] == 0 for v in lost if v != a), (kernel, tour.tolist(), t.tolist(), dlb.tolist())


@SETTINGS
@given(instance_and_candidates(symmetric=True, with_time=True), st.sampled_from([1e12, -3.0, 1e-9, 7.5]))
def test_kernels_never_read_the_diagonal(inst, diag):
    """SPEC §3.1: every kernel gives bit-identical results whatever finite value sits on the diagonal
    of C and T — including the greedy decoder on a D5-infeasible budget (first customer's round trip
    too long), whose closing leg at the depot is the depot's own diagonal entry."""
    C0, T0, tour, n, cand = inst["C"], inst["T"], inst["tour"], inst["n"], inst["cand"]
    mt, fc, d = inst["max_time"], inst["fixed_cost"], int(tour[0])
    C1, T1 = C0.copy(), T0.copy()
    np.fill_diagonal(C1, diag)
    np.fill_diagonal(T1, diag)
    bad = 0.5 * (T0[d, tour[1]] + T0[tour[1], d])  # below the first customer's round trip

    def run(C, T):
        out = {
            "tour": core.tour_cost_py(C, tour),
            "greedy": core.greedy_split_cost_py(C, T, tour, mt, fc),
            "optimal": core.optimal_split_cost_py(C, T, tour, mt, fc),
            "greedy_infeasible": core.greedy_split_cost_py(C, T, tour, bad, fc),
            "optimal_infeasible": core.optimal_split_cost_py(C, T, tour, bad, fc),
            "problem": [
                core.problem_cost_py(C, T, tour, m, fc, s) for m in (math.inf, mt, bad) for s in (GREEDY,)
            ]
            + [core.problem_cost_py(C, T, tour, mt, fc, OPTIMAL)],
            "two_opt": [core.two_opt_delta_py(C, tour, i, j) for i, j in two_opt_pairs(n)],
            "two_opt_asym": [core.two_opt_delta_asym_py(C, tour, i, j) for i, j in two_opt_pairs(n)],
            "swap": [core.swap_delta_py(C, tour, i, j) for i, j in two_opt_pairs(n)],
            "or_opt": [
                core.or_opt_delta_py(C, tour, i, L, j, r)
                for i, L, j in or_opt_moves(n)
                for r in (False, True)
            ],
        }
        for split in (GREEDY, OPTIMAL):
            starts = np.full(n + 1, -1, dtype=np.int64)
            k = core.trip_starts(T, tour, mt, split, C, fc, starts)
            costs, times = np.empty(k), np.empty(k)
            core.trip_costs(C, tour, starts[: k + 1], costs)
            core.trip_times(T, tour, starts[: k + 1], times)
            out[f"trips{split}"] = (k, starts.tolist(), costs.tolist(), times.tolist())
        starts = np.full(n + 1, -1, dtype=np.int64)
        out["trips_infeasible"] = (core.trip_starts(T, tour, bad, GREEDY, C, fc, starts), starts.tolist())
        nn = np.empty(n, dtype=np.int64)
        core.nearest_neighbour_tour(C, d, nn)
        out["nn"] = nn.tolist()
        for first in (True, False):
            t, pos, dlb = descent_buffers(inst)
            g = core.two_opt_descent(C, t, pos, cand, dlb, first, 1000)
            out[f"two_opt_descent{first}"] = (g, t.tolist(), pos.tolist(), dlb.tolist())
        t, pos, dlb = descent_buffers(inst)
        g = core.or_opt_descent(C, t, pos, cand, dlb, 3, True, 1000)
        out["or_opt_descent"] = (g, t.tolist(), pos.tolist(), dlb.tolist())
        for m, split in ((math.inf, GREEDY), (mt, GREEDY), (mt, OPTIMAL)):
            t, pos, _ = descent_buffers(inst)
            scratch_tour, dp, pred = scratch(n)
            g = core.local_search_generic(
                C, T, t, pos, cand, m, fc, split, 7, 3, 1000, scratch_tour, dp, pred
            )
            out[f"generic{m}{split}"] = (g, t.tolist(), pos.tolist())
        return out

    assert run(C1, T1) == run(C0, T0)


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
    # a depot-only tour is rejected too: tour_cost would read the diagonal and the two decoders of
    # trip_starts used to disagree on it (greedy k = 1, out = [1, 1]; optimal k = 0)
    for split in (GREEDY, OPTIMAL):
        with pytest.raises(ValueError, match="at least the depot and one customer"):
            core.trip_starts(
                C[:1, :1], np.zeros(1, dtype=np.int64), 5.0, split, C[:1, :1], 0.0, np.empty(2, np.int64)
            )
    with pytest.raises(ValueError, match="at least the depot and one customer"):
        core.problem_cost_py(C[:1, :1], C[:1, :1], np.zeros(1, dtype=np.int64), math.inf, 0.0, GREEDY)
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
    # the stub's __all__ mirrors the module's (mypy resolves ``core.__all__`` through the stub)
    stub_all = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets)
    )
    assert ast.literal_eval(stub_all.value) == list(core.__all__)
    enum_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SplitRule")
    assert [b.id for b in enum_cls.bases] == ["IntEnum"]  # type: ignore[attr-defined]


def test_every_public_name_is_documented_in_numpydoc():
    """mkdocstrings/help() show every public name: the enum has a docstring and every function a
    ``Parameters`` section (``embedsignature`` gives the signature, not the parameters' meaning)."""
    for name in core.__all__:
        doc = getattr(core, name).__doc__
        assert doc and doc.strip(), name
        if name != "SplitRule":
            assert "Parameters\n" in doc, name
    enum_doc = core.SplitRule.__doc__ or ""
    assert "SPLIT_GREEDY" in enum_doc and "SPLIT_OPTIMAL" in enum_doc


def test_docstring_examples_of_the_compiled_module_run():
    """``pytest --doctest-modules`` collects ``.py`` files only, so the examples embedded in the
    extension's docstrings are executed here."""
    result = doctest.testmod(core, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
