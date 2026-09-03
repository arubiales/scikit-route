"""Acceptance tests of the construction heuristics (SPEC §4.2): ``NearestNeighbour``, ``Insertion``,
``ClarkeWright`` and ``NRBS``. The structural battery (checks 1-11, 13) and the tolerance tests run in
``tests/test_common.py``; here live the algorithm-specific facts: pure-Python restatements of every
rule (exercised on tie-heavy integer matrices as well as on float ones, so the tie rules are pinned),
the crafted instances whose answer is known by hand, the 1.0 NRBS regression pin and the multi-trip
behaviour of ClarkeWright (savings trips checked as driven, with the decoder's arithmetic)."""

from __future__ import annotations

import json
import math
import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import reference
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import NRBS, ClarkeWright, Insertion, NearestNeighbour, RoutingProblem
from skroute.construction._clarke_wright import _closed_duration, savings_tour, savings_trips
from skroute.construction._insert import STRATEGIES, insertion_tour
from skroute.construction._nrbs import nrbs_tour, row_stats
from skroute.metrics import route_cost

PIN = Path(__file__).parent / "data" / "nrbs_barcelona_1_0.json"
STRATEGY_NAMES = sorted(STRATEGIES)
BUDGET_UNAWARE = [NearestNeighbour, Insertion, NRBS]
INSTANCES = {"float": None, "int": None}  # filled below: distinct distances vs. ties everywhere


# --------------------------------------------------------------------------- helpers
def _random_instance(n, seed, asymmetric=False):
    """Random Euclidean matrix (distinct distances almost surely), optionally made asymmetric."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    C = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    if asymmetric:
        C = C * rng.uniform(0.7, 1.3, C.shape)
        np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C)


def _integer_instance(n, seed, asymmetric=False):
    """Small integer costs: many exact ties, to exercise the tie rules."""
    rng = np.random.default_rng(seed)
    C = rng.integers(1, 5, size=(n, n)).astype(float)
    if not asymmetric:
        C = np.triu(C, 1) + np.triu(C, 1).T
    np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C)


INSTANCES.update({"float": _random_instance, "int": _integer_instance})


def _asymmetric_time(C, seed, spread=0.4):
    """A time matrix proportional to ``C`` with every arc perturbed independently: asymmetric."""
    rng = np.random.default_rng(seed)
    T = C / 10.0 * rng.uniform(1 - spread, 1 + spread, C.shape)
    np.fill_diagonal(T, 0.0)
    return np.ascontiguousarray(T)


def _is_tour(tour, n, depot):
    tour = np.asarray(tour)
    return tour.shape == (n,) and tour[0] == depot and sorted(tour.tolist()) == list(range(n))


def _driving_duration(T, depot, trip):
    """The greedy decoder's sum for one trip (``reference.greedy_split``): legs accumulated from the depot."""
    t = T[depot, trip[0]]
    for a, b in pairwise(trip):
        t += T[a, b]
    return t + T[trip[-1], depot]


def _nearest_neighbour_reference(C, depot):
    """SPEC §4.2: from the depot, always the closest unvisited node, ties by lowest index."""
    n = C.shape[0]
    tour, visited = [depot], {depot}
    while len(tour) < n:
        cur, best = tour[-1], None
        for j in range(n):
            if j not in visited and (best is None or C[cur, j] < C[cur, best]):
                best = j
        tour.append(best)
        visited.add(best)
    return tour


def _insertion_reference(C, depot, strategy):
    """O(n^3) restatement of §4.2 with the kernel's tie rules: selection ties to the lowest index
    (Python's max/min return the first extremum), position ties to the first edge met from the depot."""
    n = C.shape[0]
    others = [j for j in range(n) if j != depot]
    pick = max if strategy == "farthest" else min
    tour = [depot, pick(others, key=lambda j: C[depot, j])]

    def best_edge(j):
        best, pos = math.inf, None
        for k in range(len(tour)):
            a, b = tour[k], tour[(k + 1) % len(tour)]
            c = C[a, j] + C[j, b] - C[a, b]
            if c < best:
                best, pos = c, k
        return best, pos

    while len(tour) < n:
        unrouted = [j for j in range(n) if j not in tour]
        if strategy == "farthest":
            j = max(unrouted, key=lambda j: min(C[i, j] for i in tour))
        elif strategy == "nearest":
            j = min(unrouted, key=lambda j: min(C[i, j] for i in tour))
        else:
            j = min(unrouted, key=lambda j: best_edge(j)[0])
        _, pos = best_edge(j)
        tour.insert(pos + 1, j)
    return tour


def _savings_reference(C, depot, shape=1.0, T=None, max_time=math.inf):
    """§4.2 restated with lists, O(n^3): savings ``C[d,i] + C[j,d] - shape*C[i,j]`` descending (ties by
    ``(i, j)``); merge two trips at their endpoints when the merged trip — oriented with the endpoint
    nearer to the depot first (ties: lower index) and summed leg by leg from the depot — fits the
    budget, else in the reverse orientation if only that one fits; the merged trip keeps the smaller
    creation index; the trips are returned by creation index, as oriented."""
    n = C.shape[0]
    d = depot
    nodes = [k for k in range(n) if k != d]

    def orient(p):
        first, last = p[0], p[-1]
        return p[::-1] if C[d, last] < C[d, first] or (C[d, last] == C[d, first] and last < first) else p

    pairs = sorted(
        ((i, j) for x, i in enumerate(nodes) for j in nodes[x + 1 :]),
        key=lambda ij: (-(C[d, ij[0]] + C[ij[1], d] - shape * C[ij[0], ij[1]]), ij),
    )
    trips = {k: [k] for k in nodes}
    for i, j in pairs:
        ti = next(k for k, p in trips.items() if i in p)
        tj = next(k for k, p in trips.items() if j in p)
        if ti == tj:
            continue
        A, B = trips[ti], trips[tj]
        if i not in (A[0], A[-1]) or j not in (B[0], B[-1]):
            continue  # an interior node
        if A[-1] != i:
            A = A[::-1]
        if B[0] != j:
            B = B[::-1]
        merged = orient(A + B)
        if math.isfinite(max_time) and _driving_duration(T, d, merged) > max_time:
            merged = merged[::-1]
            if _driving_duration(T, d, merged) > max_time:
                continue
        keep, drop = min(ti, tj), max(ti, tj)
        trips[keep] = merged
        del trips[drop]
    return [trips[k] for k in sorted(trips)]


def _row_stats_reference(C):
    """The 2020 loops as Python <= 3.11 ran them: left-to-right float sums and the float ``**``.

    1.0 used the builtin ``sum()`` for the mean; since Python 3.12 ``sum()`` compensates (Neumaier),
    so the sequential loop is spelled out here to state the 2020 arithmetic exactly.
    """
    n = C.shape[0]
    means, stds = [], []
    for i in range(n):
        values = [float(v) for v in C[i]]
        total = 0
        for v in values:
            total += v
        mean = total / n
        acc = 0
        for v in values:
            acc += (v - mean) ** 2
        means.append(mean)
        stds.append((acc / n) ** 0.5)
    return np.array(means), np.array(stds)


def _edges(tour):
    """Undirected edge set of a closed tour."""
    tour = list(tour)
    return {frozenset((tour[k], tour[(k + 1) % len(tour)])) for k in range(len(tour))}


# --------------------------------------------------------------------------- NearestNeighbour
def test_nearest_neighbour_matches_reference_on_tiny(tiny_instance):
    C = tiny_instance["C"]
    est = NearestNeighbour().fit(C)
    assert est.tour_.tolist() == _nearest_neighbour_reference(C, 0)
    assert est.cost_ == pytest.approx(reference.tour_cost(C, est.tour_))
    assert est.cost_ / tiny_instance["optimum"] - 1 <= 0.50


@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("depot", [0, 3, 9])
def test_nearest_neighbour_ties_go_to_the_lowest_index(asymmetric, depot):
    C = _integer_instance(10, seed=7, asymmetric=asymmetric)  # costs in 1..4: ties everywhere
    est = NearestNeighbour().fit(C, depot=depot)
    assert est.tour_.tolist() == _nearest_neighbour_reference(C, depot)
    assert int(est.route_[0]) == int(est.route_[-1]) == depot


def test_nearest_neighbour_reproduces_the_measured_baselines(fast_instance):
    """The §4.2 baselines were measured with exactly this tie rule: 31.8 % on wi29, 46.4 % on dj38."""
    measured = {"wi29": 0.318, "dj38": 0.464}
    est = NearestNeighbour().fit(fast_instance["C"], labels=fast_instance["labels"])
    gap = est.cost_ / fast_instance["optimum"] - 1
    assert gap == pytest.approx(measured[fast_instance["name"]], abs=5e-4)


def test_nearest_neighbour_has_no_parameters():
    est = NearestNeighbour()
    assert repr(est) == "NearestNeighbour()" and est.get_params() == {}
    assert not est._get_tags().budget_aware and est._get_tags().kind == "construction"


# --------------------------------------------------------------------------- Insertion
@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_insertion_matches_reference_on_tiny(tiny_instance, strategy):
    C = tiny_instance["C"]
    est = Insertion(strategy=strategy).fit(C)
    assert est.tour_.tolist() == _insertion_reference(C, 0, strategy)
    assert est.cost_ == pytest.approx(route_cost(C, est.route_))
    if strategy != "nearest":  # the two strategies of the tolerance table (§6)
        assert est.cost_ / tiny_instance["optimum"] - 1 <= 0.30


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
@pytest.mark.parametrize("kind", sorted(INSTANCES))
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("depot", [0, 5, 19])
def test_insertion_matches_reference_from_any_depot(strategy, kind, asymmetric, depot):
    C = INSTANCES[kind](20, seed=20 + depot, asymmetric=asymmetric)
    tour = insertion_tour(C, depot, strategy)
    assert tour.dtype == np.int64 and _is_tour(tour, 20, depot)
    assert tour.tolist() == _insertion_reference(C, depot, strategy)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_insertion_position_ties_go_to_the_first_edge_from_the_depot(strategy, asymmetric):
    """Costs in 1..4: insertion costs tie constantly. The kernel's incremental cache (cheapest) must
    still answer the first minimum-cost edge met walking from the depot, exactly like the O(n^3) rule.
    Regression: the cache used to keep its cached edge on an exact tie with a newly created edge that
    came earlier in the walk (n=15, seed=45, depot=14: a 24.0 tour instead of the rule's 23.0)."""
    for n, seed, depot in ((15, 45, 14), (13, 0, 0), *((n, s, s % n) for n in (6, 9, 13) for s in range(30))):
        C = _integer_instance(n, seed=seed, asymmetric=asymmetric)
        assert insertion_tour(C, depot, strategy).tolist() == _insertion_reference(C, depot, strategy), (
            n,
            seed,
            depot,
        )


def test_insertion_is_direction_aware():
    """On an asymmetric matrix the insertion costs are read in driving direction: the tour of the
    kernel prices every insertion with C[a, j] + C[j, b] - C[a, b], never with the transposed arcs."""
    C = _random_instance(9, seed=99, asymmetric=True)
    forward = Insertion().fit(C)
    assert forward.cost_ == pytest.approx(reference.tour_cost(C, forward.tour_))
    # the reference restated with directional costs agrees; the transposed matrix is a different instance
    assert forward.tour_.tolist() == _insertion_reference(C, 0, "farthest")
    backward = Insertion().fit(np.ascontiguousarray(C.T))
    assert backward.cost_ == pytest.approx(reference.tour_cost(C.T, backward.tour_))


def test_insertion_fast_tier_gaps(fast_instance):
    """Measured: farthest 1.9 % / 0.0 %, cheapest 10.0 % / 17.7 %, nearest 10.0 % / 29.2 % on wi29 / dj38."""
    C, labels, opt = fast_instance["C"], fast_instance["labels"], fast_instance["optimum"]
    bounds = {"farthest": 0.25, "cheapest": 0.30, "nearest": 0.35}
    for strategy, bound in bounds.items():
        est = Insertion(strategy=strategy).fit(C, labels=labels)
        assert opt <= est.cost_ + 1e-9
        assert est.cost_ / opt - 1 <= bound, f"{strategy} on {fast_instance['name']}"


def test_insertion_rejects_unknown_strategy():
    C = _random_instance(5, seed=5)
    with pytest.raises(ValueError, match="The 'strategy' parameter of Insertion must be a str among"):
        Insertion(strategy="both").fit(C)
    with pytest.raises(ValueError, match="strategy must be 'farthest', 'cheapest' or 'nearest'"):
        insertion_tour(C, 0, "both")
    with pytest.raises(ValueError, match="depot must be in"):
        insertion_tour(C, 5, "farthest")
    with pytest.raises(ValueError, match=r"got shape \(3, 4\)$"):  # the real shape, not the memoryview's
        insertion_tour(np.ones((3, 4)), 0, "farthest")
    with pytest.raises(ValueError, match=r"at least 2 nodes, got shape \(1, 1\)$"):
        insertion_tour(np.zeros((1, 1)), 0, "farthest")


def test_insertion_default_and_repr():
    assert Insertion().strategy == "farthest" and repr(Insertion()) == "Insertion()"
    assert repr(Insertion(strategy="cheapest")) == "Insertion(strategy='cheapest')"


@settings(derandomize=True, deadline=None, max_examples=150)
@given(
    n=st.integers(3, 12),
    seed=st.integers(0, 10_000),
    asymmetric=st.booleans(),
    kind=st.sampled_from(sorted(INSTANCES)),
    strategy=st.sampled_from(STRATEGY_NAMES),
    depot_seed=st.integers(0, 100),
)
def test_insertion_kernel_always_returns_a_tour(n, seed, asymmetric, kind, strategy, depot_seed):
    C = INSTANCES[kind](n, seed, asymmetric)
    depot = depot_seed % n
    tour = insertion_tour(C, depot, strategy)
    assert _is_tour(tour, n, depot)
    assert tour.tolist() == _insertion_reference(C, depot, strategy)


# --------------------------------------------------------------------------- ClarkeWright
def test_clarke_wright_refuses_asymmetric_matrices():
    C = _random_instance(6, seed=6, asymmetric=True)
    with pytest.raises(ValueError, match="ClarkeWright requires a symmetric cost matrix"):
        ClarkeWright().fit(C)
    tags = ClarkeWright()._get_tags()
    assert tags.requires_symmetric and tags.budget_aware and tags.kind == "construction"


def test_clarke_wright_hand_checked_plain_tsp():
    """Savings (2,3)=16, (1,2)=10, (1,3)=7: merge [2,3], then [1,2,3]; 1 is nearer the depot -> [0,1,2,3]."""
    C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    est = ClarkeWright().fit(C)
    assert est.tour_.tolist() == [0, 1, 2, 3] and est.cost_ == 22.0 and est.n_trips_ == 1


def test_clarke_wright_hand_checked_multi_trip():
    """The §3.4 example: [2,3] fits (4 h), [2,3]+[4] (5 h) and [3]+[4] (5 h) do not -> two trips, 41.0."""
    cost = {
        1: {1: 0, 2: 5, 3: 9, 4: 10},
        2: {1: 5, 2: 0, 3: 4, 4: 8},
        3: {1: 9, 2: 4, 3: 0, 4: 3},
        4: {1: 10, 2: 8, 3: 3, 4: 0},
    }
    hours = {
        1: {1: 0, 2: 1, 3: 2, 4: 2},
        2: {1: 1, 2: 0, 3: 1, 4: 2},
        3: {1: 2, 2: 1, 3: 0, 4: 1},
        4: {1: 2, 2: 2, 3: 1, 4: 0},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # budget-aware: no UserWarning
        est = ClarkeWright().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
    assert est.route_.tolist() == [1, 2, 3, 1, 4, 1] and est.cost_ == 41.0 and est.n_trips_ == 2
    assert est.trip_times_.tolist() == [4.0, 4.0]
    assert est.cost_ == pytest.approx(
        reference.brute_force(
            np.array([[cost[i][j] for j in cost] for i in cost], float),
            np.array([[hours[i][j] for j in hours] for i in hours], float),
            max_time_work=4.0,
            extra_cost=3.0,
        )[0]
    )


def test_clarke_wright_orientation_by_nearer_endpoint_and_budget_refusal():
    """Only [1,2] fits the budget; its endpoint 2 is nearer the depot -> giant tour [0, 2, 1, 3]."""
    C = np.array([[0, 6, 5, 6], [6, 0, 1, 9], [5, 1, 0, 9], [6, 9, 9, 0]], dtype=float)
    assert savings_tour(C, 0, 1.0, T=C, max_time=12.5).tolist() == [0, 2, 1, 3]
    assert savings_tour(C, 0, 1.0).tolist() == [0, 2, 1, 3]  # no budget: [2,1]+[3] merges, 2 still first
    est = ClarkeWright().fit(C, time_matrix=C, max_time_work=12.5)
    assert est.route_.tolist() == [0, 2, 1, 0, 3, 0] and est.cost_ == 24.0


def test_clarke_wright_orientation_ties_go_to_the_smaller_index():
    """Savings (1,4)=(2,3)=9 > (3,4)=8: the path is built as [2,3,4,1]; C[d,1] == C[d,2] -> 1 first."""
    C = np.array(
        [[0, 5, 5, 5, 5], [5, 0, 9, 9, 1], [5, 9, 0, 1, 9], [5, 9, 1, 0, 2], [5, 1, 9, 2, 0]], dtype=float
    )
    assert savings_tour(C, 0, 1.0).tolist() == [0, 1, 4, 3, 2]
    assert ClarkeWright().fit(C).cost_ == 14.0


def test_clarke_wright_merges_everything_without_a_budget():
    for n, seed in ((6, 1), (15, 2), (40, 3)):
        C = _random_instance(n, seed)
        est = ClarkeWright().fit(C, depot=n // 2)
        assert est.n_trips_ == 1 and _is_tour(est.problem_.to_index_tour(est.tour_), n, n // 2)


def test_clarke_wright_alicante_under_greedy_split(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = ClarkeWright().fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
    opt = alicante["optimum"]["greedy"]
    assert opt <= est.cost_ + 1e-9
    assert est.cost_ / opt - 1 <= 0.25
    assert est.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, est.problem_.to_index_tour(est.tour_), kw["max_time_work"], 20.0
        )
    )
    # the savings trips are re-decoded: the optimal split of the same tour is never dearer
    p_opt = RoutingProblem(d.cost, time_matrix=d.time, split="optimal", **kw)
    assert p_opt.evaluate(est.problem_.to_index_tour(est.tour_)) <= est.cost_ + 1e-9


@pytest.mark.parametrize("kind", sorted(INSTANCES))
@pytest.mark.parametrize("time", ["none", "sym", "asym"])
def test_clarke_wright_matches_reference(kind, time):
    """savings_trips is the literal §4.2 rule (sorted savings, endpoint merges, smaller creation index,
    orientation, budget as driven), on float and tie-heavy integer matrices, with no budget, a symmetric
    and an asymmetric time matrix; savings_tour is the depot followed by the trips."""
    for n, seed in ((4, 1), (7, 2), (12, 3), (20, 4), (25, 5)):
        C = INSTANCES[kind](n, seed=seed)
        depot = seed % n
        if time == "none":
            T, budgets = None, [math.inf]
        else:
            T = C / 10.0 if time == "sym" else _asymmetric_time(C, seed)
            round_trip = float((T[depot, :] + T[:, depot]).max())
            budgets = [factor * round_trip for factor in (1.0, 1.3, 2.0)]
        for shape in (1.0, 0.5):
            for budget in budgets:
                trips = savings_trips(C, depot, shape, T=T, max_time=budget)
                expected = _savings_reference(C, depot, shape, T=T, max_time=budget)
                assert trips == expected, (n, seed, shape, budget)
                tour = savings_tour(C, depot, shape, T=T, max_time=budget)
                assert tour.tolist() == [depot, *(v for trip in trips for v in trip)]
                assert _is_tour(tour, n, depot)


def test_clarke_wright_budget_respected_by_savings_trips(alicante):
    """Every savings trip fits the budget on its own, in driving direction and with the decoder's own
    sum; on a metric time matrix (alicante's is) the greedy decoder can then only merge them further,
    so n_trips_ <= number of savings trips."""
    d, kw = alicante["bunch"], alicante["kwargs"]
    budget = kw["max_time_work"]
    trips = savings_trips(d.cost, 0, 1.0, T=d.time, max_time=budget)
    assert len(trips) > 1 and sorted(v for trip in trips for v in trip) == list(range(1, d.cost.shape[0]))
    assert all(_driving_duration(d.time, 0, trip) <= budget for trip in trips)
    est = ClarkeWright().fit(d.cost, time_matrix=d.time, **kw)
    assert est.problem_.to_index_tour(est.tour_).tolist() == [0, *(v for trip in trips for v in trip)]
    assert est.n_trips_ <= len(trips)
    # the same on random metric instances, at three budgets each
    for seed in range(30):
        C = _random_instance(10, seed=seed)
        T = C / 10.0
        for factor in (1.0, 1.3, 2.0):
            budget = factor * float((T[0, :] + T[:, 0]).max())
            trips = savings_trips(C, 0, 1.0, T=T, max_time=budget)
            assert all(_driving_duration(T, 0, trip) <= budget for trip in trips)
            est = ClarkeWright().fit(C, time_matrix=T, max_time_work=budget)
            assert est.n_trips_ <= len(trips), (seed, factor)


def test_clarke_wright_asymmetric_time_keeps_the_orientation_that_fits():
    """Symmetric costs, asymmetric hours: the trip [1, 2, 3] lasts 4 h as built and 8 h reversed. The
    orientation rule prefers 3 first (C[0, 3] = 5 < C[0, 1] = 7) but that direction breaks the 6 h
    budget, so the trip is kept as built and driven as one trip. Regression: the merge used to be
    checked in build orientation and the trip emitted reversed, which the decoder then split
    (n_trips_ = 2, cost 124 with extra_cost=100, instead of one trip costing 14.5)."""
    C = np.array([[0, 7, 6, 5], [7, 0, 1, 9], [6, 1, 0, 1.5], [5, 9, 1.5, 0]])
    T = np.array([[0, 1, 1, 5], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)
    assert savings_trips(C, 0, 1.0, T=T, max_time=6.0) == [[1, 2, 3]]
    est = ClarkeWright().fit(C, time_matrix=T, max_time_work=6.0, extra_cost=100.0)
    assert est.n_trips_ == 1 and est.route_.tolist() == [0, 1, 2, 3, 0] and est.cost_ == 14.5
    assert savings_trips(C, 0, 1.0) == [[3, 2, 1]]  # without a budget the nearer endpoint goes first
    # a random instance where only the reverse of the preferred orientation fits (n=6, seed=2)
    C = _random_instance(6, seed=2)
    T = _asymmetric_time(C, 2)
    budget = 1.6 * float((T[0, :] + T[:, 0]).max())
    trips = savings_trips(C, 0, 1.0, T=T, max_time=budget)
    (long_trip,) = [trip for trip in trips if len(trip) > 1]
    assert _driving_duration(T, 0, long_trip) <= budget < _driving_duration(T, 0, long_trip[::-1])
    assert C[0, long_trip[-1]] < C[0, long_trip[0]]  # the reverse would have been preferred by C[d, .]
    # in general: every savings trip fits as driven, and the decoder never splits one (T stays metric
    # enough at +-40 % for these instances: asserted, not assumed)
    for seed in range(60):
        C = _random_instance(8, seed=seed)
        T = _asymmetric_time(C, seed)
        budget = 1.6 * float((T[0, :] + T[:, 0]).max())
        trips = savings_trips(C, 0, 1.0, T=T, max_time=budget)
        assert all(_driving_duration(T, 0, trip) <= budget for trip in trips), seed
        est = ClarkeWright().fit(C, time_matrix=T, max_time_work=budget)
        assert np.all(est.trip_times_ <= budget)


def test_clarke_wright_budget_equal_to_a_trip_duration_is_still_one_trip():
    """The merge test uses the decoder's arithmetic (legs accumulated from the depot in driving
    direction), so a budget copied from a trip's own duration keeps that trip whole. Regression: the
    savings phase summed ``T[d, head] + path + T[a, b] + path + T[tail, d]`` in build orientation, one
    ulp away from the decoder's sum of the reversed trip, and ``[1, 5]`` was split (6 trips, 548.87)."""
    C = _random_instance(7, seed=2)
    rng = np.random.default_rng(2)
    T = rng.uniform(0.1, 1.0, (7, 7))
    T = (T + T.T) / 2
    np.fill_diagonal(T, 0.0)
    budget = 1.3970773560739378
    trips = savings_trips(C, 0, 1.0, T=T, max_time=budget)
    assert trips == [[1, 5], [2], [3], [4], [6]]
    est = ClarkeWright().fit(C, time_matrix=T, max_time_work=budget, extra_cost=10.0)
    assert est.n_trips_ == 5 and est.route_.tolist() == [0, 1, 5, 0, 2, 0, 3, 0, 4, 0, 6, 0]
    # in general: set the budget to the longest savings trip's duration, ulp for ulp
    for seed in range(40):
        C = _random_instance(9, seed=seed)
        T = C / 10.0
        loose = 1.5 * float((T[0, :] + T[:, 0]).max())
        tight = max(_driving_duration(T, 0, trip) for trip in savings_trips(C, 0, 1.0, T=T, max_time=loose))
        trips = savings_trips(C, 0, 1.0, T=T, max_time=tight)
        assert all(_driving_duration(T, 0, trip) <= tight for trip in trips)
        p = RoutingProblem(C, time_matrix=T, depot=0, max_time_work=tight)
        tour = np.array([0, *(v for trip in trips for v in trip)])
        assert len(p.trip_starts(tour)) - 1 <= len(trips), seed


def test_clarke_wright_decoder_may_split_a_savings_trip_on_a_non_metric_time_matrix():
    """When T violates the triangle inequality a prefix of a feasible trip may not be able to return
    to the depot in time, and the greedy decoder splits it: n_trips_ can exceed the number of savings
    trips (documented). Every reported trip still fits."""
    C = _random_instance(7, seed=35)
    rng = np.random.default_rng(35)
    T = rng.uniform(0.5, 3.0, (7, 7))
    T = (T + T.T) / 2
    np.fill_diagonal(T, 0.0)
    budget = 1.6 * float((T[0, :] + T[:, 0]).max())
    trips = savings_trips(C, 0, 1.0, T=T, max_time=budget)
    assert trips == [[5, 6, 1, 4], [3, 2]]
    assert all(_driving_duration(T, 0, trip) <= budget for trip in trips)
    est = ClarkeWright().fit(C, time_matrix=T, max_time_work=budget)
    assert est.n_trips_ == 3 > len(trips)
    assert np.all(est.trip_times_ <= budget + 1e-9)


def test_clarke_wright_asymmetric_time_matrix_with_symmetric_cost(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    rng = np.random.default_rng(3)
    T = np.ascontiguousarray(d.time * rng.uniform(0.9, 1.1, d.time.shape))
    np.fill_diagonal(T, 0.0)
    budget = 1.5 * float((T[0, :] + T[:, 0]).max())
    est = ClarkeWright().fit(
        d.cost, time_matrix=T, labels=kw["labels"], depot=kw["depot"], max_time_work=budget
    )
    assert np.all(est.trip_times_ <= budget + 1e-9) and est.n_trips_ >= 1


def test_clarke_wright_shape_parameter():
    C = _random_instance(25, seed=25)
    tours = {shape: ClarkeWright(shape=shape).fit(C).tour_.tolist() for shape in (0.0, 0.5, 1.0, 2.0)}
    assert all(_is_tour(t, 25, 0) for t in tours.values())
    assert len({tuple(t) for t in tours.values()}) > 1, "shape must influence the savings order"
    with pytest.raises(
        ValueError, match="The 'shape' parameter of ClarkeWright must be a float in the range"
    ):
        ClarkeWright(shape=-0.1).fit(C)
    with pytest.raises(ValueError, match="needs the time matrix"):
        savings_tour(C, 0, 1.0, T=None, max_time=5.0)


@settings(derandomize=True, deadline=None, max_examples=100)
@given(
    n=st.integers(3, 12),
    seed=st.integers(0, 10_000),
    factor=st.floats(1.0, 3.0),
    depot_seed=st.integers(0, 100),
)
def test_clarke_wright_kernel_always_returns_a_tour(n, seed, factor, depot_seed):
    C = _random_instance(n, seed)
    depot = depot_seed % n
    T = C / 10.0
    budget = factor * float((T[depot, :] + T[:, depot]).max())
    assert _is_tour(savings_tour(C, depot, 1.0), n, depot)
    tour = savings_tour(C, depot, 1.0, T=T, max_time=budget)
    assert _is_tour(tour, n, depot)
    trips = savings_trips(C, depot, 1.0, T=T, max_time=budget)
    assert tour.tolist() == [depot, *(v for trip in trips for v in trip)]
    assert all(_closed_duration(T, depot, trip) <= budget for trip in trips)
    p = RoutingProblem(C, time_matrix=T, depot=depot, max_time_work=budget)
    assert len(p.trip_starts(tour)) - 1 <= len(trips)  # metric T: the decoder only merges further
    assert np.all(p.trip_times(tour, p.trip_starts(tour)) <= budget + 1e-9)


# --------------------------------------------------------------------------- NRBS
def test_nrbs_reproduces_the_1_0_barcelona_result(barcelona):
    """The plain tour SEQUENCE and cost of 1.0.0a2 (commit 533f320, float64 tour_cost), all exponents 0.5."""
    pin = json.loads(PIN.read_text())
    assert {"cost", "route", "provenance"} <= set(pin)
    assert "533f320" in pin["provenance"]["source"]
    est = NRBS(0.5, 0.5, 0.5, 0.5, 0.5).fit(barcelona.cost, labels=barcelona.labels, depot=barcelona.depot)
    assert est.route_.tolist() == pin["route"]
    assert est.cost_ == pytest.approx(pin["cost"], abs=0.01)
    assert est.cost_ == pytest.approx(route_cost(barcelona.cost, est.route_, labels=barcelona.labels))


def test_nrbs_row_stats_are_the_2020_arithmetic():
    for n, seed in ((19, 1), (37, 2), (64, 3)):
        C = _random_instance(n, seed)
        mean, std = row_stats(C)
        ref_mean, ref_std = _row_stats_reference(C)
        assert np.array_equal(mean, ref_mean) and np.array_equal(std, ref_std)


def test_nrbs_valid_on_tiny(tiny_instance):
    C = tiny_instance["C"]
    est = NRBS().fit(C)
    assert _is_tour(est.problem_.to_index_tour(est.tour_), tiny_instance["n"], 0)
    assert est.cost_ == pytest.approx(reference.tour_cost(C, est.tour_))
    assert tiny_instance["optimum"] <= est.cost_ + 1e-9


def test_nrbs_defaults_accept_ints_and_reject_negatives():
    C = _random_instance(12, seed=12)
    assert repr(NRBS()) == "NRBS()"
    assert repr(NRBS(distance_weight=2.0)) == "NRBS(distance_weight=2.0)"
    by_int, by_float = NRBS(1, 1, 1, 1, 1).fit(C), NRBS().fit(C)  # 1.0 rejected ints
    assert by_int.tour_.tolist() == by_float.tour_.tolist()
    assert NRBS().get_params() == dict.fromkeys(
        ("mean_priority", "std_priority", "mean_connection", "std_connection", "distance_weight"), 1.0
    )
    with pytest.raises(
        ValueError, match="The 'std_connection' parameter of NRBS must be a float in the range"
    ):
        NRBS(std_connection=-1.0).fit(C)  # 1.0.0a2 validated only the type: negatives were accepted there


def test_nrbs_depot_only_rotates_the_cycle():
    """The depot participates like any node: another depot gives the same undirected cycle."""
    C = _random_instance(15, seed=15)
    base = NRBS().fit(C)
    for depot in (4, 14):
        est = NRBS().fit(C, depot=depot)
        assert int(est.tour_[0]) == depot and _edges(est.tour_) == _edges(base.tour_)
        assert est.cost_ == pytest.approx(base.cost_)


def test_nrbs_coincident_points_are_linked_first():
    """C[i, j] == 0 gives the maximum connection score (no inf/nan): the twins end up adjacent."""
    xy = np.random.default_rng(8).random((10, 2)) * 100
    xy[7] = xy[3]  # node 7 is a copy of node 3
    C = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    tour = nrbs_tour(np.ascontiguousarray(C), 0)
    assert _is_tour(tour, 10, 0)
    assert frozenset((3, 7)) in _edges(tour)
    est = NRBS().fit(C)
    assert math.isfinite(est.cost_)


def test_nrbs_two_passes_always_close_the_cycle():
    """Whatever the ties, two passes leave every node with two neighbours (a node with fewer always finds
    an endpoint of another path, or the other end of the Hamiltonian path): all-zero matrices, costs in
    0..2, a zero row and column, half the points coincident — sym and asym — all give a permutation."""
    exponents = [(1.0,) * 5, (0.0,) * 5, (0.5,) * 5, (0.0, 0.0, 0.0, 0.0, 1.0)]
    for n in (3, 4, 5, 7, 9, 12):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            cases = [np.zeros((n, n)), rng.integers(0, 3, (n, n)).astype(float)]
            cases.append(np.triu(cases[1], 1) + np.triu(cases[1], 1).T)
            zero_line = np.ones((n, n))
            zero_line[rng.integers(0, n), :] = 0.0
            zero_line[:, rng.integers(0, n)] = 0.0
            cases.append(zero_line)
            xy = rng.random((n, 2))
            xy[: n // 2] = xy[0]
            cases.append(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)))
            for C in cases:
                np.fill_diagonal(C, 0.0)
                C = np.ascontiguousarray(C)
                for params in exponents:
                    assert _is_tour(nrbs_tour(C, seed % n, *params), n, seed % n), (n, seed, params)


def test_nrbs_rejects_nan_scores_and_accepts_infinite_ones():
    """mu^a of a negative mean (negative costs, fractional exponent) and inf / inf (costs so large that
    the powers overflow) are NaN: a ValueError, never a self-loop in the candidate lists. Negative
    costs with integer exponents and overflow to plain inf are fine and give a valid tour, quietly."""
    rng = np.random.default_rng(3)
    C = rng.uniform(-5, 5, (8, 8))
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0.0)
    C = np.ascontiguousarray(C)
    with pytest.raises(ValueError, match="well-defined node statistics"):
        NRBS(0.5, 0.5, 0.5, 0.5, 0.5).fit(C)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = NRBS().fit(C)  # integer exponents: negative means are fine
    assert _is_tour(est.problem_.to_index_tour(est.tour_), 8, 0)
    huge = _random_instance(6, seed=0) * 1e160
    with pytest.raises(ValueError, match="overflow to NaN"):
        NRBS(1.0, 1.0, 2.0, 1.0, 2.0).fit(huge)
    with pytest.raises(ValueError, match="overflow to NaN"):
        nrbs_tour(_random_instance(6, seed=0) * 1e200, 0, 2.0, 2.0, 2.0, 2.0, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for scale in (1e160, 1e300):  # inf scores sort first; no RuntimeWarning leaks
            assert _is_tour(nrbs_tour(_random_instance(6, seed=0) * scale, 0), 6, 0)


def test_nrbs_tour_below_three_nodes():
    """Below the estimator's minimum the kernel returns the only tour there is (no two-node 'cycle')."""
    assert nrbs_tour(np.array([[0.0, 1.0], [1.0, 0.0]]), 1).tolist() == [1, 0]
    assert nrbs_tour(np.zeros((1, 1)), 0).tolist() == [0]


def test_nrbs_asymmetric_and_smallest_sizes():
    for n in (3, 4, 8):
        for asym in (False, True):
            C = _random_instance(n, seed=n, asymmetric=asym)
            est = NRBS().fit(C, depot=n - 1)
            assert _is_tour(est.problem_.to_index_tour(est.tour_), n, n - 1)
            assert est.cost_ == pytest.approx(reference.tour_cost(C, est.problem_.to_index_tour(est.tour_)))


def test_nrbs_fast_tier_gap(fast_instance):
    """Measured: 18.3 % on wi29, 7.8 % on dj38 (tolerance 0.50)."""
    est = NRBS().fit(fast_instance["C"], labels=fast_instance["labels"])
    assert fast_instance["optimum"] <= est.cost_ + 1e-9
    assert est.cost_ / fast_instance["optimum"] - 1 <= 0.50


# --------------------------------------------------------------------------- the budget rule (D6)
@pytest.mark.parametrize("Router", BUDGET_UNAWARE, ids=lambda s: s.__name__)
def test_budget_unaware_solvers_warn_and_still_split(Router, alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    with pytest.warns(UserWarning, match=f"{Router.__name__} ignores max_time_work"):
        est = Router().fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert est.cost_ == pytest.approx(
        reference.problem_cost(
            d.cost, d.time, est.problem_.to_index_tour(est.tour_), kw["max_time_work"], 20.0
        )
    )
    assert alicante["optimum"]["greedy"] <= est.cost_ + 1e-9


def test_construction_solvers_are_deterministic_and_not_iterative():
    C = _random_instance(20, seed=1)
    for est in (NearestNeighbour(), Insertion(), Insertion(strategy="cheapest"), ClarkeWright(), NRBS()):
        tags = est._get_tags()
        assert tags.kind == "construction" and not tags.stochastic and not tags.iterative and not tags.exact
        a, b = est.fit(C).tour_.copy(), est.fit(C).tour_
        assert np.array_equal(a, b)
        assert not any(hasattr(est, attr) for attr in ("history_", "n_iter_", "stop_reason_", "is_optimal_"))
