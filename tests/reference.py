"""Pure-Python reference implementations ("oracles") for the scikit-route test-suite.

Everything here is deliberately simple, slow and independent of the Cython core:
these functions define what the kernels and the solvers must compute, so they are
written from the specification, never copied from ``skroute``. Nothing in
``skroute`` may import this module.

Conventions (SPEC §3.1): nodes are ``0..n-1``; a *tour* is a permutation of all
nodes whose first element is the depot; the objective always includes the
return to the depot; ``max_time = inf`` means plain TSP; ``fixed_cost`` is
``people * extra_cost`` and is charged once per trip beyond the first.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = [
    "brute_force",
    "double_bridge",
    "greedy_split",
    "optimal_split",
    "or_opt_apply",
    "or_opt_delta_by_recompute",
    "ox",
    "pmx",
    "problem_cost",
    "route_cost_from_labels",
    "swap_apply",
    "swap_delta_by_recompute",
    "tour_cost",
    "trip_starts",
    "two_opt_apply",
    "two_opt_delta_by_recompute",
]


# --------------------------------------------------------------------------- evaluation
def tour_cost(C, tour):
    """Cost of the closed tour ``tour[0] -> tour[1] -> ... -> tour[-1] -> tour[0]``."""
    C = np.asarray(C, dtype=float)
    tour = [int(v) for v in tour]
    n = len(tour)
    return float(sum(C[tour[k], tour[(k + 1) % n]] for k in range(n)))


def greedy_split(C, T, tour, max_time, fixed_cost):
    """Greedy decoder of SPEC D1.

    Leg ``a -> b`` joins the open trip iff the trip can still return to the depot
    within ``max_time``; otherwise the trip is closed at ``a`` and a new one is
    opened ``depot -> b``.

    Returns ``(cost, starts)`` where ``starts`` is the list of trip start
    positions (``starts[0] == 1`` and ``starts[-1] == n``, so trip ``k`` is
    ``tour[starts[k]:starts[k + 1]]``).
    """
    C = np.asarray(C, dtype=float)
    tour = [int(v) for v in tour]
    n = len(tour)
    d = tour[0]
    if not math.isfinite(max_time):
        return tour_cost(C, tour), [1, n]
    T = np.asarray(T, dtype=float)
    t = 0.0
    cost = 0.0
    starts = [1]
    for k in range(n - 1):
        a, b = tour[k], tour[k + 1]
        if t + T[a, b] + T[b, d] <= max_time:
            t += T[a, b]
            cost += C[a, b]
        else:
            # close the trip at a (nothing to add when a is the depot itself: never read the diagonal)
            cost += (C[a, d] if k > 0 else 0.0) + C[d, b]
            t = T[d, b]
            starts.append(k + 1)
    cost += C[tour[n - 1], d]
    starts.append(n)
    n_trips = len(starts) - 1
    return float(cost + (n_trips - 1) * fixed_cost), starts


def optimal_split(C, T, tour, max_time, fixed_cost):
    """Optimal decoder of SPEC D1 (Prins 2004): the minimum-cost partition of the
    giant tour into consecutive trips that each fit ``max_time`` including the
    return leg.

    Implemented as a shortest path on the DAG of feasible trips, O(n^2) here (the
    kernel prunes with the monotone outbound time). No triangle inequality is
    assumed: every consecutive block whose closed duration fits is a candidate.

    Returns ``(cost, starts)`` as :func:`greedy_split`.
    """
    C = np.asarray(C, dtype=float)
    tour = [int(v) for v in tour]
    n = len(tour)
    d = tour[0]
    if not math.isfinite(max_time):
        return tour_cost(C, tour), [1, n]
    T = np.asarray(T, dtype=float)
    cust = tour[1:]
    m = n - 1
    dp = [math.inf] * (m + 1)
    pred = [-1] * (m + 1)
    dp[0] = 0.0
    for j in range(m):
        if not math.isfinite(dp[j]):
            continue
        path_time = 0.0
        path_cost = 0.0
        for i in range(j + 1, m + 1):  # trip covers customers cust[j .. i-1]
            if i == j + 1:
                path_time = T[d, cust[j]]
                path_cost = C[d, cust[j]]
            else:
                path_time += T[cust[i - 2], cust[i - 1]]
                path_cost += C[cust[i - 2], cust[i - 1]]
            if path_time > max_time:
                break  # outbound time only grows: no longer block can fit
            if path_time + T[cust[i - 1], d] > max_time:
                continue  # this block does not fit, a longer one still might
            cand = dp[j] + path_cost + C[cust[i - 1], d] + (fixed_cost if j > 0 else 0.0)
            if cand < dp[i]:
                dp[i] = cand
                pred[i] = j
    assert math.isfinite(dp[m]), "every single-customer trip must be feasible (SPEC D5)"
    bounds = []
    i = m
    while i > 0:
        bounds.append(i)
        i = pred[i]
    bounds.append(0)
    starts = sorted(b + 1 for b in bounds)  # customer index j lives at tour position j + 1
    return float(dp[m]), starts


def trip_starts(C, T, tour, max_time, fixed_cost, split="greedy"):
    """Trip start positions of a tour under the chosen decoder."""
    fn = greedy_split if split == "greedy" else optimal_split
    return fn(C, T, tour, max_time, fixed_cost)[1]


def problem_cost(C, T, tour, max_time=math.inf, fixed_cost=0.0, split="greedy"):
    """Objective of SPEC D1 for an index tour (plain TSP when ``max_time`` is inf)."""
    if T is None or not math.isfinite(max_time):
        return tour_cost(C, tour)
    if split == "greedy":
        return greedy_split(C, T, tour, max_time, fixed_cost)[0]
    if split == "optimal":
        return optimal_split(C, T, tour, max_time, fixed_cost)[0]
    raise ValueError(f"split must be 'greedy' or 'optimal', got {split!r}")


def route_cost_from_labels(
    C, route, labels, depot, T=None, max_time=math.inf, fixed_cost=0.0, split="greedy"
):
    """Objective of a label-space route (depot first, possibly repeated between
    trips and at the end): depot occurrences are dropped and the giant tour is
    re-decoded, exactly like ``RoutingProblem.to_index_tour`` followed by
    ``evaluate``."""
    index = {label: i for i, label in enumerate(list(labels))}
    d = index[depot]
    body = [index[x] for x in route if index[x] != d]
    assert sorted(body) == [i for i in range(len(labels)) if i != d], "route must visit every label once"
    return problem_cost(C, T, [d] + body, max_time, fixed_cost, split)


# --------------------------------------------------------------------------- exact oracle
def brute_force(C, T=None, *, depot=0, max_time_work=None, extra_cost=0.0, people=1, split="greedy"):
    """Exhaustive optimum over all giant tours. Ties: the lexicographically first tour.

    Returns ``(cost, tour)`` with ``tour`` an ``int64`` array starting at ``depot``.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    max_time = math.inf if max_time_work is None else float(max_time_work)
    fixed = float(extra_cost) * int(people)
    others = [i for i in range(n) if i != depot]
    best_cost, best_tour = math.inf, None
    for perm in itertools.permutations(others):
        tour = [depot, *perm]
        c = problem_cost(C, T, tour, max_time, fixed, split)
        if c < best_cost:
            best_cost, best_tour = c, tour
    return float(best_cost), np.asarray(best_tour, dtype=np.int64)


# ------------------------------------------------------------ moves (positions 1 <= i, j <= n-1)
def two_opt_apply(tour, i, j):
    """Reverse ``tour[i..j]`` (inclusive, ``i < j``); returns a new list."""
    tour = list(tour)
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def two_opt_delta_by_recompute(C, tour, i, j):
    return tour_cost(C, two_opt_apply(tour, i, j)) - tour_cost(C, tour)


def or_opt_apply(tour, i, L, j, reverse=False):
    """Move the segment ``tour[i .. i+L-1]`` so that it follows the node at
    position ``j`` (``j`` outside ``[i-1, i+L-1]``), optionally reversed; returns
    a new list. Position 0 (the depot) never moves."""
    tour = list(tour)
    n = len(tour)
    assert i >= 1 and i + L <= n and 1 <= L <= 3
    assert not (i - 1 <= j <= i + L - 1), "j must not be inside or just before the segment"
    seg = tour[i : i + L]
    if reverse:
        seg = seg[::-1]
    anchor = tour[j]
    rest = tour[:i] + tour[i + L :]
    k = rest.index(anchor)
    return rest[: k + 1] + seg + rest[k + 1 :]


def or_opt_delta_by_recompute(C, tour, i, L, j, reverse=False):
    return tour_cost(C, or_opt_apply(tour, i, L, j, reverse)) - tour_cost(C, tour)


def swap_apply(tour, i, j):
    tour = list(tour)
    tour[i], tour[j] = tour[j], tour[i]
    return tour


def swap_delta_by_recompute(C, tour, i, j):
    return tour_cost(C, swap_apply(tour, i, j)) - tour_cost(C, tour)


def double_bridge(tour, p1, p2, p3):
    """``A B C D -> A C B D`` with ``A = tour[0:p1]``, ``B = tour[p1:p2]``,
    ``C = tour[p2:p3]``, ``D = tour[p3:]`` and ``1 <= p1 < p2 < p3 <= n-1``."""
    tour = list(tour)
    assert 1 <= p1 < p2 < p3 <= len(tour) - 1
    return tour[:p1] + tour[p2:p3] + tour[p1:p2] + tour[p3:]


# ------------------------------------------------------------ crossovers (chromosome = tour[1:])
def ox(p1, p2, a, b):
    """Order crossover: keep ``p1[a..b]`` (inclusive) in place, fill the remaining
    positions, starting after ``b`` and wrapping, with the genes of ``p2`` in
    ``p2``'s order skipping those already present."""
    p1, p2 = list(p1), list(p2)
    m = len(p1)
    assert 0 <= a <= b < m
    child = [None] * m
    child[a : b + 1] = p1[a : b + 1]
    present = set(child[a : b + 1])
    fill = [g for g in p2[b + 1 :] + p2[: b + 1] if g not in present]
    pos = (b + 1) % m
    for g in fill:
        child[pos] = g
        pos = (pos + 1) % m
    return child


def pmx(p1, p2, a, b):
    """Partially mapped crossover: copy ``p1[a..b]``, then place every gene of
    ``p2`` outside the segment, following the mapping ``p1[k] -> p2[k]`` until a
    free gene is found."""
    p1, p2 = list(p1), list(p2)
    m = len(p1)
    assert 0 <= a <= b < m
    child = [None] * m
    child[a : b + 1] = p1[a : b + 1]
    mapping = {p1[k]: p2[k] for k in range(a, b + 1)}
    segment = set(child[a : b + 1])
    for k in list(range(a)) + list(range(b + 1, m)):
        g = p2[k]
        while g in segment:
            g = mapping[g]
        child[k] = g
    return child
