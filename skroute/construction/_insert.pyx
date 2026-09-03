# skroute/construction/_insert.pyx -- the insertion construction kernel (SPEC §4.2).
#
# Farthest, cheapest and nearest insertion over a dense cost matrix, direction-aware (every
# insertion cost reads C[a, j] + C[j, b] - C[a, b] with the arcs in driving direction, so the
# heuristic is exact on ATSP). The partial tour is a singly linked list ``succ[node]`` closed on
# the depot; an edge is identified by the node it leaves from. Selection ties are broken by the
# lowest node index and position ties by the first edge met when walking the tour from the depot,
# so the result is a deterministic function of the matrix.
#
# Complexity: farthest/nearest O(n^2) (one incremental ``min_dist`` update and one O(n) edge scan
# per insertion); cheapest O(n^2) in practice -- every unrouted node caches its best edge and only
# the nodes whose cached edge was split are rescanned -- and O(n^3) in the adversarial worst case.
"""Insertion construction kernel: :func:`insertion_tour` (see :class:`skroute.construction.Insertion`)."""

from libc.math cimport INFINITY
from libc.stdint cimport int64_t, uint8_t

import numpy as np

__all__ = ["STRATEGIES", "insertion_tour"]

# strategy codes of the kernel; the estimator validates the names
STRATEGIES = {"farthest": 0, "cheapest": 1, "nearest": 2}

cdef enum:
    FARTHEST = 0
    CHEAPEST = 1
    NEAREST = 2


cdef inline double _ins_cost(const double[:, ::1] C, int64_t a, int64_t j, int64_t b) noexcept nogil:
    # cost of inserting j between the consecutive nodes a -> b (direction-aware)
    return C[a, j] + C[j, b] - C[a, b]


cdef inline int64_t _best_edge(const double[:, ::1] C, const int64_t[::1] succ, int64_t depot,
                               int64_t j, double* best) noexcept nogil:
    # cheapest insertion edge for j over the whole tour, walking from the depot; the first strict
    # minimum wins. Returns the node the edge leaves from and writes its cost to *best.
    cdef int64_t a = depot, b, after = depot
    cdef double c, best_c = INFINITY
    while True:
        b = succ[a]
        c = _ins_cost(C, a, j, b)
        if c < best_c:
            best_c = c
            after = a
        a = b
        if a == depot:
            break
    best[0] = best_c
    return after


cdef void _insertion(const double[:, ::1] C, int64_t depot, int strategy,
                     int64_t[::1] succ, uint8_t[::1] routed, double[::1] min_dist,
                     double[::1] best_cost, int64_t[::1] best_after, int64_t[::1] out) noexcept nogil:
    cdef Py_ssize_t n = C.shape[0], m, k
    cdef int64_t j, a, b, chosen, after
    cdef double best, c

    for j in range(n):
        routed[j] = 0
        min_dist[j] = C[depot, j]          # distance from the one-node tour {depot}
        best_after[j] = depot
        best_cost[j] = INFINITY
    routed[depot] = 1

    # ---- the second node: farthest from the depot (farthest), nearest to it (cheapest, nearest)
    chosen = -1
    if strategy == FARTHEST:
        best = -INFINITY
        for j in range(n):
            if j != depot and C[depot, j] > best:
                best = C[depot, j]
                chosen = j
    else:
        best = INFINITY
        for j in range(n):
            if j != depot and C[depot, j] < best:
                best = C[depot, j]
                chosen = j
    if chosen < 0:                          # unreachable with finite data; keep the kernel total
        chosen = 0 if depot != 0 else 1
    succ[depot] = chosen
    succ[chosen] = depot
    routed[chosen] = 1
    for j in range(n):
        if routed[j]:
            continue
        if C[chosen, j] < min_dist[j]:
            min_dist[j] = C[chosen, j]
        if strategy == CHEAPEST:
            best_after[j] = _best_edge(C, succ, depot, j, &best_cost[j])

    # ---- main loop: select by the strategy, insert at the cheapest position, update the caches
    m = 2
    while m < n:
        chosen = -1
        if strategy == FARTHEST:
            best = -INFINITY
            for j in range(n):
                if not routed[j] and min_dist[j] > best:
                    best = min_dist[j]
                    chosen = j
        elif strategy == NEAREST:
            best = INFINITY
            for j in range(n):
                if not routed[j] and min_dist[j] < best:
                    best = min_dist[j]
                    chosen = j
        else:
            best = INFINITY
            for j in range(n):
                if not routed[j] and best_cost[j] < best:
                    best = best_cost[j]
                    chosen = j
        if chosen < 0:                      # unreachable with finite data
            for j in range(n):
                if not routed[j]:
                    chosen = j
                    break
        if strategy == CHEAPEST:
            after = best_after[chosen]
        else:
            after = _best_edge(C, succ, depot, chosen, &c)
        b = succ[after]
        succ[after] = chosen
        succ[chosen] = b
        routed[chosen] = 1
        m += 1
        for j in range(n):
            if routed[j]:
                continue
            if C[chosen, j] < min_dist[j]:
                min_dist[j] = C[chosen, j]
            if strategy == CHEAPEST:
                if best_after[j] == after:  # its cached edge (after -> b) no longer exists
                    best_after[j] = _best_edge(C, succ, depot, j, &best_cost[j])
                else:
                    c = _ins_cost(C, after, j, chosen)
                    if c < best_cost[j]:
                        best_cost[j] = c
                        best_after[j] = after
                    c = _ins_cost(C, chosen, j, b)
                    if c < best_cost[j]:
                        best_cost[j] = c
                        best_after[j] = chosen

    # ---- read the linked list out as an index tour, depot first
    a = depot
    for k in range(n):
        out[k] = a
        a = succ[a]


def insertion_tour(const double[:, ::1] C, int64_t depot, str strategy):
    """Build an insertion tour over ``C`` from ``depot``.

    Parameters
    ----------
    C : ndarray of shape (n, n), float64, C-contiguous
        Cost matrix; rows are origins. The diagonal is never read.
    depot : int
        Index of the depot, ``0 <= depot < n``.
    strategy : {"farthest", "cheapest", "nearest"}
        Selection rule of the next node to insert (see :class:`skroute.construction.Insertion`).

    Returns
    -------
    tour : ndarray of shape (n,), int64
        A permutation of ``range(n)`` with ``depot`` at position 0.
    """
    cdef Py_ssize_t n = C.shape[0]
    cdef int code
    cdef int64_t[::1] succ_v, best_after_v, out_v
    cdef uint8_t[::1] routed_v
    cdef double[::1] min_dist_v, best_cost_v
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be 'farthest', 'cheapest' or 'nearest', got {strategy!r}")
    code = STRATEGIES[strategy]
    if n < 2 or C.shape[1] != n:
        raise ValueError(f"C must be a square matrix with at least 2 nodes, got shape {tuple(C.shape)}")
    if depot < 0 or depot >= n:
        raise ValueError(f"depot must be in [0, {n}), got {depot}")
    out = np.empty(n, dtype=np.int64)
    succ = np.empty(n, dtype=np.int64)
    best_after = np.empty(n, dtype=np.int64)
    routed = np.zeros(n, dtype=np.uint8)
    min_dist = np.empty(n, dtype=np.float64)
    best_cost = np.empty(n, dtype=np.float64)
    succ_v = succ
    best_after_v = best_after
    out_v = out
    routed_v = routed
    min_dist_v = min_dist
    best_cost_v = best_cost
    with nogil:
        _insertion(C, depot, code, succ_v, routed_v, min_dist_v, best_cost_v, best_after_v, out_v)
    return out
