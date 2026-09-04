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
# the nodes whose cached edge was split, or whose cached cost is exactly tied by one of the two new
# edges (the walk decides which comes first), are rescanned -- and O(n^3) in the adversarial worst
# case. The cache therefore holds, for every unrouted node, the first minimum-cost edge met in the
# walk from the depot: inserting a node never reorders the surviving edges, a strictly cheaper new
# edge wins outright, and an exact tie is resolved by a full walk.
#
# Insertion order (D31): on request the kernel records, for each of the n - 1 steps, the node placed
# and the tour node it was placed after (the seed counts as step 0, placed after the depot), so the
# estimator can replay the partial cycles for its callback. Two stores per step behind one flag: the
# tour is bit-identical with and without the recording.
"""Insertion construction kernel: `insertion_tour` (see `skroute.construction.Insertion`)."""

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
                     double[::1] best_cost, int64_t[::1] best_after, int64_t[::1] out,
                     int64_t[::1] ins_node, int64_t[::1] ins_after, bint record) noexcept nogil:
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
    if record:                              # step 0: the seed, placed after the depot
        ins_node[0] = chosen
        ins_after[0] = depot
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
        if record:                          # step m - 1: `chosen` placed between `after` and `b`
            ins_node[m - 1] = chosen
            ins_after[m - 1] = after
        m += 1
        for j in range(n):
            if routed[j]:
                continue
            if C[chosen, j] < min_dist[j]:
                min_dist[j] = C[chosen, j]
            if strategy == CHEAPEST:
                if best_after[j] == after:  # its cached edge (after -> b) no longer exists
                    best_after[j] = _best_edge(C, succ, depot, j, &best_cost[j])
                    continue
                c = _ins_cost(C, after, j, chosen)
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_after[j] = after
                elif c == best_cost[j]:  # tied with the cached edge: the walk says which is first
                    best_after[j] = _best_edge(C, succ, depot, j, &best_cost[j])
                    continue
                c = _ins_cost(C, chosen, j, b)
                if c < best_cost[j]:
                    best_cost[j] = c
                    best_after[j] = chosen
                elif c == best_cost[j] and best_after[j] != after:
                    best_after[j] = _best_edge(C, succ, depot, j, &best_cost[j])

    # ---- read the linked list out as an index tour, depot first
    a = depot
    for k in range(n):
        out[k] = a
        a = succ[a]


def insertion_tour(const double[:, ::1] C, int64_t depot, str strategy, order=None, after=None):
    """Build an insertion tour over ``C`` from ``depot``.

    Parameters
    ----------
    C : ndarray of shape (n, n), float64, C-contiguous
        Cost matrix; rows are origins. The diagonal is never read.
    depot : int
        Index of the depot, ``0 <= depot < n``.
    strategy : {"farthest", "cheapest", "nearest"}
        Selection rule of the next node to insert (see `skroute.construction.Insertion`).
    order : ndarray of shape (n - 1,), int64, C-contiguous, optional
        Output: the node placed at each construction step (D31). Step 0 is the seed; step ``k``
        is the ``k``-th insertion into the partial cycle. Both arrays or neither.
    after : ndarray of shape (n - 1,), int64, C-contiguous, optional
        Output: the tour node each step's node was placed after (``depot`` at step 0); its
        successor at that moment is the other neighbour of the inserted node.

    Returns
    -------
    tour : ndarray of shape (n,), int64
        A permutation of ``range(n)`` with ``depot`` at position 0 — the same array whether or
        not the order is recorded.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.construction._insert import insertion_tour
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> order, after = np.empty(3, dtype=np.int64), np.empty(3, dtype=np.int64)
    >>> insertion_tour(C, 0, "farthest", order, after).tolist(), order.tolist(), after.tolist()
    ([0, 1, 2, 3], [3, 1, 2], [0, 0, 1])
    """
    cdef Py_ssize_t n = C.shape[0]
    cdef int code
    cdef bint record
    cdef int64_t[::1] succ_v, best_after_v, out_v, ins_node_v, ins_after_v
    cdef uint8_t[::1] routed_v
    cdef double[::1] min_dist_v, best_cost_v
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be 'farthest', 'cheapest' or 'nearest', got {strategy!r}")
    code = STRATEGIES[strategy]
    if n < 2 or C.shape[1] != n:
        raise ValueError(
            f"C must be a square matrix with at least 2 nodes, got shape {(C.shape[0], C.shape[1])}"
        )
    if depot < 0 or depot >= n:
        raise ValueError(f"depot must be in [0, {n}), got {depot}")
    record = order is not None or after is not None
    if record:
        if order is None or after is None:
            raise ValueError("order and after must be given together (both arrays or neither)")
        if order.shape != (n - 1,) or after.shape != (n - 1,):
            raise ValueError(
                f"order and after must have shape ({n - 1},), got {order.shape} and {after.shape}"
            )
        ins_node_v = order
        ins_after_v = after
    else:  # nothing is written: one-element stand-ins keep the kernel signature total
        ins_node_v = np.empty(1, dtype=np.int64)
        ins_after_v = np.empty(1, dtype=np.int64)
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
        _insertion(C, depot, code, succ_v, routed_v, min_dist_v, best_cost_v, best_after_v, out_v,
                   ins_node_v, ins_after_v, record)
    return out
