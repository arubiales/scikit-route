# skroute/exact/_hk.pyx -- the bitmask dynamic programme of HeldKarp (SPEC §4.1).
"""Held-Karp dynamic programming over subsets of the ``m = n - 1`` non-depot nodes.

``dp[S][j]`` is the cheapest path that leaves the depot, visits exactly the nodes of the
bitmask ``S`` and ends at ``j`` (``j`` in ``S``); ``parent[S][j]`` is the node before ``j``
on that path. Both tables are ``malloc``ed (``2^m * m`` doubles and bytes: ~80 MB at
n = 20, ~740 MB at n = 23) and freed before returning. Every arc is read directionally,
so the optimum is exact on asymmetric matrices.
"""

from libc.math cimport INFINITY
from libc.stdint cimport int8_t, int64_t
from libc.stdlib cimport free, malloc


cdef double _held_karp(const double[:, ::1] C, const int64_t[::1] others, int64_t depot,
                       double* dp, int8_t* parent, int64_t[::1] out) noexcept nogil:
    cdef Py_ssize_t m = others.shape[0], j, k, S, S2, idx, pos
    cdef Py_ssize_t full = (<Py_ssize_t>1 << m) - 1
    cdef Py_ssize_t cells = (full + 1) * m
    cdef double v, cand, best = INFINITY
    cdef int64_t oj
    cdef Py_ssize_t last = -1, pj
    for idx in range(cells):
        dp[idx] = INFINITY
        parent[idx] = -1
    for j in range(m):
        dp[((<Py_ssize_t>1) << j) * m + j] = C[depot, others[j]]
    # Subsets in increasing numeric order: every proper subset of S is smaller than S, so
    # dp[S][*] is final when S is expanded ("push" formulation, O(2^m * m^2)).
    for S in range(1, full + 1):
        for j in range(m):
            if not (S >> j) & 1:
                continue
            v = dp[S * m + j]
            if v == INFINITY:
                continue
            oj = others[j]
            for k in range(m):
                if (S >> k) & 1:
                    continue
                S2 = S | ((<Py_ssize_t>1) << k)
                idx = S2 * m + k
                cand = v + C[oj, others[k]]
                if cand < dp[idx]:
                    dp[idx] = cand
                    parent[idx] = <int8_t>j
    for j in range(m):
        cand = dp[full * m + j] + C[others[j], depot]
        if cand < best:
            best = cand
            last = j
    # Reconstruct backwards: position n-1 holds the last node, position 1 the first.
    out[0] = depot
    S = full
    pos = m
    j = last
    while j >= 0:
        out[pos] = others[j]
        pj = parent[S * m + j]
        S ^= (<Py_ssize_t>1) << j
        j = pj
        pos -= 1
    return best


def held_karp_search(const double[:, ::1] C, const int64_t[::1] others, int64_t depot,
                     int64_t[::1] out):
    """Optimal closed tour from ``depot`` through ``others`` by dynamic programming.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix, read directionally.
    others : (n - 1,) int64
        The non-depot node indices, in any order (it fixes the tie-breaking).
    depot : int
        Index of the depot.
    out : (n,) int64
        Output: the optimal tour, ``depot`` at position 0.

    Returns
    -------
    best_cost : float
        Cost of the closed tour written to ``out``.

    Raises
    ------
    ValueError
        If the sizes do not match or ``others`` has more than 40 nodes (the tables would
        exceed any memory; ``parent`` is ``int8`` and the mask a ``Py_ssize_t``).
    MemoryError
        If the ``2^(n-1) * (n-1)`` tables cannot be allocated.
    """
    cdef Py_ssize_t m = others.shape[0]
    cdef size_t cells
    cdef double* dp
    cdef int8_t* parent
    cdef double best
    if m < 2 or m > 40:
        raise ValueError(f"HeldKarp needs between 3 and 41 nodes, got {m + 1}")
    if out.shape[0] != m + 1:
        raise ValueError("out must have length len(others) + 1")
    cells = (<size_t>1 << m) * <size_t>m
    dp = <double*>malloc(cells * sizeof(double))
    if dp == NULL:
        raise MemoryError(f"HeldKarp cannot allocate its {cells * 8 / 1e9:.1f} GB table for n = {m + 1}")
    parent = <int8_t*>malloc(cells * sizeof(int8_t))
    if parent == NULL:
        free(dp)
        raise MemoryError(f"HeldKarp cannot allocate its {cells / 1e9:.1f} GB parent table for n = {m + 1}")
    try:
        with nogil:
            best = _held_karp(C, others, depot, dp, parent, out)
    finally:
        free(dp)
        free(parent)
    return best
