# skroute/exact/_brute.pyx -- the enumeration kernel of BruteForce (SPEC §4.1).
"""Exhaustive search over the ``(n - 1)!`` giant tours, in lexicographic order.

``brute_force_search`` walks the permutations of positions ``1..n-1`` with the
lexicographic *next permutation* (Knuth, TAOCP 7.2.1.2, Algorithm L, O(1) amortised) —
not Heap's algorithm, whose order is not lexicographic — so the sequence is exactly the
one ``itertools.permutations`` produces over the ascending non-depot nodes and a tie
resolves to the lexicographically first optimum, as ``tests/reference.py`` does. Every
tour is priced with the core's ``problem_cost``, so the search is exact for the plain
tour and for the multi-trip objective under either split rule.
"""

from libc.math cimport INFINITY, fabs
from libc.stdint cimport int64_t

from skroute._core._routing cimport problem_cost


cdef inline void _reverse(int64_t[::1] a, Py_ssize_t lo, Py_ssize_t hi) noexcept nogil:
    # reverse a[lo..hi] inclusive
    cdef int64_t tmp
    while lo < hi:
        tmp = a[lo]
        a[lo] = a[hi]
        a[hi] = tmp
        lo += 1
        hi -= 1


cdef inline bint _next_permutation(int64_t[::1] a, Py_ssize_t lo, Py_ssize_t hi) noexcept nogil:
    # Rearrange a[lo..hi] (inclusive) into the lexicographically next permutation (Algorithm L).
    # Returns 0 when the segment was the last permutation; it is then left in ascending order.
    cdef Py_ssize_t j = hi - 1, l = hi
    cdef int64_t tmp
    while j >= lo and a[j] >= a[j + 1]:      # L2: largest j with a[j] < a[j+1]
        j -= 1
    if j < lo:
        _reverse(a, lo, hi)
        return 0
    while a[j] >= a[l]:                      # L3: largest l with a[j] < a[l]
        l -= 1
    tmp = a[j]
    a[j] = a[l]
    a[l] = tmp
    _reverse(a, j + 1, hi)                   # L4
    return 1


cdef inline double _reversed_cost(const double[:, ::1] C, const int64_t[::1] tour) noexcept nogil:
    # Plain cost of the REVERSED orientation (d, a[n-1], ..., a[1]) on a symmetric matrix, summed in
    # the order tour_cost would sum it, so the value is bit-identical to pricing the reversed tour.
    cdef Py_ssize_t n = tour.shape[0], k
    cdef int64_t d = tour[0]
    cdef double total = C[d, tour[n - 1]]
    for k in range(n - 1, 1, -1):
        total += C[tour[k], tour[k - 1]]
    return total + C[tour[1], d]


cdef inline bint _lex_less(const int64_t[::1] a, const int64_t[::1] b) noexcept nogil:
    # a < b lexicographically (position 0 is the depot in both)
    cdef Py_ssize_t k, n = a.shape[0]
    for k in range(1, n):
        if a[k] != b[k]:
            return a[k] < b[k]
    return 0


cdef inline bint _reversed_lex_less(const int64_t[::1] tour, const int64_t[::1] b) noexcept nogil:
    # reversed(tour) < b lexicographically, without materialising the reversal
    cdef Py_ssize_t k, n = tour.shape[0]
    for k in range(1, n):
        if tour[n - k] != b[k]:
            return tour[n - k] < b[k]
    return 0


cdef double _search(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                    int64_t[::1] best, double max_time, double fixed_cost, int split,
                    bint halve, double[::1] dp, int64_t[::1] pred,
                    int64_t* n_evaluated) noexcept nogil:
    # Invariant: best_cost is the smallest value seen and best the lexicographically first tour
    # attaining it -- exactly the pair itertools.permutations + a strict `<` would end with.
    cdef Py_ssize_t n = tour.shape[0], k
    cdef double best_cost = INFINITY, c, c_rev
    cdef int64_t count = 0
    while True:
        if not halve:
            c = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
            count += 1
            if c < best_cost:                # strictly better: ties keep the earlier (smaller) tour
                best_cost = c
                for k in range(n):
                    best[k] = tour[k]
        elif tour[1] < tour[n - 1]:
            # Symmetric plain TSP: the reversed orientation has the same cost mathematically but its
            # floating-point sum may differ in the last bit, and the reference keeps the bit-smaller
            # one. Price the kept orientation (lexicographically the smaller of the pair) always, and
            # the reversal only when this pair is within 1e-9 of the incumbent; exact ties resolve
            # lexicographically, so the answer is the one the unhalved enumeration would give.
            c = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
            count += 1
            if c < best_cost:
                best_cost = c
                for k in range(n):
                    best[k] = tour[k]
            elif c == best_cost and _lex_less(tour, best):
                for k in range(n):
                    best[k] = tour[k]
            if c - best_cost <= 1e-9 * (fabs(best_cost) + 1.0):
                c_rev = _reversed_cost(C, tour)
                count += 1
                if c_rev < best_cost or (c_rev == best_cost and _reversed_lex_less(tour, best)):
                    best_cost = c_rev
                    best[0] = tour[0]
                    for k in range(1, n):
                        best[k] = tour[n - k]
        if not _next_permutation(tour, 1, n - 1):
            break
    n_evaluated[0] = count
    return best_cost


def brute_force_search(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                       int64_t[::1] best, double max_time, double fixed_cost, int split,
                       bint halve, double[::1] dp, int64_t[::1] pred):
    """Enumerate every giant tour and write the lexicographically first optimum to ``best``.

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost matrix and the matrix the decoders read as durations (``problem.time_or_cost``).
    tour : (n,) int64
        Work buffer: the depot at position 0 followed by the other nodes in **ascending**
        order (the first permutation). Left in ascending order on return.
    best : (n,) int64
        Output: the best tour found.
    max_time, fixed_cost, split
        The decoder's arguments (``inf`` and any split for a plain TSP).
    halve : bool
        Skip permutations with ``tour[1] > tour[n-1]``; valid only when the matrix is
        symmetric and there is no budget (a reversal changes the split).
    dp, pred : float64[n], int64[n]
        Scratch for the optimal split; zero-length views otherwise.

    Returns
    -------
    best_cost : float
        Objective of ``best``.
    n_evaluated : int
        Number of tours priced (``(n-1)!``, or half of it under ``halve``).
    """
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t n_evaluated = 0
    cdef double best_cost
    if n < 3:
        raise ValueError("brute_force_search needs at least 3 nodes")
    if best.shape[0] != n:
        raise ValueError("best must have the same length as tour")
    with nogil:
        best_cost = _search(C, T, tour, best, max_time, fixed_cost, split, halve, dp, pred,
                            &n_evaluated)
    return best_cost, n_evaluated
