"""Compiled core of scikit-route: evaluation, move deltas, moves, descents and construction.

This module implements the non-inline half of the frozen contract of ``_routing.pxd``
(SPEC §3.5); the O(1) primitives (``tour_cost``, ``greedy_split_cost``, ``problem_cost``,
the four move deltas, ``reverse_segment*`` and ``swap_positions*``) live with their bodies in
the ``.pxd`` so that every solver's ``.pyx`` inlines them. Solvers ``cimport`` this module;
Python code (``RoutingProblem``, tests, benchmarks) calls the ``cpdef`` functions and the
``*_py`` wrappers below.

Calling conventions
-------------------
* ``C`` and ``T`` are ``(n, n)`` **C-contiguous float64** matrices (rows are origins); a
  ``tour`` is a **C-contiguous int64** permutation of ``0..n-1`` with the depot at position 0;
  ``pos``, ``cand``, ``starts``, ``out`` and the scratch buffers are int64 (``dont_look`` is
  uint8). A typed memoryview rejects any other dtype or layout with ``ValueError`` /
  ``TypeError`` (``RoutingProblem`` coerces first with ``np.ascontiguousarray``).
* Positions are 0-based; every ``[i..j]`` range is inclusive except in ``double_bridge``.
  The successor of position ``n - 1`` is position 0. Position 0 never moves.
* ``max_time == inf`` means plain TSP (``T`` is then never read); ``fixed_cost`` is
  ``people * extra_cost`` and is charged once per trip beyond the first.
* Every function is ``noexcept nogil`` except :func:`problem_cost_py`, :func:`trip_starts`
  and the ``*_py`` wrappers, which hold the GIL, validate their arguments and may raise. The
  ``noexcept nogil`` kernels **cannot** validate: an ill-shaped buffer or an out-of-domain
  position is undefined behaviour there — respect the documented domains.
* Scratch memory is ``malloc``/``free``'d from ``libc.stdlib``; a kernel that cannot raise
  falls back to a documented safe result when allocation fails (see
  :func:`nearest_neighbour_tour`).

The pure-Python oracles these kernels are tested against live in ``tests/reference.py``.
"""

from libc.math cimport INFINITY, fabs, isfinite
from libc.stdint cimport int64_t, uint8_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

__all__ = [
    "SplitRule",
    "problem_cost_py",
    "trip_starts",
    "trip_costs",
    "trip_times",
    "double_bridge",
    "rebuild_pos",
    "two_opt_descent",
    "or_opt_descent",
    "local_search_generic",
    "nearest_neighbour_tour",
    "tour_cost_py",
    "greedy_split_cost_py",
    "optimal_split_cost_py",
    "two_opt_delta_py",
    "two_opt_delta_asym_py",
    "or_opt_delta_py",
    "swap_delta_py",
    "reverse_segment_py",
    "reverse_segment_pos_py",
    "swap_positions_py",
    "swap_positions_pos_py",
    "move_segment_py",
    "move_segment_pos_py",
]

# Relative improvement threshold of SPEC §4.0: new < best - 1e-9 * max(1, |best|).
cdef double REL_EPS = 1e-9


cdef inline bint _improves(double delta, double scale) noexcept nogil:
    # delta = cost(after) - cost(before) of a move; scale = a local magnitude (the removed edges
    # for O(1) deltas, the current cost for full evaluations). Strictly negative beyond rounding.
    if scale < 1.0:
        scale = 1.0
    return delta < -REL_EPS * scale


# ====================================================================== evaluation
cdef double optimal_split_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                               double max_time, double fixed_cost,
                               double[::1] dp, int64_t[::1] pred) noexcept nogil:
    # Prins (2004) shortest path over the DAG of feasible consecutive trips.
    # Customers c_1..c_m = tour[1..n-1]; dp[i] = best cost serving c_1..c_i, dp[0] = 0.
    # For a trip opened at c_{j+1} the OUTBOUND path time is monotone non-decreasing in the last
    # customer i (T >= 0), so the extension loop breaks once it exceeds max_time; a block whose
    # CLOSED duration exceeds max_time is only skipped (no triangle inequality assumed).
    # O(n * L), L = longest span whose outbound path fits. Returns +inf if no partition exists
    # (a single-customer trip does not fit -- excluded by D5 at the RoutingProblem level).
    cdef Py_ssize_t n = tour.shape[0], m = n - 1, i, j
    cdef int64_t d = tour[0]
    cdef double path_time, path_cost, cand, base
    dp[0] = 0.0
    pred[0] = -1
    for i in range(1, m + 1):
        dp[i] = INFINITY
        pred[i] = -1
    for j in range(m):
        base = dp[j]
        if base == INFINITY:
            continue
        path_time = T[d, tour[j + 1]]
        path_cost = C[d, tour[j + 1]]
        for i in range(j + 1, m + 1):          # the trip covers tour[j+1 .. i]
            if i > j + 1:
                path_time += T[tour[i - 1], tour[i]]
                path_cost += C[tour[i - 1], tour[i]]
            if path_time > max_time:
                break                          # outbound time only grows: no longer block fits
            if path_time + T[tour[i], d] > max_time:
                continue                       # this block does not fit, a longer one still might
            cand = dp[j] + path_cost + C[tour[i], d]
            if j > 0:
                cand += fixed_cost
            if cand < dp[i]:
                dp[i] = cand
                pred[i] = j
    return dp[m]


cdef int _check_square(const double[:, ::1] M, Py_ssize_t n, str name) except -1:
    if M.shape[0] != n or M.shape[1] != n:
        raise ValueError(f"{name} must be an ({n}, {n}) matrix, got ({M.shape[0]}, {M.shape[1]})")
    return 0


cdef int _check_tour(const int64_t[::1] tour) except -1:
    if tour.shape[0] < 1:
        raise ValueError("tour must contain at least the depot")
    return 0


cpdef double problem_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                             double max_time, double fixed_cost, int split):
    """Objective of SPEC D1 for an index tour: the Python entry point of ``problem_cost``.

    Dispatches exactly like the inline ``problem_cost``: ``max_time == inf`` gives the plain
    closed-tour cost (``T`` is never read), ``split == SplitRule.SPLIT_GREEDY`` the greedy
    decoder, anything else the optimal (Prins) decoder. Holds the GIL and allocates its own
    ``dp``/``pred`` scratch.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix, rows are origins.
    T : (n, n) float64, C-contiguous
        Time matrix; pass ``C`` itself for plain TSP (``RoutingProblem.time_or_cost``).
    tour : (n,) int64, C-contiguous
        Permutation of ``0..n-1`` with the depot at position 0.
    max_time : float
        Per-trip budget in the units of ``T``; ``inf`` for plain TSP.
    fixed_cost : float
        ``people * extra_cost``, charged per trip beyond the first.
    split : int
        ``int(SplitRule.SPLIT_GREEDY)`` or ``int(SplitRule.SPLIT_OPTIMAL)``.

    Returns
    -------
    float
        Travel cost of the decoded trips plus ``fixed_cost * (n_trips - 1)``; ``inf`` when the
        optimal split finds no feasible partition (impossible after the D5 check).

    Raises
    ------
    ValueError
        If ``C`` or ``T`` is not ``(n, n)`` for ``n = len(tour)``.
    MemoryError
        If the scratch allocation fails.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> C = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float64)
    >>> tour = np.array([0, 1, 2], dtype=np.int64)
    >>> core.problem_cost_py(C, C, tour, np.inf, 0.0, int(core.SplitRule.SPLIT_GREEDY))
    6.0
    >>> core.problem_cost_py(C, C, tour, 4.0, 10.0, int(core.SplitRule.SPLIT_GREEDY))
    16.0
    """
    cdef Py_ssize_t n = tour.shape[0]
    cdef double* dp_buf
    cdef int64_t* pred_buf
    cdef double[::1] dp
    cdef int64_t[::1] pred
    cdef double result
    _check_tour(tour)
    _check_square(C, n, "C")
    if max_time != INFINITY:
        _check_square(T, n, "T")
    dp_buf = <double*> malloc(n * sizeof(double))
    if dp_buf == NULL:
        raise MemoryError()
    pred_buf = <int64_t*> malloc(n * sizeof(int64_t))
    if pred_buf == NULL:
        free(dp_buf)
        raise MemoryError()
    try:
        dp = <double[:n]> dp_buf
        pred = <int64_t[:n]> pred_buf
        with nogil:
            result = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    finally:
        free(dp_buf)
        free(pred_buf)
    return result


cpdef Py_ssize_t trip_starts(const double[:, ::1] T, const int64_t[::1] tour, double max_time, int split,
                             const double[:, ::1] C, double fixed_cost, int64_t[::1] out):
    """Trip start positions of a tour under the chosen decoder.

    Writes ``out[0..k]`` with ``out[0] == 1`` and ``out[k] == n`` and returns ``k``, the
    number of trips; trip ``t`` is ``tour[out[t]:out[t + 1]]``. Plain TSP (``max_time ==
    inf``) writes ``[1, n]`` and returns 1. The greedy decoder starts a trip at every
    position where the D1 rule closed the previous one; the optimal decoder follows the
    ``pred`` chain of the Prins DAG, for which this function allocates its own scratch
    while holding the GIL.

    Parameters
    ----------
    T : (n, n) float64, C-contiguous
        Time matrix (``C`` itself for plain TSP).
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    max_time : float
        Per-trip budget; ``inf`` for plain TSP.
    split : int
        ``int(SplitRule.SPLIT_GREEDY)`` or ``int(SplitRule.SPLIT_OPTIMAL)``.
    C : (n, n) float64, C-contiguous
        Cost matrix, read only by the optimal split.
    fixed_cost : float
        ``people * extra_cost``, read only by the optimal split.
    out : (n + 1,) int64, C-contiguous
        Output buffer (only ``out[0..k]`` is written).

    Returns
    -------
    int
        ``k``, the number of trips (``1`` for plain TSP).

    Raises
    ------
    ValueError
        If ``out`` is shorter than ``n + 1``, if a matrix is not ``(n, n)``, or if the
        optimal split has no feasible partition (a node's round trip exceeds ``max_time``).
    MemoryError
        If the optimal split's scratch allocation fails.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> C = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float64)
    >>> tour = np.array([0, 1, 2], dtype=np.int64)
    >>> out = np.empty(4, dtype=np.int64)
    >>> k = core.trip_starts(C, tour, 4.0, int(core.SplitRule.SPLIT_GREEDY), C, 0.0, out)
    >>> k, out[: k + 1].tolist()
    (2, [1, 2, 3])
    """
    cdef Py_ssize_t n = tour.shape[0], k, m, i, idx
    cdef int64_t d, a, b
    cdef double t
    cdef double* dp_buf
    cdef int64_t* pred_buf
    cdef double[::1] dp
    cdef int64_t[::1] pred
    cdef double total
    _check_tour(tour)
    if out.shape[0] < n + 1:
        raise ValueError(f"out must have length n + 1 = {n + 1}, got {out.shape[0]}")
    if max_time == INFINITY:
        out[0] = 1
        out[1] = n
        return 1
    _check_square(T, n, "T")
    d = tour[0]
    if split == SPLIT_GREEDY:
        k = 0
        out[0] = 1
        t = 0.0
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            if t + T[a, b] + T[b, d] <= max_time:
                t += T[a, b]
            else:
                t = T[d, b]
                k += 1
                out[k] = i + 1
        k += 1
        out[k] = n
        return k
    # optimal split: run the DP, then walk the predecessor chain backwards.
    _check_square(C, n, "C")
    m = n - 1
    dp_buf = <double*> malloc(n * sizeof(double))
    if dp_buf == NULL:
        raise MemoryError()
    pred_buf = <int64_t*> malloc(n * sizeof(int64_t))
    if pred_buf == NULL:
        free(dp_buf)
        raise MemoryError()
    try:
        dp = <double[:n]> dp_buf
        pred = <int64_t[:n]> pred_buf
        with nogil:
            total = optimal_split_cost(C, T, tour, max_time, fixed_cost, dp, pred)
        if not isfinite(total):
            raise ValueError(
                "no feasible optimal split: a node's round trip exceeds max_time "
                "(RoutingProblem raises InfeasibleProblemError for this at construction)"
            )
        k = 0
        i = m
        while i > 0:
            k += 1
            i = pred[i]
        out[k] = n
        idx = k
        i = m
        while i > 0:
            i = pred[i]
            idx -= 1
            out[idx] = i + 1
    finally:
        free(dp_buf)
        free(pred_buf)
    return k


cpdef void trip_costs(const double[:, ::1] C, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil:
    """Closed travel cost of every trip: ``out[t] = C[d, first] + ... + C[last, d]``.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    starts : (k + 1,) int64, C-contiguous
        Trip boundaries as written by :func:`trip_starts` (``starts[0] == 1``, ``starts[k] == n``).
    out : (k,) float64, C-contiguous
        Receives one closed-trip cost per trip. For plain TSP (``starts == [1, n]``) ``out[0]``
        equals ``tour_cost`` bit for bit (same summation order).

    Notes
    -----
    ``noexcept nogil``: the shapes are not validated. O(n).
    """
    cdef Py_ssize_t k, s, e, i, n_trips = starts.shape[0] - 1
    cdef int64_t d = tour[0]
    cdef double total
    for k in range(n_trips):
        s = starts[k]
        e = starts[k + 1]
        total = C[d, tour[s]]
        for i in range(s, e - 1):
            total += C[tour[i], tour[i + 1]]
        out[k] = total + C[tour[e - 1], d]


cpdef void trip_times(const double[:, ::1] T, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil:
    """Closed duration of every trip: ``out[t] = T[d, first] + ... + T[last, d]``.

    Same contract as :func:`trip_costs` with the time matrix; under either decoder every
    value is ``<= max_time`` (up to rounding) when every single-customer trip fits (D5).

    Notes
    -----
    ``noexcept nogil``: the shapes are not validated. O(n).
    """
    cdef Py_ssize_t k, s, e, i, n_trips = starts.shape[0] - 1
    cdef int64_t d = tour[0]
    cdef double total
    for k in range(n_trips):
        s = starts[k]
        e = starts[k + 1]
        total = T[d, tour[s]]
        for i in range(s, e - 1):
            total += T[tour[i], tour[i + 1]]
        out[k] = total + T[tour[e - 1], d]


# ====================================================================== apply moves
cdef void move_segment(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j,
                       bint reverse) noexcept nogil:
    # Or-opt move matching or_opt_delta: the segment S = tour[i..i+L-1] ends up right after the node
    # at position j. Implemented as a rotation of the affected span by three reversals (block swap),
    # O(|i - j| + L), no scratch memory. With reverse=True the segment lands reversed.
    #   j > i+L-1: span [i..j] = S Y  ->  Y S      (Y = tour[i+L..j])
    #   j < i-1  : span [j+1..i+L-1] = X S -> S X  (X = tour[j+1..i-1])
    # "X Y -> Y X" is reverse(X); reverse(Y); reverse(XY); skipping the reversal of S yields S reversed.
    cdef Py_ssize_t end = i + L - 1
    if j > end:
        if not reverse:
            reverse_segment(tour, i, end)
        reverse_segment(tour, end + 1, j)
        reverse_segment(tour, i, j)
    else:
        reverse_segment(tour, j + 1, i - 1)
        if not reverse:
            reverse_segment(tour, i, end)
        reverse_segment(tour, j + 1, end)


cdef void move_segment_pos(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t L,
                           Py_ssize_t j, bint reverse) noexcept nogil:
    # move_segment keeping pos[node] == position consistent.
    cdef Py_ssize_t end = i + L - 1
    if j > end:
        if not reverse:
            reverse_segment_pos(tour, pos, i, end)
        reverse_segment_pos(tour, pos, end + 1, j)
        reverse_segment_pos(tour, pos, i, j)
    else:
        reverse_segment_pos(tour, pos, j + 1, i - 1)
        if not reverse:
            reverse_segment_pos(tour, pos, i, end)
        reverse_segment_pos(tour, pos, j + 1, end)


cpdef void double_bridge(const int64_t[::1] tour, Py_ssize_t p1, Py_ssize_t p2, Py_ssize_t p3,
                         int64_t[::1] out) noexcept nogil:
    """Double-bridge kick ``A B C D -> A C B D`` (the ILS perturbation of Martin, Otto and Felten).

    ``A = tour[0:p1]``, ``B = tour[p1:p2]``, ``C = tour[p2:p3]``, ``D = tour[p3:n]`` with
    ``1 <= p1 < p2 < p3 <= n - 1`` (half-open ranges, the only ones in this module). Every
    segment keeps its orientation, so the move is exact on asymmetric matrices and cannot be
    undone by 2-opt or Or-opt.

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Source permutation with the depot at position 0 (kept at position 0 since ``p1 >= 1``).
    p1, p2, p3 : int
        Cut positions, ``1 <= p1 < p2 < p3 <= n - 1``.
    out : (n,) int64, C-contiguous
        Receives the kicked tour; must not alias ``tour``.

    Notes
    -----
    ``noexcept nogil``: the positions are not validated. O(n).

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> tour = np.arange(8, dtype=np.int64)
    >>> out = np.empty(8, dtype=np.int64)
    >>> core.double_bridge(tour, 2, 4, 6, out)
    >>> out.tolist()
    [0, 1, 4, 5, 2, 3, 6, 7]
    """
    cdef Py_ssize_t n = tour.shape[0], s, k = 0
    for s in range(p1):
        out[k] = tour[s]
        k += 1
    for s in range(p2, p3):
        out[k] = tour[s]
        k += 1
    for s in range(p1, p2):
        out[k] = tour[s]
        k += 1
    for s in range(p3, n):
        out[k] = tour[s]
        k += 1


cpdef void rebuild_pos(const int64_t[::1] tour, int64_t[::1] pos) noexcept nogil:
    """Fill ``pos`` so that ``pos[tour[k]] == k`` for every position ``k``.

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        A permutation of ``0..n-1``.
    pos : (n,) int64, C-contiguous
        Receives the inverse permutation.

    Notes
    -----
    ``noexcept nogil``: the shapes are not validated. O(n).

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> tour = np.array([0, 3, 1, 2], dtype=np.int64)
    >>> pos = np.empty(4, dtype=np.int64)
    >>> core.rebuild_pos(tour, pos)
    >>> pos.tolist()
    [0, 2, 3, 1]
    """
    cdef Py_ssize_t k, n = tour.shape[0]
    for k in range(n):
        pos[tour[k]] = k


# ====================================================================== descents (symmetric, O(1) deltas)
cdef bint _two_opt_improve_node(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                                const int64_t[:, ::1] cand, uint8_t[::1] dont_look, int64_t a,
                                bint first_improvement, double* gain) noexcept nogil:
    # Bentley's neighbour-list scan for node a: for each of its two tour edges (a, succ a) and
    # (pred a, a) look for a candidate c with C[a, c] < C[edge] (lists are sorted ascending, so
    # the scan breaks at the first failure) and price the 2-opt move that adds (a, c). The reversed
    # path is always the one that excludes position 0, so the depot never moves. Applies the first
    # (or the best) improving move, resets the four endpoints' don't-look bits and returns True.
    cdef Py_ssize_t n = tour.shape[0], K = cand.shape[1], i = pos[a], k, m, lo, hi
    cdef Py_ssize_t best_lo = 0, best_hi = 0
    cdef int64_t b, c, d, best_b = 0, best_c = 0, best_d = 0
    cdef double g1, edge, delta, best_delta = 0.0
    cdef bint found = False

    # ---- direction 1: remove (a, succ a) = (a, b) and (c, succ c) = (c, d); add (a, c), (b, d)
    b = tour[i + 1] if i + 1 < n else tour[0]
    g1 = C[a, b]
    for m in range(K):
        c = cand[a, m]
        if c == a:
            continue
        if C[a, c] >= g1:
            break
        if c == b:
            continue
        k = pos[c]
        d = tour[k + 1] if k + 1 < n else tour[0]
        if d == a:
            continue
        edge = C[c, d]
        delta = C[a, c] + C[b, d] - g1 - edge
        if delta < best_delta and _improves(delta, g1 + edge):
            if k > i:
                lo = i + 1
                hi = k
            else:
                lo = k + 1
                hi = i
            found = True
            best_delta = delta
            best_lo = lo
            best_hi = hi
            best_b = b
            best_c = c
            best_d = d
            if first_improvement:
                break
    if not (found and first_improvement):
        # ---- direction 2: remove (pred a, a) = (b, a) and (pred c, c) = (d, c); add (a, c), (b, d)
        b = tour[i - 1] if i > 0 else tour[n - 1]
        g1 = C[b, a]
        for m in range(K):
            c = cand[a, m]
            if c == a:
                continue
            if C[a, c] >= g1:
                break
            if c == b:
                continue
            k = pos[c]
            d = tour[k - 1] if k > 0 else tour[n - 1]
            if d == a:
                continue
            edge = C[d, c]
            delta = C[a, c] + C[b, d] - g1 - edge
            if delta < best_delta and _improves(delta, g1 + edge):
                if k > i:
                    if i >= 1:
                        lo = i
                        hi = k - 1
                    else:            # a is the depot: reverse the complementary path instead
                        lo = k
                        hi = n - 1
                else:
                    if k >= 1:
                        lo = k
                        hi = i - 1
                    else:            # c is the depot: reverse the complementary path instead
                        lo = i
                        hi = n - 1
                found = True
                best_delta = delta
                best_lo = lo
                best_hi = hi
                best_b = b
                best_c = c
                best_d = d
                if first_improvement:
                    break
    if not found:
        return False
    reverse_segment_pos(tour, pos, best_lo, best_hi)
    gain[0] += best_delta
    dont_look[a] = 0
    dont_look[best_b] = 0
    dont_look[best_c] = 0
    dont_look[best_d] = 0
    return True


cpdef double two_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                             const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                             bint first_improvement, int max_passes) noexcept nogil:
    """2-opt descent with candidate lists and don't-look bits (symmetric matrices, plain TSP).

    Bentley's neighbour-list 2-opt: for every node ``a`` whose don't-look bit is active and
    for each of its two tour edges, the candidates ``c`` of ``a`` are scanned in ascending
    order of ``C[a, c]`` while ``C[a, c] < C[a, succ(a)]`` (resp. ``C[pred(a), a]``); the
    2-opt move adding edge ``(a, c)`` is priced in O(1) with the symmetric formula and applied
    when it improves. On improvement the bits of the four touched endpoints are reset;
    when no move improves, ``a``'s bit is set. The reversed path is always the one that does
    not contain position 0, so the depot stays at position 0.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        **Symmetric** cost matrix (on an asymmetric matrix the O(1) deltas are wrong; use
        :func:`local_search_generic`).
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation (``pos[tour[k]] == k``), kept consistent; build it once with
        :func:`rebuild_pos`.
    cand : (n, k) int64, C-contiguous
        Candidate lists sorted ascending by ``C[i, :]`` (``RoutingProblem.neighbours(k)``).
    dont_look : (n,) uint8, C-contiguous
        Don't-look bits, ``0`` = active. Zero them before the first call; they persist across
        calls so an iterative caller (``LocalSearch`` with ``max_passes=1``) resumes where it
        stopped.
    first_improvement : bool
        ``True`` applies the first improving move found for a node; ``False`` scans the
        node's whole candidate neighbourhood (both directions) and applies the best.
    max_passes : int
        Maximum number of sweeps over the active nodes; ``<= 0`` returns immediately.

    Returns
    -------
    float
        ``cost_after - cost_before`` (``<= 0``); ``0.0`` means nothing changed, i.e. the
        tour is 2-opt-optimal for the candidate neighbourhood.

    Notes
    -----
    A move is applied when ``delta < -1e-9 * max(1, removed)`` with ``removed`` the cost of the
    two removed edges (the local form of the §4.0 improvement test). One sweep is O(n * k)
    delta evaluations plus O(n) per applied reversal. ``noexcept nogil``.

    References
    ----------
    J. L. Bentley, "Fast algorithms for geometric traveling salesman problems", ORSA
    Journal on Computing 4(4), 1992.
    D. S. Johnson and L. A. McGeoch, "The traveling salesman problem: a case study in local
    optimization", in Local Search in Combinatorial Optimization, 1997.
    """
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t a
    cdef double gain = 0.0
    cdef int passes = 0
    cdef bint improved_pass
    if n < 4 or cand.shape[1] == 0:
        return 0.0
    while passes < max_passes:
        passes += 1
        improved_pass = False
        for a in range(n):
            if dont_look[a]:
                continue
            while _two_opt_improve_node(C, tour, pos, cand, dont_look, a, first_improvement, &gain):
                improved_pass = True
            dont_look[a] = 1
        if not improved_pass:
            break
    return gain


cdef bint _or_opt_try(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos, uint8_t[::1] dont_look,
                      Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, bint reverse,
                      int64_t p, int64_t q, int64_t s0, int64_t sL, int64_t c, double* gain) noexcept nogil:
    # Price the Or-opt move (i, L, j, reverse) if j is in its domain; apply it when it improves.
    cdef Py_ssize_t n = tour.shape[0], end = i + L - 1
    cdef int64_t d
    cdef double delta
    if j >= i - 1 and j <= end:
        return False
    d = tour[j + 1] if j + 1 < n else tour[0]
    delta = or_opt_delta(C, tour, i, L, j, reverse)
    if not _improves(delta, C[p, s0] + C[sL, q] + C[c, d]):
        return False
    move_segment_pos(tour, pos, i, L, j, reverse)
    gain[0] += delta
    dont_look[p] = 0
    dont_look[q] = 0
    dont_look[s0] = 0
    dont_look[sL] = 0
    dont_look[c] = 0
    dont_look[d] = 0
    return True


cdef bint _or_opt_improve_node(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                               const int64_t[:, ::1] cand, uint8_t[::1] dont_look, int64_t a,
                               int max_segment, bint allow_reverse, double* gain) noexcept nogil:
    # Or-opt for the segments that START at node a (lengths 1..max_segment): each segment end is
    # moved next to one of its candidates c when C[end, c] is below the edge removed at that end
    # (Bentley's pruning; lists sorted ascending so the scan breaks at the first failure).
    #   from s0's list: forward after c            (new edge c -> s0), or reversed before c (s0 -> c)
    #   from sL's list: forward before c           (new edge sL -> c), or reversed after c (c -> sL)
    # First-improvement: the first improving move is applied and True returned.
    cdef Py_ssize_t n = tour.shape[0], K = cand.shape[1], i = pos[a], k, m, L, end, j
    cdef int64_t p, q, s0 = a, sL, c
    cdef double g1
    if i == 0:
        return False
    p = tour[i - 1]
    for L in range(1, max_segment + 1):
        end = i + L - 1
        if end > n - 1:
            break
        sL = tour[end]
        q = tour[end + 1] if end + 1 < n else tour[0]
        # ---- candidates of the segment start s0
        g1 = C[p, s0]
        for m in range(K):
            c = cand[s0, m]
            if c == s0:
                continue
            if C[s0, c] >= g1:
                break
            k = pos[c]
            if k >= i and k <= end:
                continue
            if _or_opt_try(C, tour, pos, dont_look, i, L, k, False, p, q, s0, sL, c, gain):
                return True
            if allow_reverse and L > 1:
                j = k - 1 if k > 0 else n - 1
                if _or_opt_try(C, tour, pos, dont_look, i, L, j, True, p, q, s0, sL, c, gain):
                    return True
        # ---- candidates of the segment end sL
        g1 = C[sL, q]
        for m in range(K):
            c = cand[sL, m]
            if c == sL:
                continue
            if C[sL, c] >= g1:
                break
            k = pos[c]
            if k >= i and k <= end:
                continue
            j = k - 1 if k > 0 else n - 1
            if _or_opt_try(C, tour, pos, dont_look, i, L, j, False, p, q, s0, sL, c, gain):
                return True
            if allow_reverse and L > 1:
                if _or_opt_try(C, tour, pos, dont_look, i, L, k, True, p, q, s0, sL, c, gain):
                    return True
    return False


cpdef double or_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                            const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                            int max_segment, bint allow_reverse, int max_passes) noexcept nogil:
    """Or-opt descent with candidate lists and don't-look bits (symmetric matrices, plain TSP).

    For every active node ``a`` and every segment ``tour[i..i+L-1]`` starting at ``a``
    (``L = 1..max_segment``), the candidates of each segment end are scanned in ascending order
    while ``C[end, c] < C[removed edge at that end]`` (Bentley's pruning) and the segment is
    re-inserted next to ``c`` — after ``c`` or before it, forward or (``allow_reverse``)
    reversed — whenever the O(1) delta improves. First-improvement; the bits of the six
    touched nodes are reset on improvement and ``a``'s bit is set when nothing improves.
    The depot (position 0) is never part of a segment.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        **Symmetric** cost matrix (asymmetric matrices need :func:`local_search_generic`).
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation, kept consistent.
    cand : (n, k) int64, C-contiguous
        Candidate lists sorted ascending by ``C[i, :]``.
    dont_look : (n,) uint8, C-contiguous
        Don't-look bits, ``0`` = active; persistent across calls.
    max_segment : int
        Longest segment moved (Or-opt proper uses 3); ``<= 0`` returns immediately.
    allow_reverse : bool
        Also try the reversed insertion of segments longer than one node.
    max_passes : int
        Maximum number of sweeps over the active nodes; ``<= 0`` returns immediately.

    Returns
    -------
    float
        ``cost_after - cost_before`` (``<= 0``); ``0.0`` means nothing changed.

    Notes
    -----
    Improvement test ``delta < -1e-9 * max(1, removed)`` over the three removed edges. One
    sweep is O(n * k * max_segment) delta evaluations plus O(|i - j| + L) per applied move.
    ``noexcept nogil``.

    References
    ----------
    I. Or, "Traveling salesman-type combinatorial problems and their relation to the
    logistics of regional blood banking", PhD thesis, Northwestern University, 1976.
    J. L. Bentley, "Fast algorithms for geometric traveling salesman problems", 1992.
    """
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t a
    cdef double gain = 0.0
    cdef int passes = 0
    cdef bint improved_pass
    if n < 3 or max_segment <= 0 or cand.shape[1] == 0:
        return 0.0
    while passes < max_passes:
        passes += 1
        improved_pass = False
        for a in range(n):
            if dont_look[a]:
                continue
            while _or_opt_improve_node(C, tour, pos, cand, dont_look, a, max_segment, allow_reverse, &gain):
                improved_pass = True
            dont_look[a] = 1
        if not improved_pass:
            break
    return gain


# ====================================================================== generic descent (full evaluation)
cdef inline double _eval_scratch(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                 int64_t[::1] scratch, double max_time, double fixed_cost, int split,
                                 double[::1] dp, int64_t[::1] pred) noexcept nogil:
    # Copy tour into scratch (the caller then applies a move on scratch) -- kept separate so the
    # three move kinds share one evaluation path.
    memcpy(&scratch[0], &tour[0], tour.shape[0] * sizeof(int64_t))
    return problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)


cdef bint _generic_reverse(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                           int64_t[::1] pos, Py_ssize_t lo, Py_ssize_t hi, double max_time, double fixed_cost,
                           int split, int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                           double* cur) noexcept nogil:
    cdef double new
    if lo < 1 or hi <= lo:
        return False
    memcpy(&scratch[0], &tour[0], tour.shape[0] * sizeof(int64_t))
    reverse_segment(scratch, lo, hi)
    new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
    if not _improves(new - cur[0], fabs(cur[0])):
        return False
    reverse_segment_pos(tour, pos, lo, hi)
    cur[0] = new
    return True


cdef bint _generic_or_opt(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
                          Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, double max_time, double fixed_cost,
                          int split, int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                          double* cur) noexcept nogil:
    cdef Py_ssize_t n = tour.shape[0]
    cdef double new
    if i < 1 or i + L - 1 > n - 1 or j < 0 or j > n - 1 or (j >= i - 1 and j <= i + L - 1):
        return False
    memcpy(&scratch[0], &tour[0], n * sizeof(int64_t))
    move_segment(scratch, i, L, j, False)
    new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
    if not _improves(new - cur[0], fabs(cur[0])):
        return False
    move_segment_pos(tour, pos, i, L, j, False)
    cur[0] = new
    return True


cdef bint _generic_swap(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
                        Py_ssize_t x, Py_ssize_t y, double max_time, double fixed_cost, int split,
                        int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred, double* cur) noexcept nogil:
    cdef double new
    if x < 1 or y < 1 or x == y:
        return False
    memcpy(&scratch[0], &tour[0], tour.shape[0] * sizeof(int64_t))
    swap_positions(scratch, x, y)
    new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
    if not _improves(new - cur[0], fabs(cur[0])):
        return False
    swap_positions_pos(tour, pos, x, y)
    cur[0] = new
    return True


cdef bint _generic_try_pair(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                            int64_t[::1] pos,
                            int64_t a, int64_t c, double max_time, double fixed_cost, int split, int moves,
                            int max_segment, int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                            double* cur) noexcept nogil:
    # Every move of the enabled kinds that makes a and c adjacent; first improvement wins.
    cdef Py_ssize_t n = tour.shape[0], i = pos[a], k = pos[c], lo, hi, L, j
    if i < k:
        lo = i
        hi = k
    else:
        lo = k
        hi = i
    if moves & 1:
        # reverse [lo+1..hi]: new edges (tour[lo], tour[hi]) and (tour[lo+1], succ(hi))
        if _generic_reverse(C, T, tour, pos, lo + 1, hi, max_time, fixed_cost, split, scratch, dp, pred, cur):
            return True
        if lo >= 1:
            # reverse [lo..hi-1]: new edges (pred(lo), tour[hi-1]) and (tour[lo], tour[hi])
            if _generic_reverse(C, T, tour, pos, lo, hi - 1, max_time, fixed_cost, split, scratch, dp, pred,
                                cur):
                return True
        elif hi < n - 1:
            # lo is the depot: the mirror image, reverse [hi..n-1], closes the tour with (tour[hi], depot)
            if _generic_reverse(C, T, tour, pos, hi, n - 1, max_time, fixed_cost, split, scratch, dp, pred,
                                cur):
                return True
    if moves & 2:
        for L in range(1, max_segment + 1):
            # segment starting at a, inserted after c
            if _generic_or_opt(C, T, tour, pos, i, L, k, max_time, fixed_cost, split, scratch, dp, pred, cur):
                return True
            # segment ending at a, inserted before c (after pred c)
            j = k - 1 if k > 0 else n - 1
            if _generic_or_opt(C, T, tour, pos, i - L + 1, L, j, max_time, fixed_cost, split, scratch, dp,
                               pred, cur):
                return True
            # segment starting at c, inserted after a
            if _generic_or_opt(C, T, tour, pos, k, L, i, max_time, fixed_cost, split, scratch, dp, pred, cur):
                return True
            # segment ending at c, inserted before a (after pred a)
            j = i - 1 if i > 0 else n - 1
            if _generic_or_opt(C, T, tour, pos, k - L + 1, L, j, max_time, fixed_cost, split, scratch, dp,
                               pred, cur):
                return True
    if moves & 4:
        # a takes the place of c's predecessor / successor, and c the place of a's
        j = k - 1 if k > 0 else n - 1
        if _generic_swap(C, T, tour, pos, i, j, max_time, fixed_cost, split, scratch, dp, pred, cur):
            return True
        j = k + 1 if k + 1 < n else 0
        if _generic_swap(C, T, tour, pos, i, j, max_time, fixed_cost, split, scratch, dp, pred, cur):
            return True
        j = i - 1 if i > 0 else n - 1
        if _generic_swap(C, T, tour, pos, k, j, max_time, fixed_cost, split, scratch, dp, pred, cur):
            return True
        j = i + 1 if i + 1 < n else 0
        if _generic_swap(C, T, tour, pos, k, j, max_time, fixed_cost, split, scratch, dp, pred, cur):
            return True
    return False


cpdef double local_search_generic(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                                  int64_t[::1] pos, const int64_t[:, ::1] cand, double max_time,
                                  double fixed_cost, int split, int moves, int max_segment, int max_passes,
                                  int64_t[::1] scratch_tour, double[::1] dp,
                                  int64_t[::1] pred) noexcept nogil:
    """First-improvement descent by full re-evaluation: the multi-trip and asymmetric path.

    For every node ``a`` and every candidate ``c`` of ``a`` the moves of the enabled kinds
    that make ``a`` and ``c`` adjacent are built on ``scratch_tour``, priced with the full
    objective (``problem_cost`` — plain, greedy or optimal split, directional reads, so it is
    exact on ATSP and under a budget) and the first one that improves is applied to ``tour``.
    A pass sweeps every node; the descent stops after a pass without improvement or after
    ``max_passes`` passes. No don't-look bits: every pass is a full sweep.

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost and time matrices (``T`` is never read when ``max_time == inf``).
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation, kept consistent.
    cand : (n, k) int64, C-contiguous
        Candidate lists (``RoutingProblem.neighbours(k)``).
    max_time, fixed_cost, split : float, float, int
        The objective, as in :func:`problem_cost_py`.
    moves : int
        Bit mask: ``1`` = 2-opt (segment reversals), ``2`` = Or-opt without reversal
        (segment lengths ``1..max_segment``), ``4`` = swap of two nodes.
    max_segment : int
        Longest Or-opt segment (3 for Or-opt proper).
    max_passes : int
        Maximum number of sweeps; ``<= 0`` returns immediately.
    scratch_tour : (n,) int64, C-contiguous
        Caller-owned scratch tour (never shared across threads).
    dp, pred : (n,) float64 / int64, C-contiguous
        Caller-owned scratch of the optimal split; zero-length views are fine for the greedy
        split and plain TSP.

    Returns
    -------
    float
        ``cost_after - cost_before`` (``<= 0``), computed as the difference of the two full
        evaluations; ``0.0`` means nothing changed.

    Notes
    -----
    Improvement test ``new < cur - 1e-9 * max(1, |cur|)`` (§4.0). O(n) per candidate move
    (O(n * L) under the optimal split); documented ceiling around 2 000 nodes. Moves for the
    pair ``(a, c)``: the two (three when ``a`` is the depot) reversals that create the edge,
    the Or-opt insertions of a segment starting or ending at either node next to the other,
    and the swaps putting either node in the other's neighbouring position. ``noexcept nogil``.
    """
    cdef Py_ssize_t n = tour.shape[0], K = cand.shape[1], m
    cdef int64_t a, c
    cdef double cost0, cur
    cdef int passes = 0
    cdef bint improved_pass
    if n < 3 or K == 0 or max_passes <= 0 or moves == 0:
        return 0.0
    cur = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    cost0 = cur
    while passes < max_passes:
        passes += 1
        improved_pass = False
        for a in range(n):
            for m in range(K):
                c = cand[a, m]
                if c == a:
                    continue
                while _generic_try_pair(C, T, tour, pos, a, c, max_time, fixed_cost, split, moves,
                                        max_segment, scratch_tour, dp, pred, &cur):
                    improved_pass = True
        if not improved_pass:
            break
    return cur - cost0


# ====================================================================== construction
cpdef void nearest_neighbour_tour(const double[:, ::1] C, int64_t depot, int64_t[::1] out) noexcept nogil:
    """Nearest-neighbour tour from the depot: always the closest unvisited node, ties by lowest index.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix (rows are origins; asymmetric matrices are read directionally).
    depot : int
        Index of the depot, ``0 <= depot < n``.
    out : (n,) int64, C-contiguous
        Receives the tour, ``out[0] == depot``.

    Notes
    -----
    O(n^2) with an n-byte ``visited`` scratch from ``malloc``. If that allocation fails the
    function cannot raise (``noexcept nogil``) and writes the trivial tour ``depot`` followed by
    the other nodes in index order, which is still a valid permutation.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> C = np.array([[0, 5, 1, 4], [5, 0, 2, 9], [1, 2, 0, 7], [4, 9, 7, 0]], dtype=np.float64)
    >>> out = np.empty(4, dtype=np.int64)
    >>> core.nearest_neighbour_tour(C, 0, out)
    >>> out.tolist()
    [0, 2, 1, 3]
    """
    cdef Py_ssize_t n = C.shape[0], k, j, best
    cdef int64_t cur
    cdef double bestd
    cdef uint8_t* seen = <uint8_t*> malloc(n)
    out[0] = depot
    if seen == NULL:
        k = 1
        for j in range(n):
            if j != depot:
                out[k] = j
                k += 1
        return
    memset(seen, 0, n)
    seen[depot] = 1
    for k in range(1, n):
        cur = out[k - 1]
        best = -1
        bestd = INFINITY
        for j in range(n):
            if seen[j]:
                continue
            if best < 0 or C[cur, j] < bestd:
                bestd = C[cur, j]
                best = j
        out[k] = best
        seen[best] = 1
    free(seen)


# ====================================================================== Python wrappers (inline primitives)
# The cdef inline kernels of the .pxd are invisible from Python; these thin wrappers exist for the
# test-suite, diagnostics and benchmarks. They validate their arguments (ValueError) before calling.

cdef int _check_two_opt(Py_ssize_t n, Py_ssize_t i, Py_ssize_t j) except -1:
    if not (1 <= i < j <= n - 1):
        raise ValueError(f"2-opt/swap positions must satisfy 1 <= i < j <= n - 1, got i={i}, j={j}, n={n}")
    return 0


cdef int _check_or_opt(Py_ssize_t n, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j) except -1:
    if L < 1 or i < 1 or i + L - 1 > n - 1:
        raise ValueError(f"Or-opt segment must satisfy 1 <= i and i + L - 1 <= n - 1 with L >= 1, "
                         f"got i={i}, L={L}, n={n}")
    if j < 0 or j > n - 1 or (i - 1 <= j <= i + L - 1):
        raise ValueError(f"Or-opt insertion position must satisfy 0 <= j <= n - 1 and j not in "
                         f"[i - 1, i + L - 1], got i={i}, L={L}, j={j}, n={n}")
    return 0


cdef int _check_pos(const int64_t[::1] tour, const int64_t[::1] pos) except -1:
    if pos.shape[0] != tour.shape[0]:
        raise ValueError(f"pos must have the tour's length {tour.shape[0]}, got {pos.shape[0]}")
    return 0


def tour_cost_py(const double[:, ::1] C, const int64_t[::1] tour):
    """Closed-tour travel cost ``sum C[tour[k], tour[k+1]] + C[tour[n-1], tour[0]]`` (wraps ``tour_cost``).

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
    tour : (n,) int64, C-contiguous

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> C = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float64)
    >>> core.tour_cost_py(C, np.array([0, 1, 2], dtype=np.int64))
    6.0
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    return tour_cost(C, tour)


def greedy_split_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                         double max_time, double fixed_cost):
    """Greedy decoder cost of D1 (wraps the inline ``greedy_split_cost``).

    Leg ``a -> b`` joins the open trip iff ``t + T[a, b] + T[b, depot] <= max_time``; otherwise
    the trip closes at ``a`` and a new one opens ``depot -> b``. O(n).

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost and time matrices.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    max_time : float
        Per-trip budget in the units of ``T`` (a finite value; ``inf`` degenerates to one trip).
    fixed_cost : float
        ``people * extra_cost``, charged per trip beyond the first.

    Returns
    -------
    float
        Travel cost of the decoded trips plus ``fixed_cost * (n_trips - 1)``.

    Raises
    ------
    ValueError
        If ``C`` or ``T`` is not ``(n, n)`` or ``tour`` is empty.
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_square(T, tour.shape[0], "T")
    return greedy_split_cost(C, T, tour, max_time, fixed_cost)


def optimal_split_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                          double max_time, double fixed_cost):
    """Optimal (Prins 2004) decoder cost of D1 (wraps ``optimal_split_cost`` with its own scratch).

    The minimum-cost partition of the giant tour into consecutive trips that each fit
    ``max_time`` including the return leg: a shortest path on the DAG of feasible trips, O(n * L)
    with ``L`` the longest span whose outbound path fits. No triangle inequality is assumed.

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost and time matrices.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    max_time : float
        Per-trip budget in the units of ``T``.
    fixed_cost : float
        ``people * extra_cost``, charged per trip beyond the first.

    Returns
    -------
    float
        The optimal decoded cost, never above :func:`greedy_split_cost_py` for the same tour;
        ``inf`` when no feasible partition exists (a customer's round trip exceeds the budget).

    Raises
    ------
    ValueError
        If ``C`` or ``T`` is not ``(n, n)`` or ``tour`` is empty.
    MemoryError
        If the ``dp``/``pred`` scratch allocation fails.

    References
    ----------
    C. Prins, "A simple and effective evolutionary algorithm for the vehicle routing problem",
    Computers & Operations Research 31(12), 2004.
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_square(T, tour.shape[0], "T")
    return problem_cost_py(C, T, tour, max_time, fixed_cost, SPLIT_OPTIMAL)


def two_opt_delta_py(const double[:, ::1] C, const int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j):
    """Delta of reversing ``tour[i..j]`` (inclusive), exact for symmetric ``C`` (wraps ``two_opt_delta``).

    O(1): ``C[a, c] + C[b, d] - C[a, b] - C[c, d]`` with ``a = tour[i-1]``, ``b = tour[i]``,
    ``c = tour[j]`` and ``d = tour[j+1]`` (the depot when ``j == n - 1``).

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix; the value is exact only when ``C`` is symmetric.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    i, j : int
        Segment bounds, ``1 <= i < j <= n - 1``.

    Returns
    -------
    float
        ``cost(after) - cost(before)`` of the plain closed tour.

    Raises
    ------
    ValueError
        If the positions leave their domain or ``C`` is not ``(n, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> C = np.array([[0, 2, 9, 1], [2, 0, 1, 9], [9, 1, 0, 2], [1, 9, 2, 0]], dtype=np.float64)
    >>> tour = np.array([0, 2, 1, 3], dtype=np.int64)   # cost 9 + 1 + 9 + 1 = 20
    >>> core.two_opt_delta_py(C, tour, 1, 2)             # -> [0, 1, 2, 3], cost 2 + 1 + 2 + 1 = 6
    -14.0
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_two_opt(tour.shape[0], i, j)
    return two_opt_delta(C, tour, i, j)


def two_opt_delta_asym_py(const double[:, ::1] C, const int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j):
    """Delta of reversing ``tour[i..j]``, exact for asymmetric ``C`` (wraps ``two_opt_delta_asym``).

    Same move as :func:`two_opt_delta_py`, plus the direction change of every inner arc of the
    reversed segment; O(j - i).

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix, read directionally.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    i, j : int
        Segment bounds, ``1 <= i < j <= n - 1``.

    Returns
    -------
    float
        ``cost(after) - cost(before)`` of the plain closed tour.

    Raises
    ------
    ValueError
        If the positions leave their domain or ``C`` is not ``(n, n)``.
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_two_opt(tour.shape[0], i, j)
    return two_opt_delta_asym(C, tour, i, j)


def or_opt_delta_py(const double[:, ::1] C, const int64_t[::1] tour, Py_ssize_t i, Py_ssize_t L,
                    Py_ssize_t j, bint reverse=False):
    """Delta of moving ``tour[i..i+L-1]`` after position ``j`` (wraps ``or_opt_delta``).

    The segment ends up right after the node at position ``j`` (``j == 0``: right after the
    depot; ``j == n - 1``: at the end of the tour), optionally reversed. O(1): three edges are
    removed and three added. ``reverse=False`` is exact for asymmetric matrices;
    ``reverse=True`` only for symmetric ones (the inner arcs change direction).

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    i : int
        First position of the segment, ``i >= 1``.
    L : int
        Segment length, ``1 <= L`` and ``i + L - 1 <= n - 1`` (Or-opt proper uses ``L <= 3``).
    j : int
        Insertion anchor, ``0 <= j <= n - 1`` and ``j`` not in ``[i - 1, i + L - 1]`` (those
        values would leave the tour unchanged).
    reverse : bool, default False
        Insert the segment reversed.

    Returns
    -------
    float
        ``cost(after) - cost(before)`` of the plain closed tour; the move itself is
        :func:`move_segment_py` with the same arguments.

    Raises
    ------
    ValueError
        If ``(i, L, j)`` leaves the domain above or ``C`` is not ``(n, n)``.
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_or_opt(tour.shape[0], i, L, j)
    return or_opt_delta(C, tour, i, L, j, reverse)


def swap_delta_py(const double[:, ::1] C, const int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j):
    """Delta of exchanging the nodes at positions ``i < j``, exact on ATSP (wraps ``swap_delta``).

    O(1); the adjacent case ``j == i + 1`` (three edges instead of four) is handled.

    Parameters
    ----------
    C : (n, n) float64, C-contiguous
        Cost matrix, read directionally.
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0.
    i, j : int
        Positions, ``1 <= i < j <= n - 1``.

    Returns
    -------
    float
        ``cost(after) - cost(before)`` of the plain closed tour.

    Raises
    ------
    ValueError
        If the positions leave their domain or ``C`` is not ``(n, n)``.
    """
    _check_tour(tour)
    _check_square(C, tour.shape[0], "C")
    _check_two_opt(tour.shape[0], i, j)
    return swap_delta(C, tour, i, j)


def reverse_segment_py(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j):
    """Reverse ``tour[i..j]`` (inclusive) in place: the move priced by :func:`two_opt_delta_py`.

    Wraps the inline ``reverse_segment``; ``pos`` is not touched. O(j - i).

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    i, j : int
        Segment bounds, ``1 <= i < j <= n - 1``.

    Raises
    ------
    ValueError
        If the positions leave their domain.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> tour = np.arange(6, dtype=np.int64)
    >>> core.reverse_segment_py(tour, 2, 4)
    >>> tour.tolist()
    [0, 1, 4, 3, 2, 5]
    """
    _check_two_opt(tour.shape[0], i, j)
    reverse_segment(tour, i, j)


def reverse_segment_pos_py(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t j):
    """:func:`reverse_segment_py` keeping ``pos[node] == position`` (wraps ``reverse_segment_pos``).

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation (see :func:`rebuild_pos`); updated in place.
    i, j : int
        Segment bounds, ``1 <= i < j <= n - 1``.

    Raises
    ------
    ValueError
        If the positions leave their domain or ``pos`` has not the tour's length.
    """
    _check_two_opt(tour.shape[0], i, j)
    _check_pos(tour, pos)
    reverse_segment_pos(tour, pos, i, j)


def swap_positions_py(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j):
    """Exchange the nodes at positions ``i`` and ``j`` in place: the move priced by :func:`swap_delta_py`.

    Wraps the inline ``swap_positions``; ``pos`` is not touched. O(1).

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    i, j : int
        Positions, ``1 <= i < j <= n - 1``.

    Raises
    ------
    ValueError
        If the positions leave their domain.
    """
    _check_two_opt(tour.shape[0], i, j)
    swap_positions(tour, i, j)


def swap_positions_pos_py(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t j):
    """:func:`swap_positions_py` keeping ``pos[node] == position`` (wraps ``swap_positions_pos``).

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation; updated in place.
    i, j : int
        Positions, ``1 <= i < j <= n - 1``.

    Raises
    ------
    ValueError
        If the positions leave their domain or ``pos`` has not the tour's length.
    """
    _check_two_opt(tour.shape[0], i, j)
    _check_pos(tour, pos)
    swap_positions_pos(tour, pos, i, j)


def move_segment_py(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, bint reverse=False):
    """Move ``tour[i..i+L-1]`` so that it follows the node at position ``j`` (wraps ``move_segment``).

    The move priced by :func:`or_opt_delta_py` with the same arguments (positions after the
    segment shift by ``L`` when ``j > i``). Implemented as a rotation of the affected span by three
    reversals, O(|i - j| + L), no scratch memory; ``pos`` is not touched.

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    i, L, j : int
        Segment start, length and insertion anchor, in the domain of :func:`or_opt_delta_py`.
    reverse : bool, default False
        Insert the segment reversed.

    Raises
    ------
    ValueError
        If ``(i, L, j)`` leaves its domain.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute._core import _routing as core
    >>> tour = np.arange(6, dtype=np.int64)
    >>> core.move_segment_py(tour, 1, 2, 4)
    >>> tour.tolist()
    [0, 3, 4, 1, 2, 5]
    >>> core.move_segment_py(tour, 3, 2, 0, reverse=True)
    >>> tour.tolist()
    [0, 2, 1, 3, 4, 5]
    """
    _check_or_opt(tour.shape[0], i, L, j)
    move_segment(tour, i, L, j, reverse)


def move_segment_pos_py(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j,
                        bint reverse=False):
    """:func:`move_segment_py` keeping ``pos[node] == position`` (wraps ``move_segment_pos``).

    Parameters
    ----------
    tour : (n,) int64, C-contiguous
        Permutation with the depot at position 0; modified in place.
    pos : (n,) int64, C-contiguous
        Inverse permutation; updated in place.
    i, L, j : int
        Segment start, length and insertion anchor, in the domain of :func:`or_opt_delta_py`.
    reverse : bool, default False
        Insert the segment reversed.

    Raises
    ------
    ValueError
        If ``(i, L, j)`` leaves its domain or ``pos`` has not the tour's length.
    """
    _check_or_opt(tour.shape[0], i, L, j)
    _check_pos(tour, pos)
    move_segment_pos(tour, pos, i, L, j, reverse)
