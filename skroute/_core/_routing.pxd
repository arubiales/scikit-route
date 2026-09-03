# skroute/_core/_routing.pxd -- the frozen contract of the compiled core (SPEC §3.5).
#
# Every ``cdef inline`` function below is DEFINED here, with its body: that is the only way
# Cython inlines a primitive across extension modules (a body-less inline declaration breaks
# every cimporting module in the C compiler). The non-inline ``cdef``/``cpdef`` functions are
# declared here and defined in ``_routing.pyx``; cimporting modules reach them through the
# module's C-API. ``_routing.pyx`` never re-defines the inline functions.
#
# Conventions (SPEC §3.1): C and T are (n, n) C-contiguous float64 matrices, rows are origins;
# a tour is an int64 permutation of 0..n-1 whose position 0 holds the depot (kernels read
# ``depot = tour[0]``); the objective is always the CLOSED tour; positions are Py_ssize_t;
# the successor of position n-1 is position 0; the diagonal is never read.
#
# ``noexcept nogil`` applies to every function except ``problem_cost_py`` and ``trip_starts``:
# those two validate their arguments and malloc/free their own dp/pred scratch for the optimal
# split with the GIL held (MemoryError on failure), then release the GIL around the kernel call
# (amendment of 2026-09-03), so concurrent callers do not serialise on them.
from libc.math cimport INFINITY
from libc.stdint cimport int64_t, uint8_t


cpdef enum SplitRule:          # Python: _routing.SplitRule.SPLIT_GREEDY / .SPLIT_OPTIMAL (IntEnum)
    """Decoder of a giant tour into trips (SPEC D1): ``SPLIT_GREEDY`` (0) or ``SPLIT_OPTIMAL`` (1).

    Reached from Python as the ``IntEnum`` class ``_routing.SplitRule``; its members are not
    module attributes. C code compares an ``int split`` argument against the two values
    (``problem_cost`` dispatches on ``split == SPLIT_GREEDY``, anything else is the optimal split).
    """
    SPLIT_GREEDY = 0
    SPLIT_OPTIMAL = 1


# ------------------------------------------------------------------ evaluation
# max_time: +inf means plain TSP (T is then never read). fixed_cost = people * extra_cost.
# dp (float64[n]) and pred (int64[n]) are caller-owned scratch buffers used ONLY by the
# optimal split; pass zero-length views when split == SPLIT_GREEDY. Never share scratch across threads.

cdef inline double tour_cost(const double[:, ::1] C, const int64_t[::1] tour) noexcept nogil:
    # Travel cost of the closed tour tour[0] -> ... -> tour[n-1] -> tour[0]. O(n).
    cdef Py_ssize_t k, n = tour.shape[0]
    cdef double total = 0.0
    for k in range(n - 1):
        total += C[tour[k], tour[k + 1]]
    return total + C[tour[n - 1], tour[0]]


cdef inline double greedy_split_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                     double max_time, double fixed_cost) noexcept nogil:
    # Greedy decoder (D1): leg a->b joins the open trip iff t + T[a,b] + T[b,d] <= max_time,
    # else the trip closes at a and a new one opens d->b. Returns travel cost + (trips-1)*fixed_cost. O(n).
    # At k == 0 the "closing" leg a->d is the depot's own diagonal entry (the first customer's round
    # trip does not fit -- excluded by D5); it is skipped so that the diagonal is never read (§3.1).
    cdef Py_ssize_t k, n = tour.shape[0]
    cdef int64_t d = tour[0], a, b
    cdef int64_t trips = 1
    cdef double cost = 0.0, t = 0.0
    for k in range(n - 1):
        a = tour[k]
        b = tour[k + 1]
        if t + T[a, b] + T[b, d] <= max_time:
            t += T[a, b]
            cost += C[a, b]
        else:
            if k > 0:
                cost += C[a, d]
            cost += C[d, b]
            t = T[d, b]
            trips += 1
    cost += C[tour[n - 1], d]
    return cost + (trips - 1) * fixed_cost


# Optimal decoder (Prins 2004), no triangle inequality assumed. Defined in _routing.pyx.
cdef double optimal_split_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                               double max_time, double fixed_cost,
                               double[::1] dp, int64_t[::1] pred) noexcept nogil


cdef inline double problem_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                double max_time, double fixed_cost, int split,
                                double[::1] dp, int64_t[::1] pred) noexcept nogil:
    # dispatch: max_time == inf -> tour_cost; split == SPLIT_GREEDY -> greedy; else optimal
    if max_time == INFINITY:
        return tour_cost(C, tour)
    if split == SPLIT_GREEDY:
        return greedy_split_cost(C, T, tour, max_time, fixed_cost)
    return optimal_split_cost(C, T, tour, max_time, fixed_cost, dp, pred)


# Python entry point; validates with the GIL held, malloc/frees its own dp/pred scratch for the optimal
# split only (MemoryError on failure) and releases the GIL around the kernel call. NOT noexcept nogil.
# Rejects tours shorter than 2 (a depot-only tour would read the diagonal). Used by
# RoutingProblem.evaluate and tests.
cpdef double problem_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                             double max_time, double fixed_cost, int split)

# writes out[0..k] (out[0] == 1, out[k] == n) and returns k = n_trips; out has length n + 1.
# C and fixed_cost are needed only by the optimal split, for which it malloc/frees its own dp/pred
# scratch with the GIL held and releases the GIL around the DP (NOT noexcept nogil, MemoryError on
# failure, ValueError when no partition is feasible). Plain TSP -> k == 1. Rejects tours shorter than 2.
cpdef Py_ssize_t trip_starts(const double[:, ::1] T, const int64_t[::1] tour, double max_time, int split,
                             const double[:, ::1] C, double fixed_cost, int64_t[::1] out)

cpdef void trip_costs(const double[:, ::1] C, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil       # closed-trip travel cost per trip

cpdef void trip_times(const double[:, ::1] T, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil       # closed-trip duration per trip


# ------------------------------------------------------------------ move deltas
# Position domains (position 0 is the depot and never moves):
#   2-opt and swap: 1 <= i < j <= n-1.
#   Or-opt: 1 <= i, i + L - 1 <= n-1 (the segment never wraps), 0 <= j <= n-1 (j == 0 places the segment
#           right after the depot), j not in [i-1, i+L-1].
# Every [i..j] range in this file is INCLUSIVE, except double_bridge, whose segments are half-open.
# delta = cost(after) - cost(before) for the PLAIN closed tour. Successor of position n-1 is position 0.

cdef inline double two_opt_delta(const double[:, ::1] C, const int64_t[::1] tour,
                                 Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    # reverse tour[i..j], i < j. Exact iff the matrix is symmetric. O(1).
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t a = tour[i - 1], b = tour[i], c = tour[j]
    cdef int64_t d = tour[j + 1] if j + 1 < n else tour[0]
    return C[a, c] + C[b, d] - C[a, b] - C[c, d]


cdef inline double two_opt_delta_asym(const double[:, ::1] C, const int64_t[::1] tour,
                                      Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    # same move, exact for asymmetric matrices, O(j - i): the reversed inner arcs change direction.
    cdef Py_ssize_t n = tour.shape[0], k
    cdef int64_t a = tour[i - 1], b = tour[i], c = tour[j]
    cdef int64_t d = tour[j + 1] if j + 1 < n else tour[0]
    cdef double delta = C[a, c] + C[b, d] - C[a, b] - C[c, d]
    for k in range(i, j):
        delta += C[tour[k + 1], tour[k]] - C[tour[k], tour[k + 1]]
    return delta


cdef inline double or_opt_delta(const double[:, ::1] C, const int64_t[::1] tour,
                                Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, bint reverse) noexcept nogil:
    # move segment tour[i..i+L-1] (L in 1..3) so that it follows the node at position j
    # (j not in [i-1, i+L-1]); optionally reversed. reverse=False is exact for asymmetric
    # matrices; reverse=True is exact only when symmetric. O(1).
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t p = tour[i - 1], s0 = tour[i], sL = tour[i + L - 1]
    cdef int64_t q = tour[i + L] if i + L < n else tour[0]
    cdef int64_t c = tour[j]
    cdef int64_t d = tour[j + 1] if j + 1 < n else tour[0]
    cdef double removed = C[p, s0] + C[sL, q] + C[c, d]
    if reverse:
        return C[p, q] + C[c, sL] + C[s0, d] - removed
    return C[p, q] + C[c, s0] + C[sL, d] - removed


cdef inline double swap_delta(const double[:, ::1] C, const int64_t[::1] tour,
                              Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    # exchange the nodes at positions i and j; i < j required; adjacent (j == i+1) handled.
    # Exact for asymmetric matrices. O(1).
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t p = tour[i - 1], a = tour[i], b = tour[j]
    cdef int64_t s = tour[j + 1] if j + 1 < n else tour[0]
    cdef int64_t q, r
    if j == i + 1:
        return (C[p, b] + C[b, a] + C[a, s]) - (C[p, a] + C[a, b] + C[b, s])
    q = tour[i + 1]
    r = tour[j - 1]
    return (C[p, b] + C[b, q] + C[r, a] + C[a, s]) - (C[p, a] + C[a, q] + C[r, b] + C[b, s])


# ------------------------------------------------------------------ apply moves (in place)
# `_pos` variants keep pos[node] == position consistent; the plain variants do not touch pos.

cdef inline void reverse_segment(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    # reverses tour[i..j] INCLUSIVE, i < j -- the move priced by two_opt_delta(C, tour, i, j)
    cdef int64_t tmp
    while i < j:
        tmp = tour[i]
        tour[i] = tour[j]
        tour[j] = tmp
        i += 1
        j -= 1


cdef inline void reverse_segment_pos(int64_t[::1] tour, int64_t[::1] pos,
                                     Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    # same, inclusive
    cdef int64_t tmp
    while i < j:
        tmp = tour[i]
        tour[i] = tour[j]
        tour[j] = tmp
        pos[tour[i]] = i
        pos[tour[j]] = j
        i += 1
        j -= 1


cdef inline void swap_positions(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int64_t tmp = tour[i]
    tour[i] = tour[j]
    tour[j] = tmp


cdef inline void swap_positions_pos(int64_t[::1] tour, int64_t[::1] pos,
                                    Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef int64_t tmp = tour[i]
    tour[i] = tour[j]
    tour[j] = tmp
    pos[tour[i]] = i
    pos[tour[j]] = j


# the Or-opt move matching or_opt_delta; O(|i - j| + L) by rotating the affected span
cdef void move_segment(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j,
                       bint reverse) noexcept nogil

cdef void move_segment_pos(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t L,
                           Py_ssize_t j, bint reverse) noexcept nogil

# A B C D -> A C B D with A = tour[0..p1), B = [p1..p2), C = [p2..p3), D = [p3..n);
# 1 <= p1 < p2 < p3 <= n-1.
# Orientation-preserving: exact on ATSP. Writes to out (length n).
cpdef void double_bridge(const int64_t[::1] tour, Py_ssize_t p1, Py_ssize_t p2, Py_ssize_t p3,
                         int64_t[::1] out) noexcept nogil

cpdef void rebuild_pos(const int64_t[::1] tour, int64_t[::1] pos) noexcept nogil


# ------------------------------------------------------------------ descents
# Return value = cost_after - cost_before, always <= 0 (callers do `cost += returned`); 0.0 means "local
# optimum for this move, nothing changed". A pass = one sweep over the nodes whose don't-look bit is active.
# cand: int64 (n, k) candidate lists from RoutingProblem.neighbours(k); dont_look: uint8[n] (0 = active).
# Both use Bentley's neighbour-list scan with the pruning `C[a, succ(a)] > C[a, c]` and reset the
# don't-look bits of the touched endpoints on improvement (the four endpoints of a reversal, the six of
# a segment move: both ends of the three removed edges). Stop at a local optimum or after max_passes.
# "Local optimum" is the neighbour-list / don't-look-bit one: a move is found only from a node whose
# bit is active and whose new edge is shorter than its removed edge, and bits are reset only for the
# touched endpoints, so a node whose bit is set may miss a move that a later change elsewhere made
# available. With full lists, clearing the bits before every pass makes two_opt_descent an exact
# 2-opt local optimum; or_opt_descent additionally never scans from the two nodes whose gap closes
# (p, q), so it may stop short of a full Or-opt optimum even then. or_opt_descent examines, for an
# active node a, the segments starting and ending at a (a as a segment end) and the segments starting
# or ending at a's candidates inserted next to a (a as the insertion anchor, the depot included).
# The pos/cand/dont_look buffers are caller-owned and persist across calls (LocalSearch calls with
# max_passes=1).

cpdef double two_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                             const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                             bint first_improvement, int max_passes) noexcept nogil

cpdef double or_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                            const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                            int max_segment, bint allow_reverse, int max_passes) noexcept nogil

# Full-re-evaluation FIRST-IMPROVEMENT descent over the candidate neighbourhoods for the
# multi-trip objective and/or asymmetric matrices. moves is a bit mask: 1 = two_opt,
# 2 = or_opt (no reversal, segment lengths 1..max_segment), 4 = swap.
# O(n) per candidate move; documented ceiling ~2000 nodes.
cpdef double local_search_generic(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                                  int64_t[::1] pos, const int64_t[:, ::1] cand, double max_time,
                                  double fixed_cost, int split, int moves, int max_segment, int max_passes,
                                  int64_t[::1] scratch_tour, double[::1] dp, int64_t[::1] pred) noexcept nogil


# ------------------------------------------------------------------ construction
cpdef void nearest_neighbour_tour(const double[:, ::1] C, int64_t depot, int64_t[::1] out) noexcept nogil
