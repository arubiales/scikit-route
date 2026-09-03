# TEMPORARY STUB — replaced by the core work package at merge
"""Minimal implementation of the Python-visible part of the core contract (SPEC §3.5).

Only the ``cpdef`` functions the spine needs are defined here, with exactly the
signatures of ``_routing.pxd`` and the decoder semantics stated in §3.5, written
plainly (no inline primitives, no speed). The lead discards this file when the
real core lands.
"""
from libc.math cimport INFINITY, isfinite
from libc.stdint cimport int64_t
from libc.stdlib cimport free, malloc


cpdef enum SplitRule:
    SPLIT_GREEDY = 0
    SPLIT_OPTIMAL = 1


# ------------------------------------------------------------------ evaluation
cdef double _tour_cost(const double[:, ::1] C, const int64_t[::1] tour) noexcept nogil:
    cdef Py_ssize_t k, n = tour.shape[0]
    cdef double total = 0.0
    for k in range(n - 1):
        total += C[tour[k], tour[k + 1]]
    return total + C[tour[n - 1], tour[0]]


cdef double _greedy_split(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                          double max_time, double fixed_cost, int64_t* starts,
                          Py_ssize_t* n_trips) noexcept nogil:
    """Greedy decoder of D1. Writes the trip starts into ``starts`` when it is not NULL."""
    cdef Py_ssize_t k, n = tour.shape[0], trips = 1
    cdef int64_t d = tour[0], a, b
    cdef double t = 0.0, cost = 0.0
    if starts != NULL:
        starts[0] = 1
    for k in range(n - 1):
        a = tour[k]
        b = tour[k + 1]
        if t + T[a, b] + T[b, d] <= max_time:
            t += T[a, b]
            cost += C[a, b]
        else:
            cost += C[a, d] + C[d, b]
            t = T[d, b]
            if starts != NULL:
                starts[trips] = k + 1
            trips += 1
    cost += C[tour[n - 1], d]
    if starts != NULL:
        starts[trips] = n
    n_trips[0] = trips
    return cost + (trips - 1) * fixed_cost


cdef double _optimal_split(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                           double max_time, double fixed_cost, double* dp, int64_t* pred) noexcept nogil:
    """Optimal decoder of D1 (Prins 2004) over caller-owned ``dp``/``pred`` of length n."""
    cdef Py_ssize_t n = tour.shape[0], m = n - 1, i, j
    cdef int64_t d = tour[0], cj, ci, cprev
    cdef double open_time, path_cost, cand
    dp[0] = 0.0
    for i in range(1, m + 1):
        dp[i] = INFINITY
        pred[i] = -1
    pred[0] = -1
    for j in range(m):
        if not isfinite(dp[j]):
            continue
        cj = tour[j + 1]
        open_time = T[d, cj]
        path_cost = C[d, cj]
        i = j + 1
        while i <= m:
            ci = tour[i]
            if i > j + 1:
                cprev = tour[i - 1]
                open_time += T[cprev, ci]
                path_cost += C[cprev, ci]
            if open_time > max_time:
                break
            if open_time + T[ci, d] <= max_time:
                cand = dp[j] + path_cost + C[ci, d]
                if j > 0:
                    cand += fixed_cost
                if cand < dp[i]:
                    dp[i] = cand
                    pred[i] = j
            i += 1
    return dp[m]


cpdef double problem_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                             double max_time, double fixed_cost, int split):
    """Objective of D1 for an index tour: plain tour cost when ``max_time`` is +inf, else the
    greedy or the optimal decoder. Holds the GIL; owns its scratch."""
    cdef Py_ssize_t n = tour.shape[0], n_trips
    cdef double* dp
    cdef int64_t* pred
    cdef double result
    if not isfinite(max_time):
        return _tour_cost(C, tour)
    if split == SPLIT_GREEDY:
        return _greedy_split(C, T, tour, max_time, fixed_cost, NULL, &n_trips)
    dp = <double*> malloc(n * sizeof(double))
    pred = <int64_t*> malloc(n * sizeof(int64_t))
    if dp == NULL or pred == NULL:
        free(dp)
        free(pred)
        raise MemoryError()
    result = _optimal_split(C, T, tour, max_time, fixed_cost, dp, pred)
    free(dp)
    free(pred)
    return result


cpdef Py_ssize_t trip_starts(const double[:, ::1] T, const int64_t[::1] tour, double max_time, int split,
                             const double[:, ::1] C, double fixed_cost, int64_t[::1] out):
    """Writes ``out[0..k]`` (``out[0] == 1``, ``out[k] == n``) and returns ``k = n_trips``."""
    cdef Py_ssize_t n = tour.shape[0], m = n - 1, k, i, count
    cdef int64_t* starts
    cdef double* dp
    cdef int64_t* pred
    if not isfinite(max_time):
        out[0] = 1
        out[1] = n
        return 1
    if split == SPLIT_GREEDY:
        starts = <int64_t*> malloc((n + 1) * sizeof(int64_t))
        if starts == NULL:
            raise MemoryError()
        _greedy_split(C, T, tour, max_time, fixed_cost, starts, &k)
        for i in range(k + 1):
            out[i] = starts[i]
        free(starts)
        return k
    dp = <double*> malloc(n * sizeof(double))
    pred = <int64_t*> malloc(n * sizeof(int64_t))
    if dp == NULL or pred == NULL:
        free(dp)
        free(pred)
        raise MemoryError()
    _optimal_split(C, T, tour, max_time, fixed_cost, dp, pred)
    # follow pred from m: the block boundaries in customer space are m, pred[m], ..., 0
    count = 0
    i = m
    while i > 0:
        count += 1
        i = pred[i]
    # boundaries b_0 = 0 < b_1 < ... < b_count = m; trip starts are b + 1
    k = count
    i = m
    while i > 0:
        out[count] = i + 1
        count -= 1
        i = pred[i]
    out[0] = 1
    free(dp)
    free(pred)
    return k


cpdef void trip_costs(const double[:, ::1] C, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil:
    """Travel cost of every closed trip ``depot -> tour[a..b-1] -> depot``."""
    cdef Py_ssize_t t, k, a, b, n_trips = starts.shape[0] - 1
    cdef int64_t d = tour[0]
    cdef double c
    for t in range(n_trips):
        a = starts[t]
        b = starts[t + 1]
        c = C[d, tour[a]]
        for k in range(a, b - 1):
            c += C[tour[k], tour[k + 1]]
        c += C[tour[b - 1], d]
        out[t] = c


cpdef void trip_times(const double[:, ::1] T, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil:
    """Duration of every closed trip ``depot -> tour[a..b-1] -> depot``."""
    trip_costs(T, tour, starts, out)


# ------------------------------------------------------------------ construction
cpdef void nearest_neighbour_tour(const double[:, ::1] C, int64_t depot, int64_t[::1] out) noexcept nogil:
    """Greedy nearest-neighbour tour from ``depot`` over ``C``; ties go to the lowest index."""
    cdef Py_ssize_t n = C.shape[0], k, j, best
    cdef double best_d
    cdef char* seen = <char*> malloc(n)
    if seen == NULL:
        return
    for j in range(n):
        seen[j] = 0
    out[0] = depot
    seen[depot] = 1
    for k in range(1, n):
        best = -1
        best_d = INFINITY
        for j in range(n):
            if not seen[j] and C[out[k - 1], j] < best_d:
                best_d = C[out[k - 1], j]
                best = j
        out[k] = best
        seen[best] = 1
    free(seen)
