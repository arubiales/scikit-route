"""Compiled kernel of :class:`~skroute.metaheuristics.TabuSearch` (SPEC §4.4).

:func:`tabu_step` performs one tabu-search iteration: it scans the candidate neighbourhood,
picks the best admissible move (even a worsening one), marks the edges it removes as tabu and
applies it in place. The GIL is released around the ``noexcept nogil`` worker.

Neighbourhood. For every position ``p`` (node ``a = tour[p]``) and every candidate ``c`` of
``a`` at position ``q = pos[c]``: the 2-opt reversals that create the edge ``{a, c}`` —
``reverse tour[p+1..q]`` when ``q >= p + 2`` and ``reverse tour[q+1..p]`` when ``p >= q + 2`` —
and, when ``or_opt`` is set, the no-reversal Or-opt relocations of the segments of length 1..3
starting or ending at ``a`` next to ``c`` and vice versa (the family of the core's
``local_search_generic``). On the symmetric plain-TSP path (``fast_path``) moves are priced with
the O(1) ``two_opt_delta``/``or_opt_delta`` and the reversal of the whole tour (an orientation
flip) is skipped; on the generic path (a budget and/or an asymmetric matrix) every move is
applied on ``scratch`` and priced with ``problem_cost``.

Tabu attributes are the edges a move REMOVES (arcs when the matrix is asymmetric):
``tabu_until[x, y] = it + tenure + 1`` (both orientations when symmetric; saturated at
``INT32_MAX``), so the edge is tabu at exactly the ``tenure`` following iterations
``it + 1 .. it + tenure``. A move is tabu when any edge it ADDS satisfies
``tabu_until[x, y] > it``, unless it beats the best cost so far (aspiration). On an asymmetric
matrix a 2-opt reversal of ``tour[i..j]`` removes every arc of the span ``tour[i-1..j+1]`` and
adds their reverses, so all of them are marked and checked (on a symmetric matrix the inner
edges are unchanged and only the two boundary edges count). When no admissible move exists the
best move overall is applied so the search never stalls. The best tour is kept in a separate
``best`` buffer, written on strict improvement only; its cost is then recomputed from the tour,
so ``state[1]`` (and ``history_``) is bit-identical to a fresh evaluation of ``best``.
"""

from libc.math cimport fabs
from libc.stdint cimport INT32_MAX, int32_t, int64_t
from libc.string cimport memcpy

from skroute._core._routing cimport (
    move_segment,
    move_segment_pos,
    or_opt_delta,
    problem_cost,
    reverse_segment,
    reverse_segment_pos,
    two_opt_delta,
)

__all__ = ["tabu_step"]

cdef double REL_EPS = 1e-9


cdef struct Move:
    bint found
    int kind          # 0 = 2-opt reversal of tour[i..j]; 1 = Or-opt of tour[i..i+L-1] after position j
    Py_ssize_t i
    Py_ssize_t j
    Py_ssize_t L
    double cost       # objective of the tour after the move


cdef inline bint _improves(double new, double best) noexcept nogil:
    cdef double scale = fabs(best)
    if scale < 1.0:
        scale = 1.0
    return new < best - REL_EPS * scale


cdef inline void _record(Move* slot, int kind, Py_ssize_t i, Py_ssize_t j, Py_ssize_t L,
                         double cost) noexcept nogil:
    # Keep the cheapest move; ties go to the first one found (deterministic scan order).
    if slot.found and cost >= slot.cost:
        return
    slot.found = True
    slot.kind = kind
    slot.i = i
    slot.j = j
    slot.L = L
    slot.cost = cost


cdef inline bint _is_tabu(const int32_t[:, ::1] until, int64_t x, int64_t y, int32_t it) noexcept nogil:
    return until[x, y] > it


cdef inline void _mark(int32_t[:, ::1] until, int64_t x, int64_t y, int32_t value,
                       bint symmetric) noexcept nogil:
    until[x, y] = value
    if symmetric:
        until[y, x] = value


cdef inline bint _two_opt_adds_tabu(const int64_t[::1] tour, const int32_t[:, ::1] until, int32_t it,
                                    Py_ssize_t i, Py_ssize_t j, bint symmetric) noexcept nogil:
    # Edges added by reversing tour[i..j]: (a, c) and (b, d); on an asymmetric matrix also the
    # reversed inner arcs (tour[k+1], tour[k]), k in i..j-1 (undirected inner edges do not change).
    cdef Py_ssize_t n = tour.shape[0], k
    cdef int64_t a = tour[i - 1], b = tour[i], c = tour[j]
    cdef int64_t d = tour[j + 1] if j + 1 < n else tour[0]
    if _is_tabu(until, a, c, it) or _is_tabu(until, b, d, it):
        return True
    if symmetric:
        return False
    for k in range(i, j):
        if _is_tabu(until, tour[k + 1], tour[k], it):
            return True
    return False


cdef inline void _consider_two_opt(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                   const int32_t[:, ::1] until, int32_t it, double cur, double best_cost,
                                   double max_time, double fixed_cost, int split, bint fast_path,
                                   bint symmetric, int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                                   Py_ssize_t i, Py_ssize_t j, Move* adm, Move* any) noexcept nogil:
    # Reverse tour[i..j], 1 <= i < j <= n-1: removes (a, b), (c, d); adds (a, c), (b, d) -- and, on an
    # asymmetric matrix, replaces every inner arc of the span by its reverse.
    cdef Py_ssize_t n = tour.shape[0]
    cdef double new
    if fast_path:
        new = cur + two_opt_delta(C, tour, i, j)
    else:
        memcpy(&scratch[0], &tour[0], n * sizeof(int64_t))
        reverse_segment(scratch, i, j)
        new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
    _record(any, 0, i, j, 0, new)
    if _improves(new, best_cost) or not _two_opt_adds_tabu(tour, until, it, i, j, symmetric):
        _record(adm, 0, i, j, 0, new)


cdef inline void _consider_or_opt(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                  const int32_t[:, ::1] until, int32_t it, double cur, double best_cost,
                                  double max_time, double fixed_cost, int split, bint fast_path,
                                  int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                                  Py_ssize_t i, Py_ssize_t L, Py_ssize_t j,
                                  Move* adm, Move* any) noexcept nogil:
    # Move tour[i..i+L-1] after position j without reversal:
    # removes (p, s0), (sL, q), (c, d); adds (p, q), (c, s0), (sL, d).
    cdef Py_ssize_t n = tour.shape[0]
    cdef int64_t p, s0, sL, q, c, d
    cdef double new
    if i < 1 or i + L - 1 > n - 1 or j < 0 or j > n - 1 or (j >= i - 1 and j <= i + L - 1):
        return
    p = tour[i - 1]
    s0 = tour[i]
    sL = tour[i + L - 1]
    q = tour[i + L] if i + L < n else tour[0]
    c = tour[j]
    d = tour[j + 1] if j + 1 < n else tour[0]
    if fast_path:
        new = cur + or_opt_delta(C, tour, i, L, j, False)
    else:
        memcpy(&scratch[0], &tour[0], n * sizeof(int64_t))
        move_segment(scratch, i, L, j, False)
        new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
    _record(any, 1, i, j, L, new)
    if _improves(new, best_cost) or not (_is_tabu(until, p, q, it) or _is_tabu(until, c, s0, it)
                                         or _is_tabu(until, sL, d, it)):
        _record(adm, 1, i, j, L, new)


cdef bint _tabu_step(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
                     const int64_t[:, ::1] cand, int32_t[:, ::1] until, int32_t it, int32_t tenure,
                     double max_time, double fixed_cost, int split, bint fast_path, bint symmetric,
                     bint or_opt, int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                     int64_t[::1] best, double[::1] state) noexcept nogil:
    cdef Py_ssize_t n = tour.shape[0], K = cand.shape[1], p, q, m, L, i, j, k
    cdef int64_t a, c, x0, x1, x2, x3, x4, x5
    # tabu at it + 1 .. it + tenure (exactly `tenure` iterations); saturated so a huge tenure is "for ever"
    cdef int64_t mark = <int64_t>it + tenure + 1
    cdef int32_t value = INT32_MAX if mark > INT32_MAX else <int32_t>mark
    cdef double cur, best_cost
    cdef Move adm, any, chosen
    adm.found = False
    any.found = False
    # The current cost is recomputed once per iteration (O(n) against O(n K) move evaluations)
    # so the O(1) deltas of the fast path never drift away from the true cost of the tour.
    cur = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    best_cost = state[1]
    for p in range(n):
        a = tour[p]
        for m in range(K):
            c = cand[a, m]
            if c == a:
                continue
            q = pos[c]
            # reversing tour[1..n-1] on a symmetric plain TSP only flips the orientation: not a move
            if q >= p + 2 and not (fast_path and p == 0 and q == n - 1):
                _consider_two_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                  fast_path, symmetric, scratch, dp, pred, p + 1, q, &adm, &any)
            if p >= q + 2 and not (fast_path and q == 0 and p == n - 1):
                _consider_two_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                  fast_path, symmetric, scratch, dp, pred, q + 1, p, &adm, &any)
            if or_opt:
                for L in range(1, 4):
                    # segment starting at a, inserted after c
                    _consider_or_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                     fast_path, scratch, dp, pred, p, L, q, &adm, &any)
                    # segment ending at a, inserted before c
                    _consider_or_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                     fast_path, scratch, dp, pred, p - L + 1, L, q - 1 if q > 0 else n - 1,
                                     &adm, &any)
                    # segment starting at c, inserted after a
                    _consider_or_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                     fast_path, scratch, dp, pred, q, L, p, &adm, &any)
                    # segment ending at c, inserted before a
                    _consider_or_opt(C, T, tour, until, it, cur, best_cost, max_time, fixed_cost, split,
                                     fast_path, scratch, dp, pred, q - L + 1, L, p - 1 if p > 0 else n - 1,
                                     &adm, &any)
    chosen = adm if adm.found else any
    if not chosen.found:
        state[0] = cur
        state[1] = best_cost
        return False
    i = chosen.i
    j = chosen.j
    if chosen.kind == 0:
        if symmetric:
            x0 = tour[i - 1]
            x1 = tour[i]
            x2 = tour[j]
            x3 = tour[j + 1] if j + 1 < n else tour[0]
            _mark(until, x0, x1, value, True)
            _mark(until, x2, x3, value, True)
        else:
            # the reversal removes every arc of the span tour[i-1..j+1] (the inner ones change direction)
            for k in range(i - 1, j + 1):
                _mark(until, tour[k], tour[k + 1] if k + 1 < n else tour[0], value, False)
        reverse_segment_pos(tour, pos, i, j)
    else:
        L = chosen.L
        x0 = tour[i - 1]
        x1 = tour[i]
        x2 = tour[i + L - 1]
        x3 = tour[i + L] if i + L < n else tour[0]
        x4 = tour[j]
        x5 = tour[j + 1] if j + 1 < n else tour[0]
        _mark(until, x0, x1, value, symmetric)
        _mark(until, x2, x3, value, symmetric)
        _mark(until, x4, x5, value, symmetric)
        move_segment_pos(tour, pos, i, L, j, False)
    cur = chosen.cost
    if _improves(cur, best_cost):
        # recomputed from the tour: the O(1) delta of the fast path carries a rounding error, and
        # state[1] must be bit-identical to what the base class recomputes into ``cost_``
        cur = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
        best_cost = cur
        memcpy(&best[0], &tour[0], n * sizeof(int64_t))
    state[0] = cur
    state[1] = best_cost
    return True


def tabu_step(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
              const int64_t[:, ::1] cand, int32_t[:, ::1] until, int it, int tenure,
              double max_time, double fixed_cost, int split, bint fast_path, bint symmetric, bint or_opt,
              int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred, int64_t[::1] best,
              double[::1] state):
    """One tabu-search iteration on ``tour`` (in place): best admissible move, tabu marks, apply.

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost and time matrices (``T`` is never read when ``max_time == inf``).
    tour, pos : (n,) int64, C-contiguous
        The current tour (depot first) and its inverse permutation; both updated in place.
    cand : (n, k) int64, C-contiguous
        Candidate lists (``RoutingProblem.neighbours(k)``).
    until : (n, n) int32, C-contiguous
        ``tabu_until[x, y]``: the edge/arc ``(x, y)`` is tabu while ``until[x, y] > it``. The
        applied move writes ``it + tenure + 1`` (saturated at ``INT32_MAX``) on every edge it
        removes, so they are tabu at the ``tenure`` following iterations ``it + 1 .. it + tenure``.
    it : int
        Iteration counter (0-based).
    tenure : int >= 1
        Tabu tenure of the edges removed by the applied move, in iterations.
    max_time, fixed_cost, split : float, float, int
        The objective, as in :func:`skroute._core._routing.problem_cost_py`.
    fast_path : bool
        ``True`` for a symmetric plain TSP (O(1) deltas); ``False`` for the full-evaluation path.
    symmetric : bool
        Mark both orientations of a removed edge when ``True``. When ``False`` (asymmetric matrix)
        a 2-opt reversal marks every arc of the reversed span and is tabu if any reversed inner
        arc is tabu, not only the two boundary arcs.
    or_opt : bool
        Add the no-reversal Or-opt relocations (segment lengths 1..3) to the neighbourhood.
    scratch : (n,) int64, C-contiguous
        Caller-owned scratch tour used by the full-evaluation path.
    dp, pred : (n,) float64 / int64
        Caller-owned scratch of the optimal split (zero-length views are fine otherwise).
    best : (n,) int64, C-contiguous
        The best tour so far; overwritten on strict improvement of ``state[1]`` only.
    state : (2,) float64
        In: ``state[1]`` = best cost so far. Out: ``state[0]`` = current cost after the move,
        ``state[1]`` = best cost so far, recomputed from the tour whenever ``best`` is written
        (bit-identical to ``problem_cost`` of ``best``).

    Returns
    -------
    bool
        ``True`` if a move was applied, ``False`` when the neighbourhood is empty.

    Notes
    -----
    ``noexcept nogil`` worker; the GIL is released. Ties between equally good moves go to the
    first one in scan order (positions ascending, candidates by increasing distance).
    """
    cdef bint applied
    cdef Py_ssize_t n = tour.shape[0]
    if pos.shape[0] != n or scratch.shape[0] != n or best.shape[0] != n:
        raise ValueError("pos, scratch and best must have the length of tour")
    if cand.shape[0] != n or until.shape[0] != n or until.shape[1] != n:
        raise ValueError("cand must be (n, k) and until (n, n)")
    if state.shape[0] != 2:
        raise ValueError("state must have length 2")
    if tenure < 1:
        raise ValueError("tenure must be >= 1")
    with nogil:
        applied = _tabu_step(C, T, tour, pos, cand, until, it, tenure, max_time, fixed_cost, split,
                             fast_path, symmetric, or_opt, scratch, dp, pred, best, state)
    return applied
