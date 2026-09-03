"""Kernels of :class:`skroute.metaheuristics.AntColony` (SPEC §4.4, D10, D11).

Tour construction of a MAX-MIN Ant System over a pre-computed choice matrix
``choice[i, j] = tau[i, j] ** alpha * (1 / C[i, j]) ** beta`` and the per-ant polish and
evaluation. Every kernel is ``noexcept nogil``; the roulette wheel consumes exactly one
pre-drawn uniform per construction step (D10), so a fit is bit-identical for a given seed.
"""

from libc.stdint cimport int64_t, uint8_t
from libc.string cimport memset

from skroute._core._routing cimport (
    local_search_generic,
    or_opt_descent,
    problem_cost,
    rebuild_pos,
    two_opt_descent,
)

cdef int MOVE_TWO_OPT = 1
cdef int MOVE_OR_OPT = 2
cdef int LS_NONE = 0
cdef int LS_SYMMETRIC = 1
cdef int LS_GENERIC = 2
cdef int MAX_PASSES = 1000000


# ------------------------------------------------------------------ construction
cpdef void construct_tours(const double[:, ::1] choice, const int64_t[:, ::1] cand, int64_t depot,
                           const double[:, ::1] u, int64_t[:, ::1] tours, uint8_t[::1] visited,
                           double[::1] w) noexcept nogil:
    """Build one tour per ant by roulette over the unvisited candidates of the current node.

    ``choice`` is the ``(n, n)`` attractiveness matrix, ``cand`` the ``(n, k)`` candidate lists,
    ``u`` the ``(n_ants, n - 1)`` pre-drawn uniforms (one per step), ``tours`` the ``(n_ants, n)``
    output (every row starts at ``depot``); ``visited`` (``n``) and ``w`` (``k``) are scratch. When
    every candidate of the current node is already visited, the wheel spins over all unvisited
    nodes instead. A zero total (unreachable with positive pheromone) falls back to the last
    unvisited candidate.
    """
    cdef Py_ssize_t A = tours.shape[0], n = tours.shape[1], K = cand.shape[1]
    cdef Py_ssize_t a, s, k, j
    cdef int64_t cur, chosen
    cdef double total, thr, acc
    for a in range(A):
        memset(&visited[0], 0, n)
        tours[a, 0] = depot
        visited[depot] = 1
        cur = depot
        for s in range(n - 1):
            chosen = -1
            total = 0.0
            for k in range(K):
                j = cand[cur, k]
                if visited[j]:
                    w[k] = 0.0
                else:
                    w[k] = choice[cur, j]
                    total += w[k]
            if total > 0.0:
                thr = u[a, s] * total
                acc = 0.0
                for k in range(K):
                    if w[k] > 0.0:
                        acc += w[k]
                        if acc > thr:
                            chosen = cand[cur, k]
                            break
                if chosen < 0:  # rounding left thr >= acc at the end: the last unvisited candidate
                    for k in range(K - 1, -1, -1):
                        if w[k] > 0.0:
                            chosen = cand[cur, k]
                            break
            else:  # candidate list exhausted: every unvisited node
                for j in range(n):
                    if not visited[j]:
                        total += choice[cur, j]
                thr = u[a, s] * total
                acc = 0.0
                for j in range(n):
                    if not visited[j]:
                        acc += choice[cur, j]
                        if acc > thr:
                            chosen = j
                            break
                if chosen < 0:
                    for j in range(n - 1, -1, -1):
                        if not visited[j]:
                            chosen = j
                            break
            tours[a, s + 1] = chosen
            visited[chosen] = 1
            cur = chosen


# ------------------------------------------------------------------ polish and evaluation
cdef void _polish(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
                  const int64_t[:, ::1] cand, uint8_t[::1] dont_look, double max_time, double fixed_cost,
                  int split, int ls_mode, int ls_moves, int64_t[::1] scratch_tour, double[::1] dp,
                  int64_t[::1] pred) noexcept nogil:
    # The listed descents run to convergence, alternating until none improves; don't-look bits are
    # reset before every call because a finished descent leaves them all set.
    cdef Py_ssize_t n = tour.shape[0]
    cdef bint improved
    if ls_mode == LS_NONE or ls_moves == 0:
        return
    rebuild_pos(tour, pos)
    if ls_mode == LS_GENERIC:
        local_search_generic(C, T, tour, pos, cand, max_time, fixed_cost, split, ls_moves, 3, MAX_PASSES,
                             scratch_tour, dp, pred)
        return
    while True:
        improved = False
        if ls_moves & MOVE_TWO_OPT:
            memset(&dont_look[0], 0, n)
            if two_opt_descent(C, tour, pos, cand, dont_look, True, MAX_PASSES) < 0.0:
                improved = True
        if ls_moves & MOVE_OR_OPT:
            memset(&dont_look[0], 0, n)
            if or_opt_descent(C, tour, pos, cand, dont_look, 3, True, MAX_PASSES) < 0.0:
                improved = True
        if not improved or ls_moves == MOVE_TWO_OPT or ls_moves == MOVE_OR_OPT:
            break


cpdef void polish_and_evaluate(const double[:, ::1] C, const double[:, ::1] T, int64_t[:, ::1] tours,
                               double max_time, double fixed_cost, int split, int ls_mode, int ls_moves,
                               const int64_t[:, ::1] cand, int64_t[::1] tour, int64_t[::1] pos,
                               uint8_t[::1] dont_look, int64_t[::1] scratch_tour, double[::1] dp,
                               int64_t[::1] pred, double[::1] costs) noexcept nogil:
    """Polish every ant's tour in place (``ls_mode`` 0 none, 1 symmetric plain, 2 generic; ``ls_moves``
    bit mask 1 = two_opt, 2 = or_opt) and write its problem objective to ``costs``.

    ``tour``, ``pos``, ``dont_look``, ``scratch_tour``, ``dp`` and ``pred`` are caller-owned scratch.
    """
    cdef Py_ssize_t A = tours.shape[0], n = tours.shape[1], a, k
    for a in range(A):
        for k in range(n):
            tour[k] = tours[a, k]
        _polish(C, T, tour, pos, cand, dont_look, max_time, fixed_cost, split, ls_mode, ls_moves,
                scratch_tour, dp, pred)
        for k in range(n):
            tours[a, k] = tour[k]
        costs[a] = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
