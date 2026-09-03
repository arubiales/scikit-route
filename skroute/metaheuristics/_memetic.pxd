# skroute/metaheuristics/_memetic.pxd -- the memetic polish shared by the Genetic and AntColony kernels.
#
# A header, not a module (there is no ``_memetic.pyx``): the constants are an enum and the polish is a
# ``cdef inline`` function DEFINED here with its body, the only way Cython inlines one piece of code
# into several extension modules (D11, R5). ``_ga.pyx`` and ``_aco.pyx`` cimport it, so the descent
# schedule of the memetic polish lives in exactly one place.
from libc.stdint cimport int64_t, uint8_t
from libc.string cimport memset

from skroute._core._routing cimport (
    local_search_generic,
    or_opt_descent,
    rebuild_pos,
    two_opt_descent,
)


cdef enum:
    # Move mask shared with local_search_generic: 1 = two_opt, 2 = or_opt.
    MOVE_TWO_OPT = 1
    MOVE_OR_OPT = 2
    # Local-search mode: 0 = none, 1 = symmetric plain TSP (O(1) deltas), 2 = generic full evaluation.
    LS_NONE = 0
    LS_SYMMETRIC = 1
    LS_GENERIC = 2
    # A descent "to convergence": the kernels stop at a local optimum long before this cap.
    MAX_PASSES = 1000000


cdef inline void memetic_polish(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                                int64_t[::1] pos, const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                                double max_time, double fixed_cost, int split, int ls_mode, int ls_moves,
                                int64_t[::1] scratch_tour, double[::1] dp, int64_t[::1] pred) noexcept nogil:
    # Polish one index tour in place: the listed descents run to convergence (SPEC §4.3 accounting with an
    # unbounded pass count), alternating until none improves; ``ls_mode`` 0 = no polish, 1 = symmetric plain
    # TSP (two_opt_descent / or_opt_descent), 2 = generic (local_search_generic, for multi-trip and
    # asymmetric problems); ``ls_moves`` is the bit mask above. Don't-look bits are reset before every
    # descent because a finished descent leaves them all set. ``cand`` is never read when ``ls_mode == 0``.
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
            break  # a single descent already ended at its own local optimum
