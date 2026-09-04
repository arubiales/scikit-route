"""Compiled kernels of `SimulatedAnnealing` (SPEC §4.4, D10).

Two entry points, both releasing the GIL around a ``noexcept nogil`` worker:

* `anneal_level` runs one temperature level: ``m`` proposals whose randomness
  (``u``, ``ri``, ``rj``, ``mv``) the Python side pre-drew from its ``Generator``;
* `sample_deltas` prices the same proposals on a fixed tour without applying them
  (the ``t0="auto"`` calibration).

Draw -> move mapping (binding, §4.4). ``mv[s]`` is a move code: ``0`` = 2-opt, ``1`` = Or-opt,
``2`` = swap; ``ri[s]``, ``rj[s]`` are positions in ``1..n-1``.

* 2-opt and swap: ``i, j = min(ri, rj), max(ri, rj)``; invalid if ``i == j``.
* Or-opt: ``i = ri``, ``L = 1 + (rj % 3)``, ``j = rj`` (segment ``tour[i..i+L-1]`` moved after
  the node at position ``j``, never reversed); invalid if ``i + L - 1 > n-1`` or
  ``j in [i-1, i+L-1]``.

An invalid draw is a rejected proposal: it counts towards the level's ``m`` proposals,
consumes its ``u`` and changes nothing.

Two evaluation paths: when ``fast_path`` is set (symmetric matrix, plain TSP) every move is
priced with the O(1) deltas of the core and applied in place on acceptance; otherwise the
move is applied on ``scratch`` and priced with ``problem_cost`` (exact on ATSP and under a
budget), and the scratch is copied back on acceptance. The best tour lives in its own
``best`` buffer and is copied into on strict improvement only, so the tour and the best can
never alias (the 1.0 bug). When a level wrote ``best``, its cost is recomputed from the buffer
once at the end of the level, so ``state[1]`` (hence ``history_``) is bit-identical to a fresh
evaluation of ``best`` and never carries the rounding of the accumulated O(1) deltas.
"""

from libc.math cimport exp, fabs, NAN
from libc.stdint cimport int64_t
from libc.string cimport memcpy

from skroute._core._routing cimport (
    move_segment,
    or_opt_delta,
    problem_cost,
    reverse_segment,
    swap_delta,
    swap_positions,
    two_opt_delta,
)

__all__ = ["anneal_level", "sample_deltas"]

cdef double REL_EPS = 1e-9


cdef inline bint _improves(double new, double best) noexcept nogil:
    # The §4.0 improvement test: new < best - 1e-9 * max(1, |best|).
    cdef double scale = fabs(best)
    if scale < 1.0:
        scale = 1.0
    return new < best - REL_EPS * scale


cdef inline bint _decode(Py_ssize_t n, int64_t ri, int64_t rj, int64_t code,
                         Py_ssize_t* i, Py_ssize_t* j, Py_ssize_t* L) noexcept nogil:
    # The binding draw -> move mapping; returns False for an invalid draw.
    if code == 1:
        L[0] = 1 + (rj % 3)
        i[0] = ri
        j[0] = rj
        if i[0] + L[0] - 1 > n - 1:
            return False
        if j[0] >= i[0] - 1 and j[0] <= i[0] + L[0] - 1:
            return False
        return True
    if ri == rj:
        return False
    if ri < rj:
        i[0] = ri
        j[0] = rj
    else:
        i[0] = rj
        j[0] = ri
    return True


cdef inline void _apply(int64_t[::1] tour, int64_t code, Py_ssize_t i, Py_ssize_t j,
                        Py_ssize_t L) noexcept nogil:
    if code == 0:
        reverse_segment(tour, i, j)
    elif code == 1:
        move_segment(tour, i, L, j, False)
    else:
        swap_positions(tour, i, j)


cdef inline double _fast_delta(const double[:, ::1] C, const int64_t[::1] tour, int64_t code,
                               Py_ssize_t i, Py_ssize_t j, Py_ssize_t L) noexcept nogil:
    if code == 0:
        return two_opt_delta(C, tour, i, j)
    if code == 1:
        return or_opt_delta(C, tour, i, L, j, False)
    return swap_delta(C, tour, i, j)


cdef Py_ssize_t _anneal_level(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                              int64_t[::1] best, const double[::1] u, const int64_t[::1] ri,
                              const int64_t[::1] rj, const int64_t[::1] mv, double temperature,
                              double max_time, double fixed_cost, int split, bint fast_path,
                              int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                              double[::1] state) noexcept nogil:
    cdef Py_ssize_t n = tour.shape[0], m = u.shape[0], s, i = 0, j = 0, L = 0, accepted = 0
    cdef size_t nbytes = n * sizeof(int64_t)
    cdef int64_t code
    cdef bint best_written = False
    cdef double cur, best_cost, delta, new
    # The current cost is recomputed once per level (O(n), against 10n proposals) so the
    # accumulated O(1) deltas cannot drift away from the true cost of the tour.
    cur = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    best_cost = state[1]
    for s in range(m):
        code = mv[s]
        if not _decode(n, ri[s], rj[s], code, &i, &j, &L):
            continue                                   # invalid draw = rejected proposal
        if fast_path:
            delta = _fast_delta(C, tour, code, i, j, L)
            if delta <= 0.0 or u[s] < exp(-delta / temperature):
                _apply(tour, code, i, j, L)
                cur += delta
                accepted += 1
                if _improves(cur, best_cost):
                    best_cost = cur
                    best_written = True
                    memcpy(&best[0], &tour[0], nbytes)
        else:
            memcpy(&scratch[0], &tour[0], nbytes)
            _apply(scratch, code, i, j, L)
            new = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred)
            delta = new - cur
            if delta <= 0.0 or u[s] < exp(-delta / temperature):
                memcpy(&tour[0], &scratch[0], nbytes)
                cur = new
                accepted += 1
                if _improves(cur, best_cost):
                    best_cost = cur
                    best_written = True
                    memcpy(&best[0], &tour[0], nbytes)
    if best_written:
        # one O(n) recompute per improving level: the best cost reported (history_) must be
        # bit-identical to what the base class recomputes into ``cost_`` from the same buffer
        best_cost = problem_cost(C, T, best, max_time, fixed_cost, split, dp, pred)
    state[0] = cur
    state[1] = best_cost
    return accepted


def anneal_level(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] best,
                 const double[::1] u, const int64_t[::1] ri, const int64_t[::1] rj, const int64_t[::1] mv,
                 double temperature, double max_time, double fixed_cost, int split, bint fast_path,
                 int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred, double[::1] state):
    """Run one temperature level of ``len(u)`` Metropolis proposals on ``tour`` (in place).

    Parameters
    ----------
    C, T : (n, n) float64, C-contiguous
        Cost and time matrices (``T`` is never read when ``max_time == inf``).
    tour : (n,) int64, C-contiguous
        The current tour, depot first; modified in place.
    best : (n,) int64, C-contiguous
        The best tour so far; overwritten only on strict improvement of ``state[1]``.
    u : (m,) float64
        Uniform draws in ``[0, 1)``, one per proposal.
    ri, rj : (m,) int64
        Position draws in ``1..n-1``.
    mv : (m,) int64
        Move codes: ``0`` = 2-opt, ``1`` = Or-opt (no reversal), ``2`` = swap.
    temperature : float > 0
        Temperature of the level.
    max_time, fixed_cost, split : float, float, int
        The objective, as in `skroute._core._routing.problem_cost_py`.
    fast_path : bool
        ``True`` for a symmetric plain TSP (O(1) deltas); ``False`` for the full-evaluation path.
    scratch : (n,) int64, C-contiguous
        Caller-owned scratch tour used by the full-evaluation path.
    dp, pred : (n,) float64 / int64
        Caller-owned scratch of the optimal split (zero-length views are fine otherwise).
    state : (2,) float64
        In: ``state[1]`` = best cost so far. Out: ``state[0]`` = current cost after the level,
        ``state[1]`` = best cost so far (recomputed from ``best`` when the level wrote it, so it is
        bit-identical to ``problem_cost`` of ``best``).

    Returns
    -------
    int
        Number of accepted proposals.

    Notes
    -----
    A proposal with ``delta <= 0`` is always accepted; otherwise it is accepted iff
    ``u < exp(-delta / temperature)``. ``noexcept nogil`` worker; the GIL is released.
    """
    cdef Py_ssize_t accepted
    if u.shape[0] != ri.shape[0] or u.shape[0] != rj.shape[0] or u.shape[0] != mv.shape[0]:
        raise ValueError("u, ri, rj and mv must have the same length")
    if best.shape[0] != tour.shape[0] or scratch.shape[0] != tour.shape[0]:
        raise ValueError("best and scratch must have the length of tour")
    if state.shape[0] != 2:
        raise ValueError("state must have length 2")
    with nogil:
        accepted = _anneal_level(C, T, tour, best, u, ri, rj, mv, temperature, max_time, fixed_cost, split,
                                 fast_path, scratch, dp, pred, state)
    return accepted


cdef void _sample_deltas(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                         const int64_t[::1] ri, const int64_t[::1] rj, const int64_t[::1] mv,
                         double max_time, double fixed_cost, int split, bint fast_path,
                         int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred,
                         double[::1] out) noexcept nogil:
    cdef Py_ssize_t n = tour.shape[0], m = ri.shape[0], s, i = 0, j = 0, L = 0
    cdef size_t nbytes = n * sizeof(int64_t)
    cdef int64_t code
    cdef double cur = 0.0
    if not fast_path:
        cur = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    for s in range(m):
        code = mv[s]
        if not _decode(n, ri[s], rj[s], code, &i, &j, &L):
            out[s] = NAN
            continue
        if fast_path:
            out[s] = _fast_delta(C, tour, code, i, j, L)
        else:
            memcpy(&scratch[0], &tour[0], nbytes)
            _apply(scratch, code, i, j, L)
            out[s] = problem_cost(C, T, scratch, max_time, fixed_cost, split, dp, pred) - cur


def sample_deltas(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                  const int64_t[::1] ri, const int64_t[::1] rj, const int64_t[::1] mv,
                  double max_time, double fixed_cost, int split, bint fast_path,
                  int64_t[::1] scratch, double[::1] dp, int64_t[::1] pred, double[::1] out):
    """Price the proposals ``(ri, rj, mv)`` on ``tour`` without applying them.

    Writes ``out[s] = cost(after move s) - cost(tour)`` under the same draw -> move mapping as
    `anneal_level`, and ``NaN`` for an invalid draw. Used by the ``t0="auto"``
    calibration (median uphill delta). Same argument conventions as `anneal_level`.
    """
    if ri.shape[0] != rj.shape[0] or ri.shape[0] != mv.shape[0] or ri.shape[0] != out.shape[0]:
        raise ValueError("ri, rj, mv and out must have the same length")
    if scratch.shape[0] != tour.shape[0]:
        raise ValueError("scratch must have the length of tour")
    with nogil:
        _sample_deltas(C, T, tour, ri, rj, mv, max_time, fixed_cost, split, fast_path, scratch, dp, pred, out)
