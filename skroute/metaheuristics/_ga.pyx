"""Kernels of :class:`skroute.metaheuristics.Genetic` (SPEC §4.4, D10, D11).

Conventions: a *chromosome* is ``tour[1:]`` — an int64 permutation of the ``n - 1`` non-depot
node indices — and a population is an int64 ``(pop_size, n - 1)`` C-contiguous array. Every
kernel is ``noexcept nogil`` and reads its randomness from arrays pre-drawn in Python by the
solver (D10): tournament indices, cut points, uniforms and mutation positions. Costs are the
problem objective (``problem_cost`` of the core: plain, greedy or optimal split) so the
budget steers the population. The crossovers reproduce ``tests/reference.py``'s ``ox``/``pmx``
gene for gene.
"""

from libc.stdint cimport int64_t, uint8_t, uint64_t
from libc.string cimport memset

from skroute._core._routing cimport problem_cost
from skroute.metaheuristics._memetic cimport memetic_polish

# FNV-1a style multiplier for the row hashes of the duplicate check (uint64 wrap-around).
cdef uint64_t HASH_MULT = 1099511628211


# ------------------------------------------------------------------ evaluation
cpdef void evaluate_population(const double[:, ::1] C, const double[:, ::1] T, const int64_t[:, ::1] pop,
                               int64_t depot, double max_time, double fixed_cost, int split,
                               int64_t[::1] tour, double[::1] dp, int64_t[::1] pred,
                               double[::1] out) noexcept nogil:
    """Objective of every chromosome of ``pop`` (rows are ``tour[1:]``), written to ``out``.

    ``tour`` (length n), ``dp`` and ``pred`` (length n, only read by the optimal split) are
    caller-owned scratch buffers.
    """
    cdef Py_ssize_t r, k, m = pop.shape[1]
    tour[0] = depot
    for r in range(pop.shape[0]):
        for k in range(m):
            tour[k + 1] = pop[r, k]
        out[r] = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)


# ------------------------------------------------------------------ crossovers
cpdef void ox(const int64_t[::1] p1, const int64_t[::1] p2, Py_ssize_t a, Py_ssize_t b,
              int64_t[::1] child, uint8_t[::1] present) noexcept nogil:
    """Order crossover (OX): ``child[a..b] = p1[a..b]`` (inclusive, ``0 <= a <= b < m``); the other
    positions, starting after ``b`` and wrapping, take the genes of ``p2`` in ``p2``'s order
    (also read from ``b + 1`` onwards, wrapping) skipping those already present.

    ``present`` is a caller-owned uint8 buffer indexed by gene value (length >= max gene + 1).
    Same result as ``tests/reference.py::ox``.
    """
    cdef Py_ssize_t m = p1.shape[0], k, dst, src
    cdef int64_t g
    memset(&present[0], 0, present.shape[0])
    for k in range(a, b + 1):
        child[k] = p1[k]
        present[p1[k]] = 1
    dst = b + 1
    if dst == m:
        dst = 0
    src = dst
    for _k in range(m):
        g = p2[src]
        src += 1
        if src == m:
            src = 0
        if not present[g]:
            child[dst] = g
            dst += 1
            if dst == m:
                dst = 0


cpdef void pmx(const int64_t[::1] p1, const int64_t[::1] p2, Py_ssize_t a, Py_ssize_t b,
               int64_t[::1] child, uint8_t[::1] present, int64_t[::1] mapping) noexcept nogil:
    """Partially mapped crossover (PMX): ``child[a..b] = p1[a..b]``; every other position ``k``
    takes ``p2[k]``, following the mapping ``p1[j] -> p2[j]`` (``j`` in the segment) while the
    gene is already in the segment.

    ``present`` and ``mapping`` are caller-owned buffers indexed by gene value. Same result as
    ``tests/reference.py::pmx``.
    """
    cdef Py_ssize_t m = p1.shape[0], k
    cdef int64_t g
    memset(&present[0], 0, present.shape[0])
    for k in range(a, b + 1):
        child[k] = p1[k]
        present[p1[k]] = 1
        mapping[p1[k]] = p2[k]
    for k in range(m):
        if a <= k <= b:
            continue
        g = p2[k]
        while present[g]:
            g = mapping[g]
        child[k] = g


# ------------------------------------------------------------------ mutations (chromosome positions)
cpdef void mutate(int64_t[::1] child, int kind, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    """Apply one mutation to a chromosome in place; ``i == j`` is a no-op.

    ``kind``: 0 = inversion (reverse ``child[min..max]``, a 2-opt move on the tour), 1 = swap
    (exchange the genes at ``i`` and ``j``), 2 = insertion (move the gene at ``i`` so that it
    lands at position ``j``).
    """
    cdef Py_ssize_t lo, hi, k
    cdef int64_t tmp
    if i == j:
        return
    if kind == 0:
        lo = i if i < j else j
        hi = j if i < j else i
        while lo < hi:
            tmp = child[lo]
            child[lo] = child[hi]
            child[hi] = tmp
            lo += 1
            hi -= 1
    elif kind == 1:
        tmp = child[i]
        child[i] = child[j]
        child[j] = tmp
    else:
        tmp = child[i]
        if i < j:
            for k in range(i, j):
                child[k] = child[k + 1]
        else:
            for k in range(i, j, -1):
                child[k] = child[k - 1]
        child[j] = tmp


# ------------------------------------------------------------------ helpers
cdef inline Py_ssize_t _tournament(const double[::1] fit, const int64_t[:, :, ::1] tourn,
                                   Py_ssize_t which, Py_ssize_t c) noexcept nogil:
    # Index of the fittest among the pre-drawn contestants tourn[which, c, :]; ties -> first drawn.
    cdef Py_ssize_t k, best = tourn[which, c, 0], idx
    for k in range(1, tourn.shape[2]):
        idx = tourn[which, c, k]
        if fit[idx] < fit[best]:
            best = idx
    return best


cdef inline uint64_t _hash(const int64_t[::1] row) noexcept nogil:
    cdef Py_ssize_t k
    cdef uint64_t h = 14695981039346656037ULL
    for k in range(row.shape[0]):
        h = (h ^ <uint64_t>(row[k] + 1)) * HASH_MULT
    return h


cdef inline bint _same_row(const int64_t[:, ::1] pop, Py_ssize_t r, const int64_t[::1] row) noexcept nogil:
    cdef Py_ssize_t k
    for k in range(row.shape[0]):
        if pop[r, k] != row[k]:
            return False
    return True


cdef inline bint _is_duplicate(const int64_t[:, ::1] new_pop, Py_ssize_t upto, const uint64_t[::1] hashes,
                               uint64_t h, const int64_t[::1] child) noexcept nogil:
    # Exact duplicate of a row already in new_pop[0..upto)? Hash first, full compare on a match.
    cdef Py_ssize_t q
    for q in range(upto):
        if hashes[q] == h and _same_row(new_pop, q, child):
            return True
    return False


cpdef double polish_tour(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour, int64_t[::1] pos,
                         const int64_t[:, ::1] cand, uint8_t[::1] dont_look, double max_time,
                         double fixed_cost, int split, int ls_mode, int ls_moves, int64_t[::1] scratch_tour,
                         double[::1] dp, int64_t[::1] pred) noexcept nogil:
    """Polish one index tour in place with the listed descents run to convergence; returns its objective.

    ``ls_mode``: 0 = no polish, 1 = symmetric plain TSP (``two_opt_descent``/``or_opt_descent``),
    2 = generic (``local_search_generic``, for multi-trip and asymmetric problems). ``ls_moves`` is
    the bit mask 1 = two_opt, 2 = or_opt. The other arguments are caller-owned scratch buffers.
    The polish itself is ``_memetic.pxd``'s ``memetic_polish``, shared with the AntColony kernel.
    """
    memetic_polish(C, T, tour, pos, cand, dont_look, max_time, fixed_cost, split, ls_mode, ls_moves,
                   scratch_tour, dp, pred)
    return problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)


# ------------------------------------------------------------------ one generation
cpdef Py_ssize_t ga_generation(
    const double[:, ::1] C, const double[:, ::1] T, double max_time, double fixed_cost, int split,
    int64_t depot,
    const int64_t[:, ::1] pop, const double[::1] fit,
    int64_t[:, ::1] new_pop, double[::1] new_fit,
    const int64_t[::1] elite_idx,
    const int64_t[:, :, ::1] tourn, const double[::1] u_cross, const int64_t[:, ::1] cuts,
    const double[::1] u_mut, const int64_t[:, ::1] mut, const int64_t[:, ::1] remut,
    double p_crossover, double p_mutation, int crossover, int mutation,
    int ls_mode, int ls_moves, const int64_t[:, ::1] cand,
    int64_t[::1] par1, int64_t[::1] par2, int64_t[::1] child,
    int64_t[::1] tour, int64_t[::1] pos, uint8_t[::1] dont_look, int64_t[::1] scratch_tour,
    double[::1] dp, int64_t[::1] pred, uint8_t[::1] present, int64_t[::1] mapping,
    uint64_t[::1] hashes,
) noexcept nogil:
    """Produce the next generation into ``new_pop``/``new_fit``; returns the number of re-mutated duplicates.

    Rows ``0..n_elite-1`` of ``new_pop`` are the parents ``pop[elite_idx]`` (elitism); every other
    row ``r = n_elite + c`` is a child: two parents by tournament over the pre-drawn contestants
    ``tourn[0, c]`` / ``tourn[1, c]`` — the fitter winner is *parent 1* and contributes the
    segment, the other contributes its order (OX) or positions (PMX) —, crossover
    (``crossover`` 0 = OX, 1 = PMX) with cut points ``cuts[c] = (a, b)`` when
    ``u_cross[c] < p_crossover`` (else a copy of parent 1), mutation
    (``mutation`` 0 = inversion, 1 = swap, 2 = insertion) at ``mut[c]`` when ``u_mut[c] < p_mutation``,
    one extra mutation at ``remut[c]`` if the child duplicates a row already produced, then the
    optional memetic polish (``ls_mode``/``ls_moves``, see :func:`polish_tour`) and the objective.

    ``pop`` and every ``const`` array are read only; the remaining arguments are caller-owned
    scratch buffers (``present``/``mapping`` indexed by gene value, ``hashes`` one per row).
    """
    cdef Py_ssize_t P = pop.shape[0], m = pop.shape[1], E = elite_idx.shape[0]
    cdef Py_ssize_t r, c, k, p1, p2, src, tmp, n_dups = 0
    cdef uint64_t h
    for r in range(E):
        src = elite_idx[r]
        for k in range(m):
            new_pop[r, k] = pop[src, k]
        new_fit[r] = fit[src]
        for k in range(m):
            child[k] = pop[src, k]
        hashes[r] = _hash(child)
    tour[0] = depot
    for c in range(P - E):
        r = E + c
        p1 = _tournament(fit, tourn, 0, c)
        p2 = _tournament(fit, tourn, 1, c)
        if fit[p2] < fit[p1]:  # parent 1 (it contributes the segment) is the fitter winner; ties: the first
            tmp = p1
            p1 = p2
            p2 = tmp
        for k in range(m):
            par1[k] = pop[p1, k]
            par2[k] = pop[p2, k]
        if u_cross[c] < p_crossover:
            if crossover == 0:
                ox(par1, par2, cuts[c, 0], cuts[c, 1], child, present)
            else:
                pmx(par1, par2, cuts[c, 0], cuts[c, 1], child, present, mapping)
        else:
            for k in range(m):
                child[k] = par1[k]
        if u_mut[c] < p_mutation:
            mutate(child, mutation, mut[c, 0], mut[c, 1])
        h = _hash(child)
        if _is_duplicate(new_pop, r, hashes, h, child):
            mutate(child, mutation, remut[c, 0], remut[c, 1])
            n_dups += 1
        for k in range(m):
            tour[k + 1] = child[k]
        memetic_polish(C, T, tour, pos, cand, dont_look, max_time, fixed_cost, split, ls_mode, ls_moves,
                       scratch_tour, dp, pred)
        for k in range(m):
            child[k] = tour[k + 1]
            new_pop[r, k] = child[k]
        hashes[r] = _hash(child)
        new_fit[r] = problem_cost(C, T, tour, max_time, fixed_cost, split, dp, pred)
    return n_dups
