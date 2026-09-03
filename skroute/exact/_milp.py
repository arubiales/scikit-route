"""``MILP``: Dantzig-Fulkerson-Johnson formulation with lazy subtour cuts on HiGHS (SPEC §4.1)."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from .._core import _routing as core
from ..base import BaseRouter, RouterTags
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["MILP"]

log = logging.getLogger("skroute")


class MILP(BaseRouter):
    """Exact TSP solver: the Dantzig-Fulkerson-Johnson integer programme with lazy subtour cuts.

    The tour is modelled with binary edge variables (symmetric matrix, ``n(n-1)/2`` of them,
    degree-2 equalities) or arc variables (asymmetric, ``n(n-1)``, in/out-degree-1 equalities)
    and solved with HiGHS through :func:`scipy.optimize.milp`. Subtour-elimination
    constraints are added lazily: after each solve the support of the integral solution is
    split into connected components and one cut ``sum(x[e] for e inside S) <= |S| - 1`` per
    component is appended, until the solution is a single Hamiltonian cycle — a proven
    optimum. The plain tour is the only objective it certifies: under a budget it raises
    (D6).

    Parameters
    ----------
    time_limit : float or None, default 60.0
        Wall-clock budget in seconds for the whole cut loop; every solve receives the time
        that remains. ``None`` runs until proven optimal. When the budget runs out the fit
        still returns a valid tour (see ``is_optimal_``).
    max_nodes : int, default 300
        Hard cap on the number of nodes (``fit`` raises ``ValueError`` above it). Realistic
        sizes within a minute: ~200 symmetric nodes, ~60 asymmetric.
    mip_rel_gap : float, default 0.0
        Relative MIP gap HiGHS accepts on each solve (``0.0`` = solve every relaxation to
        proven optimality). A positive value speeds the loop up but the final tour is then
        only guaranteed within that gap: ``lower_bound_`` comes from the dual bound and
        ``is_optimal_`` is ``True`` only when ``gap_`` is zero.

    Attributes
    ----------
    is_optimal_ : bool
        ``True`` when the last solve returned an integral single-component solution proven
        optimal (then ``lower_bound_ == cost_`` and ``gap_ == 0.0``); ``False`` on time-out.
    lower_bound_ : float
        The largest valid lower bound on the optimum seen: the trivial assignment bound to
        start with (``sum_i min_j C[i, j]``; half the two smallest entries per row when
        symmetric), the objective of every solve HiGHS finished (``status == 0`` — a DFJ
        relaxation solved to optimality is a valid bound), and the dual bound
        ``mip_dual_bound`` of a time-limited solve when HiGHS reports one. The primal value
        of a time-limited incumbent is an *upper* bound and is never used. Clipped to
        ``cost_`` (a bound above a known tour is float noise).
    gap_ : float
        ``max(0.0, (cost_ - lower_bound_) / cost_)``; ``0.0`` when optimal.
    n_solves_ : int
        Number of MILP solves run.
    n_cuts_ : int
        Number of subtour-elimination constraints added.

    Notes
    -----
    On time-out the returned tour is the best single-component integral incumbent seen or,
    failing that, the core's nearest-neighbour tour polished by 2-opt (``two_opt_descent``
    when symmetric; the full-evaluation ``local_search_generic`` with the 2-opt move when
    asymmetric, because O(1) reversal deltas are exact only on symmetric matrices).

    DFJ is preferred to the Miller-Tucker-Zemlin formulation because MTZ's relaxation is
    weak: it takes hours around n = 200 where DFJ proved qa194 (9352) in about 40 s.
    Components are found with :func:`scipy.sparse.csgraph.connected_components` (weak
    connectivity); HiGHS runs single-threaded and deterministically, so two fits with the
    same input give the same tour unless the time limit intervenes.

    Progress (status, objective, number of components and bound of every solve) is logged at
    DEBUG level on the ``skroute`` logger.

    Complexity: NP-hard in the worst case; in practice a few dozen solves of an LP-sized
    programme. Memory O(n²) for the variables and constraints.

    Supports: symmetric and asymmetric matrices; plain TSP only (raises under a budget);
    deterministic.

    References
    ----------
    .. [1] G. Dantzig, R. Fulkerson and S. Johnson, "Solution of a large-scale
       traveling-salesman problem", Journal of the Operations Research Society of America
       2 (1954) 393-410.
    .. [2] Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method",
       Mathematical Programming Computation 10 (2018) 119-142 (HiGHS).

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import MILP
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> est = MILP().fit(C)
    >>> est.cost_, est.is_optimal_, est.gap_, est.lower_bound_ == est.cost_
    (22.0, True, 0.0, True)
    >>> est.n_solves_ >= 1 and est.n_cuts_ >= 0
    True

    Western Sahara (29 cities) is proven optimal in a few seconds:

    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> est = MILP().fit(wi.distance_matrix(), labels=wi.labels)
    >>> int(est.cost_) == wi.optimal_tour_length == 27603, est.is_optimal_
    (True, True)
    """

    _parameter_constraints = {
        "time_limit": [Interval(float, 0.0, None, closed="neither"), None],
        "max_nodes": [Interval(int, 3, None, closed="left")],
        "mip_rel_gap": [Interval(float, 0.0, None, closed="left")],
    }

    def __init__(
        self, time_limit: float | None = 60.0, max_nodes: int = 300, mip_rel_gap: float = 0.0
    ) -> None:
        self.time_limit = time_limit
        self.max_nodes = max_nodes
        self.mip_rel_gap = mip_rel_gap

    def _get_tags(self) -> RouterTags:
        return RouterTags(kind="exact", exact=True, budget_aware=False, max_nodes=self.max_nodes)

    # ------------------------------------------------------------------ the cut loop
    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        t0 = perf_counter()
        C, n, depot, symmetric = problem.cost, problem.n, problem.depot, problem.symmetric
        if symmetric:
            iu, ju = np.triu_indices(n, 1)  # one variable per edge {i, j}
        else:
            iu, ju = np.nonzero(~np.eye(n, dtype=bool))  # one variable per arc (i, j)
        m = iu.size
        c = np.ascontiguousarray(C[iu, ju], dtype=np.float64)
        cols = np.arange(m)
        if symmetric:  # every node touches exactly two chosen edges
            rows = np.concatenate([iu, ju])
            degree = LinearConstraint(
                coo_matrix((np.ones(2 * m), (rows, np.tile(cols, 2))), shape=(n, m)).tocsr(), 2, 2
            )
        else:  # exactly one arc leaves and one enters every node
            rows = np.concatenate([iu, n + ju])
            degree = LinearConstraint(
                coo_matrix((np.ones(2 * m), (rows, np.tile(cols, 2))), shape=(2 * n, m)).tocsr(), 1, 1
            )
        constraints: list[LinearConstraint] = [degree]
        integrality = np.ones(m)
        bounds = Bounds(0, 1)
        lower = _trivial_bound(C, symmetric)
        n_solves = n_cuts = 0
        best_tour: np.ndarray | None = None
        proven = False
        while True:
            options: dict[str, Any] = {"mip_rel_gap": float(self.mip_rel_gap)}
            if self.time_limit is not None:
                remaining = float(self.time_limit) - (perf_counter() - t0)
                if remaining <= 0.0:
                    break
                options["time_limit"] = remaining
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
            n_solves += 1
            lower = max(lower, self._bound_of(res))
            if res.x is None:  # no incumbent: the time limit hit before a first feasible point
                log.debug("MILP solve %d: status %d, no incumbent, bound %.6g", n_solves, res.status, lower)
                break
            selected = res.x > 0.5
            support = csr_matrix((np.ones(int(selected.sum())), (iu[selected], ju[selected])), shape=(n, n))
            n_components, component = connected_components(support, directed=True, connection="weak")
            log.debug(
                "MILP solve %d: status %d, objective %.6g, %d component(s), bound %.6g",
                n_solves,
                res.status,
                res.fun,
                n_components,
                lower,
            )
            if n_components == 1:
                tour = _extract_tour(iu[selected], ju[selected], n, depot, symmetric)
                if tour is not None:
                    best_tour = tour
                    proven = res.status == 0
                break
            # one subtour-elimination cut per component: sum of the variables inside S <= |S| - 1
            cut_rows: list[np.ndarray] = []
            cut_cols: list[np.ndarray] = []
            ub: list[float] = []
            for k in range(n_components):
                inside = component == k
                idx = np.flatnonzero(inside[iu] & inside[ju])
                cut_rows.append(np.full(idx.size, len(ub)))
                cut_cols.append(idx)
                ub.append(float(inside.sum() - 1))
            A = coo_matrix(
                (
                    np.ones(sum(len(z) for z in cut_cols)),
                    (np.concatenate(cut_rows), np.concatenate(cut_cols)),
                ),
                shape=(len(ub), m),
            ).tocsr()
            constraints.append(LinearConstraint(A, -np.inf, np.asarray(ub)))
            n_cuts += len(ub)
            if res.status != 0:  # a time-limited solve: the budget is spent
                break
        if best_tour is None:
            best_tour = _fallback_tour(problem)
        cost = float(problem.evaluate(best_tour))
        if proven and self.mip_rel_gap > 0.0:
            proven = lower >= cost - 1e-9 * max(1.0, abs(cost))
        lower = cost if proven else min(lower, cost)
        self.is_optimal_ = bool(proven)
        self.lower_bound_ = float(lower)
        self.gap_ = 0.0 if proven else _gap(cost, lower)
        self.n_solves_ = int(n_solves)
        self.n_cuts_ = int(n_cuts)
        return best_tour

    def _bound_of(self, res: Any) -> float:
        """Valid lower bound carried by one solve, or ``-inf`` when it offers none."""
        bound = -math.inf
        if res.status == 0 and self.mip_rel_gap == 0.0 and res.fun is not None:
            bound = float(res.fun)  # a relaxation solved to optimality
        dual = getattr(res, "mip_dual_bound", None)
        if dual is not None and math.isfinite(dual):
            bound = max(bound, float(dual))
        return bound


# ---------------------------------------------------------------------- helpers
def _trivial_bound(C: np.ndarray, symmetric: bool) -> float:
    """Assignment-style bound: every node has one outgoing arc (two incident edges when symmetric)."""
    off = C.copy()
    np.fill_diagonal(off, np.inf)
    if symmetric:
        two = np.partition(off, 1, axis=1)[:, :2]
        return float(two.sum() / 2.0)
    return float(off.min(axis=1).sum())


def _gap(cost: float, lower: float) -> float:
    if cost == 0.0:
        return 0.0 if lower >= 0.0 else math.inf
    return max(0.0, (cost - lower) / abs(cost))


def _extract_tour(ii: np.ndarray, jj: np.ndarray, n: int, depot: int, symmetric: bool) -> np.ndarray | None:
    """Walk the single-component support into an index tour; ``None`` if it is not a Hamiltonian cycle."""
    tour = np.empty(n, dtype=np.int64)
    tour[0] = depot
    if symmetric:
        adj = np.full((n, 2), -1, dtype=np.int64)
        deg = np.zeros(n, dtype=np.int64)
        for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
            if deg[a] == 2 or deg[b] == 2:
                return None
            adj[a, deg[a]] = b
            adj[b, deg[b]] = a
            deg[a] += 1
            deg[b] += 1
        if not np.all(deg == 2):
            return None
        prev, cur = -1, depot
        for k in range(1, n):
            nxt = adj[cur, 0] if adj[cur, 0] != prev else adj[cur, 1]
            prev, cur = cur, int(nxt)
            tour[k] = cur
    else:
        succ = np.full(n, -1, dtype=np.int64)
        for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
            if succ[a] != -1:
                return None
            succ[a] = b
        cur = depot
        for k in range(1, n):
            cur = int(succ[cur])
            if cur < 0:
                return None
            tour[k] = cur
    if not np.array_equal(np.sort(tour), np.arange(n)):
        return None
    return tour


def _fallback_tour(problem: RoutingProblem) -> np.ndarray:
    """Nearest neighbour from the depot polished by 2-opt (the time-out path)."""
    C, n = problem.cost, problem.n
    tour = np.empty(n, dtype=np.int64)
    core.nearest_neighbour_tour(C, problem.depot, tour)
    pos = np.empty(n, dtype=np.int64)
    core.rebuild_pos(tour, pos)
    cand = problem.neighbours(min(10, n - 1))
    if problem.symmetric:
        dont_look = np.zeros(n, dtype=np.uint8)
        core.two_opt_descent(C, tour, pos, cand, dont_look, False, 1000)
    else:
        scratch = np.empty(n, dtype=np.int64)
        core.local_search_generic(
            C,
            C,
            tour,
            pos,
            cand,
            np.inf,
            0.0,
            0,
            1,
            3,
            1000,
            scratch,
            np.empty(0),
            np.empty(0, dtype=np.int64),
        )
    return tour
