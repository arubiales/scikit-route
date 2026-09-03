"""The problem model: ``RoutingProblem``, one immutable instance in index space (SPEC §3.3).

Every solver receives a [`RoutingProblem`][skroute.RoutingProblem] (built by
[`BaseRouter.fit`][skroute.base.BaseRouter.fit] or by the user) and works in *index space*
(nodes ``0..n-1`` in matrix row order, depot ``problem.depot``); the public attributes of a
fitted solver are translated back to *label space* with
[`to_label_tour`][skroute.RoutingProblem.to_label_tour].
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from ._core import _routing as core
from .exceptions import InfeasibleProblemError
from .utils.validation import coerce_labels, coerce_matrix

__all__ = ["RoutingProblem"]

# Cython 3 exposes a cpdef enum to Python as the IntEnum class `_routing.SplitRule`; its members are
# NOT module-level names (`core.SPLIT_GREEDY` raises AttributeError — verified with Cython 3.3).
_SPLIT = {"greedy": int(core.SplitRule.SPLIT_GREEDY), "optimal": int(core.SplitRule.SPLIT_OPTIMAL)}


def _is_number(x: Any) -> bool:
    """A real scalar (Python or numpy), never a bool, a string or an array."""
    return isinstance(x, Real) and not isinstance(x, (bool, np.bool_))


class RoutingProblem:
    """One instance in index space. Immutable after construction; shareable across solvers and threads.

    Parameters
    ----------
    X : (n, n) array-like, DataFrame or dict-of-dicts
        Cost matrix. Rows are origins, columns destinations. Must be square, n >= 3, all finite.
    time_matrix : same kinds as X, optional, keyword-only (D7)
        Durations, same shape and labels as X, all finite and >= 0. Required iff max_time_work is given.
    depot : label, optional
        Label of the depot (a position for plain arrays without labels=). Default: first node.
    coords : (n, 2) array-like, optional
        Coordinates in row order of X. Needed by SOM; carried, never validated beyond shape.
    labels : sequence of n hashables, optional
        Labels for a plain ndarray X. Must equal the labels X already carries, if any.
    max_time_work : float > 0, optional
        Per-trip budget in the units of time_matrix. None = plain TSP.
    extra_cost : float >= 0, default 0.0
        Fixed charge per trip beyond the first.
    people : int >= 1, default 1
        Multiplies extra_cost only.
    split : {"greedy", "optimal"}, default "greedy"
        Decoder of the giant tour into trips (see D1).

    Attributes
    ----------
    cost : ndarray of shape (n, n), float64, C-contiguous
        The coerced cost matrix.
    time : ndarray of shape (n, n) or None
        The coerced time matrix; ``None`` for plain TSP.
    n : int
        Number of nodes.
    labels : ndarray of shape (n,), dtype int64 or object
        Label of every node, in matrix row order.
    depot : int
        Index of the depot.
    max_time_work : float
        The per-trip budget; ``inf`` for plain TSP.
    extra_cost : float
    people : int
    split : {"greedy", "optimal"}
    coords : ndarray of shape (n, 2) or None
    symmetric : bool
        Whether ``cost`` equals its transpose.

    Notes
    -----
    The objective (D1) is the travel cost of the decoded trips plus
    ``people * extra_cost * (n_trips - 1)``. Under the greedy decoder a leg ``a -> b``
    joins the open trip iff the trip can still return to the depot within
    ``max_time_work``; the optimal decoder (Prins, 2004) is the minimum-cost
    partition of the giant tour into consecutive feasible trips. Both are O(n)
    and O(n L) respectively, with L the longest feasible open path.

    References
    ----------
    .. [1] C. Prins, "A simple and effective evolutionary algorithm for the vehicle
       routing problem", Computers & Operations Research 31 (2004) 1985-2002.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import RoutingProblem
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> p = RoutingProblem(C, labels=["d", "a", "b", "c"], depot="d")
    >>> p
    RoutingProblem(n=4, TSP, symmetric, depot='d')
    >>> p.to_index_tour(["d", "a", "b", "c", "d"]).tolist()
    [0, 1, 2, 3]
    >>> p.evaluate([0, 1, 2, 3])
    22.0
    >>> hours = np.array([[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]], dtype=float)
    >>> q = RoutingProblem(C, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
    >>> q.evaluate([0, 1, 2, 3]), q.trip_starts([0, 1, 2, 3]).tolist()
    (41.0, [1, 3, 4])
    """

    def __init__(
        self,
        X: Any,
        *,
        time_matrix: Any = None,
        depot: Any = None,
        coords: Any = None,
        labels: Any = None,
        max_time_work: float | None = None,
        extra_cost: float = 0.0,
        people: int = 1,
        split: str = "greedy",
    ) -> None:
        C, lab = coerce_matrix(X, "X")  # float64 C-contiguous, labels or None
        n = C.shape[0]
        if n < 3:
            raise ValueError(f"X must have at least 3 nodes, got {n}")
        if labels is not None:
            given = coerce_labels(labels, n)  # 1-D int64 or object array, n unique entries
            if lab is not None and not np.array_equal(lab, given):
                raise ValueError("labels= disagrees with the labels carried by X")
            lab = given
        self.labels: np.ndarray = np.arange(n, dtype=np.int64) if lab is None else lab
        self._index: dict[Any, int] = {label: i for i, label in enumerate(self.labels.tolist())}
        if len(self._index) != n:
            raise ValueError("labels must be unique")
        self.cost: np.ndarray = C
        self.n: int = n
        if depot is None:
            self.depot: int = 0
        else:
            try:
                self.depot = self._index[depot]
            except (KeyError, TypeError):
                raise ValueError(f"depot {depot!r} is not a label of X") from None
        if split not in _SPLIT:
            raise ValueError(f"split must be 'greedy' or 'optimal', got {split!r}")
        self.split: str = split
        self.time: np.ndarray | None
        if max_time_work is None:
            if time_matrix is not None:
                raise ValueError(
                    "time_matrix given but no max_time_work; pass max_time_work=<hours per trip>"
                )
            if extra_cost != 0.0 or people != 1 or split != "greedy":
                raise ValueError("extra_cost, people and split have no effect without max_time_work")  # D3
            self.time = None
            self.max_time_work: float = np.inf
        else:
            if time_matrix is None:
                raise ValueError(
                    "max_time_work given but no time_matrix; "
                    "pass time_matrix=X to use the cost matrix as durations"
                )
            # type first, then value: a str, a bool or a 1-element array gets THIS message, not a numpy one
            if not (_is_number(max_time_work) and np.isfinite(max_time_work) and float(max_time_work) > 0):
                raise ValueError(f"max_time_work must be a finite number > 0, got {max_time_work!r}")
            T, tlab = coerce_matrix(time_matrix, "time_matrix")
            if T.shape != C.shape:
                raise ValueError(f"time_matrix has shape {T.shape}, X has shape {C.shape}")
            if tlab is not None and not np.array_equal(tlab, self.labels):
                raise ValueError("time_matrix labels differ from the labels of X")
            if (T < 0).any():
                raise ValueError("time_matrix contains negative durations")
            self.time = T
            self.max_time_work = float(max_time_work)
            d = self.depot
            bad = T[d, :] + T[:, d] > self.max_time_work
            bad[d] = False
            if bad.any():
                raise InfeasibleProblemError(
                    f"nodes {self.labels[bad].tolist()} cannot be served in one trip: "
                    f"depot round trip exceeds max_time_work={self.max_time_work}"
                )
        if not (_is_number(extra_cost) and np.isfinite(extra_cost) and float(extra_cost) >= 0):
            raise ValueError(f"extra_cost must be a finite number >= 0, got {extra_cost!r}")
        if not isinstance(people, (int, np.integer)) or isinstance(people, bool) or people < 1:
            raise ValueError(f"people must be an integer >= 1, got {people!r}")
        self.extra_cost: float = float(extra_cost)
        self.people: int = int(people)
        self.coords: np.ndarray | None = None
        if coords is not None:
            xy = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
            if xy.shape != (n, 2):
                raise ValueError(f"coords must have shape ({n}, 2), got {xy.shape}")
            self.coords = xy
        self.symmetric: bool = bool(np.array_equal(C, C.T))
        self._neigh: dict[int, np.ndarray] = {}

    # ----- derived, read-only -----
    @property
    def multi_trip(self) -> bool:
        """Whether a time matrix and a budget were given."""
        return self.time is not None

    @property
    def fixed_cost(self) -> float:
        """``extra_cost * people``: the charge per trip beyond the first."""
        return self.extra_cost * self.people

    @property
    def time_or_cost(self) -> np.ndarray:
        """The matrix kernels receive as ``T``: the time matrix, or the cost matrix when there is no budget.

        Without a budget (``max_time_work == inf``) the kernels never read ``T``; passing the cost matrix
        keeps every call signature uniform.
        """
        return self.cost if self.time is None else self.time

    @property
    def split_code(self) -> int:
        """Integer code of the split rule, as the core's ``SplitRule`` enum."""
        return _SPLIT[self.split]

    @property
    def depot_label(self) -> Any:
        """The depot's label."""
        return self.labels[self.depot]

    # ----- label <-> index -----
    def index_of(self, label: Any) -> int:
        """Index of ``label`` in matrix row order; ``ValueError`` if it is not a label of X."""
        try:
            return self._index[label]
        except (KeyError, TypeError):
            raise ValueError(f"{label!r} is not a label of X") from None

    def to_index_tour(self, seq: Any) -> np.ndarray:
        """Labels -> int64 tour with the depot at position 0.

        Accepts an open tour (n labels), a closed route (depot repeated at the end)
        or a multi-trip route (depot repeated between trips): every occurrence of the
        depot is removed, then the depot is prepended. Raises ValueError unless the
        remaining labels are exactly the non-depot labels, each once.
        """
        seq = list(seq)
        idx = np.fromiter((self.index_of(x) for x in seq), dtype=np.int64, count=len(seq))
        body = idx[idx != self.depot]
        expected = np.delete(np.arange(self.n), self.depot)
        if not np.array_equal(np.sort(body), expected):
            raise ValueError("init tour must contain every label exactly once (the depot may repeat)")
        return np.concatenate(([self.depot], body)).astype(np.int64)

    def to_label_tour(self, tour: Any) -> np.ndarray:
        """Index tour -> label array (same length, label dtype)."""
        return self.labels[np.asarray(tour, dtype=np.int64)]

    # ----- kernels -----
    # Typed memoryviews accept only C-contiguous int64 arrays (a list or an int32 array raises TypeError /
    # "Buffer dtype mismatch"); these public methods coerce first.
    @staticmethod
    def _as_index(a: Any) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.int64)

    def evaluate(self, tour: Any) -> float:
        """Objective of an index tour (D1). O(n) greedy / plain, O(n*L) optimal."""
        return core.problem_cost_py(
            self.cost,
            self.time_or_cost,
            self._as_index(tour),
            self.max_time_work,
            self.fixed_cost,
            self.split_code,
        )

    def trip_starts(self, tour: Any) -> np.ndarray:
        """Trip start positions of an index tour.

        Returns an int64 array of shape ``(n_trips + 1,)`` with ``starts[0] == 1`` and ``starts[-1] == n``;
        trip ``k`` is ``tour[starts[k]:starts[k + 1]]``. Plain TSP gives ``[1, n]``.
        """
        out = np.empty(self.n + 1, dtype=np.int64)
        k = core.trip_starts(
            self.time_or_cost,
            self._as_index(tour),
            self.max_time_work,
            self.split_code,
            self.cost,
            self.fixed_cost,
            out,
        )
        return out[: k + 1]

    def trip_costs(self, tour: Any, starts: Any) -> np.ndarray:
        """Travel cost of each closed trip (fixed charge excluded), float64 ``(n_trips,)``."""
        out = np.empty(len(starts) - 1)
        core.trip_costs(self.cost, self._as_index(tour), self._as_index(starts), out)
        return out

    def trip_times(self, tour: Any, starts: Any) -> np.ndarray:
        """Duration of each closed trip, float64 ``(n_trips,)``. Requires a time matrix."""
        if self.time is None:
            raise ValueError("trip_times needs a time matrix; this problem is a plain TSP")
        out = np.empty(len(starts) - 1)
        core.trip_times(self.time, self._as_index(tour), self._as_index(starts), out)
        return out

    def neighbours(self, k: int = 10) -> np.ndarray:
        """k nearest neighbours of every node by C[i, :], as int64 (n, k), sorted ascending; cached.

        The diagonal is excluded regardless of its value (only finiteness is required of it) and
        ties are broken by node index (stable sort). Six bundled instances have coincident points
        (lu980: 346 duplicate rows, ho14473: 7 370), so zero off-diagonal distances are normal input.
        """
        k = min(int(k), self.n - 1)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k not in self._neigh:
            n = self.n
            dm = self.cost.copy()  # the ONE transient (n, n) copy; accepted
            np.fill_diagonal(dm, np.inf)
            out = np.empty((n, k), dtype=np.int64)
            # Row blocks: argpartition materialises an int64 array as large as its input, so partitioning
            # the whole matrix at once would double the peak (~1 GB more at n = 10 639). A block keeps that
            # scratch at ~4 MiB and the peak at the accepted single copy.
            step = max(1, (4 << 20) // (8 * n))
            for a in range(0, n, step):
                block = dm[a : a + step]
                # kth = k: positions < k hold the k smallest (any order), position k the (k + 1)-th smallest
                part = np.argpartition(block, k, axis=1)
                # index order first, then a stable sort by distance: ties inside the selection -> lowest index
                sel = np.sort(part[:, :k], axis=1)
                dist = np.take_along_axis(block, sel, 1)
                chosen = np.take_along_axis(sel, np.argsort(dist, axis=1, kind="stable"), 1)
                # argpartition chooses arbitrarily among nodes tied AT the k-th distance (coincident points,
                # integer TSPLIB distances): a tie straddles the boundary iff the (k + 1)-th smallest equals
                # the k-th. Redo exactly those rows: every node strictly closer first, then the lowest
                # indices among the tied ones.
                thr = dist.max(axis=1)
                nxt = np.take_along_axis(block, part[:, k : k + 1], 1)[:, 0]
                for r in np.flatnonzero(nxt == thr):
                    row = block[r]
                    strict = np.flatnonzero(row < thr[r])
                    tied = np.flatnonzero(row == thr[r])
                    keep = np.concatenate((strict, tied[: k - strict.size]))
                    chosen[r] = keep[np.argsort(row[keep], kind="stable")]
                out[a : a + step] = chosen
            self._neigh[k] = out
        return self._neigh[k]

    def __repr__(self) -> str:
        kind = "multi-trip" if self.multi_trip else "TSP"
        sym = "symmetric" if self.symmetric else "asymmetric"
        label = self.depot_label
        if isinstance(label, np.generic):  # numpy 2 would print np.int64(0); show the plain label
            label = label.item()
        return f"RoutingProblem(n={self.n}, {kind}, {sym}, depot={label!r})"
