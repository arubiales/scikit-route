"""``initial_tour``: the warm-start helper shared by every solver with an ``init`` parameter (SPEC §3.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._core import _routing as core

if TYPE_CHECKING:  # problem.py imports utils.validation: keep this import out of the runtime graph
    from ..problem import RoutingProblem

__all__ = ["initial_tour"]


def initial_tour(problem: RoutingProblem, init: Any, rng: np.random.Generator | None) -> np.ndarray:
    """Build the starting index tour of a solver from its ``init`` parameter.

    Parameters
    ----------
    problem : RoutingProblem
        The instance being solved.
    init : {"nearest_neighbour", "random"} or array-like of labels
        ``"nearest_neighbour"`` runs the core's greedy nearest-neighbour construction
        from the depot; ``"random"`` returns the depot followed by a random permutation
        of the other nodes (requires ``rng``); an array of labels — the ``tour_`` or
        ``route_`` of another solver, open, closed or multi-trip — is converted with
        [`to_index_tour`][skroute.RoutingProblem.to_index_tour].
    rng : numpy.random.Generator or None
        The solver's generator; ``None`` for deterministic solvers.

    Returns
    -------
    tour : ndarray of shape (n,), int64
        A permutation of ``range(n)`` with ``problem.depot`` at position 0.

    Raises
    ------
    ValueError
        If ``init`` is an unknown string or not iterable at all (``None``, a number), or
        ``"random"`` is requested without ``rng`` (the solver is not stochastic), or the
        label array is not a valid tour.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import RoutingProblem
    >>> from skroute.utils import initial_tour
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> p = RoutingProblem(C, depot=3)
    >>> initial_tour(p, "nearest_neighbour", None).tolist()
    [3, 2, 1, 0]
    >>> initial_tour(p, [3, 0, 1, 2, 3], None).tolist()
    [3, 0, 1, 2]
    >>> t = initial_tour(p, "random", np.random.default_rng(0))
    >>> int(t[0]) == 3 and sorted(t.tolist()) == [0, 1, 2, 3]
    True
    """
    if isinstance(init, str):
        if init == "nearest_neighbour":
            out = np.empty(problem.n, dtype=np.int64)
            core.nearest_neighbour_tour(problem.cost, problem.depot, out)
            return out
        if init == "random":
            if rng is None:
                raise ValueError("init='random' needs a random generator: this solver is not stochastic")
            rest = np.delete(np.arange(problem.n, dtype=np.int64), problem.depot)
            return np.concatenate(([problem.depot], rng.permutation(rest))).astype(np.int64)
        raise ValueError("init must be 'nearest_neighbour', 'random' or an array of labels")
    if not hasattr(init, "__iter__"):  # init=5 or init=None: the documented message, not a bare TypeError
        raise ValueError("init must be 'nearest_neighbour', 'random' or an array of labels")
    return problem.to_index_tour(init)
