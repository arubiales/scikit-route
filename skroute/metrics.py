"""Public recomputation helpers: :func:`route_cost` and :func:`split_trips` (SPEC §3.5).

Both work in **label space**, on the ``route_``/``tour_`` arrays a fitted solver
exposes, so a user (or a test) can price any route under the objective without
touching index space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .problem import RoutingProblem

__all__ = ["route_cost", "split_trips"]


def route_cost(
    X: Any,
    route: Any,
    *,
    depot: Any = None,
    labels: Any = None,
    time_matrix: Any = None,
    max_time_work: float | None = None,
    extra_cost: float = 0.0,
    people: int = 1,
    split: str = "greedy",
) -> float:
    """Objective of a label-space route (D1), recomputed from scratch.

    Parameters
    ----------
    X : (n, n) array-like, DataFrame or dict-of-dicts
        Cost matrix (rows are origins).
    route : sequence of labels
        Open tour, closed route or multi-trip route (the depot may repeat). Its first
        label is the depot.
    depot : label, optional
        Must equal ``route[0]`` when given; ``None`` means ``route[0]`` — so a
        ``route_``/``tour_`` produced with any ``depot=`` re-evaluates without repeating it.
    labels : sequence of n hashables, optional
        Labels of a plain ndarray ``X``.
    time_matrix : same kinds as X, optional
        Durations; required iff ``max_time_work`` is given.
    max_time_work : float > 0, optional
        Per-trip budget. ``None`` = plain TSP.
    extra_cost : float >= 0, default 0.0
        Fixed charge per trip beyond the first.
    people : int >= 1, default 1
        Multiplies ``extra_cost`` only.
    split : {"greedy", "optimal"}, default "greedy"
        Decoder of the giant tour into trips.

    Returns
    -------
    cost : float

    Raises
    ------
    ValueError
        ``"depot must be the first label of route"`` when ``depot`` disagrees with
        ``route[0]``; the :class:`RoutingProblem` errors for invalid inputs.

    Examples
    --------
    >>> from skroute.metrics import route_cost
    >>> C = [[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]]
    >>> route_cost(C, [0, 1, 2, 3, 0])
    22.0
    >>> route_cost(C, [2, 3, 0, 1], labels=[0, 1, 2, 3])
    22.0
    >>> hours = [[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]]
    >>> route_cost(C, [0, 1, 2, 0, 3, 0], time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
    41.0
    """
    route = list(route)
    if not route:
        raise ValueError("route must not be empty")
    if depot is None:
        depot = route[0]
    elif depot != route[0]:
        raise ValueError("depot must be the first label of route")
    problem = RoutingProblem(
        X,
        time_matrix=time_matrix,
        depot=depot,
        labels=labels,
        max_time_work=max_time_work,
        extra_cost=extra_cost,
        people=people,
        split=split,
    )
    return float(problem.evaluate(problem.to_index_tour(route)))


def split_trips(route: Any, depot: Any = None) -> list[np.ndarray]:
    """Split a label route at depot occurrences into **closed** trips ``[depot, ..., depot]``.

    Parameters
    ----------
    route : sequence of labels
        As driven: depot first, possibly repeated between trips and at the end. An
        open tour (depot only at the front) is returned as one closed trip.
    depot : label, optional
        ``None`` means ``route[0]``.

    Returns
    -------
    trips : list of ndarray
        One closed trip per segment, in route order; empty segments (two consecutive
        depots) are dropped.

    Examples
    --------
    >>> from skroute.metrics import split_trips
    >>> [t.tolist() for t in split_trips([0, 1, 2, 0, 3, 0])]
    [[0, 1, 2, 0], [0, 3, 0]]
    >>> [t.tolist() for t in split_trips(["d", "a", "b"])]
    [['d', 'a', 'b', 'd']]
    """
    items = list(route)
    if not items:
        raise ValueError("route must not be empty")
    if depot is None:
        depot = items[0]
    elif depot != items[0]:
        raise ValueError("depot must be the first label of route")
    # same dtype rule as coerce_labels: int64 when every label is an int (never bool), else object
    int_like = all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in items)
    kind: Any = np.int64 if int_like else object
    trips: list[np.ndarray] = []
    body: list[Any] = []
    for x in items:
        if x == depot:
            if body:
                trips.append(_closed(depot, body, kind))
                body = []
        else:
            body.append(x)
    if body:
        trips.append(_closed(depot, body, kind))
    return trips


def _closed(depot: Any, body: list[Any], kind: Any) -> np.ndarray:
    out = np.empty(len(body) + 2, dtype=kind)
    out[0] = depot
    out[1:-1] = body
    out[-1] = depot
    return out
