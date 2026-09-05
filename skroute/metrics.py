"""Public recomputation helpers: [`route_cost`][skroute.metrics.route_cost],
[`split_trips`][skroute.metrics.split_trips] and the per-day
[`timetable`][skroute.metrics.timetable] (SPEC §3.5, D32).

Everything here works in **label space**, on the ``route_``/``tour_`` arrays a fitted
solver exposes, so a user (or a test) can price any route under the objective — or
turn it into arrival and departure times — without touching index space.
"""

from __future__ import annotations

import datetime
import math
import re
import warnings
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from .base import BaseRouter
from .problem import RoutingProblem
from .utils.validation import check_is_fitted

if TYPE_CHECKING:  # pandas is optional at runtime: only ``timetable(as_frame=True)`` imports it
    import pandas as pd

__all__ = ["Stop", "route_cost", "split_trips", "timetable", "timetable_summary"]

# minutes per unit of the time matrix
_UNITS = {"min": 1.0, "h": 60.0, "s": 1.0 / 60.0}
_MINUTES_PER_DAY = 24 * 60


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
    service_time: Any = None,
    split: str = "greedy",
) -> float:
    """Objective of a label-space route (D1), recomputed from scratch.

    Parameters
    ----------
    X : (n, n) array-like, DataFrame or dict-of-dicts
        Cost matrix (rows are origins).
    route : sequence of labels
        Open tour, closed route or multi-trip route (the depot may repeat). Its first
        label is the depot. Every depot occurrence is removed and the giant tour is
        re-decoded with ``split``, exactly as ``fit`` did — so a hand-made plan is priced as
        the decoder would cut it, not as driven; [`timetable`][skroute.metrics.timetable]
        keeps the days as driven instead. The two agree on every route produced by ``fit``.
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
    service_time : float or (n,) array-like, optional
        Time spent at each stop, in the units of ``time_matrix`` (D32); requires ``max_time_work``.
        The route is re-decoded with the services included, exactly as ``fit`` did.
    split : {"greedy", "optimal"}, default "greedy"
        Decoder of the giant tour into trips.

    Returns
    -------
    cost : float

    Raises
    ------
    ValueError
        ``"depot must be the first label of route"`` when ``depot`` disagrees with
        ``route[0]``; the [`RoutingProblem`][skroute.RoutingProblem] errors for invalid inputs.

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

    Half an hour at every stop makes the same tour need two trips under a five-hour day:

    >>> route_cost(C, [0, 1, 2, 3], time_matrix=hours, max_time_work=5.0, extra_cost=3.0)
    22.0
    >>> route_cost(C, [0, 1, 2, 3], time_matrix=hours, max_time_work=5.0, extra_cost=3.0, service_time=0.5)
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
        service_time=service_time,
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


# ------------------------------------------------------------------------------- timetable (D32)
def _clock(minutes: float) -> str:
    """``HH:MM`` of a number of minutes after midnight, rounded to the nearest minute, wrapping at 24 h."""
    total = math.floor(minutes + 0.5) % _MINUTES_PER_DAY
    return f"{total // 60:02d}:{total % 60:02d}"


def _start_minutes(start: Any) -> float:
    """Minutes after midnight of ``start``: an ``"HH:MM"`` string or a ``datetime.time``."""
    if isinstance(start, datetime.time):
        return start.hour * 60.0 + start.minute + start.second / 60.0 + start.microsecond / 60e6
    if isinstance(start, str):
        match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", start)
        if match is not None:
            hours, minutes = int(match.group(1)), int(match.group(2))
            if hours < 24 and minutes < 60:
                return hours * 60.0 + minutes
    raise ValueError(f"start must be 'HH:MM' or a datetime.time, got {start!r}")


@dataclass(frozen=True)
class Stop:
    """One line of a [`timetable`][skroute.metrics.timetable]: a visit, or the depot at either end of a day.

    Times are **minutes since the start of the day** (floats, so the arithmetic is exact and
    the caller decides the rounding); the ``*_time`` properties render them as ``HH:MM``
    clock strings from the day's ``start``.

    Parameters
    ----------
    day : int
        Day (trip) number, 1-based.
    order : int
        Position in the day: ``0`` is the departure from the depot, the last order the return.
    label : hashable
        Label of the node (the depot's label at both ends).
    arrival : float
        Minutes after ``start`` at which the vehicle arrives (``0.0`` at the departure stop).
    departure : float
        ``arrival + service + wait``; at the final depot stop it equals ``arrival``.
    travel : float
        Driving minutes from the previous stop (``0.0`` at the departure stop).
    service : float
        Minutes spent at the stop (``problem.service_time`` of the node; ``0.0`` on returning).
    wait : float, default 0.0
        Idle minutes before the service starts. Always ``0.0`` today — reserved for time windows.
    start : float, default 480.0
        Minutes after midnight at which the day starts (``08:00``); used by the clock strings only,
        and left out of the repr and of equality: two stops with the same minutes are equal
        (and hash alike) whatever clock they are rendered on.

    Examples
    --------
    >>> from skroute.metrics import Stop
    >>> stop = Stop(day=1, order=2, label="b", arrival=95.0, departure=125.0, travel=20.0, service=30.0)
    >>> stop.arrival_time, stop.departure_time
    ('09:35', '10:05')
    >>> Stop(1, 0, "d", 0.0, 0.0, 0.0, 0.0, start=9 * 60 + 30).departure_time
    '09:30'
    >>> Stop(1, 0, "d", 0.0, 0.0, 0.0, 0.0, start=9 * 60 + 30) == Stop(1, 0, "d", 0.0, 0.0, 0.0, 0.0)
    True
    """

    day: int
    order: int
    label: Any
    arrival: float
    departure: float
    travel: float
    service: float
    wait: float = 0.0
    start: float = field(default=8 * 60.0, repr=False, compare=False)

    @property
    def arrival_time(self) -> str:
        """``HH:MM`` of ``arrival`` (rounded to the minute; wraps past midnight)."""
        return _clock(self.start + self.arrival)

    @property
    def departure_time(self) -> str:
        """``HH:MM`` of ``departure`` (rounded to the minute; wraps past midnight)."""
        return _clock(self.start + self.departure)


def _resolve(obj: Any, route: Any) -> tuple[RoutingProblem, list[list[int]]]:
    """``(problem, trips)`` in index space — the bodies of the trips, without the depot — of a fitted
    estimator (its ``route_`` unless ``route`` overrides it) or of a problem plus a label route."""
    if isinstance(obj, BaseRouter):
        check_is_fitted(obj)
        problem = obj.problem_
        if route is None:
            route = obj.route_
    elif isinstance(obj, RoutingProblem):
        problem = obj
        if route is None:
            raise ValueError("timetable needs a route when given a RoutingProblem: timetable(problem, route)")
    else:
        raise TypeError(f"timetable expects a fitted estimator or a RoutingProblem, got {type(obj).__name__}")
    if not problem.multi_trip:
        raise ValueError("timetable needs time_matrix and max_time_work")
    items = list(route)
    if not items:
        raise ValueError("route must not be empty")
    idx = [problem.index_of(x) for x in items]
    d = problem.depot
    if d in idx[1:-1]:
        # a multi-trip route as driven: the depot occurrences ARE the day boundaries
        trips: list[list[int]] = []
        body: list[int] = []
        for i in idx:
            if i == d:
                if body:
                    trips.append(body)
                    body = []
            else:
                body.append(i)
        if body:
            trips.append(body)
        expected = [i for i in range(problem.n) if i != d]
        if sorted(i for trip in trips for i in trip) != expected:
            raise ValueError("route must contain every label exactly once (the depot may repeat)")
        return problem, trips
    # an open tour or a closed route: a giant tour, decoded with the problem's split rule
    tour = problem.to_index_tour(items)
    starts = problem.trip_starts(tour)
    return problem, [tour[a:b].tolist() for a, b in pairwise(starts.tolist())]


@overload
def timetable(
    obj: Any,
    route: Any = ...,
    *,
    start: Any = ...,
    units: str = ...,
    as_frame: Literal[False] = ...,
) -> list[list[Stop]]: ...


@overload
def timetable(
    obj: Any,
    route: Any = ...,
    *,
    start: Any = ...,
    units: str = ...,
    as_frame: Literal[True],
) -> pd.DataFrame: ...


@overload
def timetable(
    obj: Any,
    route: Any = ...,
    *,
    start: Any = ...,
    units: str = ...,
    as_frame: bool,
) -> list[list[Stop]] | pd.DataFrame: ...


def timetable(
    obj: Any,
    route: Any = None,
    *,
    start: Any = "08:00",
    units: str = "min",
    as_frame: bool = False,
) -> list[list[Stop]] | pd.DataFrame:
    """Arrival and departure times of every stop, day by day (D32).

    Parameters
    ----------
    obj : fitted estimator or RoutingProblem
        A fitted solver (its ``problem_`` and ``route_`` are used) or a
        [`RoutingProblem`][skroute.RoutingProblem] with a time matrix and a budget.
    route : sequence of labels, optional
        Required with a ``RoutingProblem``; overrides ``route_`` with an estimator. A multi-trip
        route (the depot repeated between trips, as ``route_``) is read **as driven**, one day per
        segment, whatever ``max_time_work`` says — a day that runs over the budget is reported
        (with a warning), never re-cut; an open tour or a closed route is a giant tour and is cut
        into days with the problem's split rule (``problem.split``). Note the difference with
        [`route_cost`][skroute.metrics.route_cost], which always removes the depot occurrences
        and re-decodes the giant tour: the two agree on every route produced by ``fit``.
    start : str or datetime.time, default "08:00"
        Start of the day at the depot, every day, as ``"HH:MM"`` or a ``datetime.time``; the
        vehicle leaves at ``start`` plus the depot's service time, if any.
    units : {"min", "h", "s"}, default "min"
        Units of ``problem.time`` and ``problem.service_time``; the timetable is always in minutes.
    as_frame : bool, default False
        Return one flat ``pandas.DataFrame`` instead (pandas required: ``pip install
        "scikit-route[pandas]"``), with the columns ``day``, ``order``, ``label``,
        ``arrival_time``, ``departure_time``, ``travel``, ``service``, ``wait``, ``arrival``,
        ``departure``.

    Returns
    -------
    days : list of list of Stop, or DataFrame
        One inner list per day (trip), each opening with the departure from the depot
        (``order 0``, ``departure == service at the depot``, ``0.0`` without one) and closing with
        the return (``arrival`` = minutes after ``start`` at which the day ends). Times are computed
        from ``problem.time`` (driving) and ``problem.service_time`` (stops), so the return of every
        day lands within ``max_time_work`` for a route produced by ``fit``.

    Raises
    ------
    ValueError
        ``"timetable needs time_matrix and max_time_work"`` for a plain TSP; a route that is not a
        valid tour; a ``start`` that is not a clock time.
    TypeError
        When ``obj`` is neither an estimator nor a problem.
    NotFittedError
        For an unfitted estimator.

    Warns
    -----
    UserWarning
        When a day of a route read as driven ends after ``max_time_work``.

    See Also
    --------
    timetable_summary : the totals of every day.

    Examples
    --------
    Minutes of driving between four nodes, a 30-minute service at every customer and a
    day of 200 minutes: the fitted tour needs two days.

    >>> import numpy as np
    >>> from skroute import BruteForce
    >>> from skroute.metrics import timetable
    >>> minutes = np.array([[0, 30, 60, 60], [30, 0, 30, 60], [60, 30, 0, 30], [60, 60, 30, 0]], dtype=float)
    >>> est = BruteForce().fit(
    ...     minutes,
    ...     labels=["office", "a", "b", "c"],
    ...     time_matrix=minutes,
    ...     max_time_work=200.0,
    ...     service_time=30.0,
    ... )
    >>> est.route_.tolist()
    ['office', 'a', 'b', 'office', 'c', 'office']
    >>> days = timetable(est, start="09:00")
    >>> for stop in days[0]:
    ...     print(stop.order, stop.label, stop.arrival_time, stop.departure_time, stop.travel, stop.service)
    0 office 09:00 09:00 0.0 0.0
    1 a 09:30 10:00 30.0 30.0
    2 b 10:30 11:00 30.0 30.0
    3 office 12:00 12:00 60.0 0.0
    >>> [(d[-1].arrival, d[-1].arrival_time) for d in days]  # each day ends within 200 minutes
    [(180.0, '12:00'), (150.0, '11:30')]

    The same from a problem and a route (any tour, decoded with the problem's split rule), and
    as a table when pandas is installed:

    >>> problem = est.problem_
    >>> [[s.label for s in day] for day in timetable(problem, ["office", "c", "b", "a"])]
    [['office', 'c', 'office'], ['office', 'b', 'a', 'office']]
    >>> timetable(est, as_frame=True)[["day", "label", "arrival_time", "departure_time"]]  # doctest: +SKIP
       day   label arrival_time departure_time
    0    1  office        08:00          08:00
    1    1       a        08:30          09:00
    ...
    """
    if units not in _UNITS:
        raise ValueError(f"units must be one of {sorted(_UNITS)}, got {units!r}")
    factor = _UNITS[units]
    problem, trips = _resolve(obj, route)
    start_min = _start_minutes(start)
    T = problem.time
    assert T is not None  # multi_trip was checked by _resolve
    service = problem.service_time
    labels = problem.labels.tolist()
    d = problem.depot
    budget = problem.max_time_work * factor  # the day, in minutes
    # a decoded day never exceeds the budget; the sums below associate differently from the kernel's, so
    # allow the ulps (check 8 itself accepts trip_times_ <= budget + 1e-9)
    tolerance = 1e-9 * max(1.0, budget)
    days: list[list[Stop]] = []
    for day, body in enumerate(trips, start=1):
        depot_service = float(service[d]) * factor
        stops = [
            Stop(day, 0, labels[d], 0.0, depot_service, 0.0, depot_service, start=start_min),
        ]
        clock = depot_service
        prev = d
        for order, j in enumerate(body, start=1):
            travel = float(T[prev, j]) * factor
            arrival = clock + travel
            spent = float(service[j]) * factor
            departure = arrival + spent
            stops.append(Stop(day, order, labels[j], arrival, departure, travel, spent, start=start_min))
            clock = departure
            prev = j
        travel = float(T[prev, d]) * factor
        arrival = clock + travel
        stops.append(Stop(day, len(body) + 1, labels[d], arrival, arrival, travel, 0.0, start=start_min))
        if arrival > budget + tolerance:  # only a route read as driven can get here
            warnings.warn(
                f"day {day} ends at {arrival:g} min, over max_time_work ({budget:g} min): the route was read "
                "as driven, with its depot occurrences as the day boundaries",
                UserWarning,
                stacklevel=2,
            )
        days.append(stops)
    if as_frame:
        return _to_frame(days)
    return days


def _to_frame(days: list[list[Stop]]) -> pd.DataFrame:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "timetable(as_frame=True) needs pandas: pip install 'scikit-route[pandas]'"
        ) from None
    rows = [
        {
            "day": s.day,
            "order": s.order,
            "label": s.label,
            "arrival_time": s.arrival_time,
            "departure_time": s.departure_time,
            "travel": s.travel,
            "service": s.service,
            "wait": s.wait,
            "arrival": s.arrival,
            "departure": s.departure,
        }
        for day in days
        for s in day
    ]
    columns = [
        "day",
        "order",
        "label",
        "arrival_time",
        "departure_time",
        "travel",
        "service",
        "wait",
        "arrival",
        "departure",
    ]
    return pd.DataFrame(rows, columns=columns)


def timetable_summary(days: list[list[Stop]]) -> list[dict[str, Any]]:
    """Totals of every day of a [`timetable`][skroute.metrics.timetable].

    Parameters
    ----------
    days : list of list of Stop
        As returned by ``timetable(...)`` (not the DataFrame form: group that one by ``day``).

    Returns
    -------
    summary : list of dict
        One dict per day with ``day`` (1-based), ``n_stops`` (customers visited, the depot
        excluded), ``driving`` (minutes on the road), ``service`` (minutes at the stops, the
        depot's included), ``total`` (minutes from the start to the return) and ``back_at`` (the
        return as ``HH:MM``).

    Raises
    ------
    TypeError
        When ``days`` is not a list of days of ``Stop``s — the DataFrame of ``as_frame=True``
        included.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import BruteForce
    >>> from skroute.metrics import timetable, timetable_summary
    >>> minutes = np.array([[0, 30, 60, 60], [30, 0, 30, 60], [60, 30, 0, 30], [60, 60, 30, 0]], dtype=float)
    >>> est = BruteForce().fit(minutes, time_matrix=minutes, max_time_work=200.0, service_time=30.0)
    >>> for day in timetable_summary(timetable(est)):
    ...     print(day)
    {'day': 1, 'n_stops': 2, 'driving': 120.0, 'service': 60.0, 'total': 180.0, 'back_at': '11:00'}
    {'day': 2, 'n_stops': 1, 'driving': 120.0, 'service': 30.0, 'total': 150.0, 'back_at': '10:30'}
    """
    if not isinstance(days, (list, tuple)):
        raise TypeError(
            "timetable_summary takes the list of days returned by timetable(...), not a "
            f"{type(days).__name__}; with as_frame=True group the frame by 'day' instead"
        )
    out: list[dict[str, Any]] = []
    for stops in days:
        if not isinstance(stops, (list, tuple)) or not all(isinstance(s, Stop) for s in stops):
            raise TypeError("timetable_summary: every day must be a list of Stop, as timetable(...) returns")
        if not stops:
            continue
        out.append(
            {
                "day": stops[0].day,
                "n_stops": len(stops) - 2,
                "driving": float(sum(s.travel for s in stops)),
                "service": float(sum(s.service for s in stops)),
                "total": float(stops[-1].arrival),
                "back_at": stops[-1].arrival_time,
            }
        )
    return out
