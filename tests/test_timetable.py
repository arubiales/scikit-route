"""``skroute.metrics.timetable`` (D32): arrival and departure times of every stop, day by day, on a
hand-built five-node instance (an office and four customers, minutes), from a fitted estimator and from
a problem plus a route; ``Stop``, the clock strings, the units, the DataFrame form, the summary and a
hypothesis property: every day of a fitted multi-trip solution ends within ``max_time_work``."""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import RoutingProblem, TwoOpt
from skroute.base import BaseRouter, RouterTags
from skroute.exceptions import NotFittedError
from skroute.metrics import Stop, route_cost, timetable, timetable_summary

LABELS = ["office", "a", "b", "c", "d"]
# driving minutes; the office is row 0
T5 = np.array(
    [
        [0, 20, 40, 60, 30],
        [20, 0, 25, 50, 45],
        [40, 25, 0, 30, 55],
        [60, 50, 30, 0, 35],
        [30, 45, 55, 35, 0],
    ],
    dtype=float,
)
S5 = np.array([0.0, 30.0, 45.0, 30.0, 60.0])  # minutes at every stop, none at the office
BUDGET = 240.0  # a four-hour day


class Identity(BaseRouter):
    """Matrix order from the depot; declared budget-aware so that the fits below warn about nothing."""

    def _get_tags(self):
        return RouterTags(kind="construction", budget_aware=True)

    def _solve(self, problem, rng):
        return np.roll(np.arange(problem.n, dtype=np.int64), -problem.depot)


def _fit(**overrides):
    kw = {"labels": LABELS, "depot": "office", "time_matrix": T5, "max_time_work": BUDGET, "service_time": S5}
    kw.update(overrides)
    return Identity().fit(T5, **kw)


def _rows(day):
    return [(s.order, s.label, s.arrival, s.departure, s.travel, s.service) for s in day]


# Hand arithmetic for the giant tour office-a-b-c-d under the greedy rule with the services included:
# office->a 20 + 30 = 50 (back 20: ok), a->b 25 + 45 -> 120 (back 40: 160 ok), b->c 30 + 30 -> 180
# (back 60: 240 ok), c->d 35 + 60 -> 275 > 240: the day closes at c. Day two: office->d 30 + 60, back 30.
DAY1 = [
    (0, "office", 0.0, 0.0, 0.0, 0.0),
    (1, "a", 20.0, 50.0, 20.0, 30.0),
    (2, "b", 75.0, 120.0, 25.0, 45.0),
    (3, "c", 150.0, 180.0, 30.0, 30.0),
    (4, "office", 240.0, 240.0, 60.0, 0.0),
]
DAY2 = [
    (0, "office", 0.0, 0.0, 0.0, 0.0),
    (1, "d", 30.0, 90.0, 30.0, 60.0),
    (2, "office", 120.0, 120.0, 30.0, 0.0),
]


# --------------------------------------------------------------------------- the hand-built instance
def test_timetable_of_a_fitted_estimator_by_hand():
    est = _fit()
    assert est.route_.tolist() == ["office", "a", "b", "c", "office", "d", "office"]
    days = timetable(est)
    assert isinstance(days, list) and len(days) == est.n_trips_ == 2
    assert [_rows(day) for day in days] == [DAY1, DAY2]
    assert [[s.day for s in day] for day in days] == [[1] * 5, [2] * 3]
    assert all(isinstance(s, Stop) for day in days for s in day)
    # clock strings from the default 08:00 start
    assert [(s.arrival_time, s.departure_time) for s in days[0]] == [
        ("08:00", "08:00"),
        ("08:20", "08:50"),
        ("09:15", "10:00"),
        ("10:30", "11:00"),
        ("12:00", "12:00"),
    ]
    assert [(s.arrival_time, s.departure_time) for s in days[1]] == [
        ("08:00", "08:00"),
        ("08:30", "09:30"),
        ("10:00", "10:00"),
    ]
    # the end of every day is the fitted trip time, services included
    assert [day[-1].arrival for day in days] == est.trip_times_.tolist() == [240.0, 120.0]
    assert all(s.wait == 0.0 for day in days for s in day)
    # every customer once, the depot at both ends of every day
    assert sorted(s.label for day in days for s in day[1:-1]) == ["a", "b", "c", "d"]
    assert all(day[0].label == day[-1].label == "office" for day in days)


def test_timetable_from_a_problem_and_a_route():
    est = _fit()
    problem = est.problem_
    # an open tour and a closed route are giant tours, cut with the problem's split rule
    assert [_rows(d) for d in timetable(problem, LABELS)] == [DAY1, DAY2]
    assert [_rows(d) for d in timetable(problem, [*LABELS, "office"])] == [DAY1, DAY2]
    assert [_rows(d) for d in timetable(problem, est.tour_)] == [DAY1, DAY2]
    # a multi-trip route is read as driven: the depot occurrences ARE the day boundaries
    given_days = timetable(problem, ["office", "a", "b", "office", "c", "d", "office"])
    assert [[s.label for s in day] for day in given_days] == [
        ["office", "a", "b", "office"],
        ["office", "c", "d", "office"],
    ]
    assert given_days[1][1].arrival == 60.0 and given_days[1][2].arrival == 60.0 + 30.0 + 35.0
    assert given_days[1][-1].arrival == 60.0 + 30.0 + 35.0 + 60.0 + 30.0  # 215 minutes: fits
    # the same route without the final depot, and a route whose first label is not the depot
    assert [_rows(d) for d in timetable(problem, ["office", "a", "b", "office", "c", "d"])] == [
        _rows(d) for d in given_days
    ]
    assert [[s.label for s in d] for d in timetable(problem, ["a", "b", "office", "c", "d"])] == [
        ["office", "a", "b", "office"],
        ["office", "c", "d", "office"],
    ]
    # a route overrides route_ on an estimator
    assert [_rows(d) for d in timetable(est, ["office", "a", "b", "office", "c", "d", "office"])] == [
        _rows(d) for d in given_days
    ]
    # the optimal decoder of another problem
    optimal = RoutingProblem(
        T5,
        labels=LABELS,
        depot="office",
        time_matrix=T5,
        max_time_work=BUDGET,
        service_time=S5,
        split="optimal",
    )
    idx = optimal.to_index_tour(LABELS)
    starts = optimal.trip_starts(idx)
    days = timetable(optimal, LABELS)
    assert len(days) == len(starts) - 1
    assert [d[-1].arrival for d in days] == optimal.trip_times(idx, starts).tolist()


def test_timetable_start_units_and_clock_strings():
    est = _fit()
    for start in ("06:30", datetime.time(6, 30), " 6:30 "):
        days = timetable(est, start=start)
        assert [(s.arrival_time, s.departure_time) for s in days[0]][:2] == [
            ("06:30", "06:30"),
            ("06:50", "07:20"),
        ]
        assert days[0][0].start == 390.0
    # seconds of a datetime.time count; rounding to the nearest minute; wrapping past midnight
    assert timetable(est, start=datetime.time(8, 0, 30))[0][1].arrival_time == "08:21"  # 08:00:30 + 20 min
    late = timetable(est, start="22:30")
    assert [s.arrival_time for s in late[0]] == ["22:30", "22:50", "23:45", "01:00", "02:30"]
    for bad in ("25:00", "8h", "8:60", 8, None, "08:00:00"):
        with pytest.raises(ValueError, match=r"start must be 'HH:MM' or a datetime\.time"):
            timetable(est, start=bad)
    # a problem in hours gives the same timetable in minutes with units="h"
    hours = _fit(time_matrix=T5 / 60.0, max_time_work=BUDGET / 60.0, service_time=S5 / 60.0)
    assert hours.route_.tolist() == est.route_.tolist()
    for a, b in zip(timetable(hours, units="h"), timetable(est), strict=True):
        assert [(s.arrival, s.departure, s.travel, s.service) for s in a] == pytest.approx(
            [(s.arrival, s.departure, s.travel, s.service) for s in b]
        )
        assert [(s.arrival_time, s.departure_time) for s in a] == [
            (s.arrival_time, s.departure_time) for s in b
        ]
    seconds = _fit(time_matrix=T5 * 60.0, max_time_work=BUDGET * 60.0, service_time=S5 * 60.0)
    assert [s.arrival for s in timetable(seconds, units="s")[0]] == pytest.approx([r[2] for r in DAY1])
    with pytest.raises(ValueError, match="units must be one of"):
        timetable(est, units="days")


def test_timetable_with_a_service_at_the_depot():
    # 15 minutes at the office before leaving: paid once per day, at departure, never on returning
    service = S5.copy()
    service[0] = 15.0
    est = _fit(service_time=service)
    days = timetable(est)
    for day in days:
        assert day[0].arrival == 0.0 and day[0].service == 15.0 and day[0].departure == 15.0
        assert day[0].departure_time == "08:15" and day[-1].service == 0.0
        assert day[-1].arrival == day[-1].departure
    # the day-end still equals trip_times_ (the effective matrix includes the depot's service)
    assert [d[-1].arrival for d in days] == pytest.approx(est.trip_times_.tolist())
    assert days[0][1].arrival == 35.0  # 15 at the office + 20 of driving


def test_timetable_as_frame():
    pd = pytest.importorskip("pandas")
    est = _fit()
    frame = timetable(est, as_frame=True, start="09:00")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
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
    days = timetable(est, start="09:00")
    assert len(frame) == sum(len(day) for day in days) == 8
    assert frame["day"].tolist() == [1] * 5 + [2] * 3 and frame["order"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert frame["label"].tolist() == [s.label for day in days for s in day]
    assert frame["arrival_time"].tolist() == [s.arrival_time for day in days for s in day]
    assert frame["arrival_time"].iloc[0] == "09:00" and frame["departure_time"].iloc[-1] == "11:00"
    assert frame["arrival"].tolist() == [s.arrival for day in days for s in day]
    assert frame["travel"].sum() == pytest.approx(135.0 + 60.0) and frame["service"].sum() == S5.sum()
    assert (frame["wait"] == 0.0).all()


def test_timetable_summary_by_hand():
    est = _fit()
    days = timetable(est, start="07:00")
    assert timetable_summary(days) == [
        {"day": 1, "n_stops": 3, "driving": 135.0, "service": 105.0, "total": 240.0, "back_at": "11:00"},
        {"day": 2, "n_stops": 1, "driving": 60.0, "service": 60.0, "total": 120.0, "back_at": "09:00"},
    ]
    assert timetable_summary([]) == [] and timetable_summary([[]]) == []
    assert timetable_summary(tuple(days)) == timetable_summary(days)


def test_timetable_summary_refuses_the_frame_and_other_kinds_with_a_pointer():
    # the natural mistake: both forms come out of timetable(...), only the list form is summarised
    pd = pytest.importorskip("pandas")
    est = _fit()
    frame = timetable(est, as_frame=True)
    assert isinstance(frame, pd.DataFrame)
    with pytest.raises(TypeError, match=r"not a DataFrame; with as_frame=True group the frame by 'day'"):
        timetable_summary(frame)
    with pytest.raises(TypeError, match="not a NoneType"):
        timetable_summary(None)
    with pytest.raises(TypeError, match="every day must be a list of Stop"):
        timetable_summary(["office", "a"])  # a day is not a list
    with pytest.raises(TypeError, match="every day must be a list of Stop"):
        timetable_summary([[("office", 0.0)]])  # a stop is not a Stop


def test_timetable_keeps_the_days_as_driven_and_warns_when_one_runs_over_the_budget():
    est = _fit()
    problem = est.problem_
    # day two: office->b 40 + 45, ->c 30 + 30, ->d 35 + 60 = 240, back 30 -> 270 > 240: the user's plan does
    # not fit, and the timetable says so instead of re-cutting it
    driven = ["office", "a", "office", "b", "c", "d", "office"]
    with pytest.warns(UserWarning, match=r"day 2 ends at 270 min, over max_time_work \(240 min\)") as rec:
        days = timetable(problem, driven)
    assert len(rec) == 1 and "read as driven" in str(rec[0].message)
    assert [[s.label for s in day] for day in days] == [
        ["office", "a", "office"],
        ["office", "b", "c", "d", "office"],
    ]
    assert [day[-1].arrival for day in days] == [70.0, 270.0]  # 20 + 30 of service + 20 back; 270 as above
    # route_cost re-decodes the same labels as a giant tour (office-a-b-c-d -> DAY1 + DAY2, 195 of driving)
    # while the plan as driven costs 175: the two public metrics describe different plans on purpose
    kw = {"time_matrix": T5, "max_time_work": BUDGET, "service_time": S5, "labels": LABELS}
    assert route_cost(T5, driven, **kw) == 195.0 == route_cost(T5, LABELS, **kw)
    assert sum(s.travel for day in days for s in day) == 175.0
    # the same warning in hours, and none at all for a plan that fits or for anything fit produced
    with pytest.warns(UserWarning, match=r"over max_time_work \(240 min\)"):
        timetable(_fit(time_matrix=T5 / 60.0, max_time_work=4.0, service_time=S5 / 60.0), driven, units="h")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timetable(problem, ["office", "a", "b", "office", "c", "d", "office"])  # 215 minutes: fits
        timetable(est)
        timetable(est, est.tour_)
        assert timetable(problem, LABELS)[0][-1].arrival == BUDGET  # exactly on the budget: no warning
    # the frame form warns too (same code path)
    pytest.importorskip("pandas")
    with pytest.warns(UserWarning, match="day 2 ends at 270 min"):
        timetable(problem, driven, as_frame=True)


def test_stop_is_a_frozen_dataclass_with_clock_properties():
    stop = Stop(day=1, order=2, label="b", arrival=95.0, departure=125.0, travel=20.0, service=30.0)
    assert dataclasses.is_dataclass(stop) and stop.wait == 0.0 and stop.start == 480.0
    assert stop.arrival_time == "09:35" and stop.departure_time == "10:05"
    assert [f.name for f in dataclasses.fields(Stop)] == [
        "day",
        "order",
        "label",
        "arrival",
        "departure",
        "travel",
        "service",
        "wait",
        "start",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        stop.arrival = 0.0  # type: ignore[misc]
    assert "start=" not in repr(stop) and repr(stop).startswith("Stop(day=1, order=2, label='b'")
    assert stop == Stop(1, 2, "b", 95.0, 125.0, 20.0, 30.0)
    assert (
        Stop(1, 0, 0, 0.0, 0.0, 0.0, 0.0, start=23 * 60 + 59.6).departure_time == "00:00"
    )  # rounds and wraps
    # start is left out of equality and hashing as it is of the repr: same minutes, same stop, whatever clock
    later = Stop(1, 2, "b", 95.0, 125.0, 20.0, 30.0, start=9 * 60)
    assert later == stop and hash(later) == hash(stop) and repr(later) == repr(stop)
    assert later.arrival_time == "10:35" != stop.arrival_time
    assert len({stop, later}) == 1 and Stop(1, 2, "b", 95.0, 125.0, 20.0, 30.0, wait=1.0) != stop
    est = _fit()
    assert timetable(est, start="06:00") == timetable(est, start="10:00")
    # microseconds of a datetime.time count like the seconds do
    late = timetable(est, start=datetime.time(9, 5, 30, 999999))
    assert late[0][0].start == pytest.approx(545.5 + 0.999999 / 60.0)
    assert late[0][0].start > 545.5


def test_timetable_errors():
    with pytest.raises(ValueError, match="timetable needs time_matrix and max_time_work"):
        timetable(Identity().fit(T5, labels=LABELS))
    with pytest.raises(ValueError, match="timetable needs time_matrix and max_time_work"):
        timetable(RoutingProblem(T5, labels=LABELS), LABELS)
    with pytest.raises(NotFittedError):
        timetable(Identity())
    with pytest.raises(TypeError, match="timetable expects a fitted estimator or a RoutingProblem, got list"):
        timetable([1, 2, 3])
    problem = _fit().problem_
    with pytest.raises(ValueError, match="timetable needs a route when given a RoutingProblem"):
        timetable(problem)
    with pytest.raises(ValueError, match="route must not be empty"):
        timetable(problem, [])
    with pytest.raises(ValueError, match="'z' is not a label of X"):
        timetable(problem, ["office", "a", "z"])
    with pytest.raises(ValueError, match="must contain every label exactly once"):
        timetable(problem, ["office", "a", "b", "c"])  # d is missing
    with pytest.raises(ValueError, match="must contain every label exactly once"):
        timetable(problem, ["office", "a", "office", "a", "b", "c", "d"])  # a twice, as driven


# --------------------------------------------------------------------------- property: days fit the budget
@st.composite
def multi_trip_instances(draw):
    n = draw(st.integers(4, 9))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    xy = rng.random((n, 2)) * 60.0
    T = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))  # minutes of driving, symmetric
    if draw(st.booleans()):
        T = T * rng.uniform(0.8, 1.2, T.shape)  # one-way streets
    np.fill_diagonal(T, 0.0)
    T = np.ascontiguousarray(T)
    depot = draw(st.integers(0, n - 1))
    round_trip = max(T[depot, v] + T[v, depot] for v in range(n) if v != depot)
    budget = round_trip * draw(st.floats(1.2, 3.0))
    slack = (budget - round_trip) / 2.0
    service = np.array(draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))) * slack
    if draw(st.booleans()):
        service = float(service.max()) if service.max() > 0 else 0.0  # a scalar for every customer
    split = draw(st.sampled_from(["greedy", "optimal"]))
    return {"T": T, "depot": depot, "budget": budget, "service": service, "split": split, "n": n}


@settings(derandomize=True, deadline=None, max_examples=40)
@given(multi_trip_instances())
def test_every_day_of_a_fitted_multi_trip_solution_ends_within_the_budget(inst):
    T, n, depot, budget = inst["T"], inst["n"], inst["depot"], inst["budget"]
    labels = [f"n{i}" for i in range(n)]
    est = TwoOpt().fit(
        T,
        labels=labels,
        depot=labels[depot],
        time_matrix=T,
        max_time_work=budget,
        service_time=inst["service"],
        split=inst["split"],
    )
    days = timetable(est)
    assert len(days) == est.n_trips_
    seen = []
    for k, day in enumerate(days, start=1):
        assert day[0].label == day[-1].label == labels[depot]
        assert [s.day for s in day] == [k] * len(day) and [s.order for s in day] == list(range(len(day)))
        assert day[-1].arrival <= budget + 1e-9 * max(1.0, budget)
        assert day[-1].arrival == pytest.approx(est.trip_times_[k - 1], rel=1e-12, abs=1e-12)
        for prev, cur in itertools.pairwise(day):
            assert cur.arrival == pytest.approx(prev.departure + cur.travel)
            assert cur.departure == pytest.approx(cur.arrival + cur.service + cur.wait)
            assert cur.arrival >= prev.departure and cur.travel >= 0.0 and cur.service >= 0.0
        seen.extend(s.label for s in day[1:-1])
    assert sorted(seen) == sorted(x for x in labels if x != labels[depot])
    # the giant tour re-decoded by the problem gives the same days as route_
    again = timetable(est.problem_, est.tour_)
    assert [_rows(d) for d in again] == [_rows(d) for d in days]
