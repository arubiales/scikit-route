# Multi-trip routing

A team leaves the depot in the morning, visits customers and must be back before the
working day ends; whatever is left waits for another day, and every extra day has a
price. That is the multi-trip objective of scikit-route: a per-trip time budget, a
fixed charge per trip beyond the first, and travel costs to minimise. This page works
it through on the bundled Barcelona table with the same estimators you use for a plain
TSP. The exact rules are in [the problem model](problem_model.md#the-multi-trip-objective).

## The instance

[`load_barcelona`][skroute.datasets.load_barcelona] returns a
[`Bunch`][skroute.utils.Bunch] with three `(19, 19)` matrices — `cost` in **EUR**,
`time` in **hours**, `distance` in metres — plus coordinates, the place ids as
`labels` and the depot `10000007`:

```python
>>> import numpy as np
>>> from skroute.datasets import load_barcelona
>>> bcn = load_barcelona()
>>> bcn
Bunch(DESCR, coords, cost, depot, distance, frame, labels, time, units)
>>> bcn.cost.shape, bcn.units, bcn.depot
((19, 19), {'cost': 'EUR', 'time': 'h', 'distance': 'm'}, 10000007)
>>> bcn.labels[:4].tolist(), int(bcn.labels[0]) == bcn.depot
([10000007, 1, 4, 5], True)
>>> round(float(bcn.time.max()), 2), round(float((bcn.time[0] + bcn.time[:, 0]).max()), 2)   # longest leg; longest round trip from the depot
(2.46, 3.88)

```

!!! warning "Pass `labels=bcn.labels`"
    The matrices are plain arrays. Without `labels=`, the nodes are `0..18` and the
    depot id is unknown to `fit`:

    ```python
    >>> from skroute import SimulatedAnnealing
    >>> SimulatedAnnealing().fit(bcn.cost, time_matrix=bcn.time, depot=bcn.depot, max_time_work=8.0)
    Traceback (most recent call last):
        ...
    ValueError: depot 10000007 is not a label of X

    ```

    `load_barcelona(as_frame=True)` returns labelled `DataFrame`s instead (pandas
    required), and then `labels=` is not needed.

## Fit: an eight-hour day, 12.83 EUR per extra day, two people

`max_time_work` is in the units of `time_matrix` (hours here), `extra_cost` in the
units of the cost matrix (EUR), and `people` multiplies the charge. Two budget-aware
metaheuristics, [`SimulatedAnnealing`][skroute.SimulatedAnnealing] and
[`TabuSearch`][skroute.TabuSearch], price every candidate move with the decoded trips
and their charges:

```python
>>> from skroute import SimulatedAnnealing, TabuSearch
>>> data = dict(time_matrix=bcn.time, labels=bcn.labels, depot=bcn.depot,
...             max_time_work=8.0, extra_cost=12.83, people=2)
>>> sa = SimulatedAnnealing(random_state=0).fit(bcn.cost, **data)
>>> ts = TabuSearch(random_state=0).fit(bcn.cost, **data)
>>> sa.n_trips_ == len(sa.trips_) == len(sa.trip_times_) == len(sa.trip_costs_)
True
>>> bool(np.all(sa.trip_times_ <= 8.0)) and bool(np.all(ts.trip_times_ <= 8.0))   # every trip, return included
True
>>> int(sa.route_[0]) == int(sa.route_[-1]) == 10000007
True
>>> sa.route_.tolist().count(10000007) == sa.n_trips_ + 1    # the depot opens and closes every trip
True
>>> sum(len(trip) - 2 for trip in sa.trips_)                 # every one of the 18 customers is served once
18
>>> abs(sa.cost_ - (float(sa.trip_costs_.sum()) + 2 * 12.83 * (sa.n_trips_ - 1))) < 1e-9
True

```

What the attributes hold:

| Attribute | Shape | Meaning |
|---|---|---|
| `trips_` | `list` of `n_trips_` arrays | each `[depot, ..., depot]`, in driving order |
| `route_` | `(n + n_trips_,)` | the trips concatenated, the depot shared between consecutive ones |
| `tour_` | `(n,)` | the giant tour without the intermediate depots — the warm-start format |
| `trip_times_` | `(n_trips_,)` | hours per closed trip, each `<= max_time_work` |
| `trip_costs_` | `(n_trips_,)` | EUR of travel per trip, fixed charge excluded |
| `n_trips_` | `int` | `len(trips_)` |
| `cost_` | `float` | `trip_costs_.sum() + people * extra_cost * (n_trips_ - 1)` |

On the author's machine the seed above gives two days; a different platform may
tie-break the annealing differently and land a few euros away, which is why the lines
below are illustrative:

```python
>>> sa.route_.tolist()  # doctest: +SKIP
[10000007, 47, 30, 12, 5, 31, 1, 25, 91, 4, 59, 23, 10000007, 7, 26, 27, 65, 46, 44, 32, 10000007]
>>> sa.trip_times_.round(2).tolist(), sa.trip_costs_.round(2).tolist(), round(sa.cost_, 2)  # doctest: +SKIP
([7.14, 4.08], [275.21, 144.9], 445.77)

```

Read it as: day one, eleven addresses in 7.14 h for 275.21 EUR; day two, seven
addresses in 4.08 h for 144.90 EUR; plus one extra day for two people,
$2 \times 12.83 = 25.66$ EUR; total 445.77 EUR.

## Greedy or optimal split

A solver returns a giant tour; a *split rule* cuts it into trips. `split="greedy"`
(default) closes a trip as soon as the next customer would not fit; `split="optimal"`
picks the cheapest feasible cut of the same tour (Prins, 2004) and is never worse.
Build a [`RoutingProblem`][skroute.RoutingProblem] with each rule to compare them on
the tour the annealing found:

```python
>>> from skroute import RoutingProblem
>>> greedy = RoutingProblem(bcn.cost, **data)                        # split="greedy" is the default
>>> optimal = RoutingProblem(bcn.cost, **data, split="optimal")
>>> idx = greedy.to_index_tour(sa.tour_)
>>> optimal.evaluate(idx) <= greedy.evaluate(idx) + 1e-9
True

```

The difference shows on tours that were not built with the budget in mind. The
nearest-neighbour tour, decoded under a *five*-hour day, needs three trips either way,
but the optimal cut moves a boundary and saves about 25 EUR:

```python
>>> from skroute import NearestNeighbour
>>> nn = NearestNeighbour().fit(bcn.cost, labels=bcn.labels, depot=bcn.depot)   # a plain tour, no budget
>>> short = dict(data, max_time_work=5.0)
>>> g5, o5 = RoutingProblem(bcn.cost, **short), RoutingProblem(bcn.cost, **short, split="optimal")
>>> idx = g5.to_index_tour(nn.tour_)
>>> round(g5.evaluate(idx), 2), g5.trip_times(idx, g5.trip_starts(idx)).round(2).tolist()
(555.72, [4.89, 4.4, 3.88])
>>> round(o5.evaluate(idx), 2), o5.trip_times(idx, o5.trip_starts(idx)).round(2).tolist()
(530.01, [4.89, 2.87, 4.95])

```

To search under the optimal rule, fit on a problem built with `split="optimal"` (a
ready problem is passed alone, without other `fit` arguments):

```python
>>> sa_opt = SimulatedAnnealing(random_state=0).fit(optimal)
>>> sa_opt.problem_.split, bool(np.all(sa_opt.trip_times_ <= 8.0))
('optimal', True)

```

!!! tip "Which rule?"
    Evaluating a tour costs O(n) under `greedy` and O(n L) under `optimal`, and a
    metaheuristic evaluates millions of tours, so `greedy` is the default. Two cheap
    ways to get most of the benefit: fit under `greedy` and re-decode `tour_` with an
    `optimal` problem (`optimal.evaluate(...)`, as above — it is never worse), or fit
    under `optimal` on instances of a few dozen to a few hundred nodes, where the extra
    factor is affordable.

## A budget-aware construction: Clarke–Wright

[`ClarkeWright`][skroute.ClarkeWright] is deterministic, instantaneous and the one
construction heuristic whose search sees the budget: every customer starts as its own
out-and-back trip and the pairs of trips that save the most are merged while the
merged trip still fits. It is a good first answer and a good warm start:

```python
>>> from skroute import ClarkeWright
>>> cw = ClarkeWright().fit(bcn.cost, **data)
>>> cw.n_trips_, round(cw.cost_, 2), bool(np.all(cw.trip_times_ <= 8.0))
(2, 460.99, True)
>>> polished = TabuSearch(init=cw.tour_, random_state=0).fit(bcn.cost, **data)   # warm start from the savings tour
>>> polished.cost_ <= cw.cost_ + 1e-9
True

```

The `shape` parameter weights the inter-customer leg in the saving
$s_{ij} = C_{di} + C_{jd} - \text{shape} \cdot C_{ij}$; `1.0` is the classical rule.
The trips reported can differ from the savings trips because the base class re-decodes
the giant tour with the problem's split rule; every reported trip fits the budget.

## The seven-minute stop

The Spanish tables' `time` is **not** driving time alone: the `hours` column they come
from adds a fixed stop of 7 minutes to every leg, `hours = (secs + 420) / 3600`, so an
eight-hour budget already accounts for the time spent at each address. The long table
(`as_frame=True`, pandas required) shows the rule:

```python
>>> frame = load_barcelona(as_frame=True).frame              # the 190-row long table
>>> legs = frame[frame["secs"] > 0]                          # off-diagonal rows
>>> bool(np.allclose(legs["hours"], (legs["secs"] + 420) / 3600))
True
>>> round(float(bcn.time[bcn.time > 0].min()) * 3600)       # the shortest leg in seconds: 333 s of driving + 420 s stop
753

```

When you build your own time matrix, keep it as driving time and pass the visits
through `service_time=` (next section) instead of folding them into every leg by hand: the
budget is then met on the road as well as on paper, and the timetable knows what is
driving and what is time at the door.

## Service times

A visit takes time. `fit(..., service_time=)` accepts a scalar — the same duration at
every customer, nothing at the depot — or one value per node in matrix row order, in the
units of the time matrix, and it requires `max_time_work`. The solver then searches on the
*effective* time matrix $T^{\text{eff}}_{ij} = T_{ij} + s_j$: the service of a node is paid
on arrival, nothing is paid on returning to the depot, and a service at the depot itself
is paid once per day, at departure. `trip_times_` counts the visits, `problem_.time`
keeps the raw driving times, and `problem_.time_or_cost` is the effective matrix.

Half an hour at each of the 18 addresses turns the two-day plan of
[`TwoOpt`][skroute.TwoOpt] (deterministic, budget-aware) into three days:

```python
>>> from skroute import TwoOpt
>>> two = TwoOpt().fit(bcn.cost, **data)
>>> two.n_trips_, two.trip_times_.round(2).tolist()
(2, [6.73, 4.76])
>>> visits = TwoOpt().fit(bcn.cost, service_time=0.5, **data)
>>> visits.n_trips_, visits.trip_times_.round(2).tolist(), round(visits.cost_, 2)
(3, [7.44, 7.57, 6.26], 512.03)
>>> visits.problem_.service_time.tolist() == [0.0] + [0.5] * 18
True
>>> eff, raw = visits.problem_.time_or_cost, visits.problem_.time
>>> bool(np.allclose(eff[0, 1:], raw[0, 1:] + 0.5)) and bool(np.array_equal(eff[1:, 0], raw[1:, 0]))
True

```

That effective matrix *is* the definition of the feature: folding the half hour into the
columns of the non-depot nodes by hand gives the same fit.

```python
>>> folded = bcn.time.copy()
>>> folded[:, 1:] += 0.5                                     # pay the visit on arrival, never on the way back
>>> by_hand = TwoOpt().fit(bcn.cost, **dict(data, time_matrix=folded))
>>> by_hand.tour_.tolist() == visits.tour_.tolist() and by_hand.cost_ == visits.cost_
True
>>> from skroute.metrics import route_cost
>>> route_cost(bcn.cost, visits.route_, service_time=0.5, **data) == visits.cost_
True

```

The feasibility check counts the visit too: under a four-hour day node `91` (3.88 h there
and back) cannot be served once its half hour is added, and the message says so.

```python
>>> RoutingProblem(bcn.cost, service_time=0.5, **dict(data, max_time_work=4.0))
Traceback (most recent call last):
    ...
skroute.exceptions.InfeasibleProblemError: nodes [91] cannot be served in one trip: depot round trip plus service time exceeds max_time_work=4.0 (91: travel 3.88... + service 0.5)

```

An array gives every node its own duration — a longer visit at the far address, a
quarter of an hour loading at the depot before leaving:

```python
>>> per_node = np.full(19, 0.5)
>>> per_node[bcn.labels.tolist().index(91)] = 1.0            # one hour at node 91
>>> per_node[0] = 0.25                                       # loading at the depot, once per day
>>> loaded = TwoOpt().fit(bcn.cost, service_time=per_node, **data)
>>> float(loaded.problem_.service_time[0]), bool(np.all(loaded.trip_times_ <= 8.0))
(0.25, True)

```

## Timetables

[`timetable`][skroute.metrics.timetable] turns a fitted solution into clock times: one
list of [`Stop`][skroute.metrics.Stop]s per day, opening with the departure from the depot
(order `0`) and closing with the return, computed from `problem_.time` (driving) and
`problem_.service_time` (the visits). Times are minutes since `start`, with `arrival_time`
and `departure_time` as `HH:MM`; the Barcelona matrix is in hours, hence `units="h"`
(the default, `"min"`, fits a matrix from a routing API).

```python
>>> from skroute.metrics import timetable, timetable_summary
>>> days = timetable(visits, start="08:00", units="h")
>>> len(days) == visits.n_trips_
True
>>> for stop in days[2]:                                     # the third day: three addresses
...     print(stop.order, stop.label, stop.arrival_time, stop.departure_time, round(stop.travel), stop.service)
0 10000007 08:00 08:00 0 0.0
1 4 08:46 09:16 46 30.0
2 91 10:51 11:21 95 30.0
3 25 12:56 13:26 95 30.0
4 10000007 14:16 14:16 49 0.0
>>> bool(np.allclose([day[-1].arrival / 60 for day in days], visits.trip_times_))   # the day ends when the trip does
True
>>> for row in timetable_summary(days):
...     print(row["day"], row["n_stops"], round(row["driving"]), row["service"], row["back_at"])
1 8 206 240.0 15:26
2 7 244 210.0 15:34
3 3 286 90.0 14:16

```

Read the third day as: leave at 08:00, drive 46 minutes to `4`, work there until 09:16,
95 minutes to `91`, and so on; back at the depot at 14:16, 6.26 h after leaving — the
`trip_times_[2]` above. A timetable also takes a [`RoutingProblem`][skroute.RoutingProblem]
and any label-space route: a multi-trip route (the depot repeated, as `route_`) is read as
driven, an open tour or a closed route is cut into days with the problem's split rule.
*As driven* means **your** days: a hand-made plan whose day runs over `max_time_work` is
shown as it is, with a `UserWarning` naming the day — whereas `route_cost` always removes
the depot occurrences and re-decodes the giant tour, so it prices the plan as the decoder
would cut it, not as driven. The two agree on every route `fit` produces.
With pandas installed, `as_frame=True` returns one flat table ready for `to_csv`:

```python
>>> table = timetable(visits, units="h", as_frame=True)
>>> table.shape, table.columns.tolist()[:5]
((24, 10), ['day', 'order', 'label', 'arrival_time', 'departure_time'])
>>> timetable(visits.problem_, visits.tour_, units="h")[0][-1].arrival_time    # the giant tour, re-decoded
'15:26'

```

## When a customer cannot be served

Node `91` is the farthest from the depot: 3.88 h there and back. With a three-hour day
no trip can include it, and [`RoutingProblem`][skroute.RoutingProblem] says so before
any search runs, naming the labels:

```python
>>> from skroute.exceptions import InfeasibleProblemError
>>> RoutingProblem(bcn.cost, **dict(data, max_time_work=3.0))
Traceback (most recent call last):
    ...
skroute.exceptions.InfeasibleProblemError: nodes [91] cannot be served in one trip: depot round trip exceeds max_time_work=3.0
>>> try:
...     SimulatedAnnealing(random_state=0).fit(bcn.cost, **dict(data, max_time_work=3.0))
... except InfeasibleProblemError as err:
...     unreachable = str(err).split("]")[0].split("[")[1]
>>> unreachable
'91'

```

What to do depends on the business, not on the solver: lengthen the day, move the
depot, serve the node from another depot or on its own, or drop it from the matrix
(`np.delete` on both matrices and on `labels`) and route the rest. Check the units
first — a time matrix in minutes against a budget in hours fails exactly like this.
`InfeasibleProblemError` is a `ValueError`, so a generic `except ValueError` also
catches it.

## Building your own time matrix

Two helpers in [`skroute.preprocessing`](../api/preprocessing.md) cover the usual
sources. From a **long table** of `(origin, destination, value)` rows — a routing API
export, a database query — [`pairs_to_matrix`][skroute.preprocessing.pairs_to_matrix]
pivots by labels, so the row order does not matter and a missing direction is mirrored
when `symmetric=True`:

```python
>>> from skroute.preprocessing import pairs_to_matrix
>>> hours, labels = pairs_to_matrix(frame["id_origin"], frame["id_destinity"], frame["hours"])
>>> eur, _ = pairs_to_matrix(frame["id_origin"], frame["id_destinity"], frame["cost"])
>>> hours.shape, bool(np.allclose(hours, bcn.time)), bool(np.array_equal(labels, bcn.labels))
((19, 19), True, True)

```

From **coordinates**, [`distance_matrix`][skroute.preprocessing.distance_matrix] gives
great-circle kilometres (`metric="haversine"`, decimal-degree `(latitude, longitude)`)
that an average speed turns into hours of driving; the time at each address goes
through `service_time=`. It is a rough proxy for road times — on this table the road
distance is typically 30–40 % longer than the straight line (median ratio 1.37) and
several times longer for a few pairs — but it is what you have before calling a routing
API:

```python
>>> from skroute.preprocessing import distance_matrix
>>> km = distance_matrix(bcn.coords, metric="haversine")
>>> approx = km / 40.0                                         # hours of driving at 40 km/h
>>> approx.shape, bool(np.allclose(np.diag(approx), 0.0)), bool(np.all(approx[km > 0] > 0))
((19, 19), True, True)
>>> rough = TwoOpt().fit(bcn.cost, time_matrix=approx, labels=bcn.labels, depot=bcn.depot,
...                      max_time_work=8.0, extra_cost=12.83, people=2, service_time=7 / 60)
>>> rough.n_trips_ >= 1 and bool(np.all(rough.trip_times_ <= 8.0))
True

```

A time matrix may be asymmetric (one-way streets, uphill legs) even when the cost
matrix is symmetric; only the cost matrix must be symmetric for `ClarkeWright`.

## Pitfalls

**`time_matrix` is keyword-only.** 1.0 took `fit(route, time, cost)`; a migrated call
that keeps the old positional order would swap two square matrices without any error
and price hours as euros. 2.0 refuses it:

```python
>>> SimulatedAnnealing().fit(bcn.cost, bcn.time)
Traceback (most recent call last):
    ...
TypeError: ...positional arguments but 3 were given

```

**A half-configured budget raises** instead of silently doing nothing:

```python
>>> SimulatedAnnealing().fit(bcn.cost, labels=bcn.labels, depot=bcn.depot, extra_cost=12.83)
Traceback (most recent call last):
    ...
ValueError: extra_cost, people and split have no effect without max_time_work
>>> SimulatedAnnealing().fit(bcn.cost, labels=bcn.labels, depot=bcn.depot, max_time_work=8.0)
Traceback (most recent call last):
    ...
ValueError: max_time_work given but no time_matrix; pass time_matrix=X to use the cost matrix as durations

```

**Not every solver searches under the budget.** `NearestNeighbour`, `Insertion`, `NRBS`
and `SOM` optimise the plain tour and warn (`UserWarning`); their result is still split
into feasible trips and priced. `HeldKarp` and `MILP` raise, because they cannot
certify a multi-trip optimum; `BruteForce` can, up to 11 nodes. The capability table
on the [home page](../index.md#which-solver) marks the budget-aware solvers.

**Units.** The budget is in the units of the time matrix, `extra_cost` in the units of
the cost matrix; the solver has no way to notice a mismatch except through an
`InfeasibleProblemError` (minutes against hours) or an absurd number of trips.

## Reusing the problem and running in parallel

A `RoutingProblem` is immutable and shared safely across threads.
[`MultiStart`][skroute.MultiStart] fits clones of a stochastic solver from independent
seeds on the same problem — on threads by default, since the kernels release the GIL —
and keeps the best; wrapping a budget-aware solver keeps the budget inside the search:

```python
>>> from skroute import MultiStart
>>> ms = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=2, random_state=0).fit(greedy)
>>> len(ms.estimators_), ms.n_trips_ == len(ms.trips_), bool(np.all(ms.trip_times_ <= 8.0))
(4, True, True)
>>> ms.cost_ == float(ms.costs_.min())
True

```

Where next: [choosing a solver](choosing_a_solver.md) for the trade-offs between the
budget-aware solvers, [warm starts and ensembles](warm_starts_and_ensembles.md) for
chaining them, and the [datasets](../api/datasets.md) page for the other cost tables
(Alicante–Murcia, Madrid, Valencia and Qatar).
