# The problem model

Every solver in scikit-route optimises the same object — a closed tour from a depot
over a dense cost matrix — and every multi-trip result is that tour *split* into trips
by a rule you choose. This page states the model precisely: what goes in, what a
solution is, how it is priced, and where the edges are (asymmetric matrices, nodes that
cannot be served, solvers that do not see the budget, the size ceiling). The
[getting started](../getting_started.md) page is the gentler introduction;
[multi-trip routing](multi_trip.md) is the practical walkthrough.

## Nodes, depot and the cost matrix

An instance has $n \geq 3$ nodes. Internally they are the row positions $0, \dots, n-1$
of the matrix (*index space*); to you they are *labels* — the positions themselves for
a plain array, the index of a `DataFrame`, the keys of a dict of dicts, or whatever
you pass as `labels=` (*label space*). One node is the **depot**, the first row unless
`depot=<label>` says otherwise.

The **cost matrix** $C$ is dense and square: $C_{ij}$ is the cost of travelling from
node $i$ to node $j$ — rows are origins, columns destinations — in any unit you like
(euros, kilometres, minutes). Every entry must be finite; the diagonal is never read.

[`RoutingProblem`][skroute.RoutingProblem] is the object that holds all of this. `fit`
builds one for you from its arguments, but you can build it yourself:

```python
>>> import numpy as np
>>> from skroute import RoutingProblem
>>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
>>> problem = RoutingProblem(C, labels=["d", "a", "b", "c"], depot="d")
>>> problem
RoutingProblem(n=4, TSP, symmetric, depot='d')
>>> problem.n, problem.depot, problem.labels.tolist(), problem.symmetric   # depot is an index internally
(4, 0, ['d', 'a', 'b', 'c'], True)

```

## The giant tour and the closed route

A **solution** is a permutation $\tau = (\tau_0, \tau_1, \dots, \tau_{n-1})$ of all
nodes with the depot first, $\tau_0 = \text{depot}$: the *giant tour*. Fitted solvers
expose it as `tour_` (open, depot first) and as `route_` (as driven, with the depot
repeated at the end). The plain TSP cost is the sum of the legs, **the return to the
depot included**:

$$
c(\tau) = \sum_{k=0}^{n-1} C_{\tau_k\, \tau_{k+1}}, \qquad \tau_n \equiv \tau_0 .
$$

```python
>>> problem.evaluate([0, 1, 2, 3])                                  # d -> a -> b -> c -> d: 5 + 4 + 3 + 10
22.0
>>> problem.to_index_tour(["d", "a", "b", "c", "d"]).tolist()      # a closed label route -> the index tour
[0, 1, 2, 3]

```

[`evaluate`][skroute.RoutingProblem.evaluate] works in index space;
[`to_index_tour`][skroute.RoutingProblem.to_index_tour] converts labels (the depot may
repeat, so a `route_` is accepted) and
[`route_cost`][skroute.metrics.route_cost] does both in one call for label routes.

## Symmetric and asymmetric matrices

When $C_{ij} = C_{ji}$ for every pair the instance is *symmetric* and a tour costs the
same in both directions. Otherwise it is an **asymmetric TSP** (ATSP): every kernel
reads $C_{ij}$ directionally and the reversed tour is a different, differently priced
solution. Nothing changes in the API — `problem.symmetric` tells you which case you
are in — but two things change underneath:

- reversing a segment (the 2-opt move) has an O(1) cost delta only on a symmetric
  matrix; on an asymmetric one the solvers take the *full-evaluation* path, O(n) per
  candidate move, the same path the multi-trip objective uses. Fine up to a few
  hundred nodes for the iterated searches, a few thousand for a single descent;
- [`ClarkeWright`][skroute.ClarkeWright] refuses an asymmetric cost matrix (its
  savings are undirected). Every other solver accepts one; `MILP` switches to arc
  variables and `BruteForce`/`HeldKarp` are exact for it.

```python
>>> from skroute import BruteForce, ClarkeWright
>>> A = np.array([[0, 1, 5, 5], [5, 0, 1, 5], [5, 5, 0, 1], [1, 5, 5, 0]], dtype=float)   # a one-way ring
>>> RoutingProblem(A).symmetric
False
>>> ring = BruteForce().fit(A)
>>> ring.route_.tolist(), ring.cost_                                # four cheap arcs in the allowed direction
([0, 1, 2, 3, 0], 4.0)
>>> RoutingProblem(A).evaluate([0, 3, 2, 1])                       # the same cycle driven backwards
20.0
>>> ClarkeWright().fit(A)
Traceback (most recent call last):
    ...
ValueError: ClarkeWright requires a symmetric cost matrix

```

## The multi-trip objective

Add three things to `fit` (or to `RoutingProblem`): a **time matrix** $T$ of the same
shape and labels as $C$ (durations, any non-negative unit), a **per-trip budget**
`max_time_work` in the units of $T$, and a **fixed charge** `extra_cost` per trip
beyond the first, multiplied by `people`. The solution is still a giant tour; a *split
rule* cuts it into $m$ consecutive trips, each driven from the depot and back, and the
objective becomes

$$
\text{cost}(\tau) = \sum_{r=1}^{m} c(\text{trip}_r) \;+\; \text{people} \cdot \text{extra\_cost} \cdot (m - 1),
\qquad \text{subject to} \quad t(\text{trip}_r) \leq \text{max\_time\_work} \;\; \forall r,
$$

where $t(\text{trip}_r)$ is the duration of the closed trip, **return leg included**.
A visit that takes time enters $t(\text{trip}_r)$ through `service_time=` — the same
duration at every customer, or one per node — and changes what fits in a day; see
[service times](multi_trip.md#service-times).

### A worked example

Four nodes; the depot is node `1`. Costs are in euros, durations in hours, the
working day lasts four hours and every extra trip costs 3.0:

```python
>>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
>>> hours = {1: {1: 0, 2: 1, 3: 2, 4: 2}, 2: {1: 1, 2: 0, 3: 1, 4: 2},
...          3: {1: 2, 2: 1, 3: 0, 4: 1}, 4: {1: 2, 2: 2, 3: 1, 4: 0}}
>>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> bf.route_.tolist(), bf.cost_, bf.n_trips_
([1, 2, 3, 1, 4, 1], 41.0, 2)
>>> [trip.tolist() for trip in bf.trips_], bf.trip_costs_.tolist(), bf.trip_times_.tolist()
([[1, 2, 3, 1], [1, 4, 1]], [18.0, 20.0], [4.0, 4.0])

```

Hand check. The single trip 1-2-3-4-1 would take $1 + 1 + 1 + 2 = 5$ h, more than the
budget, so the giant tour `1-2-3-4` must break. The **greedy rule** walks the tour with
a clock $t$ and lets a leg $a \to b$ join the open trip only if the trip can still
return afterwards, $t + T_{ab} + T_{b,\text{depot}} \leq \text{max\_time\_work}$:

| Leg | Clock after the leg | Return would end at | Fits 4 h? |
|---|---|---|---|
| 1 → 2 | 1 | 1 + 1 = 2 | yes |
| 2 → 3 | 2 | 2 + 2 = 4 | yes |
| 3 → 4 | 3 | 3 + 2 = 5 | **no** — close the trip at 3, reopen 1 → 4 |
| 1 → 4 | 2 | 2 + 2 = 4 | yes |

Trips `[1, 2, 3, 1]` (4 h, cost $5 + 4 + 9 = 18$) and `[1, 4, 1]` (4 h, cost 20), plus
one extra-trip charge: $18 + 20 + 3.0 = 41.0$. The six possible giant tours evaluate to

| Giant tour | Plain closed cost | Greedy trips | Multi-trip cost |
|---|---|---|---|
| 1-2-3-4 | 22 | `[1,2,3,1]` `[1,4,1]` | **41** |
| 1-2-4-3 | 25 | `[1,2,1]` `[1,4,1]` `[1,3,1]` | 54 |
| 1-3-2-4 | 31 | `[1,3,2,1]` `[1,4,1]` | **41** |
| 1-3-4-2 | 25 | `[1,3,1]` `[1,4,1]` `[1,2,1]` | 54 |
| 1-4-2-3 | 31 | `[1,4,1]` `[1,2,3,1]` | **41** |
| 1-4-3-2 | 22 | `[1,4,1]` `[1,3,2,1]` | **41** |

so 41 is the optimum and `BruteForce`, which visits permutations in lexicographic
order and keeps the first strictly better one, returns `1-2-3-4`. The code agrees with
the table:

```python
>>> from itertools import permutations
>>> problem = RoutingProblem(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> [problem.evaluate(problem.to_index_tour([1, *rest])) for rest in permutations([2, 3, 4])]
[41.0, 54.0, 41.0, 54.0, 41.0, 41.0]
>>> problem.trip_starts(problem.to_index_tour([1, 2, 3, 4])).tolist()   # trips begin at positions 1 and 3 of the giant tour
[1, 3, 4]

```

[`trip_starts`][skroute.RoutingProblem.trip_starts] describes a split as positions in
the giant tour: `starts[0] == 1`, `starts[-1] == n`, and trip $k$ is
`tour[starts[k]:starts[k+1]]`. A plain TSP is the single trip `[1, n]`.

### The return leg is always included

Under both split rules a trip closes while it can still get back: `trip_times_` is the
duration of every *closed* trip and is never above `max_time_work` (up to $10^{-9}$).
This is one of the three ways the 2.0 objective differs from 1.0, which checked the
clock before adding a leg and could overshoot by one leg and by the final return; the
[Barcelona section](#from-10-to-20-the-barcelona-example) below shows a real trip that
1.0 would have extended and 2.0 closes.

### `people` multiplies only the fixed charge

`people` scales `extra_cost` and nothing else: with two people the example costs
$18 + 20 + 2 \cdot 3.0 = 44$. A factor on the travel cost would never change which
tour is best, only the number reported, so 2.0 does not apply one (1.0 did, against
its own docstring).

```python
>>> BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0, people=2).cost_
44.0

```

### Greedy or optimal split

The **greedy** rule (`split="greedy"`, the default) is O(n): one pass over the tour, a
trip closes as soon as the next node would not fit. The **optimal** rule
(`split="optimal"`) is Prins' procedure: among all ways of cutting *this* giant tour
into consecutive feasible trips it picks the cheapest, as a shortest path in the DAG
whose arcs are the feasible trips — O(n L) with L the longest feasible trip. Both
rules produce feasible trips; the optimal one is never worse for the same tour, and is
the same rule inside every budget-aware solver, so the search itself sees the better
decoding.

On the four-node example both give 41 (the only other cut of `1-2-3-4`, `{2}` then
`{3, 4}`, needs $2 + 1 + 2 = 5$ h). They differ when packing greedily forces a worse
cut later. Take the corners of a 4 × 3 rectangle — every distance is an integer — and a
12-hour budget with one hour per unit of distance:

```python
>>> from skroute.preprocessing import distance_matrix
>>> xy = [[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 0.0]]          # d, a, b, c
>>> R = distance_matrix(xy)
>>> R.tolist()
[[0.0, 3.0, 5.0, 4.0], [3.0, 0.0, 4.0, 5.0], [5.0, 4.0, 0.0, 3.0], [4.0, 5.0, 3.0, 0.0]]
>>> spec = dict(time_matrix=R, max_time_work=12.0, labels=["d", "a", "b", "c"], depot="d")
>>> greedy, optimal = RoutingProblem(R, **spec), RoutingProblem(R, **spec, split="optimal")
>>> idx = greedy.to_index_tour(["d", "a", "b", "c"])
>>> greedy.evaluate(idx), greedy.trip_starts(idx).tolist()          # [d,a,b,d] fits exactly (12 h), then [d,c,d]: 12 + 8
(20.0, [1, 3, 4])
>>> optimal.evaluate(idx), optimal.trip_starts(idx).tolist()        # [d,a,d] then [d,b,c,d]: 6 + 12
(18.0, [1, 2, 4])

```

Greedy fills the first trip because `a` and `b` fit; the optimal split sees that
sending `a` alone and pairing `b` with `c` saves two units. The gap between the two
rules on the same tour is what a budget-aware solver gains when you fit under
`split="optimal"`; on Barcelona it is a few percent
([multi-trip routing](multi_trip.md#greedy-or-optimal-split)).

!!! note "What `BruteForce` certifies under a budget"
    Under `split="greedy"`, `BruteForce` is exact **over greedy-decoded giant tours**:
    a partition that closes a trip while the next node would still fit cannot be
    expressed by the greedy rule, whatever the tour. Under `split="optimal"` it is
    exact for the distance-constrained multi-trip problem. On the rectangle the
    difference does not bite — another giant tour, `d-b-c-a`, expresses the cheaper
    split under the greedy rule too:

    ```python
    >>> both = BruteForce().fit(R, **spec)
    >>> both.route_.tolist(), both.cost_
    (['d', 'b', 'c', 'd', 'a', 'd'], 18.0)

    ```

## When a node cannot be served: `InfeasibleProblemError`

A node whose round trip from the depot, $T_{\text{depot},i} + T_{i,\text{depot}}$,
exceeds `max_time_work` cannot be visited by any trip. `RoutingProblem` detects this
at construction — before any search runs — and raises
[`InfeasibleProblemError`][skroute.exceptions.InfeasibleProblemError] naming the
labels. With a three-hour day, nodes 3 and 4 (round trips of 4 h) are out of reach:

```python
>>> RoutingProblem(cost, time_matrix=hours, max_time_work=3.0)
Traceback (most recent call last):
    ...
skroute.exceptions.InfeasibleProblemError: nodes [3, 4] cannot be served in one trip: depot round trip exceeds max_time_work=3.0

```

The remedies are outside the solver: a longer day, a different depot, serving those
nodes separately, or checking the units of the time matrix (a matrix in minutes
against a budget in hours fails exactly like this). The exception is a `ValueError`,
so an existing `except ValueError` still catches it.

## Solvers that do not see the budget

Every solver returns feasible trips, because the base class decodes and prices its
giant tour with the problem's own rule. But only *budget-aware* solvers optimise the
multi-trip objective during their search: `BruteForce`, `ClarkeWright`, `TwoOpt`,
`OrOpt`, `LocalSearch`, `IteratedLocalSearch`, `SimulatedAnnealing`, `TabuSearch`,
`Genetic`, `AntColony`, the two `Ensemble*` wrappers, and `MultiStart` when it wraps
one of them. The capability table on the [home page](../index.md#which-solver) is
generated from the solvers' tags. For the others:

- **Heuristics** — `NearestNeighbour`, `Insertion`, `NRBS`, `SOM` — optimise the
  plain tour, warn once, and their result is still split and priced under the
  objective. A heuristic that ignores the budget is merely a heuristic.
- **Exact solvers** that cannot certify the multi-trip optimum — `HeldKarp` and
  `MILP` — **raise** when `max_time_work` is given. An "exact" answer that is not
  optimal for the objective would be a trap; a certified multi-trip optimum comes
  from `BruteForce` (up to 11 nodes) and larger instances go to the budget-aware
  heuristics.

```python
>>> import warnings
>>> from skroute import HeldKarp, NearestNeighbour
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     nn = NearestNeighbour().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> print(caught[0].message)
NearestNeighbour ignores max_time_work during its search; the result is still split into trips and priced under the multi-trip objective
>>> nn.route_.tolist(), nn.cost_, nn.n_trips_                     # decoded and priced all the same
([1, 2, 3, 1, 4, 1], 41.0, 2)
>>> HeldKarp().fit(cost, time_matrix=hours, max_time_work=4.0)
Traceback (most recent call last):
    ...
ValueError: HeldKarp optimises the plain tour and cannot certify a multi-trip optimum; use BruteForce (n <= 11) or a heuristic solver

```

## The cost is recomputed, never reported

A solver's `_solve` returns only an index tour. The base class checks that it is a
permutation with the depot first (a `RuntimeError` otherwise — a bug in the solver, not
in your data) and computes every fitted attribute from it with the problem's decoder:
`cost_ == trip_costs_.sum() + people * extra_cost * (n_trips_ - 1)` holds for every
solver, and a route can never disagree with its cost. You can redo that arithmetic on
any label route with [`route_cost`][skroute.metrics.route_cost]:

```python
>>> from skroute.metrics import route_cost
>>> route_cost(cost, bf.route_, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
41.0
>>> route_cost(cost, [1, 3, 2, 4], time_matrix=hours, max_time_work=4.0, extra_cost=3.0)   # any tour, same rule
41.0

```

## `RoutingProblem`: build once, solve many times

`fit(X, ...)` builds a `RoutingProblem` from its arguments and stores it as
`problem_`. Building it yourself pays off when several solvers attack the same
instance — the matrices are coerced and validated once, the neighbour lists are cached
on the problem, and `MultiStart` shares one object across threads — and when you want
the split rule and the budget written down in one place:

```python
>>> from skroute import SimulatedAnnealing, TabuSearch
>>> shared = RoutingProblem(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0, split="optimal")
>>> shared
RoutingProblem(n=4, multi-trip, symmetric, depot=1)
>>> [est.fit(shared).cost_ for est in (BruteForce(), SimulatedAnnealing(random_state=0), TabuSearch(random_state=0))]
[41.0, 41.0, 41.0]
>>> BruteForce().fit(shared).problem_ is shared
True

```

A ready problem must be passed **alone**; combining it with other `fit` arguments
raises. The constructor also refuses silent half-configurations, so a unit or a
keyword mistake surfaces immediately:

| You pass | Result |
|---|---|
| `max_time_work` without `time_matrix` | `ValueError` — pass `time_matrix=X` explicitly if the cost matrix *is* the duration |
| `time_matrix` without `max_time_work` | `ValueError` |
| `extra_cost`, `people` or `split` without `max_time_work` | `ValueError` — they would have no effect |
| `time_matrix` positionally, `fit(C, T)` | `TypeError` — it is keyword-only, so 1.0's `(time, cost)` order cannot swap the matrices |
| a time matrix of another shape, or with labels that differ from `X`'s | `ValueError` |
| negative durations, NaN or infinite costs, fewer than three nodes | `ValueError` |

```python
>>> RoutingProblem(cost, max_time_work=4.0)
Traceback (most recent call last):
    ...
ValueError: max_time_work given but no time_matrix; pass time_matrix=X to use the cost matrix as durations
>>> BruteForce().fit(shared, extra_cost=3.0)
Traceback (most recent call last):
    ...
ValueError: X is a RoutingProblem: pass it alone, without other fit arguments

```

## Size: the dense-matrix ceiling

scikit-route 2.0 works only on dense `float64` matrices, which puts the ceiling at
memory rather than at time. A matrix costs $8 n^2$ bytes:

| Nodes | Cost matrix | With a time matrix |
|---|---|---|
| 1 000 | 8 MB | 16 MB |
| 10 000 | 0.8 GB | 1.6 GB |
| 20 000 | 3.2 GB | 6.4 GB |
| 71 009 (`ch71009`) | 40 GB | — |

Budget for two to three copies of the matrix: the one you hold, the problem's
contiguous `float64` copy when the input was not already one, and a transient copy
while the neighbour lists are built. The practical ceiling is **about 20 000 nodes**:
[`distance_matrix`][skroute.preprocessing.distance_matrix] warns above it and
[`TSPBunch.distance_matrix`][skroute.datasets.TSPBunch] refuses unless `force=True`.
The four bundled instances above that size (`vm22775`, `sw24978`, `bm33708`,
`ch71009`) can be read and subsampled with `load_tsp(name, n_nodes=5000)` but not
solved whole; coordinate-only fitting (distances computed on the fly) is planned for
2.1.

Below the ceiling, time is set by the evaluation path. A symmetric plain TSP uses
O(1) move deltas, candidate lists and don't-look bits, and the local searches scale to
tens of thousands of nodes. Asymmetric matrices and the multi-trip objective have no
O(1) delta (a move can change where trips break), so each candidate move is priced in
O(n) — or O(n L) under `split="optimal"`: one descent is fine up to a few thousand
nodes, but the iterated searches and the metaheuristics are comfortable up to a few
hundred nodes at their default iteration counts; lower `n_iter` or set `time_limit`
beyond that. `TabuSearch` also keeps an `int32` `(n, n)` tabu matrix, about 100 MB at
5 000 nodes. The [benchmarks](../benchmarks.md) page lists measured times.

## Labels and indices: the conventions

- **Where labels come from.** `range(n)` for a plain array without `labels=`; the
  index of a `DataFrame` (which must equal its columns); the outer keys of a dict of
  dicts, in insertion order; or `labels=`, which must agree with any labels the input
  already carries. Loader matrices carry none: pass `labels=bunch.labels`.
- **Label dtype.** Labels are stored as `int64` when every one is an integer, as
  `object` otherwise (strings, tuples, mixed types), whatever the input path, so
  `tour_` arrays from different paths compare equal.
- **Uniqueness.** Labels must be unique and hashable.
- **The depot.** `depot=` is a label (a row position for an unlabelled array); it
  defaults to the first node. `depot_` is its label; `problem.depot` its index.
- **Outputs are labels.** `tour_`, `route_`, `trips_`, `labels_` and `depot_` never
  leak indices. `init=` takes labels too and accepts an open `tour_`, a closed route
  or a multi-trip `route_` — every occurrence of the depot is removed and re-inserted
  at the front.
- **Positions in a split.** `trip_starts` and the kernels speak positions in the
  giant tour; `RoutingProblem.to_index_tour` / `to_label_tour` translate.

## From 1.0 to 2.0: the Barcelona example

The [migration guide](../migration.md) lists three behaviour changes of the objective.
Here they are on the Barcelona table (19 places, cost in EUR, time in hours, depot
`10000007`) using a deterministic tour — the nearest-neighbour construction, built
without a budget and then decoded under an eight-hour day:

```python
>>> from skroute import NearestNeighbour
>>> from skroute.datasets import load_barcelona
>>> bcn = load_barcelona()
>>> nn = NearestNeighbour().fit(bcn.cost, labels=bcn.labels, depot=bcn.depot)
>>> day = dict(time_matrix=bcn.time, labels=bcn.labels, depot=bcn.depot, max_time_work=8.0)
>>> p = RoutingProblem(bcn.cost, **day, extra_cost=12.83, people=2)
>>> idx = p.to_index_tour(nn.tour_)
>>> starts = p.trip_starts(idx)
>>> len(starts) - 1, p.trip_times(idx, starts).round(3).tolist()   # two trips, hours
(2, [7.066, 4.946])

```

**1. The return leg counts.** The first trip serves fifteen places and reaches the
last of them 6.213 h after leaving; the drive back makes it 7.066 h. Adding the next
place would put 7.218 h on the clock — still under eight, so 1.0 would have added it —
but the return from there ends at 8.040 h, so 2.0 closes the trip:

```python
>>> T, d = bcn.time, p.depot
>>> first, nxt = idx[starts[0]:starts[1]], idx[starts[1]]
>>> legs = [T[d, first[0]]] + [T[a, b] for a, b in zip(first[:-1], first[1:])]
>>> at_last = float(sum(legs))
>>> len(first), round(at_last, 3), round(at_last + float(T[first[-1], d]), 3)
(15, 6.213, 7.066)
>>> one_more = at_last + float(T[first[-1], nxt])
>>> round(one_more, 3), round(one_more + float(T[nxt, d]), 3)
(7.218, 8.04)

```

**2. `people` multiplies only the fixed charge.** The two trips cost 455.57 EUR of
travel; each person adds one 12.83 EUR charge for the second day, and nothing else:

```python
>>> from skroute.metrics import route_cost
>>> travel = route_cost(bcn.cost, nn.tour_, **day)
>>> one = route_cost(bcn.cost, nn.tour_, **day, extra_cost=12.83)
>>> two = route_cost(bcn.cost, nn.tour_, **day, extra_cost=12.83, people=2)
>>> round(travel, 2), round(one - travel, 2), round(two - one, 2)
(455.57, 12.83, 12.83)

```

**3. Two split rules.** With a five-hour day the same tour needs three trips under
either rule, but the optimal split moves a boundary and saves about 25 EUR:

```python
>>> short = dict(day, max_time_work=5.0, extra_cost=12.83, people=2)
>>> round(route_cost(bcn.cost, nn.tour_, **short), 2)                     # greedy
555.72
>>> round(route_cost(bcn.cost, nn.tour_, **short, split="optimal"), 2)    # optimal
530.01

```

Because of 1 and 2, a route priced by 1.0 and by 2.0 gives different numbers even
when the trips coincide; because of 3, the tour a solver returns may split differently
depending on `split=`. The [multi-trip routing](multi_trip.md) page continues from
here with the solvers.
