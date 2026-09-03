# Getting started

scikit-route solves routing problems with the estimator API of scikit-learn: you
build a solver with its knobs, call `fit` on a cost matrix and read the answer from
trailing-underscore attributes. This page takes you from `pip install` to a solved
instance and explains what comes back.

## Install

```bash
pip install scikit-route
```

Wheels are published for CPython 3.11–3.14 on Linux, macOS and Windows, so no compiler
is needed. `pip install "scikit-route[pandas]"` adds `DataFrame` inputs and labelled
dataset loaders. Extras, source builds and supported versions:
[Installation](installation.md).

## The mental model

Four pieces, always in the same order:

| Step | What it is | Example |
|---|---|---|
| 1. A cost matrix | dense `(n, n)`, rows are origins and columns destinations: a numpy array, a pandas `DataFrame` or a dict of dicts | `C = wi.distance_matrix()` |
| 2. An estimator | the solver class, configured with its hyper-parameters (the *knobs*) | `IteratedLocalSearch(random_state=0)` |
| 3. `fit` | receives the *data* — the matrix and, optionally, labels, the depot, a time matrix and a budget — and returns the estimator | `est.fit(C, labels=wi.labels)` |
| 4. Fitted attributes | trailing underscore: `route_`, `tour_`, `cost_`, `trips_`, ... | `est.cost_` |

Nothing that describes the *instance* goes into `__init__` and nothing that
configures the *algorithm* goes into `fit` — the scikit-learn rule. Refitting an
estimator on another matrix clears the old attributes first.

## A first tour

Start from coordinates. [`distance_matrix`][skroute.preprocessing.distance_matrix]
turns `(n, 2)` points into the dense matrix every solver consumes, and
[`BruteForce`][skroute.BruteForce] prices every one of the $(n-1)!$ tours, so on five
points the answer is exact and the output deterministic:

```python
>>> import numpy as np
>>> from skroute import BruteForce
>>> from skroute.preprocessing import distance_matrix
>>> xy = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 0.0], [2.0, 1.5]])  # five points in the plane
>>> C = distance_matrix(xy)               # Euclidean, float64 (5, 5), zero diagonal
>>> C[0].tolist()                         # row 0: from the first point to every point
[0.0, 3.0, 5.0, 4.0, 2.5]
>>> bf = BruteForce().fit(C)              # the first row is the depot unless you say otherwise
>>> bf.route_.tolist(), bf.cost_, bf.is_optimal_
([0, 1, 2, 3, 4, 0], 15.0, True)

```

Now a real instance. [`load_tsp`][skroute.datasets.load_tsp] returns one of the 27
national instances bundled with the library, with the tour length published by the
University of Waterloo, and [`IteratedLocalSearch`][skroute.IteratedLocalSearch] is
the recommended default solver:

```python
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")                 # Western Sahara: 29 cities, published optimum 27603
>>> wi.coords.shape, wi.labels[:3].tolist(), wi.depot, wi.optimal_tour_length
((29, 2), [1, 2, 3], 1, 27603)
>>> C = wi.distance_matrix()              # TSPLIB EUC_2D metric (rounded to integers), cached on the bunch
>>> ils = IteratedLocalSearch(random_state=0).fit(C, labels=wi.labels)
>>> ils.cost_ / wi.optimal_tour_length < 1.03    # the tolerance the test-suite asserts; the run itself usually hits 27603
True

```

!!! warning "Loader matrices carry no labels"
    `wi.distance_matrix()` is a plain array, so without `labels=wi.labels` the nodes
    would be `0..28` and `wi.depot` (the file's id `1`) would be "not a label of X".
    Every loader-based example passes `labels=`; the
    [datasets page](api/datasets.md) says the same.

## Reading the result

```python
>>> len(ils.tour_), len(ils.route_)       # open giant tour: n labels; route as driven: back to the depot
(29, 30)
>>> int(ils.tour_[0]) == int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == 1
True
>>> ils.n_trips_, len(ils.trips_), hasattr(ils, "trip_times_")   # plain TSP: one trip and no time matrix
(1, 1, False)
>>> ils.n_iter_ == len(ils.history_) and bool(np.all(np.diff(ils.history_) <= 0))
True
>>> ils.stop_reason_ in {"patience", "max_iter", "time_limit"}
True
>>> ils.fit_time_ < 5.0                   # seconds spent in the search
True

```

| Attribute | Meaning |
|---|---|
| `tour_` | the open giant tour in label space, depot first — what `init=` accepts as a warm start |
| `route_` | the route as driven: depot, trip 1, depot, trip 2, ..., depot; for a plain TSP it is `tour_` plus the return |
| `trips_` | one closed `[depot, ..., depot]` array per trip; a single one for a plain TSP |
| `cost_` | the objective, recomputed from the tour by the base class (never reported by the solver) |
| `n_trips_`, `trip_costs_`, `trip_times_` | per-trip figures; `trip_times_` exists only when a time matrix was given |
| `history_`, `n_iter_`, `stop_reason_` | iterative solvers only: best cost after each outer iteration (non-increasing), how many ran, and one of `"converged"`, `"max_iter"`, `"patience"`, `"time_limit"` (each class documents its subset) |
| `is_optimal_` | exact solvers only (`BruteForce`, `HeldKarp`, `MILP`) |
| `problem_`, `labels_`, `depot_`, `n_nodes_`, `fit_time_` | the coerced [`RoutingProblem`][skroute.RoutingProblem], the labels in matrix order, the depot's label, `n`, seconds in the search |

`problem_` is reusable: hand it to another solver and both work on the very same
instance. Here [`MILP`][skroute.MILP] proves that the iterated local search found the
published optimum:

```python
>>> from skroute import MILP
>>> proof = MILP().fit(ils.problem_)      # a ready RoutingProblem is passed alone
>>> proof.is_optimal_, int(proof.cost_) == wi.optimal_tour_length
(True, True)

```

## Three kinds of input

`fit` accepts the cost matrix in three shapes. In every case the result comes back in
the labels of the input.

=== "numpy array"

    Nodes are the row positions `0..n-1` unless you pass `labels=`; the depot is row 0
    unless you pass `depot=`.

    ```python
    >>> C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> est = BruteForce().fit(C4)
    >>> est.labels_.tolist(), est.route_.tolist(), est.cost_
    ([0, 1, 2, 3], [0, 1, 2, 3, 0], 22.0)

    ```

=== "pandas DataFrame"

    The index provides the labels (index and columns must hold the same labels in the
    same order); the depot defaults to the first label. pandas is optional
    (`pip install "scikit-route[pandas]"`) and is recognised by duck typing, so a
    plain installation never imports it.

    ```python
    >>> import pandas as pd
    >>> names = ["depot", "a", "b", "c"]
    >>> frame = pd.DataFrame(C4, index=names, columns=names)
    >>> est = BruteForce().fit(frame)
    >>> est.labels_.tolist(), est.depot_, est.route_.tolist(), est.cost_
    (['depot', 'a', 'b', 'c'], 'depot', ['depot', 'a', 'b', 'c', 'depot'], 22.0)

    ```

=== "dict of dicts"

    The legacy 1.0 format `{origin: {destination: cost}}`. The outer keys are the
    labels, in insertion order, and the first key is the depot.

    ```python
    >>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
    ...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
    >>> BruteForce().fit(cost).route_.tolist()
    [1, 2, 3, 4, 1]

    ```

A fourth form is a ready [`RoutingProblem`][skroute.RoutingProblem] (`est.problem_`, or
one you build yourself), which must be passed alone — see
[the problem model](user_guide/problem_model.md#routingproblem-build-once-solve-many-times).
The matrix must be square with at least three nodes and finite entries; the diagonal is
never read. An asymmetric matrix is treated as an asymmetric TSP, directionally, by
every solver except `ClarkeWright`.

## Labels and the depot

`labels=` names the rows of a plain array with any hashables (ints, strings, tuples);
`depot=` is then a *label*. Everything you read back — `tour_`, `route_`, `trips_`,
`depot_` — is in label space:

```python
>>> est = BruteForce().fit(C4, labels=[10, 20, 30, 40], depot=30)
>>> int(est.depot_), est.route_.tolist(), est.cost_  # the same 22-cost cycle, driven from node 30
(30, [30, 20, 10, 40, 30], 22.0)
>>> BruteForce().fit(C4, depot=2).route_.tolist()    # without labels, the depot is a row position
[2, 1, 0, 3, 2]
>>> BruteForce().fit(C4, labels=[10, 20, 30, 40], depot=0)
Traceback (most recent call last):
    ...
ValueError: depot 0 is not a label of X

```

Labels must be unique, and `labels=` must agree with labels the input already carries
(a `DataFrame` index or dict keys). Integer-like labels are stored as `int64`, anything
else as `object`, so results compare equal whatever the input path.

## Reproducibility: `random_state`

Stochastic solvers take `random_state` (an int, a `numpy.random.Generator` or `None`).
All randomness is drawn in Python from `numpy.random.default_rng(random_state)` before
it reaches the compiled kernels, so the same seed on the same machine gives
bit-identical results:

```python
>>> a = IteratedLocalSearch(random_state=0).fit(C, labels=wi.labels)
>>> b = IteratedLocalSearch(random_state=0).fit(C, labels=wi.labels)
>>> a.tour_.tolist() == b.tour_.tolist() and a.cost_ == b.cost_
True

```

!!! note "What reproducibility does and does not promise"
    - A `Generator` passed as `random_state` is used in place and *advanced* by the
      fit, so two consecutive fits with the same generator differ.
    - `time_limit=` stops a search by the wall clock and therefore breaks bit-exact
      reproducibility.
    - Solvers whose acceptance rule calls `exp` (simulated annealing, iterated local
      search with `acceptance="metropolis"`) may tie-break differently on another
      operating system; the result is still valid, and never below the optimum.
    - Deterministic solvers (`BruteForce`, `HeldKarp`, `MILP`, the constructions and
      the plain descents) have no `random_state` at all.

## Warm starts: `init=`

Every iterative solver starts from a tour. The default `init="nearest_neighbour"`
builds it greedily from the depot; `init="random"` draws a permutation from
`random_state` (stochastic solvers only); and an array of labels — the `tour_` or
`route_` of another solver — resumes from that tour:

```python
>>> from skroute import NearestNeighbour, LocalSearch
>>> nn = NearestNeighbour().fit(C, labels=wi.labels)             # quick and poor: about 30 % above the optimum
>>> ls = LocalSearch(init=nn.tour_).fit(C, labels=wi.labels)     # 2-opt and Or-opt from that tour
>>> ls.cost_ <= nn.cost_ and ls.stop_reason_ in {"converged", "max_iter"}
True
>>> polished = LocalSearch(init=ils.tour_).fit(C, labels=wi.labels)   # an ILS result is already a local optimum
>>> polished.cost_ == ils.cost_, polished.n_iter_, polished.stop_reason_
(True, 1, 'converged')

```

`init=` also accepts a multi-trip `route_` (the depot may repeat). The
[warm starts and ensembles](user_guide/warm_starts_and_ensembles.md) page shows how to
chain solvers and how [`MultiStart`][skroute.MultiStart] runs several seeds in parallel.

## A taste of the multi-trip objective

Give `fit` a time matrix and a per-trip budget and the same estimators split the tour
into trips that fit a working day, charging `people * extra_cost` for every trip after
the first. On the eight-place Alicante–Murcia table `BruteForce` certifies the answer:

```python
>>> from skroute.datasets import load_alicante_murcia
>>> ali = load_alicante_murcia()          # cost in EUR, time in hours; depot 10000002
>>> day = BruteForce().fit(ali.cost, time_matrix=ali.time, labels=ali.labels, depot=ali.depot,
...                        max_time_work=5.0, extra_cost=12.83, people=2)
>>> day.n_trips_, round(day.cost_, 2), bool(np.all(day.trip_times_ <= 5.0))
(2, 337.96, True)
>>> day.route_.tolist().count(10000002)   # the depot opens and closes each of the two trips
3

```

`time_matrix` is keyword-only on purpose: 1.0 took the two square matrices
positionally and a swapped call priced hours as euros. Read
[multi-trip routing](user_guide/multi_trip.md) for the walkthrough and
[the problem model](user_guide/problem_model.md) for the exact rules.

## Seeing what a solver does: `verbose` and logging

Solvers never print. `verbose=1` logs progress every tenth of the run and `verbose=2`
every iteration, as INFO records on the `skroute` logger. Python shows only WARNING
and above by default, so enable them once with
[`set_log_level`][skroute.set_log_level] (or your own `logging.basicConfig`):

```python
>>> import skroute
>>> skroute.set_log_level("INFO")         # attaches a stderr handler to the "skroute" logger
>>> ils = IteratedLocalSearch(random_state=0, verbose=1).fit(C, labels=wi.labels)
>>> skroute.set_log_level("WARNING")      # back to quiet

```

The records look like this (they go to stderr, not to the solver's output):

```text
skroute INFO IteratedLocalSearch iteration 100/1000: best 27603, current 27603
skroute INFO IteratedLocalSearch: stopped (patience) after 118 iterations, best 27603
```

## What to try next

- [Choosing a solver](user_guide/choosing_a_solver.md) — which estimator for which
  size, objective and time budget, with the generated capability table.
- [The problem model](user_guide/problem_model.md) — the exact rules: giant tour,
  split rules, asymmetric matrices, infeasible nodes and the dense-matrix ceiling.
- [Multi-trip routing](user_guide/multi_trip.md) — the Barcelona walkthrough.
- [Warm starts and ensembles](user_guide/warm_starts_and_ensembles.md) — chaining
  solvers and running restarts in parallel.
- The [API reference](api/base.md) and the [benchmarks](benchmarks.md).
