# scikit-route

<p align="center">
  <img src="images/logo.png" alt="scikit-route" width="420">
</p>

**scikit-route** is a Python library of route-optimisation solvers with the API you
already know from scikit-learn: build an estimator, call `fit`, read the result from
trailing-underscore attributes. It solves the classic travelling salesman problem
(symmetric or asymmetric) and a **multi-trip** variant in which a worker returns to
the depot whenever a trip would exceed a working-time budget, paying a fixed charge
per extra trip. Every solver evaluates costs and neighbourhood moves in a shared
**Cython** core, so the same code handles 30 nodes in microseconds and 10 000 nodes
in seconds.

```bash
pip install scikit-route
```

Wheels are published for CPython 3.11–3.14 on Linux, macOS and Windows; see
[Installation](installation.md) for the extras and source builds.

## Two problems, one model

Every solver works on the same object: a dense cost matrix $C$ whose entry $C_{ij}$ is
the cost of travelling from node $i$ to node $j$, one of the nodes being the *depot*.
A solution is a closed tour that leaves the depot, visits every other node once and
comes back.

- **Plain TSP.** The cost of a tour $\tau$ is the sum of its legs, the return leg
  included: $c(\tau) = \sum_k C_{\tau_k \tau_{k+1}}$. An asymmetric matrix
  ($C_{ij} \neq C_{ji}$) is read directionally and solved as an asymmetric TSP.
- **Multi-trip.** Add a time matrix $T$ and a per-trip budget `max_time_work`. The
  same tour is *split* into consecutive trips, each of which — return leg included —
  fits the budget, and every trip after the first costs `people * extra_cost`. The
  objective is the travel cost of the trips plus those fixed charges. Two split rules
  are available: the O(n) `"greedy"` default and the exact `"optimal"` partition
  (Prins, 2004), never worse for the same tour.

The [problem model](user_guide/problem_model.md) page works both through on a
four-node example; [multi-trip routing](user_guide/multi_trip.md) is the practical
walkthrough.

## The scikit-learn contract

- **Knobs in `__init__`, data in `fit`.** Hyper-parameters (`n_iter`, `random_state`,
  `init`, ...) configure the estimator; the cost matrix, the labels, the depot, the
  time matrix and the budget are arguments of `fit`, which returns `self`.
- **Results in trailing-underscore attributes.** `tour_` (the open giant tour, depot
  first), `route_` (as driven, depot first and last), `trips_`, `cost_`, `n_trips_`,
  `trip_costs_`, `trip_times_`; iterative solvers add `history_`, `n_iter_` and
  `stop_reason_`, exact solvers `is_optimal_`.
- **Any square input.** A numpy array, a pandas `DataFrame` (its index provides the
  labels) or a dict of dicts; results are always in the labels you passed.
- **`cost_` is recomputed, never reported.** The base class validates the tour a
  solver returns and prices it with the problem's own decoder, so `route_` and `cost_`
  agree by construction.
- **`get_params` / `set_params` / `clone`**, declarative parameter validation, a
  print-changed-only `repr` and a public
  [`check_router`][skroute.utils.estimator_checks.check_router] battery, exactly as
  in scikit-learn.

## One Cython core

Cost evaluation, the two split rules, the O(1) move deltas of 2-opt, Or-opt and swap,
the local-search descents and the constructions live in one compiled core that every
solver shares. Randomness is drawn in Python from `numpy.random.default_rng(random_state)`
and handed to the kernels, so a run is reproducible bit for bit on the same machine and
[`MultiStart`][skroute.MultiStart] can run restarts on threads (the kernels release the
GIL) without copying the matrix.

## Thirty seconds

```python
>>> # 1. Plain TSP from numpy: Western Sahara, optimum 27603 -- labels are the file's 1-based ids
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")                       # TSPBunch: coords, labels, depot, optimal_tour_length, ...
>>> C = wi.distance_matrix()                    # metric="tsplib_euc_2d": nint rounding, optima match the literature
>>> ils = IteratedLocalSearch(random_state=0).fit(C, labels=wi.labels)   # route_ comparable with the published tour
>>> ils.cost_ / wi.optimal_tour_length < 1.03   # the fast-tier tolerance of the test-suite
True
>>> int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == 1
True
>>> ils.n_iter_ == len(ils.history_) and ils.stop_reason_ in {"patience", "max_iter"}
True

```

```python
>>> # 2. Multi-trip from Barcelona: 8-hour days, 12.83 EUR per extra day, two people.
>>> # Loader matrices are PLAIN ndarrays: pass labels=bcn.labels, otherwise depot=bcn.depot is not a label of X.
>>> import numpy as np
>>> from skroute import SimulatedAnnealing, TabuSearch, RoutingProblem
>>> from skroute.datasets import load_barcelona
>>> bcn = load_barcelona()                      # cost (EUR), time (h), labels (int64 ids), depot == 10000007
>>> sa = SimulatedAnnealing(random_state=0).fit(bcn.cost, time_matrix=bcn.time, labels=bcn.labels,
...                                             depot=bcn.depot, max_time_work=8.0, extra_cost=12.83, people=2)
>>> bool(np.all(sa.trip_times_ <= 8.0))          # every trip fits, return leg included
True
>>> int(sa.route_[0]) == int(sa.route_[-1]) == 10000007 and sa.n_trips_ == len(sa.trips_)
True
>>> problem = RoutingProblem(bcn.cost, time_matrix=bcn.time, labels=bcn.labels, depot=bcn.depot,
...                          max_time_work=8.0, extra_cost=12.83, people=2, split="optimal")
>>> costs = {type(est).__name__: est.fit(problem).cost_ for est in (sa, TabuSearch(random_state=0))}
>>> sorted(costs)                                 # one instance, several solvers, optimal split
['SimulatedAnnealing', 'TabuSearch']

```

```python
>>> # 3. Legacy dict-of-dicts input; the depot is the first key, as 1.0's route_example[0]. Deterministic: exact output.
>>> from skroute import BruteForce
>>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
>>> hours = {1: {1: 0, 2: 1, 3: 2, 4: 2}, 2: {1: 1, 2: 0, 3: 1, 4: 2},
...          3: {1: 2, 2: 1, 3: 0, 4: 1}, 4: {1: 2, 2: 2, 3: 1, 4: 0}}
>>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> bf.route_.tolist(), bf.cost_, bf.n_trips_
([1, 2, 3, 1, 4, 1], 41.0, 2)

```

The three snippets are the same ones the README carries, and both copies run under
the documentation's doctest job on every commit.

## Which solver?

The table is generated from the solvers' own capability tags at build time, so it can
never drift from the code. *Multi-trip aware* means the search itself prices the
decoded trips; the other solvers still return feasible trips, but optimise the plain
tour and warn. *Max nodes* is the hard cap of the exact solvers.

{% include-markdown "user_guide/_capability_table.md" %}

[`IteratedLocalSearch`][skroute.IteratedLocalSearch] is the recommended default: it
reaches the published optimum on the small bundled instances and stays within a few
percent on the 1 000-node ones. Use [`MILP`][skroute.MILP] (or
[`BruteForce`][skroute.BruteForce] below 12 nodes) when you need a certificate of
optimality, and [`SimulatedAnnealing`][skroute.SimulatedAnnealing],
[`TabuSearch`][skroute.TabuSearch] or [`Genetic`][skroute.Genetic] when the multi-trip
budget matters and you want to trade time for quality.
[`MultiStart`][skroute.MultiStart] runs any stochastic solver from several seeds in
parallel and keeps the best result. [Choosing a solver](user_guide/choosing_a_solver.md)
goes through the trade-offs.

## Where next

- [Getting started](getting_started.md) — install, the mental model, a first tour and
  how to read the result.
- User guide: [the problem model](user_guide/problem_model.md),
  [choosing a solver](user_guide/choosing_a_solver.md),
  [multi-trip routing](user_guide/multi_trip.md) and
  [warm starts and ensembles](user_guide/warm_starts_and_ensembles.md).
- [API reference](api/base.md) — every class and function, rendered from the
  docstrings: [base and problem](api/base.md), [exact solvers](api/exact.md),
  [construction](api/construction.md), [local search](api/local_search.md),
  [simulated annealing](api/simulated_annealing.md), [tabu search](api/tabu_search.md),
  [genetic algorithm](api/genetic.md), [ant colony](api/ant_colony.md),
  [self-organising map](api/som.md), [ensembles](api/ensemble.md),
  [datasets](api/datasets.md), [preprocessing](api/preprocessing.md) and
  [utilities](api/utils.md).
- [Benchmarks](benchmarks.md) — the gap of every solver to the published optimum of
  the bundled Waterloo instances, with the tolerances the test-suite enforces.
- [Migrating from 1.0](migration.md) — 2.0 is a rewrite: `fit` takes the cost matrix
  and returns `self`, problem data moved from `__init__` to `fit`, and four hard
  dependencies are gone.
- [Changelog](changelog.md), [contributing](contributing.md) and [about](about.md).
