# Warm starts, ensembles and reproducibility

The solvers of scikit-route compose. A construction heuristic can seed a local search,
whose local optimum can seed a metaheuristic; one [`RoutingProblem`][skroute.RoutingProblem]
can be handed to several solvers; a stochastic solver can be restarted from many seeds in
parallel; and every result can be reproduced from `random_state`. This page shows how,
and closes with what you need to write a solver of your own that passes the library's
own test battery.

The examples use Western Sahara (`wi29`, optimum 27603) and Djibouti (`dj38`, optimum
6656) from [`skroute.datasets`][skroute.datasets.load_tsp]:

```python
>>> import numpy as np
>>> from skroute.datasets import load_tsp
>>> wi, dj = load_tsp("wi29"), load_tsp("dj38")
>>> C_wi, C_dj = wi.distance_matrix(), dj.distance_matrix()
>>> wi.optimal_tour_length, dj.optimal_tour_length
(27603, 6656)

```

## Warm starts with `init=`

Every solver that starts from a tour has an `init` parameter (the deterministic descents,
`IteratedLocalSearch`, `SimulatedAnnealing`, `TabuSearch`, `Genetic`, the `Ensemble*`
wrappers). It accepts three things:

| `init=` | Meaning | Available on |
|---|---|---|
| `"nearest_neighbour"` (default) | the greedy walk from the depot, built in the compiled core | every solver with `init` |
| `"random"` | the depot followed by a random permutation drawn from `random_state` | stochastic solvers only; `TwoOpt`, `OrOpt` and `LocalSearch` refuse it at validation |
| an array of **labels** | the `tour_` (open) or `route_` (closed, or multi-trip with the depot repeated) of another solver, or any sequence you built | every solver with `init` |

The array form is what makes chaining possible: the output of one solver is a valid
input of the next. Labels are matched against the labels of `X`, so a tour produced
with `labels=wi.labels` must be fed to a fit that also receives `labels=wi.labels`.

```python
>>> from skroute import NearestNeighbour, LocalSearch, IteratedLocalSearch, SimulatedAnnealing
>>> nn = NearestNeighbour().fit(C_wi, labels=wi.labels)
>>> ls = LocalSearch(init=nn.tour_).fit(C_wi, labels=wi.labels)             # descend from the greedy tour
>>> ils = IteratedLocalSearch(init=ls.tour_, random_state=0).fit(C_wi, labels=wi.labels)
>>> sa = SimulatedAnnealing(init=ils.tour_, random_state=0).fit(C_wi, labels=wi.labels)
>>> nn.cost_ >= ls.cost_ >= ils.cost_ >= sa.cost_                            # a descent never loses ground
True
>>> ils.cost_ <= 1.03 * wi.optimal_tour_length
True
>>> float(sa.history_[0]) <= ils.cost_   # the annealer's best-so-far starts at the tour it was given
True

```

Two facts worth knowing about the chain above:

- `LocalSearch(init=nn.tour_)` is exactly `LocalSearch()`: the default `init` *is* the
  nearest-neighbour tour. Chaining pays off when the seed is better than that — a
  `ClarkeWright` tour for a multi-trip instance, or the `tour_` of a long run you want to
  polish with different moves.
- A tour that is already a local optimum for the moves of a descent converges in one
  iteration, because the descent's first full sweep finds nothing to change:

```python
>>> polished = LocalSearch(init=ils.tour_).fit(C_wi, labels=wi.labels)
>>> polished.n_iter_, polished.stop_reason_, polished.cost_ == ils.cost_
(1, 'converged', True)

```

A `route_` is accepted as well as a `tour_`, depot repeats included, which lets you hand
the trips of one multi-trip solver to another. The Alicante–Murcia table (8 places, costs
in EUR, times in hours) with a budget of 1.5 times the longest round trip:

```python
>>> from skroute import TabuSearch
>>> from skroute.datasets import load_alicante_murcia
>>> d = load_alicante_murcia()
>>> budget = 1.5 * float((d.time[0] + d.time[:, 0]).max())
>>> kw = dict(time_matrix=d.time, labels=d.labels, depot=d.depot, max_time_work=budget, extra_cost=10.0, people=2)
>>> sa = SimulatedAnnealing(random_state=0).fit(d.cost, **kw)
>>> ts = TabuSearch(init=sa.route_, random_state=0).fit(d.cost, **kw)      # route_: depot, trip, depot, trip, depot
>>> sa.n_trips_ >= 2 and ts.cost_ <= sa.cost_ and bool(np.all(ts.trip_times_ <= budget))
True

```

!!! warning "What `init` refuses"
    The array must contain every label of `X` exactly once (the depot may repeat).
    Anything else — a tour of another instance, positions instead of labels, a missing
    node — raises `ValueError: init tour must contain every label exactly once (the depot
    may repeat)`, or `... is not a label of X` when a value is not a label at all.

```python
>>> IteratedLocalSearch(init=wi.labels[:-1], random_state=0).fit(C_wi, labels=wi.labels)
Traceback (most recent call last):
    ...
ValueError: init tour must contain every label exactly once (the depot may repeat)
>>> from skroute import TwoOpt
>>> TwoOpt(init="random").fit(C_wi, labels=wi.labels)                   # deterministic: no random draws
Traceback (most recent call last):
    ...
ValueError: The 'init' parameter of TwoOpt must be a str among {'nearest_neighbour'} or an array-like. Got 'random' instead.

```

## Reusing a `RoutingProblem`

`fit` builds a [`RoutingProblem`][skroute.RoutingProblem] from its arguments — the
coerced matrices, the labels, the depot index, the budget — and stores it as `problem_`.
You can build it once yourself and pass it to any number of solvers: it is immutable,
shareable across threads, and it caches the nearest-neighbour candidate lists that the
local searches, the tabu search and the ant colony all need, so they are computed once.
When `X` is a `RoutingProblem` it must travel alone: any other `fit` argument raises.

```python
>>> from skroute import RoutingProblem, Genetic, AntColony
>>> problem = RoutingProblem(C_dj, labels=dj.labels)
>>> problem
RoutingProblem(n=38, TSP, symmetric, depot=1)
>>> fitted = {type(est).__name__: est.fit(problem)
...           for est in (LocalSearch(), IteratedLocalSearch(random_state=0), SimulatedAnnealing(random_state=0),
...                       TabuSearch(random_state=0), Genetic(random_state=0), AntColony(random_state=0))}
>>> all(est.problem_ is problem for est in fitted.values())
True
>>> LocalSearch().fit(problem, labels=dj.labels)
Traceback (most recent call last):
    ...
ValueError: X is a RoutingProblem: pass it alone, without other fit arguments

```

The same object prices any tour under the problem's objective — `evaluate` works in index
space, [`route_cost`][skroute.metrics.route_cost] in label space — which is how the base
class recomputes `cost_` from the tour a solver returns:

```python
>>> from skroute.metrics import route_cost
>>> ils = fitted["IteratedLocalSearch"]
>>> float(problem.evaluate(problem.to_index_tour(ils.tour_))) == ils.cost_ == route_cost(C_dj, ils.route_, labels=dj.labels)
True

```

## Reading `history_` to compare solvers

Every iterative solver records `history_`, the **best-so-far** cost after each outer
iteration (an iteration is a kick for `IteratedLocalSearch`, a temperature level for
`SimulatedAnnealing`, a generation for `Genetic`, an epoch for `SOM`, one pass of each
move for the descents). By contract it is non-increasing, `n_iter_ == len(history_)`,
`history_[-1] == cost_` and `stop_reason_` says why the run ended:

| Solver | `stop_reason_` it can emit |
|---|---|
| `TwoOpt`, `OrOpt`, `LocalSearch`, `SOM` | `"converged"`, `"max_iter"` |
| `IteratedLocalSearch`, `TabuSearch`, `Genetic`, `AntColony`, `EnsembleGenetic` | `"max_iter"`, `"patience"`, `"time_limit"` |
| `SimulatedAnnealing`, `EnsembleSimulatedAnnealing` | `"converged"`, `"patience"`, `"time_limit"` |
| `MultiStart` | whatever its best restart emitted |

Because the histories of different solvers have different lengths and different
meanings per step, compare them on the facts they share — the final gap and the shape of
the curve — not on iteration counts:

```python
>>> tolerance = {"LocalSearch": 0.12, "IteratedLocalSearch": 0.03, "SimulatedAnnealing": 0.03,
...              "TabuSearch": 0.08, "Genetic": 0.15, "AntColony": 0.08}         # tests/tolerances.py, fast tier
>>> print(f"{'solver':22s} {'within tol':>10s} {'monotone':>9s} {'last==cost_':>12s}")
solver                 within tol  monotone  last==cost_
>>> for name, est in fitted.items():
...     within = est.cost_ <= (1 + tolerance[name]) * dj.optimal_tour_length
...     monotone = bool(np.all(np.diff(est.history_) <= 0))
...     print(f"{name:22s} {within!s:>10s} {monotone!s:>9s} {(est.history_[-1] == est.cost_)!s:>12s}")
LocalSearch                  True      True         True
IteratedLocalSearch          True      True         True
SimulatedAnnealing           True      True         True
TabuSearch                   True      True         True
Genetic                      True      True         True
AntColony                    True      True         True
>>> all(est.n_iter_ == len(est.history_) for est in fitted.values())
True

```

`history_[0]` is a useful diagnostic too: for `IteratedLocalSearch` it is the best cost
after the first kick (the initial descent is not counted), for `SimulatedAnnealing` the
best cost after the first temperature level (the initial tour's, unless a downhill move
was already accepted), for `Genetic` the best individual seen up to the first generation,
initial population included.

## `MultiStart`

[`MultiStart`][skroute.MultiStart] runs a stochastic solver `n_restarts` times from
independent seeds and keeps the best tour. Each restart is a [`clone`][skroute.clone] of
`estimator` whose `random_state` is a `numpy.random.Generator` seeded from a child of one
`numpy.random.SeedSequence`; the restarts are fitted on the **shared**
`RoutingProblem` through `joblib.Parallel`, and the winner is the restart with the lowest
`cost_` (the lowest index on a tie).

| Parameter | Default | Role |
|---|---|---|
| `estimator` | — | an unfitted stochastic solver; a deterministic one is refused (`ValueError`) |
| `n_restarts` | 10 | number of independent restarts |
| `n_jobs` | `None` | joblib workers: `None` runs them one after another, `-1` uses every CPU, a positive int that many. **Never changes the result** |
| `prefer` | `"threads"` | joblib backend hint; `"processes"` or `None` |
| `random_state`, `verbose` | | as everywhere |

After the fit: `estimators_` (the fitted clones in restart order), `costs_` (their
`cost_` values), `best_index_`, `best_estimator_` (whose `tour_` is the tour returned),
and — when the estimator is iterative — `history_`, `n_iter_` and `stop_reason_` copied
from the winner.

```python
>>> from skroute import MultiStart
>>> ms = MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=0).fit(C_wi, labels=wi.labels)
>>> ms.cost_ <= 1.03 * wi.optimal_tour_length, len(ms.estimators_), ms.costs_.shape
(True, 4, (4,))
>>> ms.cost_ == float(ms.costs_.min()) and ms.best_estimator_ is ms.estimators_[ms.best_index_]
True
>>> ms.stop_reason_ == ms.best_estimator_.stop_reason_ == "converged"
True

```

**Why threads.** The solver kernels release the GIL, so threads run the restarts in
parallel without pickling anything — a 10 000-node cost matrix is 800 MB, and processes
would copy it once per worker. Threads give a near-linear speed-up for
`SimulatedAnnealing`, `IteratedLocalSearch` and `TabuSearch`, whose time is spent in
compiled code, and little for the Python-heavy `Genetic` and `SOM`; for those,
`prefer="processes"` is one keyword away and gives identical results.

**The result never depends on `n_jobs`.** The seeds are assigned by restart index before
anything runs, so the same `random_state` gives the same winner and the same tour with
one worker, four, or a different backend:

```python
>>> serial = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=1, random_state=0).fit(C_wi, labels=wi.labels)
>>> threaded = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=2, random_state=0).fit(C_wi, labels=wi.labels)
>>> bool(np.array_equal(serial.tour_, threaded.tour_)) and serial.costs_.tolist() == threaded.costs_.tolist()
True

```

The wrapped solver's knobs are reachable through the `estimator__` prefix, as in
scikit-learn pipelines, and the tags of a `MultiStart` are its estimator's: wrapping a
budget-aware solver keeps the multi-trip objective inside the search, wrapping `SOM`
still needs `coords=`, wrapping a deterministic solver is refused.

```python
>>> ms.set_params(estimator__alpha=0.99).estimator.alpha
0.99
>>> ms
MultiStart(estimator=SimulatedAnnealing(alpha=0.99), n_restarts=4, random_state=0)
>>> sorted(k for k in ms.get_params(deep=True) if k.startswith("estimator__"))[:3]
['estimator__alpha', 'estimator__init', 'estimator__moves']
>>> MultiStart(LocalSearch(), n_restarts=2).fit(C_wi)
Traceback (most recent call last):
    ...
ValueError: MultiStart needs a stochastic estimator (one with random_state)

```

### The legacy `Ensemble*` wrappers

[`EnsembleGenetic`][skroute.EnsembleGenetic] and
[`EnsembleSimulatedAnnealing`][skroute.EnsembleSimulatedAnnealing] are the ensembles of
scikit-route 1.0, kept as explicit-parameter wrappers: `n_genetics` / `n_simulateds`
restarts, the inner knobs keyword-only with the 2.0 defaults (1.0 defaulted
`n_simulateds=20`; it is now 10). They return exactly what the equivalent `MultiStart`
returns, expose the same `estimators_`/`costs_`/`best_estimator_`, and will be removed
in 3.0 — new code should use `MultiStart` directly.

```python
>>> from skroute import EnsembleSimulatedAnnealing
>>> legacy = EnsembleSimulatedAnnealing(n_simulateds=4, random_state=0).fit(C_wi, labels=wi.labels)
>>> reference = MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=0).fit(C_wi, labels=wi.labels)
>>> bool(np.array_equal(legacy.tour_, reference.tour_)) and legacy.costs_.tolist() == reference.costs_.tolist()
True

```

## `random_state` and reproducibility

Every stochastic solver draws all of its randomness from `numpy.random.default_rng(random_state)`
in Python and hands the numbers to the compiled kernels as arrays. Consequences:

- **An `int` gives bit-exact reproducibility on the same machine**: same seed, same
  input, same `tour_`, same `history_`, same `n_iter_`.
- **A `numpy.random.Generator` is used in place and advanced** by the fit, so two
  successive fits with the same generator are two different runs — the idiom for "give me
  another sample" — while `default_rng(0)` reproduces `random_state=0` exactly.
- `None` means a fresh, unseeded generator. The legacy `numpy.random.RandomState` is
  not accepted.
- **Across machines** the tour is usually the same but is not guaranteed to be: `exp` in
  the Metropolis test differs by an ulp between C libraries, and one flipped acceptance
  changes the rest of the trajectory. That is why the examples in this documentation
  print gaps below a tolerance rather than raw costs, and why the test-suite pins
  "same seed, same machine, bit-identical" rather than a float.
- **`time_limit` breaks reproducibility**: the iteration at which the clock runs out
  depends on the machine's speed.

```python
>>> a = IteratedLocalSearch(random_state=0).fit(C_wi, labels=wi.labels)
>>> b = IteratedLocalSearch(random_state=0).fit(C_wi, labels=wi.labels)
>>> bool(np.array_equal(a.tour_, b.tour_)) and bool(np.array_equal(a.history_, b.history_))
True
>>> rng = np.random.default_rng(0)
>>> state_before = rng.bit_generator.state
>>> g1 = IteratedLocalSearch(random_state=rng).fit(C_wi, labels=wi.labels)   # same draws as random_state=0
>>> bool(np.array_equal(g1.history_, a.history_)), rng.bit_generator.state != state_before
(True, True)
>>> g2 = IteratedLocalSearch(random_state=rng).fit(C_wi, labels=wi.labels)   # the generator moved on
>>> g2.n_iter_ != g1.n_iter_ or not np.array_equal(g2.history_, g1.history_)
True
>>> hurried = IteratedLocalSearch(time_limit=1e-6, random_state=0).fit(C_wi, labels=wi.labels)
>>> hurried.stop_reason_, hurried.n_iter_
('time_limit', 1)

```

`MultiStart` follows the same rules: one integer is drawn from its `random_state` and
spawns the child seeds, so the whole ensemble is reproducible from one number, whatever
`n_jobs`.

## Watching a run

Nothing in scikit-route prints. `verbose=1` logs every tenth outer iteration and the stop
to the `skroute` logger at `INFO`, `verbose=2` logs every iteration. Python's default
handler shows only `WARNING` and above, so enable the records once with
[`skroute.set_log_level`][skroute.set_log_level] (or `logging.basicConfig(level=logging.INFO)`):

```python
import skroute

skroute.set_log_level("INFO")  # attaches a stderr handler if none is configured
IteratedLocalSearch(verbose=1, random_state=0).fit(C_wi, labels=wi.labels)
```

In code that already configures logging, attach your own handler instead:

```python
>>> import logging
>>> messages = []
>>> class Collect(logging.Handler):
...     def emit(self, record):
...         messages.append(record.getMessage())
>>> log, handler, saved_level = logging.getLogger("skroute"), Collect(), logging.getLogger("skroute").level
>>> log.addHandler(handler); log.setLevel(logging.INFO)
>>> _ = IteratedLocalSearch(verbose=1, random_state=0).fit(C_wi, labels=wi.labels)
>>> log.removeHandler(handler); log.setLevel(saved_level)
>>> len(messages) >= 1 and messages[-1].startswith("IteratedLocalSearch: stopped")
True

```

## Writing your own solver: `check_router`

A solver is a subclass of [`BaseRouter`][skroute.BaseRouter] that implements one method.
The base class does everything else: it coerces the input into a `RoutingProblem`,
validates the hyper-parameters, honours the tags, hands `_solve` a seeded generator,
validates the tour that comes back, decodes it into trips and **recomputes `cost_`** — a
solver never reports a cost, so a route that does not match its cost is impossible.

The contract, in four points:

1. `__init__` stores every argument verbatim under the same name and does nothing else
   (that is what `get_params`, `set_params`, `clone` and `repr` rely on).
2. `_parameter_constraints` declares the legal values; validation runs at `fit` time
   with scikit-learn's message format. Constraints can be a Python type (`int`), the
   strings `"array-like"`, `"random_state"`, `"boolean"`, `"verbose"`, `None`, a
   callable, or the `Interval`/`Options` objects of `skroute.utils._param_validation`.
3. `_get_tags()` returns a [`RouterTags`][skroute.RouterTags] describing what the solver
   supports: `kind`, `exact`, `stochastic` (then it must have a `random_state`
   parameter), `iterative` (then `_solve` must set `history_`, `n_iter_` and
   `stop_reason_`, report each outer iteration with `self._emit(...)` and stop with
   `stop_reason_ = "callback"` once `self._stop_requested` is set — the progress-callback
   protocol of `fit(..., callback=)`), `budget_aware`, `requires_symmetric`,
   `requires_coords`, `max_nodes`.
4. `_solve(problem, rng)` returns an `int64` permutation of `range(problem.n)` with
   `problem.depot` at position 0 — **index space**, never labels. `rng` is a
   `numpy.random.Generator` for stochastic solvers and `None` otherwise;
   `problem.evaluate(tour)` prices a tour under the full objective, budget included.

[`check_router`][skroute.check_router] is the structural battery the library's own
solvers pass in `tests/test_common.py`: parameter protocol, unfitted state, fitted
attributes, recomputed cost, input kinds (ndarray, DataFrame, dict-of-dicts, string
labels), invalid inputs, tags, multi-trip, no printing, the iterative contract,
reproducibility, the smallest sizes and the progress callbacks. It takes an **unfitted** instance, builds its own
small instances, and raises `AssertionError("check N: ...")` at the first failure. Here is
a complete solver that passes it — random sampling with the best-so-far bookkeeping of
an iterative solver:

```python
>>> from skroute import BaseRouter, RouterTags, check_router
>>> class RandomSampling(BaseRouter):
...     """Draw ``n_samples`` random giant tours and keep the cheapest."""
...
...     _parameter_constraints = {"n_samples": [int], "random_state": ["random_state"]}
...
...     def __init__(self, n_samples=100, random_state=None):
...         self.n_samples = n_samples              # stored verbatim, nothing else
...         self.random_state = random_state
...
...     def _get_tags(self):
...         return RouterTags(kind="metaheuristic", stochastic=True, iterative=True, budget_aware=True)
...
...     def _solve(self, problem, rng):
...         others = np.delete(np.arange(problem.n), problem.depot)
...         best, best_cost, history, reason = None, np.inf, [], "max_iter"
...         for k in range(self.n_samples):
...             tour = np.concatenate(([problem.depot], rng.permutation(others)))
...             cost = problem.evaluate(tour)      # the problem's own objective, budget included
...             if cost < best_cost:
...                 best, best_cost = tour, cost
...             history.append(best_cost)
...             self._emit("iteration", k + 1, tour, cost, best, best_cost)  # progress callback
...             if self._stop_requested:            # the callback returned True: stop here
...                 reason = "callback"
...                 break
...         self.history_, self.n_iter_, self.stop_reason_ = history, len(history), reason
...         return best
>>> check_router(RandomSampling())              # silent when every check passes
>>> [name for name, fn in check_router.checks][:4]
['1_init_and_params', '2_not_fitted', '3_fit_results', '4_cost_recomputed']
>>> rs = RandomSampling(random_state=0).fit(C_wi, labels=wi.labels)
>>> rs.n_iter_, rs.stop_reason_, int(rs.route_[0]) == int(rs.route_[-1]) == int(rs.depot_) == 1
(100, 'max_iter', True)
>>> rs.cost_ == float(rs.history_[-1]) and rs.cost_ > wi.optimal_tour_length   # a sampler, not a solver
True
>>> RandomSampling(n_samples=5)
RandomSampling(n_samples=5)

```

`budget_aware=True` is honest here because the sampler prices every tour with
`problem.evaluate`, which sees the budget; a solver that optimises the plain tour and
lets the decoder split it afterwards must leave the tag at its default `False`, so that
`fit` warns under a budget. Leave `exact` at `False` unless the tour is provably optimal
— then set `is_optimal_` in `_solve`.

Tolerance checks (how close to the optimum a solver must get) are not part of
`check_router`: they live in the test-suite, driven by `tests/tolerances.py`, and a new
solver added to the library gets an entry there. Read
[Contributing](../contributing.md) for the full procedure.
