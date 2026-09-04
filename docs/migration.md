# Migrating from 1.0.0a2 to 2.0

scikit-route 2.0 is a rewrite. The eight solver classes of 1.0 keep their names, but
the way you hand them a problem, the way you read the result and the meaning of the
multi-trip objective all changed. This page maps every public name of 1.0.0a2 to its
2.0 counterpart and shows each change with a before/after snippet.

!!! warning "The matrix order is reversed and `time_matrix` is keyword-only"
    1.0's `fit(route_example, time_matrix, cost_matrix)` becomes
    `fit(cost_matrix, time_matrix=time_matrix, depot=route_example[0])`.
    Passing the time matrix positionally raises `TypeError`, on purpose: a migrated
    call that kept the old order would otherwise swap two square matrices silently
    and price hours as euros.

## 1. Installation and imports

```bash
pip install --upgrade scikit-route          # Python >= 3.11; wheels for Linux, macOS, Windows
pip install "scikit-route[pandas]"          # DataFrame inputs and as_frame=True loaders
```

The 2020 alpha reached PyPI under the distribution name `skroute` (1.0.0a1); 2.0 is published as `scikit-route`, the name the repository has always used, so `pip install --upgrade skroute` does not find it — install `scikit-route` instead. The import name is `skroute` in both.

TensorFlow, scikit-learn, googlemaps and tqdm are no longer dependencies. The old
import paths still work for the 2.x series and emit a `DeprecationWarning`:

| 1.0.0a2 | 2.0 | notes |
|---|---|---|
| `skroute.heuristics.brute.BruteForce` | `skroute.exact.BruteForce` | shim re-exports with `DeprecationWarning` |
| `skroute.heuristics.NRBS.NRBS` | `skroute.construction.NRBS` | the five exponents now default to `1.0` and must be >= 0 (1.0 validated only the type); `distance_weigth` → `distance_weight`; `fit(start, ids, cost)` → `fit(cost, depot=start)` |
| `skroute.metaheuristics.genetics.Genetic` | `skroute.metaheuristics.Genetic` | `p_c→p_crossover`, `p_m→p_mutation`, `pop→pop_size`, `gen→n_generations`, `k→tournament_size`, `early_stopping→patience` |
| `skroute.metaheuristics.genetics.EnsembleGenetic` | `skroute.ensemble.EnsembleGenetic` | `n_genetics` kept; or use `MultiStart(Genetic(...))` |
| `skroute.metaheuristics.simulated_annealing.SimulatedAnnealing` | `skroute.metaheuristics.SimulatedAnnealing` | `temp→t0`, `delta→alpha` (1.0 silently rescaled it into 0.9–1), `tol→t_min`, `neighbours→n_moves` |
| `skroute.metaheuristics.simulated_annealing.EnsembleSimulatedAnnealing` | `skroute.ensemble.EnsembleSimulatedAnnealing` | `n_simulateds` kept (default 20 → 10) |
| `skroute.metaheuristics.tabu_search.TabuSearch` | `skroute.metaheuristics.TabuSearch` | `searchs→n_iter`, `tabu_length/tabu_var→tenure`, `p_m` dropped |
| `skroute.metaheuristics.som.SOM` | `skroute.metaheuristics.SOM` | `units→n_units`, `lr→learning_rate`, `fit(nodes, epochs)` → `fit(X, coords=...)` with `n_iter` in `__init__` |

Every solver is also importable directly from `skroute`.

## 2. `fit` before and after

In 1.0 the problem data (`max_time_work`, `extra_cost`, `people`) lived in
`__init__` and `fit` took a "route example" whose first element was the depot. In 2.0
the algorithm knobs stay in `__init__` and **everything that describes the instance
goes to `fit`**:

```python
# 1.0.0a2
ga = Genetic(p_m=0.3, pop=400, gen=2000, k=5, early_stopping=100, max_time_work=6, extra_cost=12.83)
cost, route = ga.fit(route_example, time_matrix, cost_matrix)

# 2.0
from skroute import Genetic

ga = Genetic(
    p_mutation=0.3, pop_size=400, n_generations=2000, tournament_size=5, patience=100, random_state=0
)
ga.fit(cost_matrix, time_matrix=time_matrix, depot=route_example[0], max_time_work=6.0, extra_cost=12.83)
ga.cost_, ga.route_
```

The same shape applies to `SimulatedAnnealing`, `TabuSearch`, `BruteForce` and the
two `Ensemble*` wrappers. `NRBS.fit(start_point_id, ids_route, cost_matrix)` becomes
`NRBS().fit(cost_matrix, depot=start_point_id)`. `SOM.fit(nodes, epochs)` becomes
`SOM(n_iter=epochs).fit(cost_matrix, coords=xy)` where `xy` is the `(n, 2)` array of
coordinates (2.0 evaluates every solution on a dense cost matrix; see section 9).

## 3. The return value became attributes

`fit` returns the estimator. The result lives in trailing-underscore attributes:

| attribute | meaning |
|---|---|
| `tour_` | the open giant tour, depot first (labels) — the warm-start format |
| `route_` | the route as driven: depot, trip 1, depot, trip 2, …, depot |
| `trips_` | one closed array per trip |
| `cost_` | the objective, always recomputed from `route_` |
| `n_trips_`, `trip_costs_`, `trip_times_` | per-trip figures (`trip_times_` only with a time matrix) |
| `history_`, `n_iter_`, `stop_reason_` | iterative solvers only |
| `is_optimal_` | exact solvers only |

1.0's `history_` held raw costs per generation; 2.0's `history_` is the best cost so
far after each outer iteration, so it never increases.

## 4. `route_example` is gone

1.0 needed a list of node ids whose first element was the depot. 2.0 identifies nodes
by the rows of the matrix (or by the labels of a `DataFrame`/dict, or by `labels=`),
takes the depot with `depot=<label>` (default: the first node) and takes an optional
starting tour with `init=` in `__init__`: `"nearest_neighbour"` (default), `"random"`,
or the `tour_`/`route_` of another solver.

## 5. The objective changed in three ways

1. A trip may never exceed `max_time_work`, **including the return leg**. 1.0 checked
   the elapsed time before adding a leg and could overshoot the budget by one leg and
   never checked the final return.
2. `people` multiplies only `extra_cost` (as 1.0's docstring said; its code also
   multiplied travel cost, which never changed the argmin but changed reported costs).
3. Two decoders of the giant tour into trips are available: `split="greedy"`
   (default; the corrected successor of the 1.0 rule) and `split="optimal"` (the
   minimum-cost partition into feasible trips, Prins 2004). `optimal` is never worse
   than `greedy` for the same tour.

Because of 1 and 2, costs reported by 2.0 differ from 1.0 on the same route; the
[problem model](user_guide/problem_model.md#from-10-to-20-the-barcelona-example) page works
the Barcelona example through.

## 6. Hyper-parameters renamed and defaults that changed

Renames are in the table of section 1. Results obtained with **default parameters
differ from 1.0** because the defaults changed:

| class | 1.0 default | 2.0 default |
|---|---|---|
| `BruteForce`, `TabuSearch` | `max_time_work=8`, `extra_cost`, `people` in `__init__` | plain TSP unless `max_time_work=` is passed to `fit` |
| `Genetic` | `p_c=0.6, p_m=0.4, pop=400, gen=1600, k=3` | `p_crossover=0.9, p_mutation=0.2, pop_size=100, n_generations=500, tournament_size=3, patience=100` |
| `EnsembleGenetic` | `pop=400, gen=1000` | `pop_size=100, n_generations=500` |
| `SimulatedAnnealing` | `temp=12.0, neighbours=250, delta=0.78, tol=1.29` | `t0="auto", n_moves=10·n, alpha=0.995, t_min="auto", patience=None` |
| `EnsembleSimulatedAnnealing` | `n_simulateds=20` | `n_simulateds=10` |
| `TabuSearch` | `searchs=1250, p_m=0.6, tabu_length=45, tabu_var=10` | `n_iter=1000, tenure="auto"` (`p_m` dropped) |
| `SOM` | `radius_decay=0.9991, lr_decay=0.9991, fit(..., epochs=10_000)` | `radius_decay=0.9997, lr_decay=0.99997, n_iter=100_000` |
| `NRBS` | no defaults (`distance_weigth`), negative exponents accepted | all five `1.0` (`distance_weight`), exponents >= 0 |

## 7. Every public name of 1.0.0a2

| 1.0 name | status | 2.0 replacement | reason |
|---|---|---|---|
| the 8 classes of section 1 | kept (moved) | see section 1 | — |
| `EnsembleGenetic`, `EnsembleSimulatedAnnealing` | kept (moved) | `skroute.ensemble.*`; or `MultiStart(Genetic(...))` | explicit-parameter wrappers over `MultiStart` |
| `preprocessing.dfcolumn_to_dict` | replaced | `pairs_to_matrix` | the result depended on row order |
| `preprocessing.matrix_to_dict` | replaced | `to_dict_of_dicts` | referenced an undefined variable |
| `preprocessing.normalize(df, lat, lon)` | replaced | `normalize_coords(coords)` | numpy in, numpy out |
| `preprocessing.df_to_tuple` | removed | `Bunch.coords`/`labels`, or `df[[...]].to_numpy()` | one-line pandas idiom |
| `preprocessing.matrix_parse` | removed | `pairs_to_matrix` + `distance_matrix` | superseded |
| `preprocessing.DataLossWarning` | removed | — | nothing emits it any more |
| `preprocessing.CostScraper` | deprecated wrapper | `GoogleDistanceMatrix(api_key).fetch(coords, labels)` | removed in 3.0 |
| `cluster.KMeansTruncate` | removed | `sklearn.cluster.KMeans(max_iter=1)` | scikit-learn is no longer a dependency |
| `cluster.{KMeans, DBSCAN, AffinityPropagation, GaussianMixture}` | removed | import from scikit-learn | same |
| 27 `load_<country>` + 5 cost loaders | kept | same names; return a `Bunch` | `"DataFrame"` → `frame` (with `as_frame=True`), `"feature_names"` dropped, `DESCR` kept |
| `load_costs_qatar` | deprecated alias | `load_qatar_costs` | 1.0 loaded Valencia |
| `mode=` in TSP loaders | deprecated | `n_nodes=` | 1.0 returned 0 rows for wi29 `"small"` |
| `fit(route_example, time_matrix, cost_matrix)` | changed | `fit(cost, time_matrix=..., depot=..., init=...)` | see the warning at the top |

## 8. Datasets

Loaders return a `Bunch` (a dict with attribute access) instead of a dict with a
`"DataFrame"` key. The cost datasets expose square matrices ready for `fit`:

```python
from skroute.datasets import load_barcelona

bcn = load_barcelona()  # cost (EUR), time (h), distance (m), coords, labels, depot, units, DESCR
bcn.cost.shape, bcn.depot  # (19, 19), 10000007
```

Loader matrices are plain arrays: pass `labels=bcn.labels` to `fit` so that
`depot=bcn.depot` and the returned `route_` use the dataset's ids, or call
`load_barcelona(as_frame=True)` to get labelled DataFrames (requires pandas).
The 27 national instances are loaded with `load_tsp("wi29")` (or the old
`load_sahara()` names) and carry `coords`, `labels`, `optimal_tour_length` and a
cached `distance_matrix()` method; `mode="small"|"medium"|"big"` still works with a
`DeprecationWarning` — use `n_nodes=` instead.

## 9. SOM needs coordinates and a matrix

1.0's SOM took a tuple of `(id, lat, lon)` tuples and TensorFlow. 2.0's SOM is numpy
only and, like every other solver, evaluates its tours on the dense cost matrix:
`SOM(random_state=0).fit(C, coords=xy)`. That matrix is the practical ceiling of 2.0
(about 20 000 nodes, 3.2 GB); the four bundled instances above it can be read and
subsampled with `load_tsp(name, n_nodes=5000)` but not solved whole. Coordinate-only
fitting is planned for 2.1.

## 10. `Ensemble*` → `MultiStart`

`EnsembleGenetic` and `EnsembleSimulatedAnnealing` remain as thin wrappers, but the
general tool is `MultiStart(estimator, n_restarts=10, n_jobs=None)`, which runs any
stochastic solver from independent seeds in parallel (threads by default, since the
kernels release the GIL) and keeps the best result. Results do not depend on `n_jobs`.

## 11. Extras and dropped dependencies

| 1.0 | 2.0 |
|---|---|
| tensorflow (SOM) | gone — numpy implementation |
| scikit-learn (`cluster` re-exports) | gone — import scikit-learn yourself |
| googlemaps (hard) | optional: `pip install "scikit-route[google]"` |
| pandas (hard) | optional: `pip install "scikit-route[pandas]"` |
| tqdm | gone — progress goes to the `skroute` logger (`skroute.set_log_level("INFO")`) |

## 12. `CostScraper` → `GoogleDistanceMatrix`

`GoogleDistanceMatrix(api_key, mode="driving").fetch(coords, labels)` batches the
requests (1.0 issued one request per pair — 18 336 for Qatar) and returns a `Bunch`
with `distance` (m), `time` (h), `labels` and `units`. `CostScraper` still accepts the
1.0 constructor and `scrap()`/`pandas()` but warns; `to_pickle()` is gone. Both live
behind the `google` extra.

## A worked, deterministic example

The dict-of-dicts input of 1.0 is still accepted; the depot is the first key, as
`route_example[0]` used to be:

```python
>>> from skroute import BruteForce
>>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
>>> hours = {1: {1: 0, 2: 1, 3: 2, 4: 2}, 2: {1: 1, 2: 0, 3: 1, 4: 2},
...          3: {1: 2, 2: 1, 3: 0, 4: 1}, 4: {1: 2, 2: 2, 3: 1, 4: 0}}
>>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> bf.route_.tolist(), bf.cost_, bf.n_trips_
([1, 2, 3, 1, 4, 1], 41.0, 2)

```

The single trip 1-2-3-4-1 would need 5 hours, so the tour breaks after node 3:
trips `[1, 2, 3, 1]` (cost 18) and `[1, 4, 1]` (cost 20) plus one extra-trip charge
of 3.0 give 41.0.
