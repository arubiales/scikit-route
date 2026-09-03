# Construction heuristics

`skroute.construction` builds a tour from scratch, deterministically, in O(n²) to O(n² log n):
no `random_state`, no iterations, no `history_`. These solvers are the natural warm starts of the
local searches and metaheuristics (`init=est.tour_`, or simply `init="nearest_neighbour"`) and
honest baselines when a tolerance has to be judged.

| Solver | Rule | Asymmetric matrices | Sees the multi-trip budget | Typical gap (wi29 / dj38) |
|---|---|---|---|---|
| [`NearestNeighbour`](#skroute.construction.NearestNeighbour) | greedy walk from the depot | yes | no (warns; result still split) | 31.8 % / 46.4 % |
| [`Insertion`](#skroute.construction.Insertion) | farthest / cheapest / nearest insertion at the cheapest position | yes (direction-aware) | no (warns; result still split) | 1.9 % / 0.0 % (farthest) |
| [`ClarkeWright`](#skroute.construction.ClarkeWright) | parallel savings, merges refused when the merged trip would not fit | **no** (`ValueError`) | **yes** | 4.9 % / 0.1 % |
| [`NRBS`](#skroute.construction.NRBS) | the 2020 Node Ranking Based on Stats heuristic | yes | no (warns; result still split) | 18.3 % / 7.8 % |

All four accept every input kind of the base class (ndarray, DataFrame, dict-of-dicts, or a ready
`RoutingProblem`) and return the fitted attributes of [`BaseRouter`][skroute.base.BaseRouter]:
`tour_`, `route_`, `trips_`, `cost_`, ... The cost is always recomputed from the tour by the base
class.

```python
>>> from skroute import ClarkeWright, Insertion, NearestNeighbour
>>> from skroute.datasets import load_tsp
>>> dj = load_tsp("dj38")
>>> C = dj.distance_matrix()
>>> gaps = {type(est).__name__: est.fit(C, labels=dj.labels).cost_ / dj.optimal_tour_length - 1
...         for est in (NearestNeighbour(), Insertion(), ClarkeWright())}
>>> gaps["Insertion"] <= gaps["ClarkeWright"] <= gaps["NearestNeighbour"]
True

```

Under a budget, `ClarkeWright` is the one construction heuristic whose *search* sees
`max_time_work`: the others build a plain tour, warn once, and hand it to the decoder, which
still splits it into feasible trips and prices it under the multi-trip objective.

::: skroute.construction.NearestNeighbour

::: skroute.construction.Insertion

::: skroute.construction.ClarkeWright

::: skroute.construction.NRBS
