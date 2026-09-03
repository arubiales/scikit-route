# Ant colony

`AntColony` is a MAX–MIN Ant System (Stützle & Hoos, 2000). Every iteration a colony of
`n_ants` ants builds tours from the depot: at each step an ant picks the next node by a
roulette wheel over the unvisited nodes of its `n_candidates` nearest-neighbour list
(all unvisited nodes once the list is exhausted), weighting node `j` by
`tau[i, j] ** alpha * (1 / C[i, j]) ** beta`. Each tour is polished by the descents listed
in `local_search` (2-opt by default) and priced with the problem objective, so a
working-time budget steers the colony. The pheromone then evaporates (`rho`) and the
iteration-best ant — the best-so-far ant every fifth iteration — deposits `1 / cost` on its
arcs; the trail is clipped to `[tau_max / (2 n), tau_max]` with `tau_max = 1 / (rho * L_best)`,
which keeps the colony from stagnating.

The trail is stored as an `(n, n)` matrix (`pheromone_` after the fit) and the transition
weights are recomputed once per iteration as a second `(n, n)` matrix, so the practical
ceiling is a few thousand nodes. On an asymmetric matrix the trail is directional.

```python
>>> from skroute import AntColony
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")
>>> aco = AntColony(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
>>> aco.cost_ / wi.optimal_tour_length < 1.08
True
>>> int(aco.route_[0]) == int(aco.route_[-1]) == 1
True

```

::: skroute.metaheuristics.AntColony
