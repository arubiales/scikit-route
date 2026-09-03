# Self-organising map

`skroute.metaheuristics.SOM` is the neural approach to the TSP: a one-dimensional
Kohonen ring of neurons is pulled towards the cities, one random city at a time,
until it settles into a closed curve that visits every city; reading the ring
in order gives the tour. It is the only solver that **needs coordinates**
(`fit(X, coords=...)`) and it never reads the cost matrix during its search — the
matrix prices the tour decoded after every epoch, which is what `history_`
records (best-so-far, SPEC R8).

Under a multi-trip budget the map ignores `max_time_work` (it warns, D6); the
returned tour is still split into trips and priced under the multi-trip
objective. Like every 2.0 solver it evaluates on a dense `(n, n)` cost matrix,
so the practical ceiling is about 20 000 nodes; coordinate-only fitting with
distances computed on the fly is a 2.1 item (D18).

```python
>>> from skroute.metaheuristics import SOM
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")
>>> som = SOM(random_state=0).fit(wi.distance_matrix(), coords=wi.coords, labels=wi.labels)
>>> som.cost_ / wi.optimal_tour_length < 1.15
True
>>> som.stop_reason_ in {"converged", "max_iter"} and som.n_iter_ == len(som.history_)
True

```

::: skroute.metaheuristics.SOM
    options:
      show_root_heading: true
