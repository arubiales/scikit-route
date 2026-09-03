# Local search

`skroute.local_search` holds the improvement heuristics of the library: solvers that start from
a tour (the nearest-neighbour construction by default, or the `tour_`/`route_` of another solver
through `init=`) and apply moves while they shorten it.

- [`TwoOpt`](#skroute.local_search.TwoOpt) reverses tour segments (2-opt).
- [`OrOpt`](#skroute.local_search.OrOpt) relocates segments of one to three nodes (Or-opt).
- [`LocalSearch`](#skroute.local_search.LocalSearch) alternates both descents until neither
  improves.
- [`IteratedLocalSearch`](#skroute.local_search.IteratedLocalSearch) kicks a local optimum with a
  double bridge, descends again and keeps the best tour: the recommended default solver.

All four are *budget-aware*: under `max_time_work=` the search prices every move with the
multi-trip objective, so the trips it returns are the ones it optimised. Asymmetric matrices are
accepted everywhere; they and the multi-trip objective take the full-evaluation path (O(n) per
candidate move: one descent is fine up to a few thousand nodes, but `IteratedLocalSearch`
re-descends after every kick, so on those problems keep to a few hundred nodes at the default
`n_iter`, or set `time_limit`), while a symmetric plain TSP uses O(1) move deltas, candidate lists
and don't-look bits and scales to tens of thousands of nodes.

The three descents are deterministic and expose `history_` (the cost after each outer
iteration), `n_iter_` and `stop_reason_` (`"converged"` or `"max_iter"`). One outer iteration is
one pass of each listed move; the buffers persist across passes, so `max_passes` bounds the work
without changing the local optimum reached when the descent converges earlier. Because don't-look
bits skip nodes scanned earlier, a descent is declared converged only when a pass that started
with every node active changes nothing — the same confirming sweep closes every iteration of the
iterated search.

```python
>>> from skroute import IteratedLocalSearch, LocalSearch
>>> from skroute.datasets import load_tsp
>>> dj = load_tsp("dj38")  # Djibouti, optimum 6656
>>> C = dj.distance_matrix()
>>> ls = LocalSearch().fit(C, labels=dj.labels)
>>> ls.cost_ / dj.optimal_tour_length < 1.12
True
>>> ils = IteratedLocalSearch(random_state=0).fit(C, labels=dj.labels)
>>> ils.cost_ <= ls.cost_ and ils.stop_reason_ in {"patience", "max_iter"}
True
>>> polished = LocalSearch(init=ils.tour_).fit(C, labels=dj.labels)  # warm start: already a local optimum
>>> polished.n_iter_, polished.stop_reason_
(1, 'converged')

```

::: skroute.local_search.TwoOpt

::: skroute.local_search.OrOpt

::: skroute.local_search.LocalSearch

::: skroute.local_search.IteratedLocalSearch
