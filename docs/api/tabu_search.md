# Tabu search

`TabuSearch` applies, at every iteration, the best admissible move of a candidate
neighbourhood — 2-opt reversals and Or-opt relocations that join a node to one of its `k`
nearest neighbours — even when that move worsens the tour, and forbids re-adding the edges the
move removed for exactly `tenure` iterations (a tabu move is still allowed when it beats the best
tour so far). With `tenure="auto"` the tenure is redrawn every iteration from
`[ceil(sqrt(n)), 2 ceil(sqrt(n))]` (Taillard's robust tabu search); `history_` records the
best cost after every iteration and the best tour is kept in its own buffer (`history_[-1]`
equals `cost_` exactly).

Plain symmetric TSPs use the O(1) move deltas of the core; asymmetric matrices and the
multi-trip objective use full evaluation of every candidate move, so the search sees the
decoded trips and their fixed charges. Tabu attributes are edges — arcs on an asymmetric matrix,
where a 2-opt reversal removes every arc of the reversed span — and live in an `int32` `(n, n)`
matrix, which bounds the practical size at about 5 000 nodes.

```python
>>> from skroute import TabuSearch
>>> from skroute.datasets import load_tsp
>>> dj = load_tsp("dj38")
>>> ts = TabuSearch(random_state=0).fit(dj.distance_matrix(), labels=dj.labels)
>>> ts.cost_ / dj.optimal_tour_length < 1.08
True

```

::: skroute.metaheuristics.TabuSearch
