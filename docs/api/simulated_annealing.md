# Simulated annealing

`SimulatedAnnealing` walks the space of giant tours with random 2-opt, Or-opt and swap
proposals: a downhill move is always accepted, an uphill move with probability
`exp(-delta / T)`, and the temperature cools geometrically from `t0` (calibrated on the
initial tour by default) down to `t_min`. One outer iteration is one temperature level of
`n_moves` proposals; `history_` records the best cost seen after every level and the best
tour lives in its own buffer, so `fit` never returns a tour that does not match its cost.

The same estimator solves plain symmetric TSPs (O(1) move deltas), asymmetric matrices and
the multi-trip objective (full evaluation of every proposal under the chosen split rule).
All randomness comes from `random_state`: the same seed on the same machine gives
bit-identical results, and `MultiStart` runs several seeds in parallel.

```python
>>> from skroute import SimulatedAnnealing
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")
>>> sa = SimulatedAnnealing(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
>>> sa.cost_ / wi.optimal_tour_length < 1.03
True

```

::: skroute.metaheuristics.SimulatedAnnealing
