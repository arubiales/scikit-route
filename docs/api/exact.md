# Exact solvers

Three solvers that certify the optimum and set `is_optimal_`. They are the yardstick of the
library: every heuristic is tested against `BruteForce` on small instances, and `MILP` proves
the published optima of the small Waterloo instances.

| Solver | Objective it certifies | Cap (`max_nodes`) | Typical time |
|---|---|---|---|
| `BruteForce` | plain tour **and** multi-trip, under `split="greedy"` or `"optimal"` | 11 | a fraction of a second at n = 11 |
| `HeldKarp` | plain tour, symmetric or asymmetric | 20 | under a second at n = 20 (~90 MB of tables) |
| `MILP` | plain tour, symmetric or asymmetric | 300 | wi29 and dj38 in well under a second; qa194 proved in about a minute |

`HeldKarp` and `MILP` raise `ValueError` when `max_time_work` is given: an "exact" result that
is not optimal for the objective would be a trap, so a certified multi-trip optimum comes only
from `BruteForce` (n ≤ 11) and larger multi-trip instances go to the budget-aware heuristics.

Under `split="greedy"`, `BruteForce` is exact over greedy-decoded giant tours (a partition that
closes a trip while the next node would still fit is not representable); under `split="optimal"`
it is exact for the distance-constrained multi-trip problem.

```python
>>> from skroute import BruteForce, HeldKarp, MILP
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")
>>> est = MILP().fit(wi.distance_matrix(), labels=wi.labels)
>>> int(est.cost_), est.is_optimal_, est.gap_
(27603, True, 0.0)

```

::: skroute.exact.BruteForce

::: skroute.exact.HeldKarp

::: skroute.exact.MILP
