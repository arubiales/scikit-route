# Genetic algorithm

`Genetic` evolves a population of giant tours. A chromosome is the tour without its
depot, so every individual is a valid solution and its fitness is the problem objective
itself — the plain tour cost, or the greedy/optimal-split cost when a working-time budget
is given, so the multi-trip objective steers the population directly.

Each generation keeps the `n_elite` fittest parents, then fills the population with
children: two parents by tournament selection, a **real permutation crossover** (`"ox"`,
order crossover, or `"pmx"`, partially mapped crossover — the 1.0 "crossover" was a
rotation), a mutation (`"inversion"`, `"swap"` or `"insertion"`), one more mutation if the
child exactly duplicates an individual already in the new generation, and optionally the
**memetic polish** `local_search=("two_opt",)` or `("two_opt", "or_opt")`, which runs the
descents of `skroute.local_search` to convergence on every child. The plain GA is the weakest
metaheuristic of the library on large instances (≈ 19 % above the optimum on `qa194` with
the defaults); the memetic configuration is competitive with `IteratedLocalSearch`.

Every random quantity of a generation is drawn beforehand from `random_state` and handed to
the `nogil` kernel, so a fit is bit-identical for a given seed on a given machine.

```python
>>> from skroute import Genetic
>>> from skroute.datasets import load_tsp
>>> dj = load_tsp("dj38")
>>> ga = Genetic(local_search=("two_opt",), random_state=0).fit(dj.distance_matrix(), labels=dj.labels)
>>> ga.cost_ / dj.optimal_tour_length < 1.05
True
>>> ga.stop_reason_ in {"patience", "max_iter"}
True

```

A truncation ("top X %") selection, requested in issue #37, is planned for 2.1 as
`Genetic(selection="truncation")`.

::: skroute.metaheuristics.Genetic
