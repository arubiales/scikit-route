# Ensembles

`skroute.ensemble` runs a stochastic solver several times from independent seeds and keeps the
best tour. The general tool is `MultiStart(estimator, n_restarts=10, n_jobs=None)`: it clones
the estimator once per restart, seeds every clone from a child of one
`numpy.random.SeedSequence`, fits the clones on the **shared** `RoutingProblem` through
`joblib.Parallel` and returns the cheapest result (ties go to the lowest restart index). The
winner is the same for any `n_jobs` and backend, because the seeds are assigned by restart
index before anything runs — `random_state` alone reproduces a run.

Threads are the default (`prefer="threads"`): the Cython kernels release the GIL and a large
cost matrix is never pickled once per worker. They give a near-linear speed-up for
`SimulatedAnnealing`, `IteratedLocalSearch` and `TabuSearch` and little for the Python-heavy
`Genetic` and `SOM`; `prefer="processes"` is one keyword away and gives identical results.
The tags of a `MultiStart` are its estimator's with `kind="ensemble"`: wrapping a
budget-aware solver keeps the multi-trip objective inside the search, wrapping `SOM` still
needs `coords=`. A deterministic estimator is refused (`ValueError`), and `MultiStart` is not
returned by `all_solvers()` because it needs an estimator.

`EnsembleGenetic` and `EnsembleSimulatedAnnealing` are the 1.0 ensembles kept as
explicit-parameter wrappers over `MultiStart` (`n_genetics` / `n_simulateds` restarts, the
inner knobs keyword-only with the 2.0 defaults). They return exactly what the equivalent
`MultiStart(Genetic(...))` / `MultiStart(SimulatedAnnealing(...))` returns and will be removed
in 3.0; new code should use `MultiStart` directly.

```python
>>> import numpy as np
>>> from skroute import MultiStart, SimulatedAnnealing, EnsembleSimulatedAnnealing
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
>>> C = wi.distance_matrix()
>>> ms = MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=0).fit(C, labels=wi.labels)
>>> ms.cost_ / wi.optimal_tour_length < 1.03  # the fast-tier tolerance of the wrapped solver
True
>>> len(ms.estimators_), ms.best_estimator_ is ms.estimators_[ms.best_index_]
(4, True)
>>> parallel = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=2, random_state=0).fit(C, labels=wi.labels)
>>> bool(np.array_equal(parallel.tour_, ms.tour_))  # the result never depends on n_jobs
True
>>> legacy = EnsembleSimulatedAnnealing(n_simulateds=4, random_state=0).fit(C, labels=wi.labels)
>>> bool(np.array_equal(legacy.tour_, ms.tour_))  # the wrapper is the MultiStart it builds
True
>>> ms.set_params(estimator__alpha=0.99).estimator.alpha  # inner knobs through the estimator__ prefix
0.99

```

::: skroute.ensemble.MultiStart

::: skroute.ensemble.EnsembleGenetic

::: skroute.ensemble.EnsembleSimulatedAnnealing
