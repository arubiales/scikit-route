# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-09-03

A complete rewrite. Version 1.0.0a2 (2021) is the last release of the old code
base; see the [migration guide](https://arubiales.github.io/scikit-route/migration/)
for a symbol-by-symbol map.

### Added
- One problem model for every solver: a closed tour from a depot over a dense cost
  matrix, optionally decoded into trips under a per-trip working-time budget with a
  fixed charge per extra trip (`RoutingProblem`, `split="greedy"` or `"optimal"`).
- Solvers: `MILP` (Dantzig–Fulkerson–Johnson with lazy subtour cuts on HiGHS),
  `NearestNeighbour`, `Insertion`, `ClarkeWright`, `TwoOpt`, `OrOpt`, `LocalSearch`,
  `IteratedLocalSearch`, `MultiStart`; `HeldKarp` and `AntColony` when shipped.
- `skroute.metrics` (`route_cost`, `split_trips`), `skroute.check_router` (the
  estimator battery), `skroute.all_solvers`, `skroute.set_log_level`.
- Datasets return `Bunch` objects; `load_tsp(name)` and a TSPLIB reader
  (`read_tsplib`, `read_tsplib_tour`); `distance_matrix` with the TSPLIB metrics
  (`EUC_2D`, `CEIL_2D`, `MAN_2D`, `ATT`, `GEO`), Euclidean, Manhattan and haversine.
- Wheels for CPython 3.11–3.14 on Linux, macOS and Windows; typed (`py.typed`).

### Changed
- `fit()` takes the cost matrix (numpy, DataFrame or dict-of-dicts) and returns
  `self`; `time_matrix` is keyword-only; results live in `route_`, `tour_`, `trips_`,
  `cost_`, `history_`, `n_iter_` and friends.
- Objective: trips never exceed `max_time_work` including the return leg; `people`
  multiplies only `extra_cost`.
- Default hyper-parameters changed for every solver; results with defaults differ
  from 1.0 (table in the migration guide).
- `NRBS`: the five exponents must be >= 0 (1.0 validated only the type and accepted
  negative values). `Insertion(strategy="cheapest")` honours a first-edge tie rule.
  `ClarkeWright` checks a merge on the trip as it will be driven and may flip a trip when
  only the reverse orientation fits the budget.
- `Genetic`: real OX/PMX crossover. `SimulatedAnnealing`: the route/cost aliasing bug
  is fixed, three move types, automatic temperature. `TabuSearch`: rewritten with edge
  tabu attributes (a tenure lasts exactly `tenure` iterations; on asymmetric matrices a
  reversal marks every reversed arc). `NRBS`: union-find cycle check, the hard-coded route length is
  gone, `distance_weigth` → `distance_weight`. `SOM`: numpy only.
- Datasets: pickled DataFrames replaced by CSV; `frame`/`as_frame=True` instead of
  `"DataFrame"`; `feature_names` dropped.
- Dependencies: numpy, scipy and joblib only; pandas and googlemaps are optional
  extras; tensorflow, scikit-learn and tqdm are gone. Python >= 3.11, numpy 2 ready.
- `EnsembleGenetic` and `EnsembleSimulatedAnnealing` remain as explicit-parameter
  wrappers over `MultiStart` (threads by default; results independent of `n_jobs`).
- Every solver is checked by the public `check_router` battery and by tolerance tests
  against the published optima of the bundled instances (`tests/tolerances.py`,
  `docs/benchmarks.md`).

### Removed
- `skroute.cluster` (use scikit-learn directly), `preprocessing.df_to_tuple`,
  `preprocessing.matrix_parse`, `preprocessing.DataLossWarning`,
  `CostScraper.to_pickle`, the tuple return value of `fit()`, `route_example`.

### Fixed
- `load_costs_qatar` loaded Valencia; `matrix_to_dict` referenced an undefined
  variable; the datasets `__all__` was missing a comma; the TSPLIB reader used the
  deprecated `sep="\s"`; the 1.0 Cython code did not compile with Cython 3.

### Deprecated
- `skroute.heuristics.*` and the legacy `skroute.metaheuristics.<subpackage>` paths
  (import from `skroute.exact`, `skroute.construction`, `skroute.metaheuristics`,
  `skroute.ensemble`), `CostScraper`, `load_costs_qatar`, `mode=` in the TSP loaders.
  Removed in 3.0.

## [1.0.0a2] - 2021-06-21

Last release of the original code base (Cython 0.29, TensorFlow SOM, googlemaps).

[Unreleased]: https://github.com/arubiales/scikit-route/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/arubiales/scikit-route/compare/V1.0.0a1...v2.0.0
[1.0.0a2]: https://github.com/arubiales/scikit-route/releases/tag/V1.0.0a1
