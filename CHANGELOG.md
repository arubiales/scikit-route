# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - unreleased
### Added
- Service times: `fit(..., service_time=)` adds the time spent at each stop to the per-trip budget;
  `skroute.metrics.timetable` (with `Stop`, `timetable_summary` and `units=`) turns a solution into a
  per-day timetable (D32).
- `skroute.preprocessing.maps`: real road travel times (`travel_time_matrix`, OSRM or Google),
  `geocode` (Nominatim or Google) and `fetch_pois` (OpenStreetMap through Overpass) (D33).
  `GoogleDistanceMatrix(departure_time=)` asks for traffic-aware durations; a `network` pytest marker
  (deselected by default, run nightly) exercises the live services.
- `skroute.viz.google_maps`: the plan on Google Maps as Directions URLs, a KML for Google My Maps and
  a standalone Maps JavaScript page; `plot_route_map(names=, trip_names=)` (D34).
- Worked case `examples/technician_madrid.py`: a maintenance technician covering every Burger King of the
  Madrid region from an office in Leganés, with real driving times, 30-minute visits and 8-hour days; the
  data are committed under `examples/data/` (OpenStreetMap + OSRM, 2026-09-05) and the user guide page
  *A real case: the technician's plan* narrates it (D35).

## [2.0.0] - 2026-09-04

A complete rewrite. Version 1.0.0a2 (2021) is the last release of the old code
base; see the [migration guide](https://arubiales.github.io/scikit-route/migration/)
for a symbol-by-symbol map.

### Added
- One problem model for every solver: a closed tour from a depot over a dense cost
  matrix, optionally decoded into trips under a per-trip working-time budget with a
  fixed charge per extra trip (`RoutingProblem`, `split="greedy"` or `"optimal"`).
- Eighteen solvers over the same core: exact `BruteForce`, `HeldKarp` and `MILP` (Dantzig–Fulkerson–Johnson
  with lazy subtour cuts on HiGHS); construction `NearestNeighbour`, `Insertion`, `ClarkeWright` and `NRBS`;
  local search `TwoOpt`, `OrOpt`, `LocalSearch` and `IteratedLocalSearch`; metaheuristics `SimulatedAnnealing`,
  `TabuSearch`, `Genetic` (OX/PMX, optional memetic descent), `AntColony` (MAX-MIN) and `SOM`; the `MultiStart`
  wrapper and the legacy `Ensemble*` names.
- `skroute.metrics` (`route_cost`, `split_trips`), `skroute.check_router` (the
  estimator battery), `skroute.all_solvers`, `skroute.set_log_level`.
- Datasets return `Bunch` objects; `load_tsp(name)` and a TSPLIB reader
  (`read_tsplib`, `read_tsplib_tour`); `distance_matrix` with the TSPLIB metrics
  (`EUC_2D`, `CEIL_2D`, `MAN_2D`, `ATT`, `GEO`), Euclidean, Manhattan and haversine.
- Wheels for CPython 3.11–3.14 on Linux, macOS and Windows; typed (`py.typed`).
- Progress callbacks: `fit(..., callback=)` and `skroute.RouteEvent` report `start`/`iteration`/`end`
  events with label-space tours from every solver; returning `True` stops an iterative solver
  (`stop_reason_ == "callback"`); `MultiStart` forwards sequential restarts; `check_router` gains check 14.
  The `end` event reports `n_iter_` as its iteration; a `fit` that raises (in the kernel or in the callback)
  never leaves the estimator looking fitted; `_emit` refuses a second `start` or an `end` from a solver
  (the base class emits both) while a callback is set.
- `skroute.viz` (extra `viz`, maps with `viz-map`): `plot_route`, `plot_history`, `LivePlot` (watch the
  current and best tour while `fit` runs, in scripts and notebooks), `Recorder` (record a run, animate it,
  save a GIF or a Plotly figure with a slider) and `plot_route_map` on OpenStreetMap tiles.
- Watch the search being built: construction solvers report the partial structure after every step
  (`extra["edges"]`), `AntColony` its strongest pheromone trails (`extra["edge_weights"]`) and `SOM` its
  ring (`extra["ring"]`); `LivePlot` draws them (`show=`, `trail=`), `Recorder` keeps wall-clock timestamps
  and replays a run at time-lapse speed (`replay(speed=)`, `animate(speed=, fps=)`, `save()` to GIF or MP4,
  a Plotly speed menu); `examples/live_demo.py --record/--speed/--set`.
  `LivePlot(map=True)` picks the Plotly backend by itself (`backend=None`); `Recorder.replay` paces on a
  target clock (short gaps accumulate, drawing time is absorbed, gaps over two seconds are cut) and
  `Recorder.animate(speed=)` really paces the frames under `plt.show()`. A callback answering `True` at
  `start` of a construction solver silences every step event (the result never depends on the callback).

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
