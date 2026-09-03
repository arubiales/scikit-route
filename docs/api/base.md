# Base and problem

Every solver in scikit-route is an *estimator*: `__init__` stores the hyper-parameters (the
"knobs"), `fit(X, ...)` receives the data and returns `self`, and the results live in
trailing-underscore attributes (`tour_`, `route_`, `trips_`, `cost_`, ...). The contract has
three parts:

- [`RoutingProblem`](#skroute.RoutingProblem) — one immutable instance in index space: the coerced
  cost matrix, the optional time matrix and budget, the depot and the labels. `fit` builds it for
  you, or you build it once and pass it to several solvers (`Other().fit(est.problem_)`).
- [`BaseRouter`](#skroute.base.BaseRouter) — the template method: it validates the inputs and the
  hyper-parameters, honours the solver's [`RouterTags`](#skroute.base.RouterTags), calls the
  solver's `_solve(problem, rng)`, validates the returned tour and **recomputes `cost_`** from it,
  so a route that does not match its cost is impossible by construction.
- the helpers [`clone`](#skroute.clone), [`is_router`](#skroute.is_router),
  [`all_solvers`](#skroute.all_solvers) and [`set_log_level`](#skroute.set_log_level), the
  exceptions and the public recomputation helpers of `skroute.metrics`.

Plain TSP is `est.fit(C)`; the multi-trip objective is
`est.fit(C, time_matrix=T, max_time_work=8, extra_cost=12.83, people=2)` — `time_matrix` is
keyword-only, so the two square matrices can never be swapped by accident.

::: skroute.RoutingProblem

::: skroute.base.BaseRouter

::: skroute.base.RouterTags

::: skroute.clone

::: skroute.is_router

::: skroute.all_solvers

::: skroute.set_log_level

::: skroute.exceptions

::: skroute.metrics
