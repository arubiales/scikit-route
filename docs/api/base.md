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
- [`RouteEvent`](#skroute.base.RouteEvent) — the progress report a solver hands to the `callback`
  of `fit`, so you can watch (or stop) a search while it runs.
- the helpers [`clone`](#skroute.clone), [`is_router`](#skroute.is_router),
  [`all_solvers`](#skroute.all_solvers) and [`set_log_level`](#skroute.set_log_level), the
  exceptions and the public recomputation helpers of `skroute.metrics`.

Plain TSP is `est.fit(C)`; the multi-trip objective is
`est.fit(C, time_matrix=T, max_time_work=8, extra_cost=12.83, people=2)` — `time_matrix` is
keyword-only, so the two square matrices can never be swapped by accident.

## Watching a solver work

Every `fit` accepts `callback=`: any callable taking one `RouteEvent`. The solver calls it once
at `"start"`, once after every outer iteration — exactly where `history_` grows, with the tour it
is working on, the best tour so far and their costs — and once at `"end"` with the fitted tour.
Return `True` to stop the search after the current outer iteration (`stop_reason_ == "callback"`).
The tours arrive in label space, depot first, and `event.route`/`event.trips` decode the best one
with the problem's own split rule, so a multi-trip instance can be drawn trip by trip. Every
solver documents the keys of `event.extra` it fills (the temperature of an annealing level, the
kick of an iterated local search, the LP support of a `MILP` cut round...). Inside a
`MultiStart` the inner solvers report under their own name with `extra["restart"]`, and only when
the restarts run sequentially (`n_jobs=None` or `1` — then in the calling thread, whatever an
enclosing `joblib.parallel_config` says, so the callback never runs in a worker). The callback
runs inside the timed search (`fit_time_` includes it), and an exception it raises propagates out
of `fit` and leaves the estimator unfitted, at the `"end"` event as at any other. The
`skroute.viz` package builds live plots and animations on top of this protocol.

```pycon
>>> import numpy as np
>>> from skroute import SimulatedAnnealing
>>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
>>> seen = []
>>> sa = SimulatedAnnealing(random_state=0).fit(C, callback=seen.append)
>>> seen[0].stage, seen[0].iteration, seen[-1].stage, seen[-1].iteration == sa.n_iter_
('start', 0, 'end', True)
>>> len(seen) == sa.n_iter_ + 2 and seen[-1].best_cost == sa.cost_
True
>>> sorted(seen[1].extra)
['accepted', 'n_moves', 'temperature']
>>> short = SimulatedAnnealing(random_state=0).fit(C, callback=lambda e: e.iteration >= 5)
>>> short.n_iter_, short.stop_reason_
(5, 'callback')

```

::: skroute.RoutingProblem

::: skroute.base.BaseRouter

::: skroute.base.RouterTags

::: skroute.base.RouteEvent

::: skroute.clone

::: skroute.is_router

::: skroute.all_solvers

::: skroute.set_log_level

::: skroute.exceptions

::: skroute.metrics
