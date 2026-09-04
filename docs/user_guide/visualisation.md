# Live visualisation

Route optimisation is a spatial problem, and a solver's progress is easiest to judge by
eye: are the crossings disappearing, is the best tour still changing, did the annealer
freeze too early? `skroute.viz` draws the answer — a fitted solution as a static picture,
and a running solver **live**, through the `fit(..., callback=)` protocol every solver
implements: the solver hands a `RouteEvent` to your callback at the start of the search,
after every outer iteration and at the end; [`LivePlot`][skroute.viz.LivePlot] redraws
the current and best tours on each event, [`Recorder`][skroute.viz.Recorder] keeps them
for later.

<p align="center">
  <img src="../../images/live_demo.gif" alt="SimulatedAnnealing untangling a random tour of the 38 cities of Djibouti" width="480">
</p>

## Installing

The package is optional and nothing else in scikit-route imports it:

```bash
pip install "scikit-route[viz]"       # matplotlib: plot_route, plot_history, LivePlot, Recorder
pip install "scikit-route[viz-map]"   # + plotly: OpenStreetMap tiles, the plotly backend, sliders
```

`import skroute.viz` needs neither library — matplotlib and plotly are loaded at the
first call — and a missing one raises an `ImportError` naming the extra to install.

Coordinates are always `(n, 2)` arrays in the row order of the cost matrix. The
matplotlib tools draw column 0 as x and column 1 as y; the map tools expect
`(latitude, longitude)`. For geographic data on plain axes pass `coords[:, ::-1]`.

## A picture of a solution

[`plot_route`][skroute.viz.plot_route] takes a fitted estimator and draws the nodes,
the depot (a star) and the closed route, one colour per trip; the coordinates come from
`fit(..., coords=)` or from its `coords=` argument. [`plot_history`][skroute.viz.plot_history]
draws `history_`, the best-so-far cost per outer iteration.

```python
>>> import matplotlib
>>> matplotlib.use("Agg")  # this page runs headless; drop the line on a desktop
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> from skroute.viz import plot_history, plot_route
>>> dj = load_tsp("dj38")  # Djibouti, 38 cities, optimum 6656
>>> ils = IteratedLocalSearch(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, coords=dj.coords)
>>> ax = plot_route(ils)
>>> len(ax.lines), len(ax.lines[0].get_xdata())  # one closed trip through the 38 cities
(1, 39)
>>> ax.get_title().startswith("IteratedLocalSearch | cost ")
True
>>> ax = plot_history(ils)
>>> len(ax.lines[0].get_ydata()) == ils.n_iter_
True

```

A multi-trip solution gets one line per trip. `plot_route` also accepts a route array
whose entries index the rows of `coords` (the `tour_` or `route_` of a solver fitted
without `labels=`; a repeated depot separates trips) and any `RouteEvent`:

```python
>>> import numpy as np
>>> xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
>>> ax = plot_route([0, 1, 0, 2, 3], xy, labels=True)  # two trips: 0-1-0 and 0-2-3-0
>>> len(ax.lines), [t.get_text() for t in ax.texts]
(2, ['0', '1', '2', '3'])

```

## Watching a run in a script

[`LivePlot`][skroute.viz.LivePlot] is a callable: pass it as the `callback` of `fit`.
On the `"start"` event it opens the figure; on every `every`-th `"iteration"` event it
redraws the current tour (thin, light) and the best tour so far (thick), and puts the
solver, the iteration, both costs and the solver's own facts (the temperature of an
annealer, the tenure of a tabu search...) in the title; on `"end"` it draws the final
route with trip colours. It refreshes the window with `plt.pause`, so it never blocks
and never calls `plt.show()` — add that yourself after `fit` to keep the window open.

```python
>>> import matplotlib.pyplot as plt
>>> from skroute import SimulatedAnnealing
>>> from skroute.viz import LivePlot
>>> live = LivePlot(dj.coords, every=10)  # redraw every tenth temperature level
>>> sa = SimulatedAnnealing(random_state=0).fit(
...     dj.distance_matrix(), labels=dj.labels, callback=live
... )  # doctest: +SKIP
>>> plt.show()  # doctest: +SKIP

```

A redraw costs a few milliseconds and a fast solver emits thousands of outer iterations
per second, so `every=` is the knob that keeps the search fast: `SimulatedAnnealing` runs
about 1 800 levels, `IteratedLocalSearch` up to `n_iter` kicks, `TabuSearch` `n_iter`
moves. Headless machines work too — on the `Agg` backend the figure is updated without
being shown, which is how this page and the test-suite run.

`examples/live_demo.py` in the repository wraps this in a small command line:

```bash
python examples/live_demo.py --instance dj38 --solver IteratedLocalSearch
python examples/live_demo.py --instance barcelona --solver SimulatedAnnealing --every 10
```

## In a notebook

The same object works in Jupyter. `LivePlot` detects the kernel and picks the refresh
that fits the matplotlib backend:

- `%matplotlib inline` (the default) — every redraw replaces the cell output through
  `IPython.display.display(fig, clear=True)`; keep `every=` generous, each redraw
  renders a PNG.
- `%matplotlib widget` (ipympl) — the canvas is updated in place, the way a desktop
  window is.
- `LivePlot(coords, backend="plotly")` — a Plotly `FigureWidget` displayed once and
  updated in place; `map=True` puts it on OpenStreetMap tiles. Needs
  `scikit-route[viz-map]` and `anywidget` (Plotly's widget dependency); without a
  kernel the plotly backend builds a plain `Figure` and shows it once at `"end"`.

```python
>>> live = LivePlot(dj.coords, backend="plotly", every=5)  # doctest: +SKIP
>>> IteratedLocalSearch(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=live)  # doctest: +SKIP

```

## Stopping a run interactively

Any callback that returns `True` asks the solver to stop after its current outer
iteration; the fit then completes normally with `stop_reason_ == "callback"` and the
best tour found so far. `LivePlot.stop()` arms exactly that: call it from another cell
(run the fit in a thread so the kernel stays responsive) or from a key handler.

```python
>>> import threading
>>> live = LivePlot(dj.coords, every=5)
>>> sa = SimulatedAnnealing(random_state=0)
>>> worker = threading.Thread(
...     target=sa.fit, args=(dj.distance_matrix(),), kwargs={"labels": dj.labels, "callback": live}
... )
>>> worker.start()  # doctest: +SKIP
>>> live.stop()  # from another cell, a few seconds later  # doctest: +SKIP
>>> worker.join(); sa.stop_reason_  # doctest: +SKIP
'callback'

```

Redraws happen on the thread that runs `fit`, and matplotlib is not thread-safe, which
is why `MultiStart` forwards the callback to its inner estimators only with `n_jobs=1`.

## Recording a run: GIF, MP4, Plotly slider

[`Recorder`][skroute.viz.Recorder] draws nothing while the solver runs; it copies every
event (`every=` thins the iterations) and replays them afterwards:

```python
>>> from skroute.viz import Recorder
>>> rec = Recorder(every=5)
>>> ils = IteratedLocalSearch(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=rec)  # doctest: +SKIP
>>> rec.best_costs[-1] == ils.cost_  # doctest: +SKIP
True
>>> anim = rec.animate(dj.coords, interval=80)  # matplotlib FuncAnimation, one frame per kept event  # doctest: +SKIP
>>> anim.save("dj38.gif", writer="pillow", dpi=80)  # doctest: +SKIP
>>> anim.save("dj38.mp4")  # needs ffmpeg on the PATH  # doctest: +SKIP
>>> fig = rec.to_plotly(dj.coords)  # frames, Play/Pause and a slider over the iterations  # doctest: +SKIP
>>> fig.write_html("dj38.html")  # doctest: +SKIP
>>> ax = rec.plot_history()  # the best-so-far curve of the kept events  # doctest: +SKIP

```

`rec.events` holds the copies (`stage`, `iteration`, `cost`, `best_cost`, `tour`,
`best_tour`, `extra`, a reference to the problem); `rec.costs`, `rec.best_costs` and
`rec.iterations` are the same facts as arrays. A frame is drawn for every kept event
with a best tour, so `Recorder(every=30)` on an 1 800-level annealing gives about 60
frames. The GIF at the top of this page is `SimulatedAnnealing(init="random",
random_state=0)` on Djibouti (`dj38`) recorded that way and saved at 80 dpi: a random
tour of 27 722 becomes the optimum 6656 (`python examples/live_demo.py --instance dj38
--solver SimulatedAnnealing --init random --every 30 --gif live_demo.gif`).

## On a map

The five road-cost tables of [`skroute.datasets`][skroute.datasets.load_barcelona]
carry `(latitude, longitude)` coordinates. [`plot_route_map`][skroute.viz.plot_route_map]
draws a solution on OpenStreetMap tiles with Plotly's `Scattermap`, and `map=True` does
the same for `LivePlot(backend="plotly")` and `Recorder.to_plotly`:

```python
>>> from skroute.datasets import load_barcelona
>>> from skroute.viz import plot_route_map
>>> bcn = load_barcelona()  # 19 places, costs in EUR, coords are (lat, lon)
>>> ils = IteratedLocalSearch(random_state=0).fit(bcn.cost, labels=bcn.labels, coords=bcn.coords)
>>> fig = plot_route_map(ils)
>>> [t.type for t in fig.data], fig.layout.map.style
(['scattermap', 'scattermap', 'scattermap'], 'open-street-map')
>>> fig.show()  # doctest: +SKIP
>>> live = LivePlot(bcn.coords, backend="plotly", map=True, every=10)  # doctest: +SKIP
>>> ax = plot_route(ils, bcn.coords[:, ::-1])  # the same route on plain axes: x = longitude

```

Multi-trip solutions get one `Scattermap` line per trip. The tiles are fetched by the
browser that renders the figure, so a saved HTML file needs a network connection to
show the map; the routes themselves are embedded.

## What each solver reports in `extra`

Every event carries the same core fields — `solver`, `stage`, `iteration`, `cost`
(the objective of the current tour, `nan` when the solver has none), `best_cost`,
`tour` and `best_tour` (label-space open giant tours, depot first), `problem`, and the
decoded `route` and `trips` of the best tour — plus `extra`, a dict of solver-specific
facts. `LivePlot` prints the scalar ones in the title and skips the rest:

| Solver | `extra` keys | Meaning |
|---|---|---|
| `SimulatedAnnealing` | `temperature` | the level's temperature |
| `TabuSearch` | `tenure` | the tenure drawn for the iteration |
| `Genetic`, `EnsembleGenetic` | `generation`, `n_evaluations` | generation index, objective evaluations so far |
| `IteratedLocalSearch` | `kick`, `moves_applied` | the kick applied, moves applied by the descent |
| `SOM` | `radius` | the neighbourhood radius of the epoch |
| `MILP` | `edges` | the edges of the current LP support (one event per cut round) |
| `MultiStart` | `restart` | index of the restart that emitted the event |

Construction and exact solvers emit `"start"` and `"end"` only (MILP also one
`"iteration"` per cut round), so `LivePlot` shows their result and `Recorder` keeps two
events. Writing your own callback is the same one-argument function:

```python
>>> def log_improvements(event):
...     if event.stage == "iteration" and event.iteration % 100 == 0:
...         print(event.iteration, round(event.best_cost, 1), event.extra)
...     return False  # True would stop the run
>>> SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=log_improvements)  # doctest: +SKIP

```
