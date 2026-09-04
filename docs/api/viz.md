# Visualisation

`skroute.viz` draws what the solvers do: the nodes and the closed route of a solution
(one colour per trip), the best-so-far cost per iteration, the same route on
OpenStreetMap tiles, and — through the `fit(..., callback=)` protocol — the search
**while it runs**: [`LivePlot`][skroute.viz.LivePlot] redraws the current and best tours
at every iteration, together with the structure a solver is building when it reports one
(the edges of a construction heuristic, the pheromone trails of the ant colony, the SOM
ring — D31); [`Recorder`][skroute.viz.Recorder] keeps every event with a wall-clock stamp
and replays the run at time-lapse speed as a matplotlib animation, a live replay, a GIF
or MP4, or a Plotly figure with Play/Pause, a speed menu and a slider. The
[user guide page](../user_guide/visualisation.md) walks through the workflows.

The package is optional and imported by nothing else in scikit-route. matplotlib is
loaded on first use — `pip install "scikit-route[viz]"` — and plotly only by the map
tools — `pip install "scikit-route[viz-map]"`; a missing library raises an
`ImportError` that names the extra. Coordinates are `(n, 2)` arrays in matrix row
order: column 0 is x and column 1 is y for the matplotlib tools, `(latitude,
longitude)` for the map tools.

```python
>>> import matplotlib
>>> matplotlib.use("Agg")  # headless; any interactive backend works the same
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> from skroute.viz import Recorder, plot_history, plot_route
>>> dj = load_tsp("dj38")
>>> rec = Recorder(every=5)
>>> ils = IteratedLocalSearch(random_state=0).fit(
...     dj.distance_matrix(), labels=dj.labels, coords=dj.coords, callback=rec
... )
>>> ax = plot_route(ils)  # coordinates come from the fit
>>> len(ax.lines), len(ax.lines[0].get_xdata())  # one closed trip through the 38 cities
(1, 39)
>>> ax = plot_history(ils)
>>> len(ax.lines[0].get_ydata()) == ils.n_iter_
True
>>> rec.n_frames == len(rec) and bool((rec.frame_delays(speed=10) >= 10).all())  # a frame per kept event
True
>>> live = rec.replay(speed=1e9)  # the run again, through a LivePlot, as fast as it can be drawn
>>> live.n_events == len(rec)
True

```

::: skroute.viz.plot_route

::: skroute.viz.plot_history

::: skroute.viz.plot_route_map

::: skroute.viz.LivePlot

::: skroute.viz.Recorder

::: skroute.viz.RecordedEvent
