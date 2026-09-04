"""Watch the solvers work: static route pictures, a live callback and a recorder (D30).

Everything here is optional and imported by nothing else in scikit-route. matplotlib is loaded
on first use (extra ``viz``: ``pip install scikit-route[viz]``), plotly only for the map tools
(extra ``viz-map``); importing this package needs neither.

- [`plot_route`][skroute.viz.plot_route] — the nodes and the closed route of a fitted solver, a
  progress event or a route array, one colour per trip.
- [`plot_history`][skroute.viz.plot_history] — the best-so-far cost per outer iteration.
- [`plot_route_map`][skroute.viz.plot_route_map] — the same route on OpenStreetMap tiles (Plotly).
- [`LivePlot`][skroute.viz.LivePlot] — a ``fit(..., callback=)`` that redraws the current and best
  tours while the solver runs, in a script or a notebook.
- [`Recorder`][skroute.viz.Recorder] — a callback that keeps every event and replays them as a
  matplotlib animation (GIF/MP4) or a Plotly figure with a slider.
"""

from ._live import LivePlot
from ._map import plot_route_map
from ._record import RecordedEvent, Recorder
from ._static import plot_history, plot_route

__all__ = ["LivePlot", "RecordedEvent", "Recorder", "plot_history", "plot_route", "plot_route_map"]
