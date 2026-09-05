"""Watch the solvers work: static route pictures, a live callback and a recorder that replays (D30, D31).

Everything here is optional and imported by nothing else in scikit-route. matplotlib is loaded
on first use (extra ``viz``: ``pip install scikit-route[viz]``), plotly only for the map tools
(extra ``viz-map``); importing this package needs neither.

- [`plot_route`][skroute.viz.plot_route] — the nodes and the closed route of a fitted solver, a
  progress event (or the edges of one construction step) or a route array, one colour per trip.
- [`plot_history`][skroute.viz.plot_history] — the best-so-far cost per outer iteration.
- [`plot_route_map`][skroute.viz.plot_route_map] — the same route on OpenStreetMap tiles (Plotly),
  with ``names=`` hover labels and ``trip_names=`` in the legend.
- [`google_maps_urls`][skroute.viz.google_maps_urls], [`to_kml`][skroute.viz.to_kml] and
  [`google_maps_html`][skroute.viz.google_maps_html] — the plan on Google Maps (D34): Directions
  links per day, a KML for Google My Maps and Google Earth, and a standalone Maps JavaScript page
  with real road directions (the only one that needs an API key). ``skroute.viz.google_maps``
  needs none of the viz extras: neither matplotlib nor plotly.
- [`LivePlot`][skroute.viz.LivePlot] — a ``fit(..., callback=)`` that redraws the current and best
  tours — and the structure being built: edges, pheromone trails, the SOM ring — while the solver
  runs, in a script or a notebook.
- [`Recorder`][skroute.viz.Recorder] — a callback that keeps every event with its clock and replays
  the run at time-lapse speed: a matplotlib animation, a live replay, a GIF/MP4 or a Plotly figure
  with Play/Pause, a speed menu and a slider.
"""

from ._live import LivePlot
from ._map import plot_route_map
from ._record import RecordedEvent, Recorder
from ._static import plot_history, plot_route
from .google_maps import google_maps_html, google_maps_urls, to_kml

__all__ = [
    "LivePlot",
    "RecordedEvent",
    "Recorder",
    "google_maps_html",
    "google_maps_urls",
    "plot_history",
    "plot_route",
    "plot_route_map",
    "to_kml",
]
