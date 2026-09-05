"""Plotly drawings: ``plot_route_map`` on OpenStreetMap tiles, the Plotly backend of ``LivePlot`` and the
slider figure of ``Recorder.to_plotly``. plotly is imported inside the functions (extra ``viz-map``)."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np

from ._static import (
    BEST_COLOR,
    CURRENT_COLOR,
    DEPOT_COLOR,
    EDGE_COLOR,
    PLOTLY_WEIGHT_MIN,
    POINT_COLOR,
    RING_COLOR,
    closed_trips,
    coords_array,
    event_edges,
    event_ring,
    event_weights,
    graph_objects,
    resolve,
    route_index,
    route_title,
)
from .google_maps import PALETTE, node_names, trip_labels

if TYPE_CHECKING:
    from ._live import LivePlot
    from ._record import RecordedEvent

__all__ = ["plot_route_map"]

_TAB10 = PALETTE  # one colour per trip, the same as the Google Maps exports
_OSM = "open-street-map"
SPEED_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0)  # the speed menu of ``Recorder.to_plotly``
# Trace positions shared by the live view and the slider figure: nodes, depot, then the four live traces.
CURRENT, BEST, EDGES, RING = 2, 3, 4, 5


# --------------------------------------------------------------------------- geometry
def auto_zoom(latlon: np.ndarray) -> float:
    """A map zoom that frames the points: 360 degrees of longitude across 512 pixels at zoom 0 (the
    MapLibre convention behind Plotly's ``map``, not the 256-pixel one of Leaflet), halving per level, so
    the route fills about three quarters of a 700-pixel map."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    span_lat = float(lat.max() - lat.min())
    span_lon = float(lon.max() - lon.min()) * math.cos(math.radians(float(lat.mean())))
    span = max(span_lat, span_lon)
    if span <= 0.0:
        return 12.0
    return float(np.clip(math.log2(360.0 / span), 1.0, 18.0))


def _center(latlon: np.ndarray) -> dict[str, float]:
    return {"lat": float(latlon[:, 0].mean()), "lon": float(latlon[:, 1].mean())}


def _xy_kwargs(xy: np.ndarray, idx: Any, *, map: bool) -> dict[str, Any]:
    """The coordinate keywords of a trace: ``lat``/``lon`` on a map, ``x``/``y`` otherwise."""
    pts = xy[idx]
    if map:
        return {"lat": pts[:, 0], "lon": pts[:, 1]}
    return {"x": pts[:, 0], "y": pts[:, 1]}


def _pairs_kwargs(a: list[Any], b: list[Any], *, map: bool) -> dict[str, Any]:
    return {"lat": a, "lon": b} if map else {"x": a, "y": b}


def _polyline(xy: np.ndarray, trips: list[np.ndarray], *, map: bool) -> dict[str, Any]:
    """Every closed trip in one line trace, separated by ``None`` gaps."""
    a: list[float | None] = []
    b: list[float | None] = []
    for k, trip in enumerate(trips):
        if k:
            a.append(None)
            b.append(None)
        a.extend(xy[trip, 0].tolist())
        b.extend(xy[trip, 1].tolist())
    return _pairs_kwargs(a, b, map=map)


def _segments(xy: np.ndarray, idx: np.ndarray, *, map: bool) -> dict[str, Any]:
    """Every edge of ``idx`` in one line trace, separated by ``None`` gaps."""
    a: list[float | None] = []
    b: list[float | None] = []
    for k, (u, v) in enumerate(idx.tolist()):
        if k:
            a.append(None)
            b.append(None)
        a.extend((float(xy[u, 0]), float(xy[v, 0])))
        b.extend((float(xy[u, 1]), float(xy[v, 1])))
    return _pairs_kwargs(a, b, map=map)


def _ring(ring: np.ndarray | None, *, map: bool) -> dict[str, Any]:
    """The closed polyline of a SOM ring (empty without one)."""
    if ring is None or ring.shape[0] == 0:
        return _pairs_kwargs([], [], map=map)
    closed = np.vstack((ring, ring[:1]))
    return _pairs_kwargs(closed[:, 0].tolist(), closed[:, 1].tolist(), map=map)


def _structure(
    xy: np.ndarray, problem: Any, extra: Any, *, map: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The edges and ring traces' coordinates of one event's D31 extras (empty when absent).

    Plotly draws one width per trace, so with ``edge_weights`` only the trails whose weight reaches
    ``PLOTLY_WEIGHT_MIN`` of the strongest one are kept (matplotlib fades the others instead).
    """
    idx = event_edges(problem, extra)
    if idx is None:
        edges = _pairs_kwargs([], [], map=map)
    else:
        weights = event_weights(extra, idx.shape[0])
        if weights is not None:
            strongest = float(weights.max()) if weights.size else 0.0
            idx = idx[weights >= PLOTLY_WEIGHT_MIN * strongest] if strongest > 0.0 else idx[:0]
        edges = _segments(xy, idx, map=map)
    return edges, _ring(event_ring(extra), map=map)


def _trace_cls(go: Any, *, map: bool) -> Any:
    return go.Scattermap if map else go.Scatter


def _base_traces(
    go: Any, xy: np.ndarray, depot: int, labels: Any, *, map: bool, names: Any = None
) -> list[Any]:
    """Nodes and depot; the hover text is ``names`` (a sequence in row order or a mapping by label),
    else the labels."""
    cls = _trace_cls(go, map=map)
    mask = np.ones(xy.shape[0], dtype=bool)
    mask[depot] = False
    text = np.asarray(node_names(np.asarray(labels), names), dtype=object)
    nodes = cls(
        mode="markers",
        marker={"size": 7, "color": POINT_COLOR},
        text=text[mask].tolist(),
        hoverinfo="text",
        name="nodes",
        showlegend=False,
        **_xy_kwargs(xy, mask, map=map),
    )
    dep = cls(
        mode="markers",
        marker={"size": 14, "color": DEPOT_COLOR},
        text=[text[depot]],
        hoverinfo="text",
        name="depot",
        showlegend=False,
        **_xy_kwargs(xy, [depot], map=map),
    )
    return [nodes, dep]


def _live_traces(go: Any, *, map: bool) -> list[Any]:
    """The four traces a run updates: current (thin), best (thick), edges (orange) and ring (teal)."""
    cls = _trace_cls(go, map=map)
    empty = _pairs_kwargs([], [], map=map)
    return [
        cls(
            mode="lines", line={"width": 1, "color": CURRENT_COLOR}, name="current", hoverinfo="skip", **empty
        ),
        cls(mode="lines", line={"width": 3, "color": BEST_COLOR}, name="best", hoverinfo="skip", **empty),
        cls(mode="lines", line={"width": 2, "color": EDGE_COLOR}, name="edges", hoverinfo="skip", **empty),
        cls(
            mode="lines+markers",
            line={"width": 1.5, "color": RING_COLOR},
            marker={"size": 4, "color": RING_COLOR},
            name="ring",
            hoverinfo="skip",
            **empty,
        ),
    ]


def _layout(
    go: Any, xy: np.ndarray, title: str, *, map: bool, zoom: float | None, legend: bool = False
) -> Any:
    layout = go.Layout(title={"text": title}, margin={"l": 10, "r": 10, "t": 50, "b": 10}, showlegend=legend)
    if map:
        layout.update(
            map={"style": _OSM, "center": _center(xy), "zoom": auto_zoom(xy) if zoom is None else zoom}
        )
    else:
        layout.update(
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "x"},
            plot_bgcolor="white",
        )
    return layout


# --------------------------------------------------------------------------- public
def plot_route_map(
    obj: Any,
    coords: Any = None,
    *,
    zoom: float | None = None,
    names: Any = None,
    trip_names: Any = None,
) -> Any:
    """Draw a solution on OpenStreetMap tiles with Plotly (one line per trip, the depot marked).

    Parameters
    ----------
    obj : fitted estimator, RouteEvent or array of row positions
        As in [`plot_route`][skroute.viz.plot_route].
    coords : (n, 2) array-like, optional
        ``(latitude, longitude)`` in decimal degrees, matrix row order; default: the problem's
        ``coords``.
    zoom : float, optional
        Map zoom level; default: fitted to the points.
    names : sequence of n str or mapping {label: str}, optional
        The hover text of each node — a name per row of ``coords`` (matrix row order) or a mapping
        from label to name (a label without an entry shows the label); default: the labels.
    trip_names : sequence of str, optional
        One name per trip, shown in a legend (``"Monday"``, ``"Tuesday"``...); without it the
        lines are named ``"trip 1"``, ``"trip 2"``... and the legend stays hidden.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        ``Scattermap`` traces on the ``"open-street-map"`` style: nodes, depot and one line per
        trip; call ``fig.show()`` or ``fig.write_html("route.html")``.

    Examples
    --------
    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_barcelona
    >>> from skroute.viz import plot_route_map
    >>> bcn = load_barcelona()  # coords are (latitude, longitude)
    >>> ils = IteratedLocalSearch(random_state=0).fit(bcn.cost, labels=bcn.labels, coords=bcn.coords)
    >>> fig = plot_route_map(ils)
    >>> [t.type for t in fig.data], fig.layout.map.style
    (['scattermap', 'scattermap', 'scattermap'], 'open-street-map')

    Names on hover and a legend of days for a two-day plan:

    >>> days = IteratedLocalSearch(random_state=0).fit(
    ...     bcn.cost, labels=bcn.labels, coords=bcn.coords, time_matrix=bcn.time, max_time_work=6.0
    ... )
    >>> places = {label: f"Place {label}" for label in bcn.labels}
    >>> fig = plot_route_map(days, names=places, trip_names=["Monday", "Tuesday"])
    >>> fig.data[1].text, [t.name for t in fig.data[2:]], fig.layout.showlegend
    (('Place 10000007',), ['Monday', 'Tuesday'], True)
    """
    go = graph_objects()
    xy, trips, depot, labels, name, cost = resolve(obj, coords)
    traces = _base_traces(go, xy, depot, labels, map=True, names=names)
    titles = trip_labels(trip_names, len(trips)) if trip_names is not None else None
    for k, trip in enumerate(trips):
        traces.append(
            go.Scattermap(
                mode="lines",
                line={"width": 3, "color": _TAB10[k % 10]},
                name=titles[k] if titles is not None else f"trip {k + 1}",
                hoverinfo="skip",
                **_xy_kwargs(xy, trip, map=True),
            )
        )
    layout = _layout(
        go, xy, route_title(name, cost, len(trips)), map=True, zoom=zoom, legend=titles is not None
    )
    return go.Figure(data=traces, layout=layout)


# --------------------------------------------------------------------------- LivePlot backend
class PlotlyLiveView:
    """The Plotly drawing of a ``LivePlot``: a ``FigureWidget`` updated in place in a notebook, else a
    plain ``Figure`` shown once at ``"end"``. Traces: nodes, depot, current, best, edges, ring."""

    def __init__(self, owner: LivePlot, problem: Any) -> None:
        self.owner = owner
        self.problem = problem
        self.xy = coords_array(owner.coords, problem.n)
        self.fig: Any = None
        self.ax = None
        self.widget = False

    def start(self, event: Any) -> None:
        from ._live import display_figure, in_notebook

        go = graph_objects()
        m = self.owner.map
        traces = _base_traces(go, self.xy, self.problem.depot, self.problem.labels, map=m)
        traces.extend(_live_traces(go, map=m))
        layout = _layout(go, self.xy, self._title(event), map=m, zoom=None)
        self.widget = False
        if in_notebook():
            try:
                self.fig = go.FigureWidget(data=traces, layout=layout)
                self.widget = True
            except ImportError:  # plotly >= 6 needs anywidget for FigureWidget
                self.fig = go.Figure(data=traces, layout=layout)
            if self.widget:
                display_figure(self.fig, clear=False)
        else:
            self.fig = go.Figure(data=traces, layout=layout)
        self._set(event)

    def restart(self, event: Any) -> None:
        """A nested ``"start"`` (the next restart of a MultiStart): the previous restart's drawing goes."""
        ctx = self.fig.batch_update() if self.widget else nullcontext()
        with ctx:
            for k in (CURRENT, BEST, EDGES, RING):
                _assign(self.fig.data[k], _pairs_kwargs([], [], map=self.owner.map))
        self._set(event)

    def update(self, event: Any) -> None:
        self._set(event)

    def finish(self, event: Any, *, last: bool = True) -> None:
        """Draw the final route; outside a notebook the figure is shown once, at the outermost ``"end"``."""
        self._set(event, final=True)
        if last and not self.widget:
            self.fig.show()

    @property
    def shows_frames(self) -> bool:
        """Whether each redraw is seen as it happens (the widget) or only the figure shown at ``"end"``."""
        return self.widget

    # ----- helpers
    def _title(self, event: Any) -> str:
        from ._live import status_line

        return status_line(self.owner.title or str(event.solver), event, newline="<br>")

    def _set(self, event: Any, *, final: bool = False) -> None:
        m, show = self.owner.map, self.owner.show
        ctx = self.fig.batch_update() if self.widget else nullcontext()
        with ctx:
            current, best = self.fig.data[CURRENT], self.fig.data[BEST]
            if final or event.tour is None or show == "best":
                _assign(current, _pairs_kwargs([], [], map=m))
            else:
                _assign(current, _xy_kwargs(self.xy, route_index(self.problem, event.tour), map=m))
            if event.best_tour is not None and (final or show != "current"):
                if final:
                    _assign(best, _polyline(self.xy, closed_trips(self.problem, event.best_tour), map=m))
                else:
                    _assign(best, _xy_kwargs(self.xy, route_index(self.problem, event.best_tour), map=m))
            edges, ring = _structure(self.xy, self.problem, getattr(event, "extra", None), map=m)
            _assign(self.fig.data[EDGES], edges)
            _assign(self.fig.data[RING], ring)
            self.fig.layout.title.text = self._title(event)


def _assign(trace: Any, kwargs: dict[str, Any]) -> None:
    for key, value in kwargs.items():
        setattr(trace, key, value)


# --------------------------------------------------------------------------- Recorder.to_plotly
def recorder_figure(
    frames: list[RecordedEvent], xy: np.ndarray, problem: Any, *, map: bool, show: str, fps: float
) -> Any:
    """A figure with one frame per recorded drawable event, Play/Pause, a speed menu and a slider
    (``Recorder.to_plotly``). Frame ``k`` updates the current, best, edges and ring traces. ``xy`` is
    validated against ``problem`` (the frames' instance) and every other problem the frames carry."""
    from ._live import status_line

    go = graph_objects()
    depot, labels = problem.depot, problem.labels
    empty = _pairs_kwargs([], [], map=map)

    def frame_traces(ev: RecordedEvent) -> list[Any]:
        prob = problem if ev.problem is None else ev.problem
        current, best = dict(empty), dict(empty)
        if ev.tour is not None and ev.stage != "end" and show != "best":
            current = _xy_kwargs(xy, route_index(prob, ev.tour), map=map)
        if ev.best_tour is not None and (ev.stage == "end" or show != "current"):
            best = _polyline(xy, closed_trips(prob, ev.best_tour), map=map)
        edges, ring = _structure(xy, prob, ev.extra, map=map)
        template = _live_traces(go, map=map)
        for trace, kwargs in zip(template, (current, best, edges, ring), strict=True):
            trace.update(**kwargs)
        return template

    def title(ev: RecordedEvent) -> str:
        return status_line(ev.solver, ev, newline="<br>")

    def step(ev: RecordedEvent) -> str:
        """The slider label: the iteration, prefixed by the restart index inside a MultiStart."""
        restart = (ev.extra or {}).get("restart")
        return str(ev.iteration) if restart is None else f"{restart}:{ev.iteration}"

    traces = [*_base_traces(go, xy, depot, labels, map=map), *frame_traces(frames[0])]
    fig = go.Figure(data=traces, layout=_layout(go, xy, title(frames[0]), map=map, zoom=None))
    live = [CURRENT, BEST, EDGES, RING]
    fig.frames = [
        go.Frame(
            data=frame_traces(ev),
            traces=live,
            name=str(k),
            layout={"title": {"text": title(ev)}},
        )
        for k, ev in enumerate(frames)
    ]

    def frame_args(duration: float) -> dict[str, Any]:
        return {
            "frame": {"duration": duration, "redraw": map},
            "mode": "immediate",
            "transition": {"duration": 0},
        }

    base = 1000.0 / fps
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 0.0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {**frame_args(base), "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {**frame_args(0)}],
                    },
                ],
            },
            {
                "type": "dropdown",
                "showactive": True,
                "active": SPEED_FACTORS.index(1.0),
                "x": 0.2,
                "y": 0.0,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": f"{factor:g}x",
                        "method": "animate",
                        "args": [None, {**frame_args(base / factor), "fromcurrent": True}],
                    }
                    for factor in SPEED_FACTORS
                ],
            },
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "iteration ", "visible": True},
                "pad": {"t": 30},
                "steps": [
                    {"label": step(ev), "method": "animate", "args": [[str(k)], frame_args(base)]}
                    for k, ev in enumerate(frames)
                ],
            }
        ],
    )
    return fig
