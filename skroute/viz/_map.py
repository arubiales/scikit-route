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
    POINT_COLOR,
    closed_trips,
    coords_array,
    format_number,
    graph_objects,
    resolve,
    route_index,
    route_title,
)

if TYPE_CHECKING:
    from ._live import LivePlot
    from ._record import RecordedEvent, Recorder

__all__ = ["plot_route_map"]

_TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_OSM = "open-street-map"


# --------------------------------------------------------------------------- geometry
def auto_zoom(latlon: np.ndarray) -> float:
    """A map zoom that frames the points: 360 degrees of longitude at zoom 0, halving per level."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    span_lat = float(lat.max() - lat.min())
    span_lon = float(lon.max() - lon.min()) * math.cos(math.radians(float(lat.mean())))
    span = max(span_lat, span_lon)
    if span <= 0.0:
        return 12.0
    return float(np.clip(math.log2(360.0 / span) - 1.0, 1.0, 18.0))


def _center(latlon: np.ndarray) -> dict[str, float]:
    return {"lat": float(latlon[:, 0].mean()), "lon": float(latlon[:, 1].mean())}


def _xy_kwargs(xy: np.ndarray, idx: Any, *, map: bool) -> dict[str, Any]:
    """The coordinate keywords of a trace: ``lat``/``lon`` on a map, ``x``/``y`` otherwise."""
    pts = xy[idx]
    if map:
        return {"lat": pts[:, 0], "lon": pts[:, 1]}
    return {"x": pts[:, 0], "y": pts[:, 1]}


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
    return {"lat": a, "lon": b} if map else {"x": a, "y": b}


def _trace_cls(go: Any, *, map: bool) -> Any:
    return go.Scattermap if map else go.Scatter


def _base_traces(go: Any, xy: np.ndarray, depot: int, labels: Any, *, map: bool) -> list[Any]:
    """Nodes and depot."""
    cls = _trace_cls(go, map=map)
    mask = np.ones(xy.shape[0], dtype=bool)
    mask[depot] = False
    nodes = cls(
        mode="markers",
        marker={"size": 7, "color": POINT_COLOR},
        text=[str(x) for x in np.asarray(labels)[mask]],
        hoverinfo="text",
        name="nodes",
        **_xy_kwargs(xy, mask, map=map),
    )
    dep = cls(
        mode="markers",
        marker={"size": 14, "color": DEPOT_COLOR},
        text=[str(np.asarray(labels)[depot])],
        hoverinfo="text",
        name="depot",
        **_xy_kwargs(xy, [depot], map=map),
    )
    return [nodes, dep]


def _layout(go: Any, xy: np.ndarray, title: str, *, map: bool, zoom: float | None) -> Any:
    layout = go.Layout(title={"text": title}, margin={"l": 10, "r": 10, "t": 50, "b": 10}, showlegend=False)
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
def plot_route_map(obj: Any, coords: Any = None, *, zoom: float | None = None) -> Any:
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
    """
    go = graph_objects()
    xy, trips, depot, labels, name, cost = resolve(obj, coords)
    traces = _base_traces(go, xy, depot, labels, map=True)
    for k, trip in enumerate(trips):
        traces.append(
            go.Scattermap(
                mode="lines",
                line={"width": 3, "color": _TAB10[k % 10]},
                name=f"trip {k + 1}",
                hoverinfo="skip",
                **_xy_kwargs(xy, trip, map=True),
            )
        )
    return go.Figure(
        data=traces, layout=_layout(go, xy, route_title(name, cost, len(trips)), map=True, zoom=zoom)
    )


# --------------------------------------------------------------------------- LivePlot backend
class PlotlyLiveView:
    """The Plotly drawing of a ``LivePlot``: a ``FigureWidget`` updated in place in a notebook, else a
    plain ``Figure`` shown once at ``"end"``."""

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
        cls = _trace_cls(go, map=m)
        traces = _base_traces(go, self.xy, self.problem.depot, self.problem.labels, map=m)
        traces.append(
            cls(mode="lines", line={"width": 1, "color": CURRENT_COLOR}, name="current", hoverinfo="skip")
        )
        traces.append(
            cls(mode="lines", line={"width": 3, "color": BEST_COLOR}, name="best", hoverinfo="skip")
        )
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

    def update(self, event: Any) -> None:
        self._set(event)

    def finish(self, event: Any) -> None:
        self._set(event, final=True)
        if not self.widget:
            self.fig.show()

    # ----- helpers
    def _title(self, event: Any) -> str:
        from ._live import status_line

        return status_line(self.owner.title or str(event.solver), event)

    def _set(self, event: Any, *, final: bool = False) -> None:
        m = self.owner.map
        ctx = self.fig.batch_update() if self.widget else nullcontext()
        with ctx:
            current, best = self.fig.data[2], self.fig.data[3]
            if final or event.tour is None:
                _assign(current, _polyline(self.xy, [], map=m))
            else:
                idx = route_index(self.problem, event.tour)
                _assign(current, _xy_kwargs(self.xy, idx, map=m))
            if event.best_tour is not None:
                if final:
                    _assign(best, _polyline(self.xy, closed_trips(self.problem, event.best_tour), map=m))
                else:
                    _assign(best, _xy_kwargs(self.xy, route_index(self.problem, event.best_tour), map=m))
            self.fig.layout.title.text = self._title(event)


def _assign(trace: Any, kwargs: dict[str, Any]) -> None:
    for key, value in kwargs.items():
        setattr(trace, key, value)


# --------------------------------------------------------------------------- Recorder.to_plotly
def recorder_figure(rec: Recorder, frames: list[RecordedEvent], coords: Any, *, map: bool) -> Any:
    """A figure with one frame per recorded best tour and a slider (``Recorder.to_plotly``)."""
    go = graph_objects()
    problem = rec.problem
    xy = coords_array(coords, None if problem is None else problem.n)
    depot = 0 if problem is None else problem.depot
    labels = np.arange(xy.shape[0]) if problem is None else problem.labels
    cls = _trace_cls(go, map=map)

    def line(ev: RecordedEvent) -> Any:
        trips = closed_trips(problem, ev.best_tour)
        return cls(
            mode="lines",
            line={"width": 3, "color": BEST_COLOR},
            name="best",
            hoverinfo="skip",
            **_polyline(xy, trips, map=map),
        )

    def label(ev: RecordedEvent) -> str:
        best = f" | best {format_number(ev.best_cost)}" if math.isfinite(ev.best_cost) else ""
        return f"{ev.solver} | iteration {ev.iteration}{best}"

    traces = [*_base_traces(go, xy, depot, labels, map=map), line(frames[0])]
    fig = go.Figure(data=traces, layout=_layout(go, xy, label(frames[0]), map=map, zoom=None))
    fig.frames = [
        go.Frame(data=[line(ev)], traces=[2], name=str(k), layout={"title": {"text": label(ev)}})
        for k, ev in enumerate(frames)
    ]
    frame_args = {
        "frame": {"duration": 80, "redraw": map},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
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
                        "args": [None, {**frame_args, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {**frame_args, "frame": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "iteration ", "visible": True},
                "pad": {"t": 30},
                "steps": [
                    {"label": str(ev.iteration), "method": "animate", "args": [[str(k)], frame_args]}
                    for k, ev in enumerate(frames)
                ],
            }
        ],
    )
    return fig
