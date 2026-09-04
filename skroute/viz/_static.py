"""Static pictures: ``plot_route`` and ``plot_history``, plus the helpers the live tools share.

matplotlib is imported inside the functions (never at module level), so ``import skroute.viz``
costs nothing and a missing matplotlib raises a clear ``ImportError`` at the first call.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from ..base import BaseRouter
from ..problem import RoutingProblem
from ..utils.validation import check_is_fitted

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

__all__ = ["plot_history", "plot_route"]

MATPLOTLIB_HINT = "matplotlib is required for skroute.viz: pip install scikit-route[viz]"
PLOTLY_HINT = "plotly is required for skroute.viz maps: pip install scikit-route[viz-map]"

POINT_COLOR = "#404040"
DEPOT_COLOR = "#c0392b"
CURRENT_COLOR = "#bfbfbf"
BEST_COLOR = "#1f5673"


# --------------------------------------------------------------------------- optional imports
def pyplot() -> Any:
    """``matplotlib.pyplot``, imported on first use.

    Raises
    ------
    ImportError
        With the message ``"matplotlib is required for skroute.viz: pip install scikit-route[viz]"``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(MATPLOTLIB_HINT) from exc
    return plt


def graph_objects() -> Any:
    """``plotly.graph_objects``, imported on first use (``ImportError`` names the ``viz-map`` extra)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(PLOTLY_HINT) from exc
    return go


# --------------------------------------------------------------------------- geometry helpers
def coords_array(coords: Any, n: int | None = None) -> np.ndarray:
    """``coords`` as a float64 ``(n, 2)`` array; ``ValueError`` on any other shape or row count."""
    try:
        xy = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"coords must be an (n, 2) array of x, y positions; got {type(coords).__name__}"
        ) from exc
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords must be an (n, 2) array of x, y positions; got shape {xy.shape}")
    if n is not None and xy.shape[0] != n:
        raise ValueError(f"coords has {xy.shape[0]} rows but the problem has {n} nodes")
    if not np.all(np.isfinite(xy)):
        raise ValueError("coords must be finite")
    return xy


def closed_trips(problem: RoutingProblem, tour: Any) -> list[np.ndarray]:
    """Closed index trips ``[depot, ..., depot]`` of a label-space tour, split by the problem's rule.

    ``tour`` may be an open giant tour (``tour_``), a closed route or a multi-trip route
    (``route_``): every occurrence of the depot is dropped and the giant tour is re-decoded.
    """
    idx = problem.to_index_tour(tour)
    starts = problem.trip_starts(idx)
    d = np.array([problem.depot], dtype=np.int64)
    return [np.concatenate((d, idx[a:b], d)) for a, b in pairwise(starts)]


def route_index(problem: RoutingProblem, tour: Any) -> np.ndarray:
    """The decoded route of a label-space tour as one index polyline: depot, trip 1, depot, trip 2, ..."""
    trips = closed_trips(problem, tour)
    return np.concatenate([trips[0]] + [t[1:] for t in trips[1:]])


def trips_from_array(route: Any) -> list[np.ndarray]:
    """Closed trips of a route given as row positions of ``coords`` (the depot is its first entry).

    An open tour ``[d, a, b, c]`` is one trip ``[d, a, b, c, d]``; a route with the depot repeated,
    ``[d, a, b, d, c, d]``, is split at every depot into ``[d, a, b, d]`` and ``[d, c, d]``.
    """
    r = np.asarray(route)
    if r.ndim != 1 or r.size == 0 or r.dtype.kind not in "iu":
        raise ValueError("a route array must be a 1-D integer array of row positions with the depot first")
    r = r.astype(np.int64)
    depot = r[0]
    cuts = np.flatnonzero(r == depot).tolist()
    if cuts[-1] != r.size - 1:
        cuts.append(r.size)  # an open tour: close it at the depot
    trips = []
    for a, b in pairwise(cuts):
        if b - a > 1:  # skip consecutive depots (an empty trip)
            trips.append(np.concatenate((r[a:b], [depot])))
    return trips


def is_event(obj: Any) -> bool:
    """Duck-typed test for a ``RouteEvent`` (or the recorder's copy of one)."""
    return all(hasattr(obj, name) for name in ("stage", "best_tour", "tour", "best_cost", "solver"))


def resolve(obj: Any, coords: Any) -> tuple[np.ndarray, list[np.ndarray], int, np.ndarray, str | None, float]:
    """What to draw for ``obj``: ``(xy, closed index trips, depot index, labels, name, cost)``.

    ``obj`` is a fitted estimator, a route event or a route array (row positions; needs ``coords``).
    """
    if isinstance(obj, BaseRouter):
        check_is_fitted(obj)
        problem = obj.problem_
        xy = _problem_coords(problem, coords)
        return (
            xy,
            closed_trips(problem, obj.tour_),
            problem.depot,
            problem.labels,
            type(obj).__name__,
            obj.cost_,
        )
    if is_event(obj):
        ev_problem: Any = getattr(obj, "problem", None)
        if ev_problem is None:
            raise ValueError("the event carries no RoutingProblem; nothing to decode")
        xy = _problem_coords(ev_problem, coords)
        tour = obj.best_tour if obj.best_tour is not None else obj.tour
        trips = closed_trips(ev_problem, tour) if tour is not None else []
        cost = float(obj.best_cost) if obj.best_tour is not None else float(obj.cost)
        return xy, trips, ev_problem.depot, ev_problem.labels, str(obj.solver), cost
    if coords is None:
        raise ValueError("a route array needs coords=: pass the (n, 2) positions its entries index")
    trips = trips_from_array(obj)
    if not trips:
        raise ValueError("the route visits no node besides the depot; nothing to draw")
    xy = coords_array(coords)
    if any(int(t.min()) < 0 or int(t.max()) >= xy.shape[0] for t in trips):
        raise ValueError("the route indexes rows beyond the end of coords (or negative rows)")
    return xy, trips, int(trips[0][0]), np.arange(xy.shape[0]), None, math.nan


def _problem_coords(problem: RoutingProblem, coords: Any) -> np.ndarray:
    if coords is None:
        if problem.coords is None:
            raise ValueError("no coordinates to draw: pass coords= (or fit with coords=)")
        coords = problem.coords
    return coords_array(coords, problem.n)


# --------------------------------------------------------------------------- drawing helpers
def colors_for_trips(n_trips: int, per_trip: bool, color: Any = None) -> list[Any]:
    """One colour per trip from the ``tab10`` cycle, or the same ``color`` for all."""
    if not per_trip or color is not None:
        return [BEST_COLOR if color is None else color] * n_trips
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    return [cmap(k % 10) for k in range(n_trips)]


def draw_points(
    ax: Axes, xy: np.ndarray, depot: int, labels: Any, *, show_depot: bool, show_labels: bool
) -> None:
    """Scatter the nodes; the depot as a larger star; optional text labels."""
    mask = np.ones(xy.shape[0], dtype=bool)
    if show_depot:
        mask[depot] = False
    ax.scatter(xy[mask, 0], xy[mask, 1], s=18, color=POINT_COLOR, zorder=3, linewidths=0)
    if show_depot:
        ax.scatter(xy[depot, 0], xy[depot, 1], s=160, marker="*", color=DEPOT_COLOR, zorder=4, label="depot")
    if show_labels:
        for i in range(xy.shape[0]):
            ax.annotate(str(labels[i]), xy[i], xytext=(3, 3), textcoords="offset points", fontsize=7)


def draw_trips(
    ax: Axes, xy: np.ndarray, trips: list[np.ndarray], colors: list[Any], **line_kwargs: Any
) -> list[Line2D]:
    """One polyline per closed trip; returns the ``Line2D`` artists."""
    kw: dict[str, Any] = {"linewidth": 1.6, "zorder": 2, "solid_capstyle": "round"}
    kw.update(line_kwargs)
    lines = []
    for trip, color in zip(trips, colors, strict=True):
        (line,) = ax.plot(xy[trip, 0], xy[trip, 1], color=color, **kw)
        lines.append(line)
    return lines


def frame_axes(ax: Axes, xy: np.ndarray) -> None:
    """Equal aspect in a square data window (5 % margin), no ticks: the picture is the route, not a chart."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    half = 0.55 * max(float((hi - lo).max()), 1e-9)
    cx, cy = (lo + hi) / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def format_number(x: Any) -> str:
    """Compact text of a scalar for titles (``27603``, ``446.59``, ``1.2e+06``)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int | np.integer):
        return str(int(x))
    if isinstance(x, float | np.floating):
        return f"{float(x):.6g}"
    return str(x)


def route_title(name: str | None, cost: float, n_trips: int) -> str:
    parts = [] if name is None else [name]
    if math.isfinite(cost):
        parts.append(f"cost {format_number(cost)}")
    if n_trips > 1:
        parts.append(f"{n_trips} trips")
    return " | ".join(parts)


# --------------------------------------------------------------------------- public functions
def plot_route(
    obj: Any,
    coords: Any = None,
    *,
    ax: Axes | None = None,
    labels: bool = False,
    depot: bool = True,
    trip_colors: bool = True,
    **line_kwargs: Any,
) -> Axes:
    """Draw the nodes and the closed route of a solution, one colour per trip.

    Parameters
    ----------
    obj : fitted estimator, RouteEvent or array of row positions
        A fitted solver (its ``tour_`` is decoded with its ``problem_``), a progress event (its
        ``best_tour``, or ``tour`` when there is no best yet), or a route array whose entries are
        **row positions of** ``coords`` — the ``tour_``/``route_`` of a solver fitted without
        ``labels=``. In the array the depot is the first entry and a repeated depot separates
        trips (``[0, 3, 1, 0, 2, 0]`` is two trips).
    coords : (n, 2) array-like, optional
        Positions in matrix row order; column 0 is drawn as x and column 1 as y. Default: the
        coordinates the problem carries (``fit(..., coords=)``). For ``(latitude, longitude)``
        data pass ``coords[:, ::-1]`` or use [`plot_route_map`][skroute.viz.plot_route_map].
    ax : matplotlib Axes, optional
        Draw into these axes; a new 7 x 7 inch figure otherwise.
    labels : bool, default False
        Write each node's label next to it.
    depot : bool, default True
        Mark the depot with a star (a plain point otherwise).
    trip_colors : bool, default True
        One colour per trip; ``False`` (or a ``color=`` keyword) draws every trip alike.
    **line_kwargs
        Forwarded to ``Axes.plot`` for the route lines (``linewidth``, ``alpha``, ``color``...).

    Returns
    -------
    ax : matplotlib Axes
        Equal aspect, no ticks, one ``Line2D`` per trip; the title names the solver and the cost
        when they are known.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_tsp
    >>> from skroute.viz import plot_route
    >>> dj = load_tsp("dj38")
    >>> ils = IteratedLocalSearch(random_state=0).fit(
    ...     dj.distance_matrix(), labels=dj.labels, coords=dj.coords
    ... )
    >>> ax = plot_route(ils)
    >>> len(ax.lines), len(ax.lines[0].get_xdata())  # one trip, closed at the depot
    (1, 39)
    >>> ax.get_title().startswith("IteratedLocalSearch | cost ")
    True

    A route array with its coordinates (no estimator needed):

    >>> import numpy as np
    >>> xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> ax = plot_route([0, 1, 0, 2, 3], xy)  # two trips: 0-1-0 and 0-2-3-0
    >>> len(ax.lines)
    2
    """
    plt = pyplot()
    xy, trips, depot_idx, node_labels, name, cost = resolve(obj, coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    draw_points(ax, xy, depot_idx, node_labels, show_depot=depot, show_labels=labels)
    colors = colors_for_trips(len(trips), trip_colors, line_kwargs.pop("color", None))
    draw_trips(ax, xy, trips, colors, **line_kwargs)
    frame_axes(ax, xy)
    title = route_title(name, cost, len(trips))
    if title:
        ax.set_title(title)
    return ax


def plot_history(obj_or_events: Any, ax: Axes | None = None) -> Axes:
    """Draw the best-so-far cost per outer iteration.

    Parameters
    ----------
    obj_or_events : fitted iterative estimator, Recorder or sequence of events
        An estimator with ``history_``, a [`Recorder`][skroute.viz.Recorder], or a sequence of
        progress events (the ``"iteration"`` events' ``best_cost`` is drawn against ``iteration``;
        a ``nan`` best — MILP before its first integral solution — is skipped). Events forwarded
        by ``MultiStart`` (``extra["restart"]``) restart their iteration count at every restart:
        they are drawn against the iteration counted across the restarts, and the title names
        the ensemble.
    ax : matplotlib Axes, optional
        Draw into these axes; a new figure otherwise.

    Returns
    -------
    ax : matplotlib Axes
        A step plot (``steps-post``) with the iteration on x and the best cost on y.

    Raises
    ------
    NotFittedError
        For an estimator that was not fitted.
    ValueError
        For a fitted estimator without ``history_`` (a construction or exact solver).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from skroute import SimulatedAnnealing
    >>> from skroute.datasets import load_tsp
    >>> from skroute.viz import plot_history
    >>> wi = load_tsp("wi29")
    >>> sa = SimulatedAnnealing(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
    >>> ax = plot_history(sa)
    >>> len(ax.lines[0].get_ydata()) == sa.n_iter_
    True
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('Iteration', 'Best cost')
    """
    plt = pyplot()
    xlabel = "Iteration"
    if isinstance(obj_or_events, BaseRouter):
        check_is_fitted(obj_or_events)
        if not hasattr(obj_or_events, "history_"):
            raise ValueError(
                f"{type(obj_or_events).__name__} has no history_: it is not an iterative solver "
                "(record its events with a Recorder to plot them)"
            )
    if hasattr(obj_or_events, "history_"):
        y = np.asarray(obj_or_events.history_, dtype=np.float64)
        x = np.arange(1, y.size + 1)
        name: str | None = type(obj_or_events).__name__
    else:
        events = list(getattr(obj_or_events, "events", obj_or_events))
        iters = [e for e in events if e.stage == "iteration" and math.isfinite(float(e.best_cost))]
        y = np.array([float(e.best_cost) for e in iters], dtype=np.float64)
        x, nested = _iteration_axis(iters)
        if nested:
            xlabel = "Iteration (all restarts)"
        outer = [e for e in events if e.stage == "start" and "restart" not in (e.extra or {})]
        first = outer[0] if outer else (iters[0] if iters else None)
        name = None if first is None else str(first.solver)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, drawstyle="steps-post", color=BEST_COLOR, linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best cost")
    ax.set_title(f"{name}: best-so-far cost" if name else "Best-so-far cost")
    ax.grid(True, alpha=0.3)
    return ax


def _iteration_axis(iters: list[Any]) -> tuple[np.ndarray, bool]:
    """The x of ``plot_history`` for iteration events: ``iteration`` itself, or — when the events come
    from the restarts of a ``MultiStart`` (``extra["restart"]``), each restart counting from 1 again —
    the iteration counted across the restarts, so the axis never runs backwards."""
    x = np.array([int(e.iteration) for e in iters], dtype=np.int64)
    if not any("restart" in (e.extra or {}) for e in iters):
        return x, False
    offset, previous = 0, None
    for k, e in enumerate(iters):
        restart = (e.extra or {}).get("restart")
        if k and restart != previous:
            offset = int(x[k - 1])
        previous = restart
        x[k] += offset
    return x, True
