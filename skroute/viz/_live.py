"""``LivePlot``: a callback that redraws the current and best tours — and the structure a solver is
building (D31) — while ``fit`` runs."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from ._static import (
    BEST_COLOR,
    CURRENT_COLOR,
    EDGE_COLOR,
    RING_COLOR,
    check_show,
    closed_trips,
    colors_for_trips,
    coords_array,
    draw_points,
    draw_trips,
    event_edges,
    event_ring,
    event_weights,
    format_number,
    frame_axes,
    n_trips_of,
    pyplot,
    ring_xy,
    route_index,
    segments_3d,
    segments_xy,
    trail_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["LivePlot"]

_BACKENDS = ("matplotlib", "plotly")


# --------------------------------------------------------------------------- environment probes
def ipython_shell() -> Any:
    """The running IPython shell, or ``None`` outside IPython (or without IPython installed)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def in_notebook() -> bool:
    """Whether the code runs under a Jupyter kernel (a shell with a ``kernel``; the terminal one has none)."""
    shell = ipython_shell()
    return shell is not None and getattr(shell, "kernel", None) is not None


def inline_backend() -> bool:
    """Whether the code runs under a Jupyter kernel with the ``inline`` matplotlib backend."""
    if not in_notebook():
        return False
    import matplotlib

    return "inline" in matplotlib.get_backend().lower()


def display_figure(fig: Any, *, clear: bool) -> None:
    """``IPython.display.display(fig, clear=clear)`` — only called once ``in_notebook()`` is true."""
    from IPython.display import display

    display(fig, clear=clear)


def backend_is_interactive() -> bool:
    """Whether matplotlib's current backend can show a window (so ``plt.pause`` is meaningful)."""
    import matplotlib

    name = matplotlib.get_backend().lower()
    if name.startswith("module://ipympl") or name in {"widget", "ipympl"}:
        return True
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        interactive = {b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    except ImportError:  # matplotlib < 3.9
        from matplotlib import rcsetup

        interactive = {b.lower() for b in rcsetup.interactive_bk}  # type: ignore[attr-defined]
    return name in interactive


# The one solver fact worth a place in the title, by preference; ``restart`` (the context inside a
# MultiStart) is always shown, lists (kick, moves_applied) never, the structures are counted instead.
INFORMATIVE = (
    "temperature",
    "tenure",
    "mean_cost",
    "generation",
    "iteration_best",
    "radius",
    "lower_bound",
    "gain",
    "accepted",
    "n_restarts",
    "n_units",
)
_SCALARS = (bool, int, float, str, np.integer, np.floating, np.bool_)
TITLE_WIDTH = 60  # characters per title line at fontsize 10 in a 6-7 inch figure


def status_parts(name: str, event: Any) -> list[str]:
    """The pieces of a status title: name, restart, iteration, cost, best, trips, one fact, edges, ring."""
    extra = event.extra or {}
    scalars = {k: v for k, v in extra.items() if isinstance(v, _SCALARS)}
    parts = [name]
    if "restart" in scalars:
        parts.append(f"restart {format_number(scalars.pop('restart'))}")
    parts.append(f"iteration {int(event.iteration)}")
    cost, best = float(event.cost), float(event.best_cost)
    if math.isfinite(cost):
        parts.append(f"cost {format_number(cost)}")
    if math.isfinite(best):
        parts.append(f"best {format_number(best)}")
    n_trips = n_trips_of(getattr(event, "problem", None), event.best_tour)
    if n_trips > 1:
        parts.append(f"{n_trips} trips")
    key = next((k for k in INFORMATIVE if k in scalars), next(iter(scalars), None))
    if key is not None:
        parts.append(f"{key} {format_number(scalars[key])}")
    if extra.get("edges") is not None:
        parts.append(f"edges {len(extra['edges'])}")
    if extra.get("ring") is not None:
        parts.append(f"ring {len(extra['ring'])}")
    return parts


def status_line(name: str, event: Any, *, width: int = TITLE_WIDTH, newline: str = "\n") -> str:
    """``"Solver | iteration 12 | cost 27811 | best 27603 | temperature 1.2"`` from an event.

    After the costs come the number of trips of the best tour when it is more than one, the most
    informative scalar fact of ``event.extra`` (``temperature``, ``tenure``, ``mean_cost``,
    ``radius``... — one, by the order of ``INFORMATIVE``; ``restart`` follows the name inside a
    MultiStart) and the sizes of the D31 structures — ``edges 37``, ``ring 304``. The pieces are
    packed into lines of at most ``width`` characters, joined by ``newline``, so the title fits
    the figure.
    """
    lines: list[str] = []
    current = ""
    for part in status_parts(name, event):
        candidate = part if not current else f"{current} | {part}"
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    return newline.join(lines)


# --------------------------------------------------------------------------- the callback
class LivePlot:
    """Watch a solver work: a callback that redraws the current and best tours as ``fit`` runs.

    Pass an instance as ``fit(..., callback=live)``. On the ``"start"`` event it creates the
    figure (the nodes, the depot as a star); on every ``every``-th ``"iteration"`` event it
    redraws the **current** tour as a thin light line and the **best** tour so far as a thick
    line, and refreshes the title with the solver, the iteration, the current and best costs and
    one solver-specific fact of ``event.extra`` — the most informative scalar (temperature,
    tenure, mean_cost, generation, radius...; ``restart`` always); on ``"end"`` it draws the final
    best route with one colour per trip. It never blocks and never
    calls ``plt.show()`` — in a script the window appears through ``plt.pause``, so add
    ``plt.show()`` after ``fit`` to keep it open; in Jupyter the figure is refreshed in place.

    The three optional ``extra`` keys of D31 are drawn whenever an event carries them:
    ``extra["edges"]`` — the partial structure a solver holds (the growing path of a nearest
    neighbour, the partial cycle of an insertion, the LP support of MILP) — as orange segments,
    which with ``tour=None`` *is* the picture of the route being built; ``extra["edge_weights"]``
    turns those segments into pheromone trails drawn under the tours, wider and more opaque the
    stronger the trail; ``extra["ring"]`` — SOM's neurons — as a closed teal polyline with small
    markers. An event without a key clears that drawing.

    One instance can watch several fits: a ``"start"`` that begins a new run (after the previous
    ``"end"``) opens a fresh figure and forgets a pending ``stop()``; the nested ``"start"`` and
    ``"end"`` of the restarts a ``MultiStart(n_jobs=1)`` forwards reset the lines in place, so
    every restart is drawn from scratch and its final route stays only until the next one begins.
    In a script the figure is left open after ``fit`` (``plt.close(live.fig)`` when watching many
    runs); under ``%matplotlib inline`` it is closed after the final display, so the cell shows
    the last picture once.

    Parameters
    ----------
    coords : (n, 2) array-like
        Node positions in matrix row order; column 0 is x and column 1 is y (with
        ``map=True``: latitude, longitude).
    backend : {"matplotlib", "plotly"}, optional
        ``"matplotlib"`` (the default) draws in a matplotlib figure (interactive backends show a
        window; headless ``Agg`` works silently: the artists are updated on every kept iteration
        and rendered once at ``"end"``); ``"plotly"`` — the default with ``map=True`` — draws in a
        ``FigureWidget`` updated in place inside a notebook, or in a plain ``Figure`` shown once
        at ``"end"`` elsewhere.
    map : bool, default False
        Draw on OpenStreetMap tiles (Plotly ``Scattermap``, so the plotly backend is selected;
        ``backend="matplotlib"`` with ``map=True`` is refused) with the coordinates as
        ``(latitude, longitude)``.
    every : int >= 1, default 1
        Redraw on every ``every``-th iteration event (a redraw costs milliseconds; a fast solver
        emits thousands of iterations per second).
    show : {"both", "best", "current"}, default "both"
        Which tours to draw during the run: the current attempt (thin) and the best so far
        (thick), only the best, or only the current. The final route at ``"end"`` and the D31
        structures are drawn regardless.
    trail : int >= 0, default 0
        Keep the last ``trail`` current tours on screen, fading with age (a sense of where the
        search has been); ``0`` draws the current tour alone.
    figsize : tuple of two floats, default (7, 7)
        Size of the matplotlib figure in inches.
    title : str, optional
        Text that replaces the solver's name in the status line.
    pause : float, default 0.001
        Seconds handed to ``plt.pause`` after each redraw (interactive matplotlib backends only).
    trip_colors : bool, default True
        One colour per trip on the final route; ``False`` draws every trip alike.

    Attributes
    ----------
    fig : matplotlib Figure, plotly Figure or None
        Created on the first event.
    ax : matplotlib Axes or None
        The axes (matplotlib backend only).
    n_events : int
        Events received so far.
    n_redraws : int
        Times the picture was refreshed (start, the kept iterations, the start of every nested
        restart and end).

    Notes
    -----
    Redraws happen on the thread that runs ``fit``: matplotlib is not thread-safe, which is why
    ``MultiStart`` forwards the callback only with ``n_jobs=1``. Calling ``stop()`` makes the
    callback return ``True`` at the next event, which asks the solver to stop after its current
    iteration (``stop_reason_ == "callback"``). With a desktop window keep ``fit`` on the main
    thread and call ``stop()`` from a key handler on ``live.fig``; running ``fit`` in a worker
    thread and stopping it from another cell is safe under ``%matplotlib inline`` and with the
    plotly backend, whose redraws are IPython display calls rather than GUI drawing.

    With the matplotlib backend the artists sit in a fixed order: ``ax.lines[0]`` is the current
    tour, ``ax.lines[1]`` the best, ``ax.lines[2]`` the edges, ``ax.lines[3]`` the ring, then the
    ``trail`` fading tours, and the final route's lines are appended at ``"end"``; the pheromone
    trails are a ``LineCollection`` in ``ax.collections``.

    Examples
    --------
    >>> from skroute import SimulatedAnnealing
    >>> from skroute.datasets import load_tsp
    >>> from skroute.viz import LivePlot
    >>> wi = load_tsp("wi29")
    >>> live = LivePlot(wi.coords, every=20)
    >>> live.n_events, live.fig is None  # nothing is drawn (or imported) before the first event
    (0, True)
    >>> sa = SimulatedAnnealing(random_state=0).fit(
    ...     wi.distance_matrix(), labels=wi.labels, callback=live
    ... )  # doctest: +SKIP
    >>> attempts = LivePlot(wi.coords, show="current", trail=5)  # the walker and its five last positions
    >>> LivePlot(wi.coords, map=True).backend  # tiles are drawn by Plotly: map=True selects it
    'plotly'
    >>> LivePlot(wi.coords, backend="matplotlib", map=True)
    Traceback (most recent call last):
        ...
    ValueError: map=True needs backend="plotly": OpenStreetMap tiles are drawn with Plotly's Scattermap
    """

    def __init__(
        self,
        coords: Any,
        *,
        backend: str | None = None,
        map: bool = False,
        every: int = 1,
        show: str = "both",
        trail: int = 0,
        figsize: tuple[float, float] = (7, 7),
        title: str | None = None,
        pause: float = 0.001,
        trip_colors: bool = True,
    ) -> None:
        if backend is None:
            backend = "plotly" if map else "matplotlib"
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}; got {backend!r}")
        if map and backend != "plotly":
            raise ValueError(
                'map=True needs backend="plotly": OpenStreetMap tiles are drawn with Plotly\'s Scattermap'
            )
        if not isinstance(every, int | np.integer) or isinstance(every, bool) or every < 1:
            raise ValueError(f"every must be an int >= 1; got {every!r}")
        if not isinstance(trail, int | np.integer) or isinstance(trail, bool) or trail < 0:
            raise ValueError(f"trail must be an int >= 0; got {trail!r}")
        if pause < 0:
            raise ValueError(f"pause must be >= 0; got {pause!r}")
        self.coords = coords_array(coords)
        self.backend = backend
        self.map = map
        self.every = int(every)
        self.show = check_show(show)
        self.trail = int(trail)
        self.figsize = figsize
        self.title = title
        self.pause = pause
        self.trip_colors = bool(trip_colors)
        self.fig: Any = None
        self.ax: Axes | None = None
        self.n_events = 0
        self.n_redraws = 0
        self._stop = False
        self._n_iterations = 0
        self._depth = 0  # "start" events minus "end" events: > 1 inside the restarts of a MultiStart
        self._view: _MatplotlibView | Any = None

    def stop(self) -> None:
        """Ask the solver to stop: the callback returns ``True`` at its next event (reset by a new run)."""
        self._stop = True

    def __call__(self, event: Any) -> bool:
        """Handle one event; returns ``True`` once ``stop()`` was called."""
        self.n_events += 1
        stage = event.stage
        if stage == "start":
            if self._view is not None and (self._depth == 0 or event.problem is not self._view.problem):
                # A new fit reusing this instance: forget the previous run (a fresh figure is created
                # below; the old one stays open for the caller) and the stop request that ended it.
                self._view, self._depth, self._stop = None, 0, False
            self._n_iterations = 0  # the ``every`` phase starts again with every run
            self._depth += 1
            if self._view is not None:  # a nested run (the next restart of a MultiStart): redraw from scratch
                self._view.restart(event)
                self.n_redraws += 1
                return self._stop
        if self._view is None:  # "start", or a callback attached to a run already in progress
            self._view = self._make_view(event)
            self._view.start(event)  # the figure, with this event's tours, structures and title
            self.fig, self.ax = self._view.fig, self._view.ax
            if stage != "end":  # an "end" still needs its final route below
                self.n_redraws += 1
                if stage == "iteration":
                    self._n_iterations = 1  # attached mid-run: this is the first kept iteration
                return self._stop
        if stage == "iteration":
            self._n_iterations += 1
            if (self._n_iterations - 1) % self.every == 0:
                self._view.update(event)
                self.n_redraws += 1
        elif stage == "end":
            self._depth = max(0, self._depth - 1)
            self._view.finish(event, last=self._depth == 0)
            self.n_redraws += 1
        return self._stop

    def _make_view(self, event: Any) -> Any:
        if self.backend == "plotly":
            from ._map import PlotlyLiveView

            return PlotlyLiveView(self, event.problem)
        return _MatplotlibView(self, event.problem)


class _MatplotlibView:
    """The matplotlib drawing of a ``LivePlot``: persistent artists updated in place.

    ``ax.lines``: current (0), best (1), edges (2), ring (3), then ``owner.trail`` fading copies of
    past current tours; the pheromone trails are a ``LineCollection``. The final route's lines are
    appended at ``finish``. ``silent=True`` (set by ``Recorder.animate``) skips every refresh call,
    so the artists can be driven by a ``FuncAnimation`` instead.
    """

    def __init__(self, owner: LivePlot, problem: Any) -> None:
        self.owner = owner
        self.problem = problem
        self.xy = coords_array(owner.coords, problem.n)
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.final_lines: list[Any] = []
        self.current: Any = None
        self.best: Any = None
        self.edges: Any = None
        self.ring: Any = None
        self.trails: Any = None
        self.fade: list[Any] = []
        self.past: deque[tuple[Any, Any]] = deque(maxlen=max(1, owner.trail))
        self.notebook = in_notebook()
        self.inline = False
        self.interactive = False
        self.silent = False

    # ----- stages
    def start(self, event: Any) -> None:
        plt = pyplot()
        import matplotlib
        from matplotlib.collections import LineCollection

        self.inline = self.notebook and "inline" in matplotlib.get_backend().lower()
        self.interactive = not self.notebook and backend_is_interactive()
        self.fig, self.ax = plt.subplots(figsize=self.owner.figsize)
        draw_points(
            self.ax, self.xy, self.problem.depot, self.problem.labels, show_depot=True, show_labels=False
        )
        (self.current,) = self.ax.plot([], [], color=CURRENT_COLOR, linewidth=0.9, zorder=1)
        (self.best,) = self.ax.plot([], [], color=BEST_COLOR, linewidth=2.2, zorder=2, solid_capstyle="round")
        (self.edges,) = self.ax.plot([], [], color=EDGE_COLOR, linewidth=1.6, zorder=1.5)
        (self.ring,) = self.ax.plot(
            [], [], color=RING_COLOR, linewidth=1.2, marker="o", markersize=2.5, zorder=1.8
        )
        k = self.owner.trail
        for i in range(k):
            (line,) = self.ax.plot(
                [], [], color=CURRENT_COLOR, linewidth=0.9, alpha=0.6 * (k - i) / (k + 1), zorder=0.8
            )
            self.fade.append(line)
        self.trails = LineCollection([], zorder=0.5)
        self.ax.add_collection(self.trails)
        frame_axes(self.ax, self.xy)
        self._set_tours(event)
        self._set_structure(event)
        self._set_title(event)
        self._flush()

    def restart(self, event: Any) -> None:
        """A nested ``"start"`` (the next restart of a MultiStart): the previous restart's route goes."""
        self._clear_final()
        self._clear_live()
        self._set_tours(event)
        self._set_structure(event)
        self._set_title(event)
        self._flush()

    def update(self, event: Any) -> None:
        self._clear_final()  # a route left by an inner "end" never stays under the live lines
        self._set_tours(event)
        self._set_structure(event)
        self._set_title(event)
        self._flush()

    @property
    def shows_frames(self) -> bool:
        """Whether each refresh is seen as it happens (a window, a notebook) or only the last one (Agg)."""
        return self.notebook or self.interactive

    def finish(self, event: Any, *, last: bool = True) -> None:
        """Draw the final route; ``last`` is false for the ``"end"`` of a restart inside a MultiStart."""
        assert self.ax is not None
        self._clear_live()
        self._clear_final()
        self._set_structure(event)  # an "end" that still describes a structure keeps it under the route
        tour = event.best_tour if event.best_tour is not None else event.tour
        trips = closed_trips(self.problem, tour) if tour is not None else []
        self.final_lines = draw_trips(
            self.ax, self.xy, trips, colors_for_trips(len(trips), self.owner.trip_colors), linewidth=2.2
        )
        self._set_title(event)
        self._flush(final=last)

    # ----- helpers
    def _clear_final(self) -> None:
        for line in self.final_lines:
            line.remove()
        self.final_lines = []

    def _clear_live(self) -> None:
        self.current.set_data([], [])
        self.best.set_data([], [])
        self.edges.set_data([], [])
        self.ring.set_data([], [])
        self.trails.set_segments(np.empty((0, 2, 2)))
        self.past.clear()
        for line in self.fade:
            line.set_data([], [])

    def _polyline(self, tour: Any) -> tuple[np.ndarray, np.ndarray]:
        idx = route_index(self.problem, tour)
        return self.xy[idx, 0], self.xy[idx, 1]

    def _set_tours(self, event: Any) -> None:
        show = self.owner.show
        if show != "best" and event.tour is not None:
            if self.fade and len(self.current.get_xdata()):
                self.past.append(self.current.get_data())
                for line, (x, y) in zip(self.fade, reversed(self.past), strict=False):
                    line.set_data(x, y)
            self.current.set_data(*self._polyline(event.tour))
        if show != "current" and event.best_tour is not None:
            self.best.set_data(*self._polyline(event.best_tour))

    def _set_structure(self, event: Any) -> None:
        """Draw (or clear) the D31 extras the event carries: edges, weighted trails and the ring."""
        extra = getattr(event, "extra", None) or {}
        idx = event_edges(self.problem, extra)
        weights = None if idx is None else event_weights(extra, idx.shape[0])
        if idx is None or weights is not None:
            self.edges.set_data([], [])
        else:
            self.edges.set_data(*segments_xy(self.xy, idx))
        if idx is None or weights is None:
            self.trails.set_segments(np.empty((0, 2, 2)))
        else:
            linewidths, colors = trail_style(weights)
            self.trails.set_segments(segments_3d(self.xy, idx))
            self.trails.set_linewidths(linewidths)
            self.trails.set_color(colors)
        ring = event_ring(extra)
        if ring is None:
            self.ring.set_data([], [])
        else:
            self.ring.set_data(*ring_xy(ring))

    def _set_title(self, event: Any) -> None:
        assert self.ax is not None
        self.ax.set_title(status_line(self.owner.title or str(event.solver), event), fontsize=10)

    def _flush(self, *, final: bool = False) -> None:
        assert self.fig is not None
        if self.silent:  # driven by a FuncAnimation (Recorder.animate): the animation refreshes
            return
        if self.inline:
            display_figure(self.fig, clear=True)
            if final:
                # The inline backend displays every open pyplot figure again when the cell ends, so the
                # last picture would appear twice; the Figure object stays reachable as ``LivePlot.fig``.
                pyplot().close(self.fig)
            return
        if self.notebook:  # ipympl and the like: a canvas updated in place
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        elif self.interactive:
            self.fig.canvas.draw_idle()
            pyplot().pause(self.owner.pause)
        elif final:
            # Headless (Agg): nobody sees the frames and draw_idle() is a full ~10 ms raster render
            # there, so the artists are only updated during the run and rendered once at the end.
            self.fig.canvas.draw_idle()
