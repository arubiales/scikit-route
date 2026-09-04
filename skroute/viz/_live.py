"""``LivePlot``: a callback that redraws the current and best tours while ``fit`` runs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._static import (
    BEST_COLOR,
    CURRENT_COLOR,
    closed_trips,
    colors_for_trips,
    coords_array,
    draw_points,
    draw_trips,
    format_number,
    frame_axes,
    pyplot,
    route_index,
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


def status_line(name: str, event: Any) -> str:
    """``"Solver | iteration 12 | cost 27811 | best 27603 | temperature 1.2"`` from an event."""
    parts = [name, f"iteration {int(event.iteration)}"]
    cost, best = float(event.cost), float(event.best_cost)
    if math.isfinite(cost):
        parts.append(f"cost {format_number(cost)}")
    if math.isfinite(best):
        parts.append(f"best {format_number(best)}")
    for key, value in (event.extra or {}).items():
        if isinstance(value, bool | int | float | str | np.integer | np.floating):
            parts.append(f"{key} {format_number(value)}")
    return " | ".join(parts)


# --------------------------------------------------------------------------- the callback
class LivePlot:
    """Watch a solver work: a callback that redraws the current and best tours as ``fit`` runs.

    Pass an instance as ``fit(..., callback=live)``. On the ``"start"`` event it creates the
    figure (the nodes, the depot as a star); on every ``every``-th ``"iteration"`` event it
    redraws the **current** tour as a thin light line and the **best** tour so far as a thick
    line, and refreshes the title with the solver, the iteration, the current and best costs and
    the solver-specific facts of ``event.extra`` (temperature, tenure, generation...); on
    ``"end"`` it draws the final best route with one colour per trip. It never blocks and never
    calls ``plt.show()`` — in a script the window appears through ``plt.pause``, so add
    ``plt.show()`` after ``fit`` to keep it open; in Jupyter the figure is refreshed in place.

    Parameters
    ----------
    coords : (n, 2) array-like
        Node positions in matrix row order; column 0 is x and column 1 is y (with
        ``map=True``: latitude, longitude).
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        ``"matplotlib"`` draws in a matplotlib figure (interactive backends show a window;
        headless ``Agg`` works silently); ``"plotly"`` draws in a ``FigureWidget`` updated in
        place inside a notebook, or in a plain ``Figure`` shown once at ``"end"`` elsewhere.
    map : bool, default False
        Draw on OpenStreetMap tiles (Plotly ``Scattermap``; requires ``backend="plotly"`` and
        coordinates as ``(latitude, longitude)``).
    every : int >= 1, default 1
        Redraw on every ``every``-th iteration event (a redraw costs milliseconds; a fast solver
        emits thousands of iterations per second).
    figsize : tuple of two floats, default (7, 7)
        Size of the matplotlib figure in inches.
    title : str, optional
        Text that replaces the solver's name in the status line.
    pause : float, default 0.001
        Seconds handed to ``plt.pause`` after each redraw (interactive matplotlib backends only).

    Attributes
    ----------
    fig : matplotlib Figure, plotly Figure or None
        Created on the first event.
    ax : matplotlib Axes or None
        The axes (matplotlib backend only).
    n_events : int
        Events received so far.
    n_redraws : int
        Times the picture was refreshed (start, the kept iterations and end).

    Notes
    -----
    Redraws happen on the thread that runs ``fit``: matplotlib is not thread-safe, which is why
    ``MultiStart`` forwards the callback only with ``n_jobs=1``. Calling :meth:`stop` — from another
    cell or thread — makes the callback return ``True`` at the next event, which asks the solver
    to stop after its current iteration (``stop_reason_ == "callback"``).

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
    >>> LivePlot(wi.coords, map=True)  # tiles need Plotly
    Traceback (most recent call last):
        ...
    ValueError: map=True needs backend="plotly": OpenStreetMap tiles are drawn with Plotly's Scattermap
    """

    def __init__(
        self,
        coords: Any,
        *,
        backend: str = "matplotlib",
        map: bool = False,
        every: int = 1,
        figsize: tuple[float, float] = (7, 7),
        title: str | None = None,
        pause: float = 0.001,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}; got {backend!r}")
        if map and backend != "plotly":
            raise ValueError(
                'map=True needs backend="plotly": OpenStreetMap tiles are drawn with Plotly\'s Scattermap'
            )
        if not isinstance(every, int | np.integer) or isinstance(every, bool) or every < 1:
            raise ValueError(f"every must be an int >= 1; got {every!r}")
        if pause < 0:
            raise ValueError(f"pause must be >= 0; got {pause!r}")
        self.coords = coords_array(coords)
        self.backend = backend
        self.map = map
        self.every = int(every)
        self.figsize = figsize
        self.title = title
        self.pause = pause
        self.fig: Any = None
        self.ax: Axes | None = None
        self.n_events = 0
        self.n_redraws = 0
        self._stop = False
        self._n_iterations = 0
        self._view: _MatplotlibView | Any = None

    def stop(self) -> None:
        """Ask the solver to stop: the callback returns ``True`` at its next event."""
        self._stop = True

    def __call__(self, event: Any) -> bool:
        """Handle one event; returns ``True`` once :meth:`stop` was called."""
        self.n_events += 1
        stage = event.stage
        if self._view is None:  # "start", or a callback attached to a run already in progress
            self._view = self._make_view(event)
            self._view.start(event)
            self.fig, self.ax = self._view.fig, self._view.ax
            self.n_redraws += 1
            if stage == "start":
                return self._stop
        if stage == "iteration":
            self._n_iterations += 1
            if (self._n_iterations - 1) % self.every == 0:
                self._view.update(event)
                self.n_redraws += 1
        elif stage == "end":
            self._view.finish(event)
            self.n_redraws += 1
        return self._stop

    def _make_view(self, event: Any) -> Any:
        if self.backend == "plotly":
            from ._map import PlotlyLiveView

            return PlotlyLiveView(self, event.problem)
        return _MatplotlibView(self, event.problem)


class _MatplotlibView:
    """The matplotlib drawing of a ``LivePlot``: two persistent lines updated in place."""

    def __init__(self, owner: LivePlot, problem: Any) -> None:
        self.owner = owner
        self.problem = problem
        self.xy = coords_array(owner.coords, problem.n)
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.final_lines: list[Any] = []
        self.current: Any = None
        self.best: Any = None
        self.notebook = in_notebook()
        self.inline = False
        self.interactive = False

    # ----- stages
    def start(self, event: Any) -> None:
        plt = pyplot()
        import matplotlib

        self.inline = self.notebook and "inline" in matplotlib.get_backend().lower()
        self.interactive = not self.notebook and backend_is_interactive()
        self.fig, self.ax = plt.subplots(figsize=self.owner.figsize)
        draw_points(
            self.ax, self.xy, self.problem.depot, self.problem.labels, show_depot=True, show_labels=False
        )
        (self.current,) = self.ax.plot([], [], color=CURRENT_COLOR, linewidth=0.9, zorder=1)
        (self.best,) = self.ax.plot([], [], color=BEST_COLOR, linewidth=2.2, zorder=2, solid_capstyle="round")
        frame_axes(self.ax, self.xy)
        self._set_tours(event)
        self._set_title(event)
        self._flush()

    def update(self, event: Any) -> None:
        self._set_tours(event)
        self._set_title(event)
        self._flush()

    def finish(self, event: Any) -> None:
        assert self.ax is not None
        self.current.set_data([], [])
        self.best.set_data([], [])
        for line in self.final_lines:
            line.remove()
        tour = event.best_tour if event.best_tour is not None else event.tour
        trips = closed_trips(self.problem, tour) if tour is not None else []
        self.final_lines = draw_trips(
            self.ax, self.xy, trips, colors_for_trips(len(trips), True), linewidth=2.2
        )
        self._set_title(event)
        self._flush()

    # ----- helpers
    def _polyline(self, tour: Any) -> tuple[np.ndarray, np.ndarray]:
        idx = route_index(self.problem, tour)
        return self.xy[idx, 0], self.xy[idx, 1]

    def _set_tours(self, event: Any) -> None:
        if event.tour is not None:
            self.current.set_data(*self._polyline(event.tour))
        if event.best_tour is not None:
            self.best.set_data(*self._polyline(event.best_tour))

    def _set_title(self, event: Any) -> None:
        assert self.ax is not None
        self.ax.set_title(status_line(self.owner.title or str(event.solver), event), fontsize=10)

    def _flush(self) -> None:
        assert self.fig is not None
        if self.inline:
            display_figure(self.fig, clear=True)
            return
        self.fig.canvas.draw_idle()
        if self.notebook:
            self.fig.canvas.flush_events()
        elif self.interactive:
            pyplot().pause(self.owner.pause)
