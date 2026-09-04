"""``Recorder``: a callback that keeps every event and replays them as an animation or a Plotly slider."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._static import (
    closed_trips,
    colors_for_trips,
    coords_array,
    draw_points,
    draw_trips,
    format_number,
    frame_axes,
    pyplot,
)
from ._static import plot_history as _plot_history

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes

__all__ = ["RecordedEvent", "Recorder"]


@dataclass(frozen=True)
class RecordedEvent:
    """The copy of a progress event a [`Recorder`][skroute.viz.Recorder] keeps.

    Attributes
    ----------
    solver : str
        Class name of the solver.
    stage : {"start", "iteration", "end"}
    iteration : int
        Outer iteration index (0 at start).
    cost : float
        Objective of ``tour`` (``nan`` when unknown).
    best_cost : float
        Best objective so far.
    tour : ndarray or None
        Label-space open giant tour of the current solution (``None`` when the solver had none,
        or when the recorder was built with ``keep_tours=False``).
    best_tour : ndarray or None
        Label-space best-so-far tour (same conventions).
    extra : dict
        Solver-specific facts (``temperature``, ``tenure``, ``generation``...).
    problem : RoutingProblem or None
        The instance the event belongs to (a reference, shared by the events of one run), so a
        recorded event can be drawn with [`plot_route`][skroute.viz.plot_route] like a live one.
    trips : list of ndarray
        ``best_tour`` decoded into closed label trips ``[depot, ..., depot]`` with the problem's
        split rule, as ``RouteEvent.trips``; ``[]`` without a tour (or a problem).
    route : ndarray or None
        ``best_tour`` as driven — depot, trip 1, depot, ..., depot — as ``RouteEvent.route``;
        ``None`` without a tour.
    """

    solver: str
    stage: str
    iteration: int
    cost: float
    best_cost: float
    tour: np.ndarray | None
    best_tour: np.ndarray | None
    extra: dict[str, Any] = field(default_factory=dict)
    problem: Any = None

    @property
    def trips(self) -> list[np.ndarray]:
        """``best_tour`` decoded into closed label trips ``[depot, ..., depot]``; ``[]`` without a tour."""
        if self.best_tour is None or self.problem is None:
            return []
        labels = np.asarray(self.problem.labels)
        return [labels[trip] for trip in closed_trips(self.problem, self.best_tour)]

    @property
    def route(self) -> np.ndarray | None:
        """``best_tour`` as driven — depot, trip 1, depot, ..., depot — or ``None`` without a tour."""
        trips = self.trips
        if not trips:
            return None
        return np.concatenate([trips[0]] + [t[1:] for t in trips[1:]])


def _copy_tour(tour: Any) -> np.ndarray | None:
    return None if tour is None else np.array(tour, copy=True)


class Recorder:
    """Record the events of a run to replay them later: an animation, a Plotly slider or a history plot.

    Pass an instance as ``fit(..., callback=rec)``. Nothing is drawn during the run; the events
    are copied into ``events`` (label-space tours included unless ``keep_tours=False``) and the
    run can then be replayed with ``animate`` (a matplotlib ``FuncAnimation``, savable as a GIF or
    MP4), ``to_plotly`` (a figure with frames and a slider, optionally on OpenStreetMap tiles) or
    summarised with ``plot_history``.

    A recorder accumulates: handed to a second ``fit`` it appends that run's events (each event
    keeps a reference to its own problem, and ``every`` counts from the run's first iteration
    again). The replays draw every kept event on one set of coordinates, so build a new recorder
    per instance you want to replay.

    Parameters
    ----------
    every : int >= 1, default 1
        Keep every ``every``-th ``"iteration"`` event (``"start"`` and ``"end"`` are always kept).
    keep_tours : bool, default True
        Store ``tour`` and ``best_tour``; ``False`` keeps costs only (``animate`` and ``to_plotly``
        then have nothing to draw and raise ``ValueError``).

    Attributes
    ----------
    events : list of RecordedEvent
        The kept events, in order.
    problem : RoutingProblem or None
        The instance of the latest run recorded (every event also carries its own).
    costs : ndarray of shape (n_events,), float64
        ``cost`` of every kept event (``nan`` when the solver reported none).
    best_costs : ndarray of shape (n_events,), float64
        ``best_cost`` of every kept event, non-increasing.
    iterations : ndarray of shape (n_events,), int64
        ``iteration`` of every kept event.
    n_frames : int
        Kept events that carry a best tour — the frames ``animate`` and ``to_plotly`` draw.

    Examples
    --------
    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_tsp
    >>> from skroute.viz import Recorder
    >>> dj = load_tsp("dj38")
    >>> rec = Recorder(every=5)
    >>> len(rec.events), rec.n_frames
    (0, 0)
    >>> ils = IteratedLocalSearch(random_state=0).fit(
    ...     dj.distance_matrix(), labels=dj.labels, callback=rec
    ... )  # doctest: +SKIP
    >>> anim = rec.animate(dj.coords)  # doctest: +SKIP
    >>> anim.save("dj38.gif", writer="pillow", dpi=80)  # doctest: +SKIP
    >>> rec.to_plotly(dj.coords).show()  # doctest: +SKIP
    """

    def __init__(self, every: int = 1, keep_tours: bool = True) -> None:
        if not isinstance(every, int | np.integer) or isinstance(every, bool) or every < 1:
            raise ValueError(f"every must be an int >= 1; got {every!r}")
        self.every = int(every)
        self.keep_tours = bool(keep_tours)
        self.events: list[RecordedEvent] = []
        self.problem: Any = None
        self._n_iterations = 0

    def __call__(self, event: Any) -> None:
        """Store a copy of ``event`` (subject to ``every``); never asks the solver to stop."""
        problem = getattr(event, "problem", None)
        if (
            event.stage == "start"
        ):  # a new run (or a restart forwarded by MultiStart): its instance, ``every`` from 1
            self.problem = problem
            self._n_iterations = 0
        elif self.problem is None:
            self.problem = problem
        if problem is None:
            problem = self.problem
        if event.stage == "iteration":
            self._n_iterations += 1
            if (self._n_iterations - 1) % self.every:
                return
        keep = self.keep_tours
        self.events.append(
            RecordedEvent(
                solver=str(event.solver),
                stage=str(event.stage),
                iteration=int(event.iteration),
                cost=float(event.cost),
                best_cost=float(event.best_cost),
                tour=_copy_tour(event.tour) if keep else None,
                best_tour=_copy_tour(event.best_tour) if keep else None,
                extra=dict(event.extra or {}),
                problem=problem,
            )
        )

    # ----- arrays
    @property
    def costs(self) -> np.ndarray:
        """``cost`` of every kept event (``nan`` when unknown)."""
        return np.array([e.cost for e in self.events], dtype=np.float64)

    @property
    def best_costs(self) -> np.ndarray:
        """``best_cost`` of every kept event."""
        return np.array([e.best_cost for e in self.events], dtype=np.float64)

    @property
    def iterations(self) -> np.ndarray:
        """``iteration`` of every kept event."""
        return np.array([e.iteration for e in self.events], dtype=np.int64)

    @property
    def n_frames(self) -> int:
        """Kept events with a best tour: the number of frames of ``animate`` and ``to_plotly``."""
        return sum(1 for e in self.events if e.best_tour is not None)

    def _frames(self) -> list[RecordedEvent]:
        frames = [e for e in self.events if e.best_tour is not None]
        if not frames:
            hint = " (built with keep_tours=False)" if not self.keep_tours else ""
            raise ValueError(f"nothing to draw: no recorded event carries a best tour{hint}")
        return frames

    # ----- replays
    def animate(
        self,
        coords: Any,
        *,
        interval: int = 60,
        figsize: tuple[float, float] = (7, 7),
        trip_colors: bool = True,
    ) -> FuncAnimation:
        """Replay the best tour of every kept event as a matplotlib animation.

        Parameters
        ----------
        coords : (n, 2) array-like
            Node positions in matrix row order (x, y).
        interval : int, default 60
            Milliseconds between frames.
        figsize : tuple of two floats, default (7, 7)
            Figure size in inches.
        trip_colors : bool, default True
            One colour per trip.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            One frame per kept event with a best tour (``n_frames``). Save it with
            ``anim.save("run.gif", writer="pillow", dpi=80)`` or ``anim.save("run.mp4")`` (ffmpeg);
            in a notebook show it with ``IPython.display.HTML(anim.to_jshtml())``. In a script the
            figure stays open, so ``plt.show()`` plays the animation; under ``%matplotlib inline``
            it is closed right away (the animation keeps its own reference), so the cell does not
            end with a stray empty picture.
        """
        plt = pyplot()
        from matplotlib.animation import FuncAnimation

        from ._live import inline_backend

        frames = self._frames()
        problem = self.problem
        xy = coords_array(coords, None if problem is None else problem.n)
        fig, ax = plt.subplots(figsize=figsize)
        depot = 0 if problem is None else problem.depot
        labels = np.arange(xy.shape[0]) if problem is None else problem.labels
        draw_points(ax, xy, depot, labels, show_depot=True, show_labels=False)
        frame_axes(ax, xy)
        lines: list[Any] = []

        def draw(k: int) -> list[Any]:
            for line in lines:
                line.remove()
            lines.clear()
            ev = frames[k]
            trips = closed_trips(problem if ev.problem is None else ev.problem, ev.best_tour)
            lines.extend(draw_trips(ax, xy, trips, colors_for_trips(len(trips), trip_colors), linewidth=2.0))
            parts = [ev.solver, f"iteration {ev.iteration}"]
            if math.isfinite(ev.best_cost):
                parts.append(f"best {format_number(ev.best_cost)}")
            ax.set_title(" | ".join(parts), fontsize=10)
            return lines

        anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval, blit=False, repeat=True)
        if inline_backend():
            plt.close(fig)  # else the kernel displays the empty figure again when the cell ends
        return anim

    def to_plotly(self, coords: Any, *, map: bool = False) -> Any:
        """Replay the run as a Plotly figure with one frame per kept event and a slider.

        Parameters
        ----------
        coords : (n, 2) array-like
            Node positions in matrix row order: (x, y), or (latitude, longitude) when ``map=True``.
        map : bool, default False
            Draw on OpenStreetMap tiles (``Scattermap``) instead of plain axes.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Nodes, depot and the best route of the first frame as base traces; ``fig.frames`` holds
            one frame per kept event; a slider and Play/Pause buttons drive them.
        """
        from ._map import recorder_figure

        return recorder_figure(self, self._frames(), coords, map=map)

    def plot_history(self, ax: Axes | None = None) -> Axes:
        """Best-so-far cost of the kept iteration events (see [`plot_history`][skroute.viz.plot_history])."""
        return _plot_history(self.events, ax=ax)

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"Recorder(every={self.every}, keep_tours={self.keep_tours}, n_events={len(self.events)})"
