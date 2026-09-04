"""``Recorder``: a callback that keeps every event — with its D31 structures and a wall-clock stamp — and
replays the run as an animation at time-lapse speed, a live replay, a GIF/MP4 or a Plotly slider."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._static import STRUCTURE_KEYS, check_show, closed_trips, coords_array, pyplot
from ._static import plot_history as _plot_history

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes

    from ._live import LivePlot

__all__ = ["RecordedEvent", "Recorder"]

MIN_DELAY_MS = 10.0  # a replayed frame never flashes faster than this...
MAX_DELAY_MS = 2000.0  # ...nor lingers longer, whatever the recorded gap (a slow cut round, a pause)


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
        Solver-specific facts (``temperature``, ``tenure``, ``generation``...). The D31
        structures are copied: ``edges`` as a list of ``(label, label)`` tuples, ``edge_weights``
        and ``ring`` as float64 arrays (all three dropped with ``keep_tours=False``).
    problem : RoutingProblem or None
        The instance the event belongs to (a reference, shared by the events of one run), so a
        recorded event can be drawn with [`plot_route`][skroute.viz.plot_route] like a live one.
    timestamp : float
        ``time.perf_counter()`` when the event was recorded, in seconds: only the differences
        between two events of one recorder mean anything (they are the gaps a replay scales).
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
    timestamp: float = 0.0

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

    @property
    def drawable(self) -> bool:
        """Whether the event carries anything to draw: a tour, a best tour, edges or a ring."""
        return (
            self.best_tour is not None
            or self.tour is not None
            or self.extra.get("edges") is not None
            or self.extra.get("ring") is not None
        )


def _copy_tour(tour: Any) -> np.ndarray | None:
    return None if tour is None else np.array(tour, copy=True)


def _copy_extra(extra: Any, keep: bool) -> dict[str, Any]:
    """A shallow copy of ``extra`` whose D31 structures are copied too (or dropped when ``keep`` is false)."""
    out = dict(extra or {})
    for key in STRUCTURE_KEYS:
        if key not in out:
            continue
        if not keep:
            del out[key]
            continue
        value = out[key]
        try:
            if key == "edges":
                out[key] = [tuple(pair) for pair in value]
            else:
                out[key] = np.array(value, dtype=np.float64)
        except (TypeError, ValueError):
            pass  # left as received: the drawing reports what is wrong with it
    return out


class Recorder:
    """Record the events of a run to replay them later: live, as an animation, a file or a Plotly slider.

    Pass an instance as ``fit(..., callback=rec)``. Nothing is drawn during the run; the events
    are copied into ``events`` (label-space tours and the D31 structures — ``extra["edges"]``,
    ``extra["edge_weights"]``, ``extra["ring"]`` — included unless ``keep_tours=False``) with a
    wall-clock stamp each, and the run can then be replayed:

    - ``animate`` — a matplotlib ``FuncAnimation`` drawing the current tour thin, the best tour
      thick and the structures, frame by frame in the recorded order, at the recorded pace
      divided by ``speed`` (a time-lapse) or at a fixed ``fps``;
    - ``replay`` — the same events driven through a [`LivePlot`][skroute.viz.LivePlot] at
      ``speed`` times the recorded pace, in a window or a notebook;
    - ``save`` — a GIF (pillow) or an MP4 (ffmpeg) of the animation;
    - ``to_plotly`` — a figure with frames, Play/Pause, a speed menu and a slider, optionally on
      OpenStreetMap tiles;
    - ``plot_history`` — the best-so-far curve of the kept events.

    A recorder accumulates: handed to a second ``fit`` it appends that run's events (each event
    keeps a reference to its own problem, and ``every`` counts from the run's first iteration
    again). The replays draw every kept event on one set of coordinates, so build a new recorder
    per instance you want to replay.

    Parameters
    ----------
    every : int >= 1, default 1
        Keep every ``every``-th ``"iteration"`` event (``"start"`` and ``"end"`` are always kept).
    keep_tours : bool, default True
        Store ``tour``, ``best_tour`` and the D31 structures; ``False`` keeps the costs and the
        scalar facts only (the replays then have nothing to draw and raise ``ValueError``).

    Attributes
    ----------
    events : list of RecordedEvent
        The kept events, in order.
    problem : RoutingProblem or None
        The instance of the latest run recorded (every event also carries its own).
    costs : ndarray of shape (n_events,), float64
        ``cost`` of every kept event (``nan`` when the solver reported none).
    best_costs : ndarray of shape (n_events,), float64
        ``best_cost`` of every kept event, non-increasing (``nan`` while unknown).
    iterations : ndarray of shape (n_events,), int64
        ``iteration`` of every kept event.
    timestamps : ndarray of shape (n_events,), float64
        ``time.perf_counter()`` of every kept event, non-decreasing; ``timestamps -
        timestamps[0]`` is the recorded clock of the run.
    n_frames : int
        Kept events with something to draw — a tour, a best tour, edges or a ring: the frames of
        ``animate``, ``save`` and ``to_plotly``.

    Examples
    --------
    A deterministic descent (three events) recorded and replayed headless:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from skroute import TwoOpt
    >>> from skroute.viz import Recorder
    >>> C = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
    >>> xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> rec = Recorder()
    >>> est = TwoOpt().fit(C, callback=rec)
    >>> [e.stage for e in rec.events], rec.n_frames
    (['start', 'iteration', 'end'], 3)
    >>> bool(np.all(np.diff(rec.timestamps) >= 0))
    True
    >>> rec.frame_delays(speed=1e9).tolist()  # microsecond gaps at 1e9x: clipped to the 10 ms floor
    [10.0, 10.0, 10.0]
    >>> anim = rec.animate(xy, fps=5)  # a FuncAnimation: plt.show() plays it  # doctest: +SKIP
    >>> live = rec.replay(xy, speed=1e9)  # drives a LivePlot through the three events, no waiting
    >>> live.n_events
    3

    A real run, recorded every fifth iteration and saved as a time-lapse GIF:

    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_tsp
    >>> dj = load_tsp("dj38")
    >>> rec = Recorder(every=5)
    >>> ils = IteratedLocalSearch(random_state=0).fit(
    ...     dj.distance_matrix(), labels=dj.labels, coords=dj.coords, callback=rec
    ... )  # doctest: +SKIP
    >>> rec.save("dj38.gif", speed=10)  # coords come from the fit; 10x time-lapse  # doctest: +SKIP
    >>> rec.to_plotly(dj.coords).show()  # Play/Pause, a speed menu and a slider  # doctest: +SKIP
    """

    def __init__(self, every: int = 1, keep_tours: bool = True) -> None:
        if not isinstance(every, int | np.integer) or isinstance(every, bool) or every < 1:
            raise ValueError(f"every must be an int >= 1; got {every!r}")
        self.every = int(every)
        self.keep_tours = bool(keep_tours)
        self.events: list[RecordedEvent] = []
        self.problem: Any = None
        self._n_iterations = 0
        self._last_live: LivePlot | None = None  # the LivePlot behind the latest animate/save (tests)

    def __call__(self, event: Any) -> None:
        """Store a copy of ``event`` (subject to ``every``) and its arrival time; never stops the solver."""
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
                extra=_copy_extra(event.extra, keep),
                problem=problem,
                timestamp=time.perf_counter(),
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
    def timestamps(self) -> np.ndarray:
        """``time.perf_counter()`` of every kept event (seconds; non-decreasing)."""
        return np.array([e.timestamp for e in self.events], dtype=np.float64)

    @property
    def n_frames(self) -> int:
        """Kept events with something to draw: the frames of ``animate``, ``save`` and ``to_plotly``."""
        return sum(1 for e in self.events if e.drawable)

    def _frames(self) -> list[RecordedEvent]:
        frames = [e for e in self.events if e.drawable]
        if not frames:
            hint = " (built with keep_tours=False)" if not self.keep_tours else ""
            raise ValueError(f"nothing to draw: no recorded event carries a tour, edges or a ring{hint}")
        return frames

    def _coords(self, coords: Any) -> np.ndarray:
        """``coords`` validated against the recorded problem; the problem's own coordinates by default."""
        problem = self.problem
        if coords is None:
            if problem is None or problem.coords is None:
                raise ValueError("no coordinates to draw: pass coords= (or fit with coords=)")
            coords = problem.coords
        return coords_array(coords, None if problem is None else problem.n)

    def frame_delays(self, *, speed: float = 1.0) -> np.ndarray:
        """Milliseconds each frame of ``animate`` stays on screen when the run is replayed at ``speed``.

        Frame ``k`` stays for the recorded gap between it and the next drawable event divided by
        ``speed`` — ``speed=10`` is a ten-fold time-lapse — clipped to ``[10, 2000]`` ms so a burst
        of iterations never flashes past and a slow step (a cut round of MILP, a pause) never
        freezes the replay; the last frame keeps the delay of the one before it.

        Parameters
        ----------
        speed : float > 0, default 1.0
            Time-lapse factor over the recorded clock.

        Returns
        -------
        delays : ndarray of shape (n_frames,), float64
            Per-frame delays in milliseconds, in ``[10, 2000]``.
        """
        if not speed > 0:
            raise ValueError(f"speed must be > 0; got {speed!r}")
        frames = self._frames()
        stamps = np.array([f.timestamp for f in frames], dtype=np.float64)
        gaps = np.diff(stamps) * 1000.0 / float(speed)
        last = gaps[-1] if gaps.size else MIN_DELAY_MS
        return np.clip(np.append(gaps, last), MIN_DELAY_MS, MAX_DELAY_MS)

    # ----- replays
    def _player(
        self,
        xy: np.ndarray,
        frames: list[RecordedEvent],
        *,
        show: str,
        trail: int,
        figsize: tuple[float, float],
        trip_colors: bool,
    ) -> tuple[LivePlot, Any]:
        """A silent ``LivePlot`` (its view never refreshes itself) and the function drawing frame ``k``."""
        from ._live import LivePlot, _MatplotlibView

        problem = frames[0].problem if frames[0].problem is not None else self.problem
        if problem is None:
            raise ValueError("the recorded events carry no RoutingProblem; record a fit to replay it")
        live = LivePlot(xy, show=show, trail=trail, figsize=figsize, trip_colors=trip_colors)
        view = _MatplotlibView(live, problem)
        view.silent = True
        view.start(frames[0])
        live.fig, live.ax, live._view = view.fig, view.ax, view
        last_drawn = [-1]

        def draw(k: int) -> None:
            if k == last_drawn[0]:
                return  # a held frame (save's time-lapse repeats): nothing changed
            last_drawn[0] = k
            ev = frames[k]
            if ev.stage == "end":
                view.finish(ev, last=True)
            elif ev.stage == "start":
                view.restart(ev)
            else:
                view.update(ev)

        self._last_live = live
        return live, draw

    def animate(
        self,
        coords: Any = None,
        *,
        show: str = "both",
        interval: float | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        hold: float = 1.0,
        trail: int = 0,
        figsize: tuple[float, float] = (7, 7),
        trip_colors: bool = True,
    ) -> FuncAnimation:
        """Replay the kept events as a matplotlib animation, in the recorded order.

        Every frame is drawn the way [`LivePlot`][skroute.viz.LivePlot] draws a live event: the
        current tour thin, the best tour thick, ``extra["edges"]`` as segments,
        ``extra["edge_weights"]`` as pheromone trails, ``extra["ring"]`` as the SOM ring, the
        status line as the title, and the final route with trip colours at ``"end"``.

        Parameters
        ----------
        coords : (n, 2) array-like, optional
            Node positions in matrix row order (x, y); default: the recorded problem's ``coords``.
        show : {"both", "best", "current"}, default "both"
            Which tours to draw (the structures are always drawn).
        interval : float, optional
            A fixed delay between frames, in milliseconds (takes precedence over ``fps`` and ``speed``).
        fps : float, optional
            A fixed frame rate: every frame lasts ``1000 / fps`` ms (takes precedence over ``speed``).
        speed : float > 0, default 1.0
            Without ``interval`` and ``fps``, the frames follow the recorded clock divided by
            ``speed`` — ``speed=10`` replays a 30 s run in about 3 s — with every delay clipped
            to ``[10, 2000]`` ms (see ``frame_delays``).
        hold : float >= 0, default 1.0
            Seconds the final picture stays before the animation loops.
        trail : int >= 0, default 0
            Fading copies of the last ``trail`` current tours (as ``LivePlot(trail=)``).
        figsize : tuple of two floats, default (7, 7)
            Figure size in inches.
        trip_colors : bool, default True
            One colour per trip on the final route.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            One frame per kept drawable event (``n_frames``). ``plt.show()`` plays it in a script;
            in a notebook show it with ``IPython.display.HTML(anim.to_jshtml())``; ``save`` writes
            it to a file (``anim.save`` works too, at one fixed frame rate). Under ``%matplotlib
            inline`` the figure is closed right away (the animation keeps its own reference), so
            the cell does not end with a stray empty picture.
        """
        plt = pyplot()
        from matplotlib.animation import FuncAnimation

        from ._live import inline_backend

        check_show(show)
        if hold < 0:
            raise ValueError(f"hold must be >= 0; got {hold!r}")
        frames = self._frames()
        xy = self._coords(coords)
        n = len(frames)
        if interval is not None:
            if not interval > 0:
                raise ValueError(f"interval must be > 0 ms; got {interval!r}")
            delays = np.full(n, float(interval))
        elif fps is not None:
            if not fps > 0:
                raise ValueError(f"fps must be > 0; got {fps!r}")
            delays = np.full(n, 1000.0 / float(fps))
        else:
            delays = self.frame_delays(speed=speed)
        live, draw = self._player(
            xy, frames, show=show, trail=trail, figsize=figsize, trip_colors=trip_colors
        )
        holder: list[Any] = []

        def step(k: int) -> list[Any]:
            draw(k)
            if holder and holder[0].event_source is not None:  # the next frame waits its own delay
                holder[0].event_source.interval = max(1, round(float(delays[k])))
            return []

        anim = FuncAnimation(
            live.fig,
            step,
            frames=n,
            interval=max(1, round(float(delays[0]))),
            blit=False,
            repeat=True,
            repeat_delay=round(float(hold) * 1000.0),
        )
        holder.append(anim)
        if inline_backend():
            plt.close(live.fig)  # else the kernel displays the empty figure again when the cell ends
        return anim

    def save(
        self,
        path: str | Path,
        coords: Any = None,
        *,
        fps: int = 20,
        dpi: int = 80,
        show: str = "both",
        speed: float | None = None,
        hold: float = 1.0,
        trail: int = 0,
        figsize: tuple[float, float] = (6, 6),
        trip_colors: bool = True,
    ) -> Path:
        """Write the replay to a file: a GIF through pillow, or any format ffmpeg writes (``.mp4``...).

        Parameters
        ----------
        path : str or Path
            Destination; the suffix picks the writer (``.gif`` -> pillow, anything else -> ffmpeg).
        coords : (n, 2) array-like, optional
            Node positions (x, y); default: the recorded problem's ``coords``.
        fps : int > 0, default 20
            Frames per second of the file: without ``speed`` every kept event is one frame.
        dpi : int, default 80
            Resolution of the frames (``figsize`` x ``dpi`` pixels: 480 x 480 by default).
        show : {"both", "best", "current"}, default "both"
            Which tours to draw.
        speed : float > 0, optional
            Time-lapse factor: each frame is held for its recorded gap divided by ``speed``,
            rounded to whole frames of ``1 / fps`` s (a GIF folds the repeated frames, so a
            time-lapse costs no extra bytes).
        hold : float >= 0, default 1.0
            Seconds the final picture stays at the end of the file.
        trail, figsize, trip_colors
            As in ``animate``.

        Returns
        -------
        path : Path
            The file written.

        Raises
        ------
        RuntimeError
            For a non-GIF suffix when ffmpeg is not on the ``PATH``.
        """
        plt = pyplot()
        from matplotlib import animation

        check_show(show)
        if not fps > 0:
            raise ValueError(f"fps must be > 0; got {fps!r}")
        if hold < 0:
            raise ValueError(f"hold must be >= 0; got {hold!r}")
        out = Path(path)
        suffix = out.suffix.lower()
        if suffix == ".gif":
            writer = "pillow"
        elif animation.writers.is_available("ffmpeg"):
            writer = "ffmpeg"
        else:
            raise RuntimeError(
                f"saving {suffix or 'a movie'} needs ffmpeg on the PATH (install it with your package "
                "manager: brew install ffmpeg, apt install ffmpeg); a .gif needs only pillow"
            )
        frames = self._frames()
        xy = self._coords(coords)
        n = len(frames)
        base = 1000.0 / float(fps)
        if speed is None:
            sequence = list(range(n))
        else:
            delays = self.frame_delays(speed=speed)
            sequence = [k for k in range(n) for _ in range(max(1, round(float(delays[k]) / base)))]
        sequence.extend([n - 1] * round(float(hold) * float(fps)))
        live, draw = self._player(
            xy, frames, show=show, trail=trail, figsize=figsize, trip_colors=trip_colors
        )

        def step(k: int) -> list[Any]:
            draw(k)
            return []

        anim = animation.FuncAnimation(
            live.fig, step, frames=sequence, interval=base, blit=False, repeat=False
        )
        anim.save(str(out), writer=writer, fps=fps, dpi=dpi)
        plt.close(live.fig)
        return out

    def replay(
        self,
        coords: Any = None,
        *,
        speed: float = 10.0,
        every: int = 1,
        backend: str = "matplotlib",
        show: str = "both",
        **live_kwargs: Any,
    ) -> LivePlot:
        """Drive a [`LivePlot`][skroute.viz.LivePlot] through the recorded events at time-lapse speed.

        The events are handed to a fresh ``LivePlot`` in the recorded order, waiting between two of
        them for their recorded gap divided by ``speed`` (never more than two seconds), so the
        replay looks like the run did — ``speed`` times faster. It blocks until the last event.

        Parameters
        ----------
        coords : (n, 2) array-like, optional
            Node positions; default: the recorded problem's ``coords``.
        speed : float > 0, default 10.0
            Time-lapse factor (``1.0`` replays in real time; a huge value skips every wait).
        every : int >= 1, default 1
            Redraw every ``every``-th iteration event (as ``LivePlot(every=)``).
        backend : {"matplotlib", "plotly"}, default "matplotlib"
            Drawing backend of the ``LivePlot``.
        show : {"both", "best", "current"}, default "both"
            Which tours to draw.
        **live_kwargs
            Other ``LivePlot`` options: ``trail``, ``map``, ``figsize``, ``title``, ``pause``,
            ``trip_colors``.

        Returns
        -------
        live : LivePlot
            The plot that was driven (``live.fig`` holds the final picture).
        """
        from ._live import LivePlot

        if not speed > 0:
            raise ValueError(f"speed must be > 0; got {speed!r}")
        if not self.events:
            raise ValueError("nothing to replay: no recorded events")
        if any(e.problem is None for e in self.events):
            raise ValueError("the recorded events carry no RoutingProblem; record a fit to replay it")
        xy = self._coords(coords)
        live = LivePlot(xy, backend=backend, every=every, show=show, **live_kwargs)
        previous: float | None = None
        for ev in self.events:
            if previous is not None:
                wait = (ev.timestamp - previous) / float(speed)
                if wait >= MIN_DELAY_MS / 1000.0:
                    time.sleep(min(wait, MAX_DELAY_MS / 1000.0))
            previous = ev.timestamp
            live(ev)
        return live

    def to_plotly(self, coords: Any = None, *, map: bool = False, show: str = "both", fps: float = 20) -> Any:
        """Replay the run as a Plotly figure: one frame per kept event, Play/Pause, a speed menu and a slider.

        Parameters
        ----------
        coords : (n, 2) array-like, optional
            Node positions in matrix row order: (x, y), or (latitude, longitude) when ``map=True``;
            default: the recorded problem's ``coords``.
        map : bool, default False
            Draw on OpenStreetMap tiles (``Scattermap``) instead of plain axes.
        show : {"both", "best", "current"}, default "both"
            Which tours to draw (the edges and the ring are always drawn).
        fps : float > 0, default 20
            Frame rate of the Play button (``1x``); the speed menu offers 0.5x, 1x, 2x, 4x and 8x.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Nodes, depot and the first frame's current, best, edges and ring traces as base traces;
            ``fig.frames`` holds one frame per kept drawable event; Play/Pause buttons, a speed
            dropdown and a slider drive them.
        """
        from ._map import recorder_figure

        check_show(show)
        if not fps > 0:
            raise ValueError(f"fps must be > 0; got {fps!r}")
        return recorder_figure(self, self._frames(), self._coords(coords), map=map, show=show, fps=float(fps))

    def plot_history(self, ax: Axes | None = None) -> Axes:
        """Best-so-far cost of the kept iteration events (see [`plot_history`][skroute.viz.plot_history])."""
        return _plot_history(self.events, ax=ax)

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"Recorder(every={self.every}, keep_tours={self.keep_tours}, n_events={len(self.events)})"
