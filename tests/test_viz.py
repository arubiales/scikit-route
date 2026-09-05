"""``skroute.viz`` (D30): static plots, the live callback, the recorder and the Plotly map tools.

Two protocols drive the callbacks. ``fake_run`` emits ``FakeEvent`` -- a frozen dataclass with exactly
the D30 fields and properties -- from a scripted descent (``start``, iterations with an improving
``best_cost``, ``end``; it stops early when the callback returns ``True``), which keeps the unit tests
fast and their expectations exact. The "real protocol" section then runs the solvers themselves through
``fit(callback=)`` -- ``RouteEvent``, the restarts ``MultiStart(n_jobs=1)`` forwards, ``stop()`` mid-run,
a reused callback -- the event shapes a fake cannot promise to reproduce.

Every figure is drawn on the Agg backend (forced at import); ``plt.pause``/``fig.show`` never run.
"""

# ruff: noqa: E402  (the optional matplotlib/pillow imports come after pytest.importorskip)

from __future__ import annotations

import gc
import math
import os
import re
import subprocess
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="skroute.viz needs the optional 'viz' extra")
pytest.importorskip("PIL", reason="pillow ships with matplotlib; needed to save GIFs")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

import skroute
from skroute import (
    SOM,
    AntColony,
    BruteForce,
    Insertion,
    IteratedLocalSearch,
    MultiStart,
    NearestNeighbour,
    RoutingProblem,
    SimulatedAnnealing,
)
from skroute.datasets import load_barcelona, load_tsp
from skroute.utils import initial_tour
from skroute.viz import LivePlot, RecordedEvent, Recorder, _record, plot_history, plot_route, plot_route_map
from skroute.viz._static import EDGE_COLOR, RING_COLOR

matplotlib.use("Agg")

go = pytest.importorskip("plotly.graph_objects")


# --------------------------------------------------------------------------- the fake protocol
@dataclass(frozen=True)
class FakeEvent:
    """The ``RouteEvent`` contract of D30, field for field."""

    solver: str
    stage: str
    iteration: int
    cost: float
    best_cost: float
    tour: np.ndarray | None
    best_tour: np.ndarray | None
    problem: RoutingProblem
    extra: dict[str, Any] = field(default_factory=dict)

    def _label_trips(self) -> list[np.ndarray]:
        idx = self.problem.to_index_tour(self.best_tour)
        starts = self.problem.trip_starts(idx)
        lab, d = self.problem.labels, self.problem.depot
        depot = lab[d : d + 1]
        return [np.concatenate((depot, lab[idx[a:b]], depot)) for a, b in pairwise(starts)]

    @property
    def route(self) -> np.ndarray:
        trips = self._label_trips()
        return np.concatenate([trips[0]] + [t[1:] for t in trips[1:]])

    @property
    def trips(self) -> list[np.ndarray]:
        return self._label_trips()


def fake_run(callback, problem, n_iter=12, seed=0, solver="FakeDescent"):
    """A scripted solver: ``start``, ``n_iter`` iterations of a random-reversal descent, ``end``.

    The current tour is the proposal just priced (so it visibly changes), the best tour improves when
    a proposal beats it; ``extra`` carries a fake temperature. Returns ``(events, stop_reason)``.
    """
    rng = np.random.default_rng(seed)
    best = initial_tour(problem, "nearest_neighbour", None)
    best_cost = problem.evaluate(best)
    cur, cur_cost = best.copy(), best_cost
    events: list[FakeEvent] = []

    def emit(stage, k, **extra):
        ev = FakeEvent(
            solver,
            stage,
            k,
            float(cur_cost),
            float(best_cost),
            problem.to_label_tour(cur),
            problem.to_label_tour(best),
            problem,
            extra,
        )
        events.append(ev)
        return callback(ev) is True

    reason, k = "max_iter", 0
    if emit("start", 0):
        reason = "callback"
    else:
        for k in range(1, n_iter + 1):
            i, j = np.sort(rng.choice(np.arange(1, problem.n), size=2, replace=False))
            cur = best.copy()
            cur[i : j + 1] = cur[i : j + 1][::-1]
            cur_cost = problem.evaluate(cur)
            if cur_cost < best_cost - 1e-9:
                best, best_cost = cur.copy(), cur_cost
            if emit("iteration", k, temperature=1.0 / k):
                reason = "callback"
                break
    emit("end", k)
    return events, reason


def closed_xy(problem, xy, label_tour):
    """The x, y polyline ``plot_route``/``LivePlot`` draw for a label tour (closed at the depot)."""
    idx = problem.to_index_tour(label_tour)
    idx = np.append(idx, idx[0])
    return xy[idx]


BASE_LINES = (
    4  # LivePlot's persistent Line2D artists: current, best, edges, ring (then ``trail`` fading tours)
)


def final_lines(live):
    """The lines of the final route a LivePlot drew at ``"end"`` (appended after the persistent artists)."""
    return live._view.final_lines


def title(ax_or_fig):
    """A figure's status title as one line (LivePlot wraps long titles; Plotly uses ``<br>``)."""
    text = ax_or_fig.get_title() if hasattr(ax_or_fig, "get_title") else ax_or_fig.layout.title.text
    return text.replace("\n", " | ").replace("<br>", " | ")


def structure_event(problem, stage="iteration", k=1, solver="Insertion", **extra):
    """A D31 event without any tour: ``cost``/``best_cost`` nan, the structure in ``extra``."""
    return FakeEvent(solver, stage, k, math.nan, math.nan, None, None, problem, extra)


def with_stamps(rec, stamps):
    """Rewrite the timestamps of a recorder's events (a controlled clock for the replay tests)."""
    from dataclasses import replace

    assert len(stamps) == len(rec.events)
    rec.events = [replace(e, timestamp=float(t)) for e, t in zip(rec.events, stamps, strict=True)]
    return rec


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def dj():
    return load_tsp("dj38")


@pytest.fixture(scope="module")
def dj_problem(dj):
    return RoutingProblem(dj.distance_matrix(), labels=dj.labels, coords=dj.coords)


@pytest.fixture(scope="module")
def ils(dj):
    return IteratedLocalSearch(n_iter=5, patience=None, random_state=0).fit(
        dj.distance_matrix(), labels=dj.labels, coords=dj.coords
    )


SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
H4 = np.array([[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]], dtype=float)


@pytest.fixture(scope="module")
def two_trips():
    """The 4-node example of the problem model: a budget of 4 h splits the tour into two trips (cost 41)."""
    X = {a: {b: C4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    T = {a: {b: H4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    est = BruteForce().fit(X, time_matrix=T, depot="d", max_time_work=4.0, extra_cost=3.0)
    assert est.n_trips_ == 2 and est.cost_ == 41.0
    return est


@pytest.fixture(scope="module")
def bcn_problem():
    b = load_barcelona()
    return RoutingProblem(b.cost, labels=b.labels, coords=b.coords)


# --------------------------------------------------------------------------- import surface
def test_public_api_and_lazy_imports():
    assert skroute.viz.__all__ == [
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
    code = "import sys, skroute.viz; sys.exit(int('matplotlib' in sys.modules or 'plotly' in sys.modules))"
    env = {**os.environ, "PYTHONPATH": str(Path(skroute.__file__).resolve().parents[1])}
    assert subprocess.run([sys.executable, "-c", code], env=env, check=False).returncode == 0


def test_matplotlib_missing_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(
        ImportError, match=re.escape("matplotlib is required for skroute.viz: pip install scikit-route[viz]")
    ):
        plot_route([0, 1, 2], SQUARE[:3])
    with pytest.raises(ImportError, match=re.escape("scikit-route[viz]")):
        plot_history([])


def test_plotly_missing_message(monkeypatch, two_trips, dj_problem):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(
        ImportError,
        match=re.escape("plotly is required for skroute.viz maps: pip install scikit-route[viz-map]"),
    ):
        plot_route_map(two_trips, SQUARE)
    live = LivePlot(dj_problem.coords, backend="plotly")
    with pytest.raises(ImportError, match=re.escape("scikit-route[viz-map]")):
        fake_run(live, dj_problem, n_iter=1)


# --------------------------------------------------------------------------- plot_route
def test_plot_route_fitted_estimator(ils, dj):
    ax = plot_route(ils)
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert len(line.get_xdata()) == dj.coords.shape[0] + 1  # closed at the depot
    np.testing.assert_allclose(line.get_xydata(), closed_xy(ils.problem_, dj.coords, ils.tour_))
    offsets = sorted(len(c.get_offsets()) for c in ax.collections)
    assert offsets == [1, 37]  # the depot star and the other 37 cities
    assert ax.get_title() == f"IteratedLocalSearch | cost {ils.cost_:.6g}"
    assert ax.get_aspect() == 1.0 and ax.get_xticks().size == 0
    assert len(ax.texts) == 0


def test_plot_route_explicit_coords_ax_labels_depot(ils, dj):
    _, ax0 = plt.subplots()
    ax = plot_route(ils, dj.coords * 2.0, ax=ax0, labels=True, depot=False, linewidth=3.0, alpha=0.5)
    assert ax is ax0
    assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == 38  # no depot star
    assert [t.get_text() for t in ax.texts] == [str(x) for x in dj.labels]
    assert ax.lines[0].get_linewidth() == 3.0 and ax.lines[0].get_alpha() == 0.5
    np.testing.assert_allclose(ax.lines[0].get_xydata(), 2.0 * closed_xy(ils.problem_, dj.coords, ils.tour_))


def test_plot_route_multi_trip_colours(two_trips):
    ax = plot_route(two_trips, SQUARE)
    assert len(ax.lines) == 2
    assert [len(line.get_xdata()) for line in ax.lines] == [len(t) for t in two_trips.trips_]
    c0, c1 = (line.get_color() for line in ax.lines)
    assert c0 != c1, "one colour per trip"
    assert ax.get_title() == "BruteForce | cost 41 | 2 trips"
    same = plot_route(two_trips, SQUARE, trip_colors=False)
    assert same.lines[0].get_color() == same.lines[1].get_color()
    black = plot_route(two_trips, SQUARE, color="k")
    assert {line.get_color() for line in black.lines} == {"k"}
    assert [t.get_text() for t in plot_route(two_trips, SQUARE, labels=True).texts] == ["d", "a", "b", "c"]


def test_plot_route_event_and_array(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=3)
    ax = plot_route(events[-1])
    np.testing.assert_allclose(
        ax.lines[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )
    assert ax.get_title() == f"FakeDescent | cost {events[-1].best_cost:.6g}"
    # a route array indexes the rows of coords: open tour, closed route and multi-trip route
    assert len(plot_route([0, 1, 2, 3], SQUARE).lines) == 1
    assert len(plot_route(np.array([0, 1, 2, 3, 0]), SQUARE).lines) == 1
    two = plot_route([0, 1, 0, 2, 3, 0], SQUARE)
    assert [len(line.get_xdata()) for line in two.lines] == [3, 4]
    assert two.get_title() == "2 trips"  # no solver, no cost: only the structure


def test_plot_route_errors(ils, two_trips, dj_problem):
    with pytest.raises(ValueError, match="needs coords="):
        plot_route([0, 1, 2])
    with pytest.raises(ValueError, match="beyond the end of coords"):
        plot_route([0, 1, 9], SQUARE)
    with pytest.raises(ValueError, match="no coordinates to draw"):
        plot_route(two_trips)  # fitted without coords=
    with pytest.raises(ValueError, match="rows but the problem has"):
        plot_route(ils, SQUARE)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        plot_route(ils, np.zeros((38, 3)))
    with pytest.raises(skroute.exceptions.NotFittedError):
        plot_route(IteratedLocalSearch(), SQUARE)
    with pytest.raises(ValueError, match="1-D integer array"):
        plot_route([0.5, 1.0], SQUARE)
    for lonely in ([0], [0, 0], [2]):  # no node besides the depot: a clear error, not an IndexError
        with pytest.raises(ValueError, match="no node besides the depot"):
            plot_route(lonely, SQUARE)
    with pytest.raises(ValueError, match="negative"):
        plot_route([0, 1, -1], SQUARE)  # -1 must not silently draw the last row
    with pytest.raises(ValueError, match=r"\(n, 2\).*dict"):
        plot_route([0, 1, 2, 3], {0: (0.0, 0.0)})


# --------------------------------------------------------------------------- plot_history
def test_plot_history_estimator_events_and_recorder(ils, dj_problem):
    ax = plot_history(ils)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), ils.history_)
    assert ax.lines[0].get_xdata().tolist() == list(range(1, ils.n_iter_ + 1))
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
        "Iteration",
        "Best cost",
        "IteratedLocalSearch: best-so-far cost",
    )
    rec = Recorder()
    events, _ = fake_run(rec, dj_problem, n_iter=6)
    for source in (events, rec, rec.events):
        ax = plot_history(source)
        assert ax.lines[0].get_xdata().tolist() == list(range(1, 7))
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [e.best_cost for e in events[1:-1]])
        assert ax.get_title() == "FakeDescent: best-so-far cost"
    # a nan best (MILP before its first integral solution) is skipped; an empty sequence draws nothing
    nan_event = FakeEvent("MILP", "iteration", 1, math.nan, math.nan, None, None, dj_problem, {"edges": []})
    assert plot_history([nan_event, *events[1:3]]).lines[0].get_xdata().tolist() == [1, 2]
    assert plot_history([]).get_title() == "Best-so-far cost"
    assert rec.plot_history().get_ylabel() == "Best cost"


def test_plot_history_rejects_estimators_without_history(dj):
    with pytest.raises(skroute.exceptions.NotFittedError):
        plot_history(IteratedLocalSearch())
    nn = NearestNeighbour().fit(dj.distance_matrix())
    with pytest.raises(ValueError, match="NearestNeighbour has no history_"):
        plot_history(nn)


def test_plot_history_counts_iterations_across_restarts(dj_problem):
    """Events forwarded by MultiStart restart ``iteration`` at 1: the x axis must not run backwards."""
    outer = FakeEvent("MultiStart", "start", 0, math.nan, math.nan, None, None, dj_problem, {"n_restarts": 2})
    inner = []
    for restart in (0, 1):
        events, _ = fake_run(lambda ev: None, dj_problem, n_iter=3, seed=restart)
        inner += [
            FakeEvent(
                e.solver,
                e.stage,
                e.iteration,
                e.cost,
                e.best_cost,
                e.tour,
                e.best_tour,
                e.problem,
                {**e.extra, "restart": restart},
            )
            for e in events
        ]
    ax = plot_history([outer, *inner])
    assert ax.lines[0].get_xdata().tolist() == [1, 2, 3, 4, 5, 6]
    assert ax.get_xlabel() == "Iteration (all restarts)"
    assert ax.get_title() == "MultiStart: best-so-far cost"
    plain = plot_history(inner[:5])  # the restart key alone: still counted across restarts
    assert plain.lines[0].get_xdata().tolist() == [1, 2, 3]


# --------------------------------------------------------------------------- LivePlot (matplotlib)
def test_liveplot_matplotlib_full_run(dj_problem, dj):
    live = LivePlot(dj.coords, figsize=(4, 4))
    assert (live.fig, live.ax, live.n_events, live.n_redraws) == (None, None, 0, 0)
    events, reason = fake_run(live, dj_problem, n_iter=9)
    assert reason == "max_iter"
    assert live.n_events == 11 and live.n_redraws == 11  # start + 9 iterations + end
    assert live.fig is not None and live.ax is not None and live.fig.get_size_inches().tolist() == [4.0, 4.0]
    ax = live.ax
    current, best = ax.lines[0], ax.lines[1]
    assert len(current.get_xdata()) == 0 and len(best.get_xdata()) == 0, "the live lines are cleared at end"
    final = final_lines(live)
    assert len(final) == 1 and ax.lines[-1] is final[0]  # one trip drawn with trip colours, appended last
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour))
    assert title(ax).startswith("FakeDescent | iteration 9 | cost ")
    assert f"best {events[-1].best_cost:.6g}" in title(ax)


def test_liveplot_lines_and_title_follow_the_events(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=4)
    live = LivePlot(dj.coords, title="watching")
    assert live(events[0]) is False
    ax = live.ax
    assert len(ax.lines) == BASE_LINES and len(ax.collections) == 3  # points, depot, pheromone trails
    np.testing.assert_allclose(
        ax.lines[1].get_xydata(), closed_xy(dj_problem, dj.coords, events[0].best_tour)
    )
    assert title(ax) == f"watching | iteration 0 | cost {events[0].cost:.6g} | best {events[0].best_cost:.6g}"
    for ev in events[1:3]:
        assert live(ev) is False
        np.testing.assert_allclose(ax.lines[0].get_xydata(), closed_xy(dj_problem, dj.coords, ev.tour))
        np.testing.assert_allclose(ax.lines[1].get_xydata(), closed_xy(dj_problem, dj.coords, ev.best_tour))
        assert title(ax) == (
            f"watching | iteration {ev.iteration} | cost {ev.cost:.6g} | best {ev.best_cost:.6g}"
            f" | temperature {1.0 / ev.iteration:.6g}"
        )
    assert ax.get_title().count("\n") == 1, "a long title is wrapped onto a second line"
    assert live.n_redraws == 3


def test_liveplot_every_skips_redraws(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=9)
    live = LivePlot(dj.coords, every=3)
    for ev in events[:-1]:  # start + 9 iterations, no end
        live(ev)
    assert live.n_events == 10
    assert live.n_redraws == 4  # start + iterations 1, 4 and 7
    assert "iteration 7 |" in title(live.ax)
    live(events[-1])
    assert live.n_redraws == 5 and "iteration 9 |" in title(live.ax)


def test_liveplot_stop_returns_true(dj_problem, dj):
    live = LivePlot(dj.coords)

    def stop_after_start(ev):
        out = live(ev)
        if ev.stage == "start":
            live.stop()
        return out

    events, reason = fake_run(stop_after_start, dj_problem, n_iter=50)
    assert reason == "callback"
    assert [e.stage for e in events] == ["start", "iteration", "end"]
    assert live.n_events == 3
    assert live(events[-1]) is True  # keeps saying so
    fresh = LivePlot(dj.coords)
    fresh.stop()
    assert fresh(events[0]) is True and fresh.fig is not None  # the first event still draws


def test_liveplot_attached_mid_run_and_multi_trip_end(two_trips):
    problem = two_trips.problem_
    events, _ = fake_run(lambda ev: None, problem, n_iter=2)
    live = LivePlot(SQUARE)
    live(events[1])  # no "start" seen: the figure is created on the first event received, drawn once
    assert live.fig is not None and live.n_redraws == 1
    assert len(live.ax.lines[0].get_xdata()) >= problem.n + 1  # ...with that event's current tour on it
    live(events[-1])
    final = final_lines(live)
    assert len(final) == len(events[-1].trips) >= 1 and live.n_redraws == 2
    assert "FakeDescent | iteration 2" in title(live.ax)
    live(events[1])  # an iteration after an "end" without a new "start": the final route must not stay
    assert len(live.ax.lines) == BASE_LINES and live._view.final_lines == []
    thinned = LivePlot(SQUARE, every=2)  # the first event received is the first kept iteration
    thinned(events[1])
    thinned(events[2])
    assert thinned.n_redraws == 1 and "iteration 1" in title(thinned.ax)
    ended = LivePlot(SQUARE)  # attached at the very end: the figure and its final route, one redraw
    ended(events[-1])
    assert ended.n_redraws == 1 and len(final_lines(ended)) == len(events[-1].trips)


def test_liveplot_nested_start_resets_the_drawing_in_place(dj_problem, dj):
    """The start/end of every restart a MultiStart forwards redraw from scratch in the same figure."""
    outer = FakeEvent("MultiStart", "start", 0, math.nan, math.nan, None, None, dj_problem, {"n_restarts": 2})
    live = LivePlot(dj.coords, every=2)
    assert live(outer) is False and title(live.ax) == "MultiStart | iteration 0 | n_restarts 2"
    fig = live.fig
    for restart in (0, 1):
        events, _ = fake_run(lambda ev: None, dj_problem, n_iter=3, seed=restart)
        for ev in events:
            live(
                FakeEvent(
                    *[
                        getattr(ev, f)
                        for f in (
                            "solver",
                            "stage",
                            "iteration",
                            "cost",
                            "best_cost",
                            "tour",
                            "best_tour",
                            "problem",
                        )
                    ],
                    {**ev.extra, "restart": restart},
                )
            )
            if ev.stage == "iteration":
                assert len(live.ax.lines) == BASE_LINES, "the previous restart's final route is gone"
                assert title(live.ax).startswith(f"FakeDescent | restart {restart} | iteration ")
        assert (
            len(live.ax.lines) == BASE_LINES + 1 and live.fig is fig
        )  # this restart's final route, same figure
    assert live.n_events == 11 and live.n_redraws == 1 + 2 * (1 + 2 + 1)  # every=2: iterations 1 and 3
    assert len(plt.get_fignums()) == 1


def test_liveplot_headless_renders_once(monkeypatch, dj_problem, dj):
    """On Agg nobody sees the frames: the artists are updated at every event, the raster drawn at ``end``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    draws: list[int] = []
    original = FigureCanvasAgg.draw

    def counting_draw(self):
        draws.append(1)
        return original(self)

    monkeypatch.setattr(FigureCanvasAgg, "draw", counting_draw)
    live = LivePlot(dj.coords)
    events, _ = fake_run(live, dj_problem, n_iter=20)
    assert live.n_redraws == 22 and len(draws) == 1
    np.testing.assert_allclose(
        final_lines(live)[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )


def test_liveplot_validation(dj):
    assert LivePlot(dj.coords).backend == "matplotlib"
    assert LivePlot(dj.coords, map=True).backend == "plotly"  # the tiles are Plotly's: map=True selects it
    with pytest.raises(ValueError, match='map=True needs backend="plotly"'):
        LivePlot(dj.coords, backend="matplotlib", map=True)
    with pytest.raises(ValueError, match="backend must be one of"):
        LivePlot(dj.coords, backend="bokeh")
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="every must be an int >= 1"):
            LivePlot(dj.coords, every=bad)
    with pytest.raises(ValueError, match="pause must be >= 0"):
        LivePlot(dj.coords, pause=-1)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        LivePlot(dj.coords[:, 0])
    with pytest.raises(ValueError, match="finite"):
        LivePlot([[0.0, np.nan], [1.0, 1.0]])


def test_liveplot_jupyter_inline_redraws_through_display(monkeypatch, tmp_path, dj_problem, dj):
    calls: list[bool] = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace(kernel=object())  # a Jupyter kernel shell
    display = types.ModuleType("IPython.display")
    display.display = lambda fig, clear=False: calls.append(clear)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline")
    live = LivePlot(dj.coords)
    fake_run(live, dj_problem, n_iter=4)
    assert calls == [True] * 6  # every redraw replaces the cell output
    # ... and the figure is closed at the end, or the kernel would display it a second time
    assert live.fig.number not in plt.get_fignums() and len(live.ax.lines) == BASE_LINES + 1
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=2)
    anim = rec.animate(dj.coords, figsize=(2, 2))
    assert plt.get_fignums() == []  # animate() closed its figure too; saving still works
    anim.save(tmp_path / "inline.gif", writer="pillow", dpi=20)
    assert (tmp_path / "inline.gif").stat().st_size > 0
    # a non-inline notebook backend (ipympl) redraws the canvas in place instead, and keeps the figure
    calls.clear()
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "module://ipympl.backend_nbagg")
    live = LivePlot(dj.coords)
    fake_run(live, dj_problem, n_iter=2)
    assert calls == [] and live.fig.number in plt.get_fignums()


def test_liveplot_never_pauses_on_a_non_interactive_backend(monkeypatch, dj_problem, dj):
    monkeypatch.setattr(plt, "pause", lambda *_: pytest.fail("plt.pause must not run under Agg"))
    monkeypatch.setattr(plt, "show", lambda *_, **__: pytest.fail("LivePlot never calls plt.show"))
    fake_run(LivePlot(dj.coords), dj_problem, n_iter=3)


def test_liveplot_pauses_on_an_interactive_backend(monkeypatch, dj_problem, dj):
    from skroute.viz import _live

    pauses: list[float] = []
    monkeypatch.setattr(_live, "backend_is_interactive", lambda: True)
    monkeypatch.setattr(plt, "pause", pauses.append)
    fake_run(LivePlot(dj.coords, pause=0.5), dj_problem, n_iter=3)
    assert pauses == [0.5] * 5


# --------------------------------------------------------------------------- LivePlot (plotly)
def test_liveplot_plotly_script_shows_once_at_end(monkeypatch, dj_problem, dj):
    shown: list[Any] = []
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: shown.append(self))
    live = LivePlot(dj.coords, backend="plotly")
    events, _ = fake_run(live, dj_problem, n_iter=4)
    assert live.ax is None and isinstance(live.fig, go.Figure)
    assert shown == [live.fig]
    assert [t.type for t in live.fig.data] == ["scatter"] * 6  # nodes, depot, current, best, edges, ring
    best = live.fig.data[3]
    np.testing.assert_allclose(
        np.column_stack([best.x, best.y]), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )
    assert live.fig.data[2].x is None or len(live.fig.data[2].x) == 0  # current line cleared at end
    assert title(live.fig).startswith("FakeDescent | iteration 4 | cost ")
    assert live.fig.layout.yaxis.scaleanchor == "x"


def test_liveplot_plotly_map(monkeypatch, bcn_problem):
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: None)
    live = LivePlot(bcn_problem.coords, backend="plotly", map=True, every=2)
    fake_run(live, bcn_problem, n_iter=5)
    assert [t.type for t in live.fig.data] == ["scattermap"] * 6
    assert live.fig.layout.map.style == "open-street-map" and 8 < live.fig.layout.map.zoom < 13
    assert live.n_redraws == 1 + 3 + 1
    assert len(live.fig.data[3].lat) == bcn_problem.n + 1  # one closed trip, lat/lon on a map


def test_liveplot_plotly_notebook_widget(monkeypatch, dj_problem, dj):
    """In a notebook the figure is a FigureWidget updated in place (or a Figure when anywidget is missing)."""
    from skroute.viz import _live

    displayed: list[Any] = []
    monkeypatch.setattr(_live, "in_notebook", lambda: True)
    monkeypatch.setattr(_live, "display_figure", lambda fig, clear: displayed.append(fig))
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: pytest.fail("no show"))
    try:
        go.FigureWidget()
        have_widget = True
    except ImportError:
        have_widget = False
    live = LivePlot(dj.coords, backend="plotly")
    if have_widget:
        fake_run(live, dj_problem, n_iter=2)
        assert isinstance(live.fig, go.FigureWidget) and displayed == [live.fig]
    else:
        # without anywidget: plain Figure, shown nowhere by us (the user displays it)
        monkeypatch.setattr(
            "plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: displayed.append("show")
        )
        fake_run(live, dj_problem, n_iter=2)
        assert isinstance(live.fig, go.Figure) and displayed == ["show"]


# --------------------------------------------------------------------------- Recorder
def test_recorder_stores_copies(dj_problem):
    rec = Recorder()
    assert repr(rec) == "Recorder(every=1, keep_tours=True, n_events=0)" and len(rec) == 0
    events, _ = fake_run(rec, dj_problem, n_iter=8)
    assert len(rec) == len(rec.events) == 10 and rec.problem is dj_problem
    assert [e.stage for e in rec.events] == ["start", *["iteration"] * 8, "end"]
    assert rec.iterations.tolist() == [0, *range(1, 9), 8]
    assert rec.costs.shape == rec.best_costs.shape == (10,) and rec.costs.dtype == np.float64
    assert np.all(np.diff(rec.best_costs) <= 0)
    np.testing.assert_allclose(rec.costs, [e.cost for e in events])
    assert all(isinstance(e, RecordedEvent) for e in rec.events)
    kept, seen = rec.events[3], events[3]
    assert kept.tour is not seen.tour and np.array_equal(kept.tour, seen.tour)
    assert kept.best_tour is not seen.best_tour and np.array_equal(kept.best_tour, seen.best_tour)
    assert kept.extra == {"temperature": 1.0 / 3} and kept.extra is not seen.extra
    assert rec.n_frames == 10
    # the copies decode like live events (D30's route and trips)
    assert kept.route.tolist() == seen.route.tolist()
    assert [t.tolist() for t in kept.trips] == [t.tolist() for t in seen.trips]
    bare = RecordedEvent("X", "start", 0, math.nan, math.nan, None, None)
    assert bare.trips == [] and bare.route is None


def test_recorder_every_and_keep_tours(dj_problem):
    rec = Recorder(every=3)
    fake_run(rec, dj_problem, n_iter=8)
    assert rec.iterations.tolist() == [0, 1, 4, 7, 8]
    bare = Recorder(keep_tours=False)
    fake_run(bare, dj_problem, n_iter=3)
    assert len(bare) == 5 and all(e.tour is None and e.best_tour is None for e in bare.events)
    assert bare.n_frames == 0 and bare.best_costs.shape == (5,)
    with pytest.raises(ValueError, match="keep_tours=False"):
        bare.animate(dj_problem.coords)
    with pytest.raises(ValueError, match="keep_tours=False"):
        bare.replay(dj_problem.coords, speed=1e9)  # no blank picture: the same error as the other replays
    with pytest.raises(ValueError, match="keep_tours=False"):
        bare.to_plotly(dj_problem.coords)
    with pytest.raises(ValueError, match="nothing to draw"):
        Recorder().to_plotly(dj_problem.coords)
    for bad in (0, 2.0, True):
        with pytest.raises(ValueError, match="every must be an int >= 1"):
            Recorder(every=bad)


def test_recorder_animate_and_save_gif(tmp_path, dj_problem, dj):
    rec = Recorder()
    events, _ = fake_run(rec, dj_problem, n_iter=3)
    assert rec.n_frames == 5
    anim = rec.animate(dj.coords, interval=50, figsize=(3, 3))
    assert isinstance(anim, FuncAnimation)
    fig = plt.gcf()  # the figure animate created
    assert fig.get_size_inches().tolist() == [3.0, 3.0]
    ax = fig.axes[0]
    assert len(ax.collections) == 3  # points, depot and the (empty) pheromone trails, drawn once
    assert anim._interval == 50 and anim._repeat_delay == 1000  # a second on the final picture per loop
    path = tmp_path / "run.gif"
    anim.save(path, writer="pillow", dpi=40)
    with Image.open(path) as im:
        # 5 frames rendered; Pillow folds the "end" frame into the identical last iteration frame
        assert im.n_frames in (4, 5)
    # the last frame drawn is the final route (the live lines cleared), titled like LivePlot's end
    final = final_lines(rec._last_live)
    assert len(final) == 1 and len(ax.lines[0].get_xdata()) == len(ax.lines[1].get_xdata()) == 0
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour))
    last = events[-1]
    assert title(ax) == f"FakeDescent | iteration 3 | cost {last.cost:.6g} | best {last.best_cost:.6g}"


def test_recorder_animate_multi_trip_colours(tmp_path, two_trips):
    rec = Recorder()
    fake_run(rec, two_trips.problem_, n_iter=2)
    anim = rec.animate(SQUARE, trip_colors=True)
    anim.save(
        tmp_path / "trips.gif", writer="pillow", dpi=30
    )  # renders every frame; the last one stays drawn
    n_trips = len(plot_route(rec.events[-1], SQUARE).lines)  # a recorded event is drawable on its own
    final = final_lines(rec._last_live)
    assert len(final) == n_trips >= 1
    assert len({line.get_color() for line in final}) == n_trips
    assert "2 trips" in title(rec._last_live.ax)
    plain = rec.animate(SQUARE, trip_colors=False)
    plain.save(tmp_path / "plain.gif", writer="pillow", dpi=30)
    assert len({line.get_color() for line in final_lines(rec._last_live)}) == 1


def test_recorder_to_plotly(dj_problem, dj, bcn_problem):
    rec = Recorder(every=2)
    fake_run(rec, dj_problem, n_iter=6)
    fig = rec.to_plotly(dj.coords)
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == rec.n_frames == 5
    assert [t.type for t in fig.data] == ["scatter"] * 6  # nodes, depot, current, best, edges, ring
    assert len(fig.layout.sliders) == 1 and len(fig.layout.sliders[0].steps) == 5
    assert [s.label for s in fig.layout.sliders[0].steps] == ["0", "1", "3", "5", "6"]
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["Play", "Pause"]
    last = rec.events[-1]
    assert fig.frames[-1].layout.title.text == (
        f"FakeDescent | iteration 6 | cost {last.cost:.6g} | best {last.best_cost:.6g}"
    )
    assert "<br>" in fig.frames[-2].layout.title.text, "a long Plotly title breaks with <br>"
    nested = Recorder()
    for restart in (0, 1):
        events, _ = fake_run(lambda ev: None, dj_problem, n_iter=2, seed=restart)
        for e in events:
            nested(
                FakeEvent(
                    e.solver,
                    e.stage,
                    e.iteration,
                    e.cost,
                    e.best_cost,
                    e.tour,
                    e.best_tour,
                    e.problem,
                    {**e.extra, "restart": restart},
                )
            )
    steps = nested.to_plotly(dj.coords).layout.sliders[0].steps
    assert [s.label for s in steps] == ["0:0", "0:1", "0:2", "0:2", "1:0", "1:1", "1:2", "1:2"]
    nested_title = nested.to_plotly(dj.coords).frames[0].layout.title.text
    assert nested_title.startswith("FakeDescent | restart 0 | iteration 0 | cost ")
    end = fig.frames[-1].data  # the "end" frame: the current trace is cleared, the best is the closed route
    assert [t.name for t in end] == ["current", "best", "edges", "ring"] and len(end[0].x) == 0
    np.testing.assert_allclose(
        np.column_stack([end[1].x, end[1].y]), closed_xy(dj_problem, dj.coords, rec.events[-1].best_tour)
    )
    geo = Recorder()
    fake_run(geo, bcn_problem, n_iter=2)
    on_map = geo.to_plotly(bcn_problem.coords, map=True)
    assert [t.type for t in on_map.data] == ["scattermap"] * 6
    assert on_map.layout.map.style == "open-street-map"
    assert len(on_map.frames) == 4 and on_map.frames[0].data[0].type == "scattermap"


# --------------------------------------------------------------------------- the real protocol
def test_real_liveplot_counts_events_and_redraws(dj):
    live = LivePlot(dj.coords, every=50)
    sa = SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=live)
    assert live.n_events == sa.n_iter_ + 2  # start, one event per level, end
    assert live.n_redraws == 2 + math.ceil(sa.n_iter_ / 50)
    final = final_lines(live)
    assert len(final) == 1
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(sa.problem_, dj.coords, sa.tour_))
    assert title(live.ax).startswith(f"SimulatedAnnealing | iteration {sa.n_iter_} | cost ")
    assert f"best {sa.cost_:.6g}" in title(live.ax)


def test_real_liveplot_stop_mid_run(dj):
    live = LivePlot(dj.coords)

    def watch(ev):
        out = live(ev)
        if ev.stage == "iteration" and ev.iteration == 7:
            live.stop()  # honoured at the next event: the solver stops after iteration 8
        return out

    sa = SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=watch)
    assert (sa.stop_reason_, sa.n_iter_, live.n_events) == ("callback", 8, 10)


def test_real_liveplot_reused_for_a_second_fit_starts_afresh(dj):
    """A new fit with the same LivePlot: fresh figure, no stale route, ``every`` phase and stop() reset."""
    C = dj.distance_matrix()
    live = LivePlot(dj.coords, every=3)
    drawn: list[int] = []
    stop_once = [True]

    def watch(ev):
        before = live.n_redraws
        out = live(ev)
        if ev.stage == "iteration" and live.n_redraws > before:
            drawn.append(ev.iteration)
        if ev.stage == "iteration" and ev.iteration == 4 and stop_once:
            stop_once.clear()
            live.stop()
        return out

    first = IteratedLocalSearch(n_iter=10, patience=None, random_state=0).fit(
        C, labels=dj.labels, callback=watch
    )
    assert first.stop_reason_ == "callback" and first.n_iter_ == 5 and drawn == [1, 4]
    fig1 = live.fig
    drawn.clear()
    second = IteratedLocalSearch(n_iter=7, patience=None, random_state=1).fit(
        C, labels=dj.labels, callback=watch
    )
    assert second.stop_reason_ == "max_iter" and second.n_iter_ == 7, "the old stop request is forgotten"
    assert drawn == [1, 4, 7], "the every phase starts again"
    assert live.fig is not fig1 and len(live.ax.lines) == BASE_LINES + 1  # a fresh figure, one final route
    np.testing.assert_allclose(
        final_lines(live)[0].get_xydata(), closed_xy(second.problem_, dj.coords, second.tour_)
    )
    assert (
        len(fig1.axes[0].lines) == BASE_LINES + 1 and len(plt.get_fignums()) == 2
    )  # the first is left as it was
    wi = load_tsp("wi29")
    with pytest.raises(ValueError, match="coords has 38 rows but the problem has 29 nodes"):
        IteratedLocalSearch(n_iter=1, patience=None).fit(
            wi.distance_matrix(), labels=wi.labels, callback=live
        )


def test_real_liveplot_under_multistart(dj):
    """MultiStart(n_jobs=1) forwards every restart's events: each restart is drawn from scratch."""
    live = LivePlot(dj.coords, every=2)
    lines_seen: list[int] = []

    def spy(ev):
        out = live(ev)
        if ev.stage == "iteration" and ev.extra.get("restart", 0) > 0:
            lines_seen.append(len(live.ax.lines))  # never the previous restart's final route underneath
        return out

    ms = MultiStart(IteratedLocalSearch(n_iter=3, patience=None), n_restarts=3, random_state=0).fit(
        dj.distance_matrix(), labels=dj.labels, callback=spy
    )
    assert live.n_events == 2 + 3 * 5  # its own start/end + 3 x (start, 3 iterations, end)
    assert live.n_redraws == 2 + 3 * 4  # every=2 counts from 1 in each restart: iterations 1 and 3
    assert lines_seen == [BASE_LINES] * 6
    assert len(plt.get_fignums()) == 1, "one figure for the whole ensemble"
    final = final_lines(live)
    assert len(final) == 1
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(ms.problem_, dj.coords, ms.tour_))
    assert title(live.ax).startswith("MultiStart | ")


def test_real_liveplot_plotly_under_multistart_shows_once(monkeypatch, dj):
    shown: list[Any] = []
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: shown.append(self))
    live = LivePlot(dj.coords, backend="plotly")
    ms = MultiStart(IteratedLocalSearch(n_iter=2, patience=None), n_restarts=2, random_state=0).fit(
        dj.distance_matrix(), labels=dj.labels, callback=live
    )
    assert shown == [live.fig]  # not once per restart
    best = live.fig.data[3]
    np.testing.assert_allclose(np.column_stack([best.x, best.y]), closed_xy(ms.problem_, dj.coords, ms.tour_))


def test_real_recorder_matches_the_fit(dj):
    rec = Recorder(every=10)
    sa = SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=rec)
    assert rec.problem is sa.problem_ and len(rec) == 2 + math.ceil(sa.n_iter_ / 10)
    assert rec.iterations[-1] == sa.n_iter_ and rec.best_costs[-1] == sa.cost_
    last = rec.events[-1]
    assert np.array_equal(last.best_tour, sa.tour_)
    assert last.route.tolist() == sa.route_.tolist()
    assert [t.tolist() for t in last.trips] == [t.tolist() for t in sa.trips_]
    assert np.all(np.diff(rec.best_costs) <= 0)
    assert np.all(np.diff(rec.timestamps) >= 0) and rec.frame_delays(speed=1e9).tolist() == [10.0] * len(rec)
    live = rec.replay(dj.coords, speed=1e9, every=10)  # no waiting at that speed
    assert live.n_events == len(rec) and live.n_redraws == 2 + math.ceil((len(rec) - 2) / 10)
    np.testing.assert_allclose(final_lines(live)[0].get_xydata(), closed_xy(sa.problem_, dj.coords, sa.tour_))


def test_real_recorder_reused_keeps_each_runs_problem(dj):
    rec = Recorder(every=2)
    IteratedLocalSearch(n_iter=3, patience=None, random_state=0).fit(
        dj.distance_matrix(), labels=dj.labels, callback=rec
    )
    b = load_barcelona()
    ils = IteratedLocalSearch(n_iter=3, patience=None, random_state=0).fit(
        b.cost, labels=b.labels, callback=rec
    )
    assert rec.iterations.tolist() == [0, 1, 3, 3] * 2, "every counts from the new run's first iteration"
    assert rec.problem is ils.problem_ and rec.events[0].problem.n == 38 and rec.events[-1].problem.n == 19
    assert len(plot_route(rec.events[-1], b.coords).lines) == 1  # decoded with its own problem
    assert rec.events[-1].route.tolist() == ils.route_.tolist()


def test_real_plot_history_and_slider_over_multistart_events(dj):
    rec = Recorder()
    MultiStart(IteratedLocalSearch(n_iter=4, patience=None), n_restarts=2, random_state=0).fit(
        dj.distance_matrix(), labels=dj.labels, callback=rec
    )
    assert rec.iterations.tolist() == [0, 0, 1, 2, 3, 4, 4, 0, 1, 2, 3, 4, 4, 4]
    ax = rec.plot_history()
    assert ax.lines[0].get_xdata().tolist() == list(range(1, 9))
    assert (ax.get_xlabel(), ax.get_title()) == ("Iteration (all restarts)", "MultiStart: best-so-far cost")
    steps = rec.to_plotly(dj.coords).layout.sliders[0].steps
    assert [s.label for s in steps] == [f"{r}:{i}" for r in (0, 1) for i in (0, 1, 2, 3, 4, 4)] + ["4"]


def test_real_plot_route_of_a_multi_trip_event(two_trips):
    X = {a: {b: C4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    T = {a: {b: H4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    events: list[Any] = []
    rec = Recorder()
    est = BruteForce().fit(
        X,
        time_matrix=T,
        depot="d",
        max_time_work=4.0,
        extra_cost=3.0,
        callback=lambda e: (events.append(e), rec(e))[1],
    )
    assert [e.stage for e in events] == ["start", "end"] and est.n_trips_ == 2
    ax = plot_route(events[-1], SQUARE)
    assert [len(line.get_xdata()) for line in ax.lines] == [len(t) for t in est.trips_]
    assert ax.get_title() == "BruteForce | cost 41 | 2 trips"
    assert [t.tolist() for t in rec.events[-1].trips] == [t.tolist() for t in events[-1].trips]
    assert rec.events[-1].route.tolist() == events[-1].route.tolist() == est.route_.tolist()


# --------------------------------------------------------------------------- the real D31 traces
def test_real_insertion_steps_are_recorded_and_drawn(dj):
    rec = Recorder()
    est = Insertion().fit(dj.distance_matrix(), labels=dj.labels, callback=rec)
    n = len(dj.labels)
    steps = [e for e in rec.events if e.stage == "iteration"]
    assert len(steps) == n - 1 and all(e.tour is None and math.isnan(e.cost) and e.drawable for e in steps)
    assert rec.n_frames == n  # n - 1 construction steps and the final tour
    assert [len(e.extra["edges"]) for e in steps] == list(range(2, n + 1))  # closed partial cycles
    closed = [*est.tour_.tolist(), est.tour_[0]]
    assert {frozenset(p) for p in steps[-1].extra["edges"]} == {frozenset(p) for p in pairwise(closed)}
    ax = plot_route(steps[3], dj.coords)
    assert ax.get_title().endswith("5 edges")
    assert any(line.get_color() == EDGE_COLOR and len(line.get_xdata()) > 0 for line in ax.lines)
    assert len(rec.frame_delays(speed=1e9)) == rec.n_frames
    live = rec.replay(dj.coords, speed=1e9)
    assert live.n_events == len(rec)
    np.testing.assert_allclose(
        final_lines(live)[0].get_xydata(), closed_xy(est.problem_, dj.coords, est.tour_)
    )


def test_real_som_ring_stays_around_the_cities(dj):
    rec = Recorder()
    est = SOM(random_state=0).fit(dj.distance_matrix(), coords=dj.coords, labels=dj.labels, callback=rec)
    rings = [e.extra["ring"] for e in rec.events if e.stage == "iteration"]
    assert len(rings) == est.n_iter_ >= 2 and all(r.shape == (rings[0].shape[0], 2) for r in rings)
    lo, hi = dj.coords.min(axis=0), dj.coords.max(axis=0)
    margin = 0.1 * (hi - lo)
    assert all(np.all(r >= lo - margin) and np.all(r <= hi + margin) for r in rings)
    ax = plot_route(rec.events[1], dj.coords)
    ring_lines = [line for line in ax.lines if line.get_color() == RING_COLOR]
    assert len(ring_lines) == 1 and len(ring_lines[0].get_xdata()) == rings[0].shape[0] + 1  # closed
    live = rec.replay(dj.coords, speed=1e9)
    assert live.n_events == len(rec)
    np.testing.assert_allclose(
        final_lines(live)[0].get_xydata(), closed_xy(est.problem_, dj.coords, est.tour_)
    )


def test_real_ant_colony_trails_are_recorded_and_drawn(dj):
    rec = Recorder()
    AntColony(n_iter=3, random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=rec)
    n = len(dj.labels)
    steps = [e for e in rec.events if e.stage == "iteration"]
    assert len(steps) == 3
    for e in steps:
        assert len(e.extra["edges"]) == len(e.extra["edge_weights"]) == min(3 * n, n * (n - 1) // 2)
        assert all(0.0 < w <= 1.0 for w in e.extra["edge_weights"])
        assert all(a in dj.labels and b in dj.labels for a, b in e.extra["edges"])
    ax = plot_route(steps[-1], dj.coords)
    assert any(
        len(c.get_segments()) == len(steps[-1].extra["edges"])
        for c in ax.collections
        if hasattr(c, "get_segments")
    )


# --------------------------------------------------------------------------- plot_route_map
def test_plot_route_map(bcn_problem, two_trips):
    b = load_barcelona()
    ils = IteratedLocalSearch(n_iter=3, patience=None, random_state=0).fit(
        b.cost, labels=b.labels, coords=b.coords
    )
    fig = plot_route_map(ils)
    assert isinstance(fig, go.Figure)
    assert [t.type for t in fig.data] == ["scattermap"] * 3
    nodes, depot, trip = fig.data
    assert len(nodes.lat) == 18 and len(depot.lat) == 1 and len(trip.lat) == 20
    assert depot.text == (str(b.depot),)
    assert fig.layout.map.style == "open-street-map" and 8 < fig.layout.map.zoom < 13
    assert fig.layout.title.text == f"IteratedLocalSearch | cost {ils.cost_:.6g}"
    assert plot_route_map(ils, zoom=9).layout.map.zoom == 9
    # explicit coords, multi-trip: one line trace per trip, distinct colours
    fig2 = plot_route_map(two_trips, SQUARE + 40.0)
    assert [t.type for t in fig2.data] == ["scattermap"] * 4
    assert fig2.data[2].line.color != fig2.data[3].line.color
    assert fig2.layout.title.text == "BruteForce | cost 41 | 2 trips"
    with pytest.raises(ValueError, match="no coordinates to draw"):
        plot_route_map(two_trips)
    events, _ = fake_run(lambda ev: None, bcn_problem, n_iter=1)
    assert len(plot_route_map(events[-1]).data) == 3
    # the legend stays hidden without trip_names, and the lines keep their generic names
    assert fig2.layout.showlegend is False and [t.name for t in fig2.data[2:]] == ["trip 1", "trip 2"]


def test_plot_route_map_names_and_trip_names(two_trips):
    # names= in matrix row order ("d", "a", "b", "c") become the hover text; the depot keeps its own
    fig = plot_route_map(two_trips, SQUARE + 40.0, names=["Office", "Ana", "Bea", "Carl"])
    nodes, depot = fig.data[0], fig.data[1]
    assert depot.text == ("Office",) and sorted(nodes.text) == ["Ana", "Bea", "Carl"]
    assert nodes.hoverinfo == "text" and depot.hoverinfo == "text"
    # a mapping by label names some nodes; the others show their label
    fig = plot_route_map(two_trips, SQUARE + 40.0, names={"d": "Office", "b": "Bea"})
    assert fig.data[1].text == ("Office",) and sorted(fig.data[0].text) == ["Bea", "a", "c"]
    # trip_names= name the lines and show the legend, in trip order, one colour each
    fig = plot_route_map(two_trips, SQUARE + 40.0, trip_names=["Monday", "Tuesday"])
    assert [t.name for t in fig.data[2:]] == ["Monday", "Tuesday"]
    assert fig.layout.showlegend is True
    assert fig.data[0].showlegend is False and fig.data[1].showlegend is False
    assert fig.data[2].line.color == "#1f77b4" and fig.data[3].line.color == "#ff7f0e"
    with pytest.raises(ValueError, match="trip_names has 1 entries but the plan has 2 trips"):
        plot_route_map(two_trips, SQUARE + 40.0, trip_names=["Monday"])
    with pytest.raises(ValueError, match="names has 3 entries but there are 4 nodes"):
        plot_route_map(two_trips, SQUARE + 40.0, names=["a", "b", "c"])


# --------------------------------------------------------------------------- D31: structures being built
def test_liveplot_draws_construction_edges(dj_problem, dj):
    """A construction step: ``tour=None``, nan costs and ``extra["edges"]`` -- the edges are the picture."""
    labels = list(dj_problem.labels)
    live = LivePlot(dj.coords)
    assert live(structure_event(dj_problem, "start", 0)) is False  # nothing to draw yet
    ax = live.ax
    current, best, edges_line = ax.lines[0], ax.lines[1], ax.lines[2]
    assert edges_line.get_color() == EDGE_COLOR and len(edges_line.get_xdata()) == 0
    assert title(ax) == "Insertion | iteration 0"
    for k in range(2, 7):
        edges = list(pairwise(labels[:k]))  # the path grows one node per step (rows 0 .. k-1)
        live(structure_event(dj_problem, "iteration", k - 1, edges=edges))
        x, y = edges_line.get_xdata(), edges_line.get_ydata()
        assert len(x) == 3 * len(edges) and np.isnan(x[2::3]).all() and np.isnan(y[2::3]).all()
        np.testing.assert_allclose(x[0::3], dj.coords[: k - 1, 0])  # x0 of every edge
        np.testing.assert_allclose(x[1::3], dj.coords[1:k, 0])  # x1 of every edge
        np.testing.assert_allclose(y[1::3], dj.coords[1:k, 1])
        assert len(current.get_xdata()) == 0 and len(best.get_xdata()) == 0
        assert title(ax) == f"Insertion | iteration {k - 1} | edges {len(edges)}"
    live(structure_event(dj_problem, "iteration", 6))  # an event without the key clears the structure
    assert len(edges_line.get_xdata()) == 0
    tour = dj_problem.to_label_tour(initial_tour(dj_problem, "nearest_neighbour", None))
    cost = dj_problem.evaluate(dj_problem.to_index_tour(tour))
    live(FakeEvent("Insertion", "end", 37, cost, cost, tour, tour, dj_problem, {}))
    assert len(final_lines(live)) == 1 and len(edges_line.get_xdata()) == 0
    assert title(ax) == f"Insertion | iteration 37 | cost {cost:.6g} | best {cost:.6g}"
    with pytest.raises(ValueError, match="pairs of the problem's labels"):
        live(structure_event(dj_problem, "iteration", 1, edges=[("nowhere", labels[0])]))
    with pytest.raises(ValueError, match="pairs of the problem's labels"):
        live(structure_event(dj_problem, "iteration", 1, edges=[labels[0]]))


def test_liveplot_draws_pheromone_trails(dj_problem, dj):
    """``extra["edge_weights"]``: the edges become a LineCollection; width and alpha follow the weight."""
    n, labels = dj_problem.n, list(dj_problem.labels)
    rng = np.random.default_rng(0)
    m = 3 * n
    a = rng.integers(0, n, m)
    b = (a + rng.integers(1, n, m)) % n
    edges = [(labels[i], labels[j]) for i, j in zip(a.tolist(), b.tolist(), strict=True)]
    weights = np.linspace(0.0, 1.0, m)
    tour = dj_problem.to_label_tour(initial_tour(dj_problem, "nearest_neighbour", None))
    cost = dj_problem.evaluate(dj_problem.to_index_tour(tour))

    def ant_event(k, **extra):
        return FakeEvent("AntColony", "iteration", k, cost, cost, tour, tour, dj_problem, extra)

    live = LivePlot(dj.coords)
    live(FakeEvent("AntColony", "start", 0, cost, cost, tour, tour, dj_problem, {"n_ants": 10}))
    live(ant_event(1, n_ants=10, edges=edges, edge_weights=weights))
    ax = live.ax
    trails = live._view.trails
    assert trails is ax.collections[2] and len(trails.get_segments()) == m
    np.testing.assert_allclose(trails.get_segments()[-1], dj.coords[[a[-1], b[-1]]])
    lw = np.asarray(trails.get_linewidths())
    assert (
        lw.shape == (m,)
        and np.all(np.diff(lw) > 0)
        and lw[0] == pytest.approx(0.4)
        and lw[-1] == pytest.approx(3.6)
    )
    alpha = np.asarray(trails.get_colors())[:, 3]
    assert np.all(np.diff(alpha) > 0) and alpha[0] == pytest.approx(0.08) and alpha[-1] == pytest.approx(1.0)
    assert trails.get_zorder() < ax.lines[0].get_zorder(), "trails go under the tours"
    assert len(ax.lines[2].get_xdata()) == 0, "weighted edges are the collection, not the plain line"
    assert len(ax.lines[0].get_xdata()) == len(ax.lines[1].get_xdata()) == n + 1  # the tours stay
    assert title(ax) == f"AntColony | iteration 1 | cost {cost:.6g} | best {cost:.6g} | n_ants 10 | edges {m}"
    live(ant_event(2, edges=edges[:3], edge_weights=[2.0, -1.0, np.nan]))  # clipped to [0, 1], nan -> 0
    np.testing.assert_allclose(trails.get_linewidths(), [3.6, 0.4, 0.4])
    with pytest.raises(ValueError, match=r"has 2 values but extra\['edges'\] has 3 pairs"):
        live(ant_event(3, edges=edges[:3], edge_weights=[0.5, 0.5]))
    live(ant_event(4, edges=edges[:3]))  # without weights: the plain line takes over
    assert len(trails.get_segments()) == 0 and len(ax.lines[2].get_xdata()) == 9
    live(ant_event(5))
    assert len(trails.get_segments()) == 0 and len(ax.lines[2].get_xdata()) == 0
    static = plot_route(
        ant_event(6, edges=edges, edge_weights=weights)
    )  # the static picture: same collection
    assert len(static.collections) == 3 and len(static.collections[2].get_segments()) == m


def test_liveplot_draws_the_som_ring(dj_problem, dj):
    m = 20
    theta = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    center, radius = dj.coords.mean(axis=0), dj.coords.std()
    ring = np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))
    live = LivePlot(dj.coords)
    live(structure_event(dj_problem, "start", 0, solver="SOM", radius=30.4, n_units=m, ring=ring))
    ax = live.ax
    ring_line = ax.lines[3]
    assert ring_line.get_color() == RING_COLOR and ring_line.get_marker() == "o"
    xy = ring_line.get_xydata()
    assert xy.shape == (m + 1, 2)
    np.testing.assert_allclose(xy[:-1], ring)
    np.testing.assert_allclose(xy[-1], ring[0])  # closed
    assert title(ax) == f"SOM | iteration 0 | radius 30.4 | ring {m}"  # radius outranks n_units
    tour = dj_problem.to_label_tour(initial_tour(dj_problem, "nearest_neighbour", None))
    cost = dj_problem.evaluate(dj_problem.to_index_tour(tour))
    live(FakeEvent("SOM", "iteration", 1, cost, cost, tour, tour, dj_problem, {"ring": ring * 0.5}))
    np.testing.assert_allclose(ring_line.get_xydata()[:-1], ring * 0.5)
    assert len(ax.lines[0].get_xdata()) == dj_problem.n + 1  # the epoch's decoded tour, too
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        live(structure_event(dj_problem, "iteration", 2, solver="SOM", ring=np.zeros((3, 3))))
    live(FakeEvent("SOM", "iteration", 3, cost, cost, tour, tour, dj_problem, {}))
    assert len(ring_line.get_xdata()) == 0, "cleared without the key"
    ringed = plot_route(FakeEvent("SOM", "iteration", 4, cost, cost, tour, tour, dj_problem, {"ring": ring}))
    assert len(ringed.lines) == 2 and len(ringed.lines[0].get_xdata()) == m + 1  # ring, then the route
    assert ringed.get_title() == f"SOM | cost {cost:.6g}"


def test_liveplot_show_and_trail_options(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=6)
    n1 = dj_problem.n + 1
    best_only = LivePlot(dj.coords, show="best")
    for ev in events[:-1]:
        best_only(ev)
    assert len(best_only.ax.lines[0].get_xdata()) == 0 and len(best_only.ax.lines[1].get_xdata()) == n1
    current_only = LivePlot(dj.coords, show="current")
    for ev in events[:-1]:
        current_only(ev)
    assert len(current_only.ax.lines[0].get_xdata()) == n1 and len(current_only.ax.lines[1].get_xdata()) == 0
    current_only(events[-1])
    assert len(final_lines(current_only)) == 1, "the final route is drawn whatever show= says"
    trailing = LivePlot(dj.coords, trail=3)
    trailing(events[0])
    fade = trailing.ax.lines[BASE_LINES : BASE_LINES + 3]
    assert len(trailing.ax.lines) == BASE_LINES + 3 and all(len(line.get_xdata()) == 0 for line in fade)
    alphas = [line.get_alpha() for line in fade]
    assert alphas == sorted(alphas, reverse=True) and 0 < alphas[-1] < alphas[0] < 1
    trailing(events[1])  # the start's current tour becomes the newest ghost
    np.testing.assert_allclose(fade[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[0].tour))
    assert len(fade[1].get_xdata()) == 0
    for ev in events[2:-1]:
        trailing(ev)
    for age, line in enumerate(fade):  # events 6 is current; 5, 4, 3 fade behind it
        np.testing.assert_allclose(line.get_xydata(), closed_xy(dj_problem, dj.coords, events[5 - age].tour))
    trailing(events[-1])
    assert all(len(line.get_xdata()) == 0 for line in fade), "cleared at the end"
    with pytest.raises(ValueError, match="show must be one of"):
        LivePlot(dj.coords, show="all")
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="trail must be an int >= 0"):
            LivePlot(dj.coords, trail=bad)


def test_plot_route_of_a_construction_step(dj_problem, dj):
    labels = list(dj_problem.labels)
    edges = [(labels[0], labels[1]), (labels[1], labels[2]), (labels[2], labels[0])]
    ax = plot_route(structure_event(dj_problem, "iteration", 3, edges=edges))
    assert len(ax.lines) == 1 and len(ax.lines[0].get_xdata()) == 9 and ax.lines[0].get_color() == EDGE_COLOR
    assert ax.get_title() == "Insertion | 3 edges"
    np.testing.assert_allclose(ax.lines[0].get_xydata()[[0, 1, 3, 4, 6, 7]], dj.coords[[0, 1, 1, 2, 2, 0]])
    scaled = plot_route(
        structure_event(dj_problem, "iteration", 3, edges=edges), dj.coords * 2.0, labels=True
    )
    np.testing.assert_allclose(scaled.lines[0].get_xydata()[0], 2.0 * dj.coords[0])
    assert len(scaled.texts) == dj_problem.n
    assert len(plot_route(structure_event(dj_problem, "start", 0)).lines) == 0  # nothing yet: just the nodes
    with pytest.raises(ValueError, match="pairs of the problem's labels"):
        plot_route(structure_event(dj_problem, "iteration", 1, edges=[(labels[0], "nowhere")]))
    # a recorded construction step draws the same way
    rec = Recorder()
    rec(structure_event(dj_problem, "iteration", 3, edges=edges))
    assert len(plot_route(rec.events[0]).lines[0].get_xdata()) == 9


# --------------------------------------------------------------------------- D31: the recorder's clock
def test_recorder_timestamps_and_structure_copies(dj_problem):
    labels = list(dj_problem.labels)
    edges = [(labels[0], labels[1]), (labels[1], labels[2])]
    weights = [0.2, 0.9]
    ring = np.zeros((4, 2))
    rec = Recorder()
    rec(structure_event(dj_problem, "start", 0))
    rec(structure_event(dj_problem, "iteration", 1, edges=edges, edge_weights=weights, ring=ring, n_units=4))
    ring[:] = 1.0  # the solver keeps working on its buffers...
    weights[0] = 0.0
    edges.append((labels[2], labels[3]))
    kept = rec.events[1].extra  # ...the copies do not move
    assert kept["edges"] == [(labels[0], labels[1]), (labels[1], labels[2])]
    assert all(isinstance(pair, tuple) for pair in kept["edges"])
    np.testing.assert_array_equal(kept["edge_weights"], [0.2, 0.9])
    np.testing.assert_array_equal(kept["ring"], np.zeros((4, 2)))
    assert kept["edge_weights"].dtype == kept["ring"].dtype == np.float64 and kept["n_units"] == 4
    stamps = rec.timestamps
    assert (
        stamps.shape == (2,)
        and stamps.dtype == np.float64
        and 0 < stamps[0] <= stamps[1] <= time.perf_counter()
    )
    assert [e.drawable for e in rec.events] == [False, True] and rec.n_frames == 1
    assert RecordedEvent("X", "start", 0, math.nan, math.nan, None, None).timestamp == 0.0
    bare = Recorder(keep_tours=False)
    bare(structure_event(dj_problem, "iteration", 1, edges=edges, edge_weights=weights, ring=ring, n_units=4))
    assert bare.events[0].extra == {"n_units": 4} and bare.n_frames == 0  # the structures go with the tours
    odd = Recorder()
    odd(structure_event(dj_problem, "iteration", 1, edges=3, ring="not an array"))  # copied as received...
    assert odd.events[0].extra == {"edges": 3, "ring": "not an array"}
    with pytest.raises(ValueError, match="pairs of the problem's labels"):  # ...and reported when drawn
        plot_route(odd.events[0])


def test_recorder_frame_delays_follow_the_clock(dj_problem):
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=3)  # five drawable events
    with_stamps(rec, [0.0, 0.05, 0.1, 3.0, 3.0005])
    np.testing.assert_allclose(
        rec.frame_delays(), [50, 50, 2000, 10, 10]
    )  # gaps in ms, clipped, last repeated
    np.testing.assert_allclose(rec.frame_delays(speed=10), [10, 10, 290, 10, 10])
    np.testing.assert_allclose(rec.frame_delays(speed=0.01), [2000] * 3 + [50] * 2)  # slow motion: capped
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="speed must be > 0"):
            rec.frame_delays(speed=bad)
    labels = list(dj_problem.labels)
    con = Recorder()  # only drawable events are frames: a construction's empty "start" is skipped
    con(structure_event(dj_problem, "start", 0))
    con(structure_event(dj_problem, "iteration", 1, edges=[(labels[0], labels[1])]))
    con(structure_event(dj_problem, "iteration", 2, edges=[(labels[0], labels[1]), (labels[1], labels[2])]))
    with_stamps(con, [0.0, 1.0, 1.5])
    assert con.n_frames == 2 and con.frame_delays().tolist() == [500.0, 500.0]
    with pytest.raises(ValueError, match="nothing to draw"):
        Recorder().frame_delays()


class FakeTimer:
    """Stands in for the animation's event source: records every interval the frames ask for."""

    def __init__(self):
        self.intervals: list[int] = []
        self._interval = 0

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        self._interval = value
        self.intervals.append(value)

    def start(self, *_):
        pass

    def stop(self, *_):
        pass

    def add_callback(self, *_):
        pass

    def remove_callback(self, *_):
        pass


def test_recorder_animate_speed_fps_and_interval(tmp_path, dj_problem, dj):
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=3)
    with_stamps(rec, [0.0, 0.05, 0.1, 3.0, 3.0005])
    lapse = rec.animate(dj.coords, speed=10, figsize=(2, 2))  # 10x: delays 10, 10, 290, 10, 10 ms
    assert lapse._interval == 10 and lapse._repeat_delay == 1000
    timer = FakeTimer()
    lapse.event_source = timer
    lapse._init_draw()  # what plt.show() does: the first frame, then one _step per timer tick
    in_force = []
    for _ in range(7):
        lapse._step()
        in_force.append(timer.interval)  # the wait the timer applies before the next frame
    # TimedAnimation._step copies the animation's interval to the timer after every frame, so a value set
    # on the timer itself would be clobbered and every frame would wait delays[0]: each frame's own delay
    # must be in force, then the hold before the loop restarts, then the first frame's delay again
    assert in_force == [10, 10, 290, 10, 10, 1000, 10]
    lapse.save(tmp_path / "lapse.gif", writer="pillow", dpi=20)  # the same frames, written in order
    assert rec._last_live.show == "both" and rec._last_live.figsize == (2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # these animations are inspected, never rendered
        fixed = rec.animate(dj.coords, fps=25, hold=0.5, show="best", trail=2)
        assert fixed._interval == 40 and fixed._repeat_delay == 500
        assert rec._last_live.show == "best" and len(rec._last_live.ax.lines) == BASE_LINES + 2
        assert rec.animate(dj.coords, interval=7, fps=25).__dict__["_interval"] == 7  # interval wins
        assert rec.animate(dj.coords).__dict__["_interval"] == 50  # speed=1: the recorded pace
        del fixed
        gc.collect()  # matplotlib warns when an animation is collected unrendered: collect it here, quietly
    for kwargs in ({"fps": 0}, {"interval": 0}, {"hold": -1}, {"show": "none"}, {"speed": 0}):
        with pytest.raises(ValueError):
            rec.animate(dj.coords, **kwargs)


def fake_clock(monkeypatch, waits):
    """A ``time`` for ``_record`` whose clock only advances when ``sleep`` is called (which is logged)."""
    now = [1000.0]

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(_record, "time", types.SimpleNamespace(perf_counter=lambda: now[0], sleep=sleep))
    return now


def interactive_window(monkeypatch):
    """Make the Agg backend look like a window to LivePlot (frames seen as drawn), without pausing."""
    from skroute.viz import _live

    monkeypatch.setattr(_live, "backend_is_interactive", lambda: True)
    monkeypatch.setattr(plt, "pause", lambda *_: None)


def test_recorder_replay_drives_a_liveplot(monkeypatch, dj_problem, dj):
    rec = Recorder()
    events, _ = fake_run(rec, dj_problem, n_iter=4)  # start, 4 iterations, end
    with_stamps(rec, [0.0, 0.5, 0.5002, 3.5002, 3.7002, 3.7002])
    waits: list[float] = []
    fake_clock(monkeypatch, waits)
    interactive_window(monkeypatch)
    live = rec.replay(dj.coords, speed=10)
    assert isinstance(live, LivePlot) and live.n_events == 6 and live.n_redraws == 6
    # gaps / 10 on a target clock: the 0.02 ms gap is too short to sleep, so it is carried into the next wait
    np.testing.assert_allclose(waits, [0.05, 0.30002, 0.02], rtol=1e-6)
    np.testing.assert_allclose(
        final_lines(live)[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )
    waits.clear()
    with_stamps(rec, [0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    rec.replay(dj.coords, speed=1)
    assert waits == [2.0], "a long gap is capped at two seconds"
    waits.clear()
    with_stamps(rec, [0.0, 100.0, 100.5, 101.0, 101.5, 102.0])
    rec.replay(dj.coords, speed=1)
    np.testing.assert_allclose(waits, [2.0, 0.5, 0.5, 0.5, 0.5])  # the clock jumps over the cut part of a gap
    waits.clear()
    fast = rec.replay(dj.coords, speed=1e9, show="best", trail=2, every=2, title="again")
    assert waits == [] and fast.show == "best" and fast.title == "again"
    assert (
        len(fast.ax.lines) == BASE_LINES + 2 + 1 and fast.n_redraws == 1 + 2 + 1
    )  # every=2: iterations 1, 3
    assert len(rec.replay(speed=1e9).ax.lines) == BASE_LINES + 1  # coords come from the recorded problem
    with pytest.raises(ValueError, match="speed must be > 0"):
        rec.replay(dj.coords, speed=0)
    with pytest.raises(ValueError, match="nothing to replay"):
        Recorder().replay(dj.coords)
    orphan = Recorder()
    orphan.events.append(RecordedEvent("X", "start", 0, math.nan, math.nan, None, None))
    with pytest.raises(ValueError, match="carry no RoutingProblem"):
        orphan.replay(dj.coords)
    bare = Recorder()
    fake_run(bare, RoutingProblem(dj.distance_matrix(), labels=dj.labels), n_iter=1)  # fitted without coords
    with pytest.raises(ValueError, match="no coordinates to draw"):
        bare.replay(speed=1e9)


def test_recorder_replay_respects_the_recorded_clock(monkeypatch, dj_problem, dj):
    """A dense run — gaps under the 10 ms sleep grain — still lasts the recorded span over speed: the short
    gaps accumulate instead of being dropped, and time spent drawing is absorbed by the next waits."""
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=118)  # 120 events
    gap = 0.0167  # ~60 events per second, the pace of a metaheuristic's outer loop
    with_stamps(rec, np.arange(len(rec)) * gap)
    span = (len(rec) - 1) * gap
    waits: list[float] = []
    now = fake_clock(monkeypatch, waits)
    interactive_window(monkeypatch)
    for speed in (10.0, 1.0):
        waits.clear()
        start = now[0]
        rec.replay(dj.coords, speed=speed)
        # every sleep is at least the grain, the sum is the scaled span (minus at most one grain, the tail)
        assert min(waits) >= 0.01 and len(waits) <= span / speed / 0.01 + 1
        assert span / speed - 0.01 < sum(waits) <= span / speed + 1e-9
        assert now[0] - start == pytest.approx(sum(waits))
    # drawing time is absorbed: a clock that also advances 5 ms per redraw changes nothing in the total
    waits.clear()
    original_call = LivePlot.__call__

    def slow_call(self, event):
        now[0] += 0.005
        return original_call(self, event)

    monkeypatch.setattr(LivePlot, "__call__", slow_call)
    start = now[0]
    rec.replay(dj.coords, speed=1.0)
    assert span - 0.01 < now[0] - start <= span + 0.005 + 1e-9
    assert sum(waits) == pytest.approx(now[0] - start - 0.005 * len(rec))
    waits.clear()
    with_stamps(rec, np.arange(len(rec)) * 0.001)  # drawn slower than its pace: no waits, no error
    rec.replay(dj.coords, speed=1.0)
    assert waits == []


def test_recorder_replay_does_not_wait_where_nothing_is_shown(monkeypatch, dj_problem, dj):
    """On Agg (and the plotly backend outside a notebook) nothing shows before "end": no sleeping."""
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=4)
    with_stamps(rec, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    waits: list[float] = []
    fake_clock(monkeypatch, waits)
    live = rec.replay(dj.coords, speed=1)
    assert waits == [] and live.n_redraws == 6 and len(final_lines(live)) == 1
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: None)
    fig = rec.replay(dj.coords, speed=1, backend="plotly").fig
    assert waits == [] and isinstance(fig, go.Figure)
    interactive_window(monkeypatch)  # a window: paced
    rec.replay(dj.coords, speed=1)
    assert waits == [0.5] * 5


def test_recorder_save_gif_and_movie(monkeypatch, tmp_path, dj_problem, dj):
    from matplotlib import animation

    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=3)
    with_stamps(rec, [0.0, 0.1, 0.2, 0.3, 0.4])
    out = rec.save(tmp_path / "run.gif", dj.coords, fps=10, dpi=20, hold=0)
    assert out == tmp_path / "run.gif" and out.stat().st_size > 0 and plt.get_fignums() == []
    with Image.open(out) as im:
        assert im.n_frames in (4, 5) and im.size == (120, 120)  # 6 in x 20 dpi
        durations = []
        for k in range(im.n_frames):
            im.seek(k)
            durations.append(im.info["duration"])
    assert sum(durations) == 500  # five frames at 10 fps
    # a time-lapse at 0.5x: 100 ms gaps become 200 ms = two frames each; identical frames fold, so the GIF
    # keeps five pictures whose durations carry the pace, plus a one-second hold on the last one
    lapse = rec.save(tmp_path / "lapse.gif", dj.coords, fps=10, speed=0.5, hold=1.0)
    with Image.open(lapse) as im:
        assert im.n_frames in (4, 5)
        total = 0
        for k in range(im.n_frames):
            im.seek(k)
            total += im.info["duration"]
    assert total == 5 * 200 + 1000
    assert len(final_lines(rec._last_live)) == 1  # the last picture is the final route
    default = rec.save(
        tmp_path / "default.gif", fps=10, dpi=10, hold=0, show="current"
    )  # coords from the fit
    assert default.stat().st_size > 0
    monkeypatch.setattr(animation.writers, "is_available", lambda name: False)
    with pytest.raises(RuntimeError, match="needs ffmpeg on the PATH"):
        rec.save(tmp_path / "run.mp4", dj.coords)
    monkeypatch.undo()
    if animation.writers.is_available("ffmpeg"):
        movie = rec.save(tmp_path / "run.mp4", dj.coords, fps=5, dpi=20, hold=0)
        assert movie.stat().st_size > 0
    for kwargs in ({"fps": 0}, {"hold": -1}, {"show": "none"}, {"speed": 0}):
        with pytest.raises(ValueError):
            rec.save(tmp_path / "bad.gif", dj.coords, **kwargs)
    with pytest.raises(ValueError, match="nothing to draw"):
        Recorder(keep_tours=False).save(tmp_path / "empty.gif", dj.coords)


def test_recorder_to_plotly_shares_the_replay_guards(dj_problem, dj, bcn_problem):
    orphan = Recorder()
    orphan.events.append(RecordedEvent("X", "end", 1, 1.0, 1.0, np.arange(38), np.arange(38), {}, None))
    with pytest.raises(ValueError, match="carry no RoutingProblem"):
        orphan.to_plotly(dj.coords)
    with pytest.raises(ValueError, match="carry no RoutingProblem"):
        orphan.animate(dj.coords)
    mixed = Recorder()  # two instances in one recorder: the coordinates cannot fit both
    fake_run(mixed, dj_problem, n_iter=1)
    fake_run(mixed, bcn_problem, n_iter=1)
    assert mixed.problem is bcn_problem
    with pytest.raises(ValueError, match="coords has 19 rows but the problem has 38 nodes"):
        mixed.to_plotly()
    with pytest.raises(ValueError, match="coords has 19 rows but the problem has 38 nodes"):
        mixed.animate()
    with pytest.raises(ValueError, match="coords has 38 rows but the problem has 19 nodes"):
        mixed.to_plotly(dj.coords)


def test_recorder_save_closes_its_figure_when_the_writer_fails(tmp_path, dj_problem, dj):
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=2)
    assert plt.get_fignums() == []
    with pytest.raises(FileNotFoundError):
        rec.save(tmp_path / "no_such_dir" / "run.gif", dj.coords, fps=5, dpi=10)
    assert plt.get_fignums() == [], "a failed save leaves no figure behind"


def test_viewers_copy_the_coordinates(dj_problem, dj):
    """Mutating the caller's array after construction moves nothing that is drawn."""
    xy = np.array(dj.coords, dtype=np.float64)
    live = LivePlot(xy)
    assert live.coords is not xy and live.coords.dtype == np.float64
    events, _ = fake_run(live, dj_problem, n_iter=1)
    xy[:] += 1000.0
    live(events[1])
    np.testing.assert_allclose(
        live.ax.lines[1].get_xydata(), closed_xy(dj_problem, dj.coords, events[1].best_tour)
    )
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=1)
    kept = rec._coords(xy)
    assert kept is not xy
    np.testing.assert_array_equal(kept, xy)


def test_plotly_keeps_the_trails_relative_to_the_strongest(dj_problem, dj):
    """The Plotly cut is a quarter of the strongest trail, not an absolute 0.25: AntColony's weights are
    ``tau / tau_max``, whose maximum sits well under 1."""
    labels = list(dj_problem.labels)
    edges = [(labels[0], labels[1]), (labels[1], labels[2]), (labels[2], labels[3])]
    rec = Recorder()
    rec(
        structure_event(
            dj_problem, "iteration", 1, solver="AntColony", edges=edges, edge_weights=[0.2, 0.12, 0.5]
        )
    )
    fig = rec.to_plotly(dj.coords)
    x0, x1, x2, x3 = dj.coords[:4, 0].tolist()
    assert list(fig.frames[0].data[2].x) == [x0, x1, None, x2, x3]  # 0.2 >= 0.125 stays; 0.12 goes
    live = LivePlot(dj.coords, backend="plotly")
    live(
        structure_event(dj_problem, "start", 0, solver="AntColony", edges=edges, edge_weights=[0.0, 0.0, 0.0])
    )
    assert list(live.fig.data[4].x) == []  # no strongest trail: nothing drawn
    live(
        structure_event(
            dj_problem, "iteration", 1, solver="AntColony", edges=edges, edge_weights=[1.0, 0.3, 0.2]
        )
    )
    assert list(live.fig.data[4].x) == [x0, x1, None, x1, x2]  # 0.3 stays, 0.2 (< 0.25) goes


def test_recorder_to_plotly_speed_menu_and_structures(dj_problem, dj):
    rec = Recorder()
    fake_run(rec, dj_problem, n_iter=2)
    fig = rec.to_plotly(dj.coords, fps=20)
    play, speed = fig.layout.updatemenus
    assert [b.label for b in play.buttons] == ["Play", "Pause"]
    assert (
        play.buttons[0].args[1]["frame"]["duration"] == 50
        and play.buttons[1].args[1]["frame"]["duration"] == 0
    )
    assert speed.type == "dropdown" and speed.active == 1
    assert [b.label for b in speed.buttons] == ["0.5x", "1x", "2x", "4x", "8x"]
    assert [b.args[1]["frame"]["duration"] for b in speed.buttons] == [100, 50, 25, 12.5, 6.25]
    assert all(b.method == "animate" for b in speed.buttons)
    assert all(list(f.traces) == [2, 3, 4, 5] for f in fig.frames), "frames update current, best, edges, ring"
    assert [t.name for t in fig.data] == ["nodes", "depot", "current", "best", "edges", "ring"]
    assert len(fig.frames[1].data[0].x) == dj_problem.n + 1  # an iteration: the current tour is drawn...
    assert not rec.to_plotly(dj.coords, show="best").frames[1].data[0].x  # ...unless only the best is wanted
    assert not rec.to_plotly(dj.coords, show="current").frames[1].data[1].x
    labels = list(dj_problem.labels)
    con = Recorder()
    con(structure_event(dj_problem, "start", 0))  # nothing to draw: not a frame
    con(structure_event(dj_problem, "iteration", 1, edges=[(labels[0], labels[1]), (labels[1], labels[2])]))
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    con(
        structure_event(
            dj_problem,
            "iteration",
            2,
            solver="SOM",
            edges=[(labels[0], labels[1]), (labels[1], labels[2])],
            edge_weights=[0.1, 0.9],
            ring=ring,
        )
    )
    fig = con.to_plotly(dj.coords)
    assert len(fig.frames) == 2
    x0, x1, x2 = dj.coords[:3, 0].tolist()
    assert list(fig.frames[0].data[2].x) == [x0, x1, None, x1, x2]  # edges, None between them
    assert list(fig.frames[1].data[2].x) == [x1, x2]  # only the trail whose weight reaches 0.25
    assert list(fig.frames[1].data[3].x) == [0.0, 1.0, 1.0, 0.0] and list(fig.frames[1].data[3].y) == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert fig.frames[1].layout.title.text == "SOM | iteration 2 | edges 2 | ring 3"
    assert fig.layout.title.text == "Insertion | iteration 1 | edges 2"
    with pytest.raises(ValueError, match="fps must be > 0"):
        rec.to_plotly(dj.coords, fps=0)
    with pytest.raises(ValueError, match="show must be one of"):
        rec.to_plotly(dj.coords, show="none")
    live = LivePlot(dj.coords, backend="plotly", show="best")  # the live Plotly view draws the same traces
    live(structure_event(dj_problem, "start", 0, solver="SOM", ring=ring))
    assert list(live.fig.data[5].x) == [0.0, 1.0, 1.0, 0.0] and [t.name for t in live.fig.data][2:] == [
        "current",
        "best",
        "edges",
        "ring",
    ]
