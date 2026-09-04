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

import math
import os
import re
import subprocess
import sys
import types
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
    BruteForce,
    IteratedLocalSearch,
    MultiStart,
    NearestNeighbour,
    RoutingProblem,
    SimulatedAnnealing,
)
from skroute.datasets import load_barcelona, load_tsp
from skroute.utils import initial_tour
from skroute.viz import LivePlot, RecordedEvent, Recorder, plot_history, plot_route, plot_route_map

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
        "plot_history",
        "plot_route",
        "plot_route_map",
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
    final = ax.lines[2:]
    assert len(final) == 1  # one trip drawn with trip colours
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour))
    assert ax.get_title().startswith("FakeDescent | iteration 9 | cost ")
    assert f"best {events[-1].best_cost:.6g}" in ax.get_title()


def test_liveplot_lines_and_title_follow_the_events(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=4)
    live = LivePlot(dj.coords, title="watching")
    assert live(events[0]) is False
    ax = live.ax
    assert len(ax.lines) == 2 and len(ax.collections) == 2
    np.testing.assert_allclose(
        ax.lines[1].get_xydata(), closed_xy(dj_problem, dj.coords, events[0].best_tour)
    )
    assert (
        ax.get_title()
        == f"watching | iteration 0 | cost {events[0].cost:.6g} | best {events[0].best_cost:.6g}"
    )
    for ev in events[1:3]:
        assert live(ev) is False
        np.testing.assert_allclose(ax.lines[0].get_xydata(), closed_xy(dj_problem, dj.coords, ev.tour))
        np.testing.assert_allclose(ax.lines[1].get_xydata(), closed_xy(dj_problem, dj.coords, ev.best_tour))
        assert ax.get_title() == (
            f"watching | iteration {ev.iteration} | cost {ev.cost:.6g} | best {ev.best_cost:.6g}"
            f" | temperature {1.0 / ev.iteration:.6g}"
        )
    assert live.n_redraws == 3


def test_liveplot_every_skips_redraws(dj_problem, dj):
    events, _ = fake_run(lambda ev: None, dj_problem, n_iter=9)
    live = LivePlot(dj.coords, every=3)
    for ev in events[:-1]:  # start + 9 iterations, no end
        live(ev)
    assert live.n_events == 10
    assert live.n_redraws == 4  # start + iterations 1, 4 and 7
    assert "iteration 7 |" in live.ax.get_title()
    live(events[-1])
    assert live.n_redraws == 5 and "iteration 9 |" in live.ax.get_title()


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
    live(events[1])  # no "start" seen: the figure is created on the first event received
    assert live.fig is not None and live.n_redraws == 2
    live(events[-1])
    final = live.ax.lines[2:]
    assert len(final) == len(events[-1].trips) >= 1
    assert "FakeDescent | iteration 2" in live.ax.get_title()
    live(events[1])  # an iteration after an "end" without a new "start": the final route must not stay
    assert len(live.ax.lines) == 2 and live._view.final_lines == []


def test_liveplot_nested_start_resets_the_drawing_in_place(dj_problem, dj):
    """The start/end of every restart a MultiStart forwards redraw from scratch in the same figure."""
    outer = FakeEvent("MultiStart", "start", 0, math.nan, math.nan, None, None, dj_problem, {"n_restarts": 2})
    live = LivePlot(dj.coords, every=2)
    assert live(outer) is False and live.ax.get_title() == "MultiStart | iteration 0 | n_restarts 2"
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
                assert len(live.ax.lines) == 2, "the previous restart's final route is gone"
                assert f"restart {restart}" in live.ax.get_title()
        assert len(live.ax.lines) == 3 and live.fig is fig  # this restart's final route, same figure
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
        live.ax.lines[2].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )


def test_liveplot_validation(dj):
    with pytest.raises(ValueError, match='map=True needs backend="plotly"'):
        LivePlot(dj.coords, map=True)
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
    assert live.fig.number not in plt.get_fignums() and len(live.ax.lines) == 3
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
    assert [t.type for t in live.fig.data] == ["scatter"] * 4
    best = live.fig.data[3]
    np.testing.assert_allclose(
        np.column_stack([best.x, best.y]), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )
    assert live.fig.data[2].x is None or len(live.fig.data[2].x) == 0  # current line cleared at end
    assert live.fig.layout.title.text.startswith("FakeDescent | iteration 4 | cost ")
    assert live.fig.layout.yaxis.scaleanchor == "x"


def test_liveplot_plotly_map(monkeypatch, bcn_problem):
    monkeypatch.setattr("plotly.basedatatypes.BaseFigure.show", lambda self, *a, **k: None)
    live = LivePlot(bcn_problem.coords, backend="plotly", map=True, every=2)
    fake_run(live, bcn_problem, n_iter=5)
    assert [t.type for t in live.fig.data] == ["scattermap"] * 4
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
    assert len(ax.collections) == 2  # points and depot, drawn once
    path = tmp_path / "run.gif"
    anim.save(path, writer="pillow", dpi=40)
    with Image.open(path) as im:
        # 5 frames rendered; Pillow folds the "end" frame into the identical last iteration frame
        assert im.n_frames in (4, 5)
    # the last frame drawn is the last best tour, one line, titled with its cost
    assert len(ax.lines) == 1
    np.testing.assert_allclose(
        ax.lines[0].get_xydata(), closed_xy(dj_problem, dj.coords, events[-1].best_tour)
    )
    assert ax.get_title() == f"FakeDescent | iteration 3 | best {events[-1].best_cost:.6g}"


def test_recorder_animate_multi_trip_colours(tmp_path, two_trips):
    rec = Recorder()
    fake_run(rec, two_trips.problem_, n_iter=2)
    anim = rec.animate(SQUARE, trip_colors=True)
    ax = plt.gcf().axes[0]
    anim.save(
        tmp_path / "trips.gif", writer="pillow", dpi=30
    )  # renders every frame; the last one stays drawn
    n_trips = len(plot_route(rec.events[-1], SQUARE).lines)  # a recorded event is drawable on its own
    assert len(ax.lines) == n_trips >= 1
    assert len({line.get_color() for line in ax.lines}) == n_trips
    plain = rec.animate(SQUARE, trip_colors=False)
    ax2 = plt.gcf().axes[0]
    plain.save(tmp_path / "plain.gif", writer="pillow", dpi=30)
    assert len({line.get_color() for line in ax2.lines}) == 1


def test_recorder_to_plotly(dj_problem, dj, bcn_problem):
    rec = Recorder(every=2)
    fake_run(rec, dj_problem, n_iter=6)
    fig = rec.to_plotly(dj.coords)
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == rec.n_frames == 5
    assert [t.type for t in fig.data] == ["scatter"] * 3
    assert len(fig.layout.sliders) == 1 and len(fig.layout.sliders[0].steps) == 5
    assert [s.label for s in fig.layout.sliders[0].steps] == ["0", "1", "3", "5", "6"]
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["Play", "Pause"]
    assert fig.frames[-1].layout.title.text.startswith("FakeDescent | iteration 6 | best ")
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
    assert (
        nested.to_plotly(dj.coords)
        .frames[0]
        .layout.title.text.startswith("FakeDescent | restart 0 | iteration 0")
    )
    np.testing.assert_allclose(
        np.column_stack([fig.frames[-1].data[0].x, fig.frames[-1].data[0].y]),
        closed_xy(dj_problem, dj.coords, rec.events[-1].best_tour),
    )
    geo = Recorder()
    fake_run(geo, bcn_problem, n_iter=2)
    on_map = geo.to_plotly(bcn_problem.coords, map=True)
    assert [t.type for t in on_map.data] == [
        "scattermap"
    ] * 3 and on_map.layout.map.style == "open-street-map"
    assert len(on_map.frames) == 4 and on_map.frames[0].data[0].type == "scattermap"


# --------------------------------------------------------------------------- the real protocol
def test_real_liveplot_counts_events_and_redraws(dj):
    live = LivePlot(dj.coords, every=50)
    sa = SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=live)
    assert live.n_events == sa.n_iter_ + 2  # start, one event per level, end
    assert live.n_redraws == 2 + math.ceil(sa.n_iter_ / 50)
    final = live.ax.lines[2:]
    assert len(final) == 1
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(sa.problem_, dj.coords, sa.tour_))
    assert live.ax.get_title().startswith(f"SimulatedAnnealing | iteration {sa.n_iter_} | cost ")
    assert f"best {sa.cost_:.6g}" in live.ax.get_title()


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
    assert live.fig is not fig1 and len(live.ax.lines) == 3  # a fresh figure: current, best, one final route
    np.testing.assert_allclose(
        live.ax.lines[2].get_xydata(), closed_xy(second.problem_, dj.coords, second.tour_)
    )
    assert len(fig1.axes[0].lines) == 3 and len(plt.get_fignums()) == 2  # the first figure is left as it was
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
    assert lines_seen == [2] * 6
    assert len(plt.get_fignums()) == 1, "one figure for the whole ensemble"
    final = live.ax.lines[2:]
    assert len(final) == 1
    np.testing.assert_allclose(final[0].get_xydata(), closed_xy(ms.problem_, dj.coords, ms.tour_))
    assert live.ax.get_title().startswith("MultiStart | ")


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
    assert [s.label for s in steps] == [f"{r}:{i}" for r in (0, 1) for i in (0, 1, 2, 3, 4, 4)] + ["0"]


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
