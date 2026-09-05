"""Smoke test of the worked case ``examples/technician_madrid.py`` (D35) on the committed data.

The example is run as a subprocess in ``--quick`` mode (a few iterations, no wall-clock budget, so the
result is deterministic), headless (``MPLBACKEND=Agg``), from a working directory outside the checkout,
writing into a temporary ``--out`` directory; the test then reads the CSVs back and checks that every
day respects the eight-hour budget.

The script and its data are located from **this file**, never from ``skroute.__file__``: the sdist and
wheel jobs import skroute from site-packages, where no ``examples/`` exists (D16), so the module skips
when the checkout is not beside it. The subprocess imports the same skroute as this process (its parent
directory goes first on ``PYTHONPATH``, as the other subprocess tests do, without dropping what was
there). The Plotly map and the PNGs are optional outputs -- the script degrades to a warning without
plotly or matplotlib, and cibuildwheel's test environment has neither -- so they are asserted only when
the extra is importable, and the degradation itself is tested by hiding both.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import skroute
from skroute.metrics import timetable, timetable_summary
from skroute.utils import Bunch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "technician_madrid.py"
DATA = ROOT / "examples" / "data"
if not EXAMPLE.is_file() or not DATA.is_dir():
    pytest.skip("examples/ is not beside tests/: not a checkout", allow_module_level=True)


def _importable(name: str) -> bool:
    """Whether the optional extra is there -- also under a finder that refuses the module outright."""
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


HAS_MATPLOTLIB = _importable("matplotlib")
HAS_PLOTLY = _importable("plotly")
PREFIX = "technician_madrid"
N_RESTAURANTS = 182
HOURS = 8.0
SERVICE = 30.0
MANDATORY = ("_timetable.csv", "_days.csv", ".kml", "_google_urls.txt")
HIDE_VIZ = "import sys\nfor name in ('matplotlib', 'plotly', 'PIL'):\n    sys.modules[name] = None\n"


def _env() -> dict[str, str]:
    """The subprocess environment: headless, no Google key, the skroute of this process first on the path."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.pop("GOOGLE_MAPS_API_KEY", None)  # the Google page is written only with a key
    package_parent = str(Path(skroute.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_parent, os.environ.get("PYTHONPATH", "")) if p)
    return env


def _run(out: Path, *extra: str, prelude: str = "") -> subprocess.CompletedProcess[str]:
    """Run the example in ``--quick`` mode into ``out``, from ``out`` (outside the checkout, like CI).

    With ``prelude`` the script runs through ``runpy`` after that code -- the way to hide optional modules.
    """
    args = ["--quick", "--data", str(DATA), "--out", str(out), *extra]
    if prelude:
        code = (
            f"{prelude}import runpy, sys\nsys.argv = [{str(EXAMPLE)!r}, *{args!r}]\n"
            f"runpy.run_path({str(EXAMPLE)!r}, run_name='__main__')\n"
        )
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [sys.executable, str(EXAMPLE), *args]
    return subprocess.run(cmd, env=_env(), cwd=out, capture_output=True, text=True, check=False, timeout=180)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("technician")
    result = _run(out)
    assert result.returncode == 0, result.stderr
    return out, result.stderr


@pytest.fixture(scope="module")
def example() -> ModuleType:
    """The script as a module, for the pieces that are tested in-process."""
    spec = importlib.util.spec_from_file_location("technician_madrid", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small(example) -> Bunch:
    """The office and the first twelve restaurants of the committed data: a one-day instance."""
    data = example.load_data(DATA)
    keep = list(range(13))
    return Bunch(
        labels=[data.labels[i] for i in keep],
        names=[data.names[i] for i in keep],
        addresses=[data.addresses[i] for i in keep],
        coords=data.coords[keep],
        time=data.time[np.ix_(keep, keep)],
        distance=data.distance[np.ix_(keep, keep)],
        rows=[data.rows[i] for i in keep],
    )


@pytest.fixture(scope="module")
def plan(example, small):
    args = example.build_parser().parse_args(["--quick", "--verbose"])
    return example.solve(args, small)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- the committed data
def test_data_files_are_consistent(example):
    points = _rows(DATA / "madrid_burger_king.csv")
    assert len(points) == N_RESTAURANTS + 1 and points[0]["label"] == "office"
    # the office row is what --refresh writes: its addr:* columns are the script's constants
    assert {c: points[0][c] for c in example.OFFICE_ADDR} == example.OFFICE_ADDR
    labels = [r["label"] for r in points]
    for name in ("madrid_burger_king_times_min.csv", "madrid_burger_king_dist_km.csv"):
        with (DATA / name).open(encoding="utf-8", newline="") as fh:
            table = list(csv.reader(fh))
        assert table[0][1:] == labels and [r[0] for r in table[1:]] == labels
        values = [float(v) for r in table[1:] for v in r[1:]]
        assert len(values) == len(labels) ** 2 and min(values) == 0.0 and all(v == v for v in values)


def test_paths_come_from_the_checkout_not_from_the_package(example, monkeypatch):
    """Regression: the sdist and wheel jobs import skroute from site-packages, which has no ``examples/``."""
    assert example.DEFAULT_DATA == DATA and EXAMPLE.parent == DATA.parent
    assert "site-packages" not in str(EXAMPLE) or "site-packages" in str(Path(__file__))
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "AIza-not-for-the-test")
    env = _env()
    path = env["PYTHONPATH"].split(os.pathsep)
    assert path[0] == str(Path(skroute.__file__).resolve().parents[1]) and path[-1] == "elsewhere"
    assert env["MPLBACKEND"] == "Agg" and "GOOGLE_MAPS_API_KEY" not in env


# --------------------------------------------------------------------------- the quick run
def test_example_writes_the_plan(run):
    out, stderr = run
    for suffix in MANDATORY:
        assert (out / f"{PREFIX}{suffix}").is_file(), suffix
    assert not (out / f"{PREFIX}_google.html").exists()  # no key given
    kml = ET.parse(out / f"{PREFIX}.kml").getroot()
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    folders = kml.findall(".//k:Folder", ns)
    assert folders and folders[0].find("k:name", ns).text == "Día 1"
    urls = (out / f"{PREFIX}_google_urls.txt").read_text(encoding="utf-8").splitlines()
    assert urls[0] == "day\tleg\turl"
    assert all(line.split("\t")[2].startswith("https://www.google.com/maps/dir/?api=1&") for line in urls[1:])
    # the optional outputs: written with the extra, a warning without it (cibuildwheel has neither extra)
    if HAS_PLOTLY:
        html = (out / f"{PREFIX}_map.html").read_text(encoding="utf-8")
        assert "open-street-map" in html
        # plotly escapes non-ASCII text as \uXXXX unless orjson is installed: accept both spellings
        assert "Día 1" in html or "D\\u00eda 1" in html
    else:
        assert not (out / f"{PREFIX}_map.html").exists() and "Plotly map skipped" in stderr
    if HAS_MATPLOTLIB:
        for suffix in ("_days.png", "_day1.png"):
            assert (out / f"{PREFIX}{suffix}").stat().st_size > 0, suffix
    else:
        assert not (out / f"{PREFIX}_days.png").exists() and "PNG pictures skipped" in stderr


def test_every_day_respects_the_budget(run):
    out, _ = run
    days = _rows(out / f"{PREFIX}_days.csv")
    timetable_rows = _rows(out / f"{PREFIX}_timetable.csv")
    assert 12 <= len(days) <= 18  # 12 is the service-only lower bound; a quick search lands a little above 15
    assert sum(int(r["n_stops"]) for r in days) == N_RESTAURANTS
    for day in days:
        assert float(day["total_min"]) <= HOURS * 60 + 1e-9
        parts = float(day["driving_min"]) + float(day["service_min"])
        assert float(day["total_min"]) == pytest.approx(parts, abs=0.02)
        assert float(day["service_min"]) == pytest.approx(int(day["n_stops"]) * SERVICE)
        # the kilometres file is read for something: the road distance of the day, at a plausible speed
        speed = float(day["driving_km"]) / float(day["driving_min"]) * 60
        assert 10 < speed < 120, day
    # the per-stop timetable tells the same story: every day starts and ends at the office within the budget
    by_day: dict[str, list[dict[str, str]]] = {}
    for row in timetable_rows:
        by_day.setdefault(row["day"], []).append(row)
    assert len(by_day) == len(days)
    visited = []
    for day, stops in by_day.items():
        assert stops[0]["label"] == "office" and stops[-1]["label"] == "office"
        assert stops[0]["arrival"] == "08:00"
        total = sum(float(s["travel_min"]) + float(s["service_min"]) for s in stops)
        assert total <= HOURS * 60 + 1e-9
        assert total == pytest.approx(float(next(d for d in days if d["day"] == day)["total_min"]), abs=0.02)
        visited.extend(s["label"] for s in stops[1:-1])
    assert len(visited) == len(set(visited)) == N_RESTAURANTS


def test_console_report(run):
    _, stderr = run
    assert "Totals:" in stderr and "Lower bound: 182 visits x 30 min" in stderr
    assert "Search: MultiStart (Or-opt relocations, optimal split) gave" in stderr
    assert "Baseline NearestNeighbour" in stderr and "Baseline Insertion" in stderr
    assert "OpenStreetMap contributors" in stderr
    assert "IteratedLocalSearch iteration" not in stderr  # the solvers' own records need --verbose


def test_example_degrades_without_the_viz_extras(tmp_path):
    """Regression: the wheel jobs have neither matplotlib nor plotly; the plan is still written, minus the
    pictures, with a warning each -- and the test suite must not demand them there."""
    result = _run(tmp_path, prelude=HIDE_VIZ)
    assert result.returncode == 0, result.stderr
    for suffix in MANDATORY:
        assert (tmp_path / f"{PREFIX}{suffix}").is_file(), suffix
    for suffix in ("_map.html", "_days.png", "_day1.png"):
        assert not (tmp_path / f"{PREFIX}{suffix}").exists(), suffix
    assert "Plotly map skipped" in result.stderr and "PNG pictures skipped" in result.stderr
    assert "Totals:" in result.stderr and "Traceback" not in result.stderr


# --------------------------------------------------------------------------- the script's pieces
def test_verbose_reaches_every_solver(example, small, plan):
    """Regression: ``--verbose`` promised the solvers' own progress and reached no estimator."""
    parser = example.build_parser()
    for solver in ("multistart", "ils", "sa", "tabu", "genetic"):
        loud = example.build_search(parser.parse_args(["--quick", "--verbose", "--solver", solver]), 10.0)
        quiet = example.build_search(parser.parse_args(["--quick", "--solver", solver]), 10.0)
        assert loud.verbose == 1 and quiet.verbose == 0, solver
        if solver == "multistart":
            assert loud.estimator.verbose == 1 and quiet.estimator.verbose == 0
    est, facts = plan  # solved with --verbose: the polish is loud too, and the report names what ran
    assert est.verbose == 1 and facts["search"] == "MultiStart (Or-opt relocations, optimal split)"
    args = parser.parse_args(["--quick", "--solver", "sa"])
    assert example.solve(args, small)[1]["search"] == "SimulatedAnnealing (optimal split)"


def test_bad_arguments_fail_before_solving(example, tmp_path, capsys):
    parser = example.build_parser()
    for bad in ("25:00", "8am", "08:00:30", "8:60"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--start", bad])
        assert exc.value.code == 2, bad
        assert "must be HH:MM" in capsys.readouterr().err
    assert parser.parse_args(["--start", " 7:30 "]).start == "7:30"
    assert parser.parse_args([]).start == "08:00"

    with pytest.raises(SystemExit) as exc:
        example.main(["--quick", "--data", str(tmp_path / "nowhere"), "--out", str(tmp_path)])
    assert exc.value.code == 2 and "not found: pass --data DIR or run --refresh" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:  # a round trip plus the service does not fit in two hours
        example.main(["--quick", "--hours", "2", "--data", str(DATA), "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "the day is too short for some restaurants" in err
    assert "raise --hours or lower --service" in err and len(err) < 1500 and "Traceback" not in err
    assert not (tmp_path / f"{PREFIX}_days.csv").exists()

    with pytest.raises(SystemExit) as exc:
        example.main(["--quick", "--limit", "5", "--data", str(DATA), "--out", str(tmp_path)])
    assert exc.value.code == 2 and "--limit N needs --refresh" in capsys.readouterr().err


def test_pngs_leave_the_backend_alone(example, small, plan, tmp_path):
    """Regression: the PNGs used to force Agg through pyplot, which killed the ``--live`` window's backend."""
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    est, _ = plan
    days = timetable(est, start="08:00")
    summary = timetable_summary(days)
    before = matplotlib.get_backend()
    matplotlib.use("pdf")  # a non-Agg backend, standing in for the GUI window of --live
    try:
        example.plot_days(est, small, summary, tmp_path / "days.png")
        example.plot_day(est, small, days, 0, tmp_path / "day1.png")
        assert matplotlib.get_backend().lower() == "pdf"
        assert plt.get_fignums() == []  # nothing went through pyplot
    finally:
        matplotlib.use(before)
    assert (tmp_path / "days.png").stat().st_size > 0 and (tmp_path / "day1.png").stat().st_size > 0


def test_unroutable_pairs_are_filled_per_matrix(example, caplog):
    """Regression: a server without a ``distances`` table left the kilometres file full of ``nan``."""
    coords = np.array([[40.0, -3.7], [40.1, -3.6], [40.2, -3.8]])
    time_ = np.array([[0.0, 10.0, np.nan], [10.0, 0.0, 12.0], [np.nan, 12.0, 0.0]])
    distance = np.full((3, 3), np.nan)
    np.fill_diagonal(distance, 0.0)
    with caplog.at_level(logging.WARNING, logger="skroute"):
        t, d = example._fill_unroutable(time_, distance, coords)
    assert np.isfinite(t).all() and np.isfinite(d).all()
    assert t[0, 1] == 10.0 and t[1, 2] == 12.0  # routed pairs untouched
    assert d[0, 1] > 5 and t[0, 2] == pytest.approx(d[0, 2] / 30 * 60)  # great circle, 30 km/h
    assert "2 unroutable times and 6 missing distances" in caplog.text
    t2, d2 = example._fill_unroutable(t, d, coords)
    assert np.array_equal(t2, t) and np.array_equal(d2, d)


def test_trip_km_sums_the_legs(example, small, plan):
    est, _ = plan
    km = example.trip_km(est, small)
    assert len(km) == est.n_trips_ and all(k > 0 for k in km)
    index = {label: i for i, label in enumerate(small.labels)}
    first = [index[label] for label in est.trips_[0].tolist()]
    assert km[0] == pytest.approx(sum(small.distance[a, b] for a, b in pairwise(first)))
