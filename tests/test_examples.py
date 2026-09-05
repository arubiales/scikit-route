"""Smoke test of the worked case ``examples/technician_madrid.py`` (D35) on the committed data.

The example is run as a subprocess in ``--quick`` mode (a few iterations, no wall-clock budget, so the
result is deterministic), headless (``MPLBACKEND=Agg``), writing into a temporary ``--out`` directory;
the test then reads the CSVs back and checks that every day respects the eight-hour budget.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import skroute

ROOT = Path(skroute.__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "technician_madrid.py"
DATA = ROOT / "examples" / "data"
PREFIX = "technician_madrid"
N_RESTAURANTS = 182
HOURS = 8.0
SERVICE = 30.0


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("technician")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT)}
    env.pop("GOOGLE_MAPS_API_KEY", None)  # the Google page is written only with a key
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--quick", "--data", str(DATA), "--out", str(out)],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return out, result.stderr


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_data_files_are_consistent():
    points = _rows(DATA / "madrid_burger_king.csv")
    assert len(points) == N_RESTAURANTS + 1 and points[0]["label"] == "office"
    labels = [r["label"] for r in points]
    for name in ("madrid_burger_king_times_min.csv", "madrid_burger_king_dist_km.csv"):
        with (DATA / name).open(encoding="utf-8", newline="") as fh:
            table = list(csv.reader(fh))
        assert table[0][1:] == labels and [r[0] for r in table[1:]] == labels
        values = [float(v) for r in table[1:] for v in r[1:]]
        assert len(values) == len(labels) ** 2 and min(values) == 0.0 and all(v == v for v in values)


def test_example_writes_the_plan(run):
    out, _ = run
    suffixes = ("_timetable.csv", "_days.csv", ".kml", "_google_urls.txt", "_map.html")
    suffixes += ("_days.png", "_day1.png")
    for suffix in suffixes:
        assert (out / f"{PREFIX}{suffix}").is_file(), suffix
    assert not (out / f"{PREFIX}_google.html").exists()  # no key given
    kml = ET.parse(out / f"{PREFIX}.kml").getroot()
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    folders = kml.findall(".//k:Folder", ns)
    assert folders and folders[0].find("k:name", ns).text == "Día 1"
    html = (out / f"{PREFIX}_map.html").read_text(encoding="utf-8")
    assert "open-street-map" in html and "Día 1" in html
    urls = (out / f"{PREFIX}_google_urls.txt").read_text(encoding="utf-8").splitlines()
    assert urls[0] == "day\tleg\turl"
    assert all(line.split("\t")[2].startswith("https://www.google.com/maps/dir/?api=1&") for line in urls[1:])


def test_every_day_respects_the_budget(run):
    out, _ = run
    days = _rows(out / f"{PREFIX}_days.csv")
    timetable = _rows(out / f"{PREFIX}_timetable.csv")
    assert 12 <= len(days) <= 18  # 12 is the service-only lower bound; a quick search lands a little above 15
    assert sum(int(r["n_stops"]) for r in days) == N_RESTAURANTS
    for day in days:
        assert float(day["total_min"]) <= HOURS * 60 + 1e-9
        parts = float(day["driving_min"]) + float(day["service_min"])
        assert float(day["total_min"]) == pytest.approx(parts, abs=0.02)
        assert float(day["service_min"]) == pytest.approx(int(day["n_stops"]) * SERVICE)
    # the per-stop timetable tells the same story: every day starts and ends at the office within the budget
    by_day: dict[str, list[dict[str, str]]] = {}
    for row in timetable:
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
    assert "Baseline NearestNeighbour" in stderr and "Baseline Insertion" in stderr
    assert "OpenStreetMap contributors" in stderr
