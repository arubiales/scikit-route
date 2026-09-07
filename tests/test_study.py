"""Checks of the committed study ``examples/technician_madrid_study/`` -- fast and offline.

Nothing here runs a solver or a search: the point is that the *record* is internally consistent,
that the plan it calls best really is feasible on the committed matrices, that the bounds really
are below it, and that the harness still knows every configuration the record contains. Under
five seconds in total; the campaign it describes took a night.

The study and its data are located from **this file**, never from ``skroute.__file__``: the sdist
and wheel jobs import skroute from site-packages, where no ``examples/`` exists (D16), so the
module skips when the checkout is not beside it -- exactly as ``tests/test_examples.py`` does.

What is *not* asserted, and why, is as informative as what is: two derived fields of the record
disagree with the plans they describe, both from the same off-by-one in the campaign's recording
code, and both are pinned here so that a future regeneration of the record has to face them --

* ``stops_per_day`` counts the *legs* of a day (the stops plus the return to the office) on every
  solver row, because the campaign recorded ``len(trip) - 1`` and ``trips_`` are closed
  ``[office, ..., office]`` arrays. The two hybrid rows carry the true stop count.
* the row fitted on the symmetrised matrix records one day more than its own plan has, because
  the campaign read ``len(trip_starts(tour))`` -- which is ``n_trips + 1`` -- as the day count.

``days``, ``driving_min`` and ``objective`` of every other row are consistent, and the objective
of that row is right too, so the ranking of the study is unaffected. See the study's README,
"What the record gets wrong".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import skroute

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "technician_madrid_study"
RESULTS = STUDY / "results"
DATA = ROOT / "examples" / "data"
if not STUDY.is_dir() or not DATA.is_dir():
    pytest.skip("examples/ is not beside tests/: not a checkout", allow_module_level=True)

N_RESTAURANTS = 182
N_NODES = N_RESTAURANTS + 1
DAY_MIN = 480.0
SERVICE_MIN = 30.0
EXTRA_DAY_MIN = 480.0
BEST_DAYS = 15
BEST_DRIVING = 1532.8
BEST_KM = 1206.7
ATSP_BOUND = 1078.32
N_ROWS = 60
REQUIRED = ("name", "family", "params", "round", "wall_s")
RESULT_KEYS = (
    "days",
    "driving_min",
    "objective",
    "stops_per_day",
    "day_minutes",
    "n_iter",
    "stop_reason",
    "fit_time_s",
    "tour_sha256",
    "budget_s",
)
BENCHMARK_ROUNDS = {"roster (2 min)": "roster", "chained (4 min)": "chained", "long run (30-60 min)": "long"}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    """Every line of the recorded campaign."""
    text = (RESULTS / "results.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def results(rows) -> list[dict[str, Any]]:
    """The rows that carry a plan (all but the recorded refusal)."""
    return [row for row in rows if "objective" in row]


@pytest.fixture(scope="module")
def best_plan() -> dict[str, Any]:
    """The plan the study reports."""
    return json.loads((RESULTS / "best_plan.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bounds() -> dict[str, Any]:
    """The bounds proved on the OSRM matrix."""
    return json.loads((RESULTS / "lower_bounds.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def closure_bounds() -> dict[str, Any]:
    """The bounds proved on the metric closure of the OSRM matrix."""
    return json.loads((RESULTS / "lower_bounds_metric_closure.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def polish() -> ModuleType:
    """``exact_polish.py`` as a module: its plan reader is what rebuilds the plan here."""
    spec = importlib.util.spec_from_file_location("study_exact_polish", STUDY / "exact_polish.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebuilt(polish, best_plan) -> tuple[Any, list[tuple[int, ...]]]:
    """The best plan cut into days again, from the committed CSVs, by the study's own reader."""
    instance = polish.load_instance(DATA)
    problem = polish.make_problem(instance)
    return problem, polish.days_from_tour(problem, best_plan["tour"])


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """A study script as a subprocess, importing the same skroute as this process."""
    package_parent = str(Path(skroute.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            path for path in (package_parent, os.environ.get("PYTHONPATH", "")) if path
        ),
    }
    return subprocess.run(
        [sys.executable, str(STUDY / script), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=ROOT,
    )


# --------------------------------------------------------------------------- the record
def test_the_record_has_the_rows_the_study_describes(rows, results):
    assert len(rows) == N_ROWS
    assert len(results) == N_ROWS - 1
    refusals = [row for row in rows if "error" in row]
    assert len(refusals) == 1
    assert refusals[0]["name"] == "MILP plain TSP (driving only)"
    assert "cannot certify a multi-trip optimum" in refusals[0]["error"]
    assert len({row["name"] for row in rows}) == N_ROWS  # a name identifies a row
    assert {row["round"] for row in rows} == {*BENCHMARK_ROUNDS, "exact polish"}


@pytest.mark.parametrize("key", REQUIRED)
def test_every_row_carries_its_identity(rows, key):
    for row in rows:
        assert key in row, row.get("name")


@pytest.mark.parametrize("key", RESULT_KEYS)
def test_every_result_row_carries_its_result(results, key):
    for row in results:
        assert key in row, row["name"]


def test_the_arithmetic_of_every_result_row(results):
    """``objective = driving + 480 * (days - 1)``, and no day over the eight hours."""
    for row in results:
        expected = row["driving_min"] + EXTRA_DAY_MIN * (row["days"] - 1)
        assert row["objective"] == pytest.approx(expected, abs=0.05), row["name"]
        assert max(row["day_minutes"]) <= DAY_MIN + 1e-6, row["name"]
        assert len(row["day_minutes"]) == len(row["stops_per_day"]), row["name"]
        assert row["driving_min"] > 0 and row["days"] > 0, row["name"]


def test_the_shape_of_every_recorded_plan(results):
    """The stops are all accounted for -- under each of the two conventions of the record."""
    for row in results:
        stops = row["stops_per_day"]
        # the hybrid rows count stops; the solver rows count legs, i.e. one more per day
        legs = 0 if row["family"] == "hybrid" else len(stops)
        assert sum(stops) == N_RESTAURANTS + legs, row["name"]
        # ... and the day count matches the plan, except on the symmetrised-matrix row
        off_by_one = 1 if "cost_sym" in row else 0
        assert len(stops) == row["days"] - off_by_one, row["name"]


def test_the_two_conventions_are_the_documented_rows(results):
    """Whoever regenerates the record must face these two rows, not discover them later."""
    hybrid = [row["name"] for row in results if row["family"] == "hybrid"]
    assert len(hybrid) == 2 and all(name.startswith("Exact per-day routes") for name in hybrid)
    symmetric = [row["name"] for row in results if "cost_sym" in row]
    assert symmetric == ["ClarkeWright (symmetrised T)"]


def test_the_best_row_of_the_record_is_the_best_plan(results, best_plan):
    best = min(results, key=lambda row: (row["objective"], row["wall_s"]))
    assert best["name"] == best_plan["found_by"]
    assert best["days"] == best_plan["days"] == BEST_DAYS
    assert best["driving_min"] == best_plan["driving_min"] == BEST_DRIVING


# --------------------------------------------------------------------------- the best plan
def test_the_best_plan_is_feasible_on_the_committed_matrices(rebuilt, best_plan):
    """Rebuilt from the CSVs by the study's own reader: 15 days, every day inside the budget."""
    problem, days = rebuilt
    assert len(days) == BEST_DAYS
    assert all(days), "a day of the plan is empty"
    assert sorted(stop for day in days for stop in day) == list(range(1, N_NODES))
    index_tour = problem.to_index_tour(best_plan["tour"])
    starts = problem.trip_starts(index_tour)
    driving = float(problem.trip_costs(index_tour, starts).sum())
    minutes = problem.trip_times(index_tour, starts)
    assert driving == pytest.approx(BEST_DRIVING, abs=0.05)
    assert float(problem.evaluate(index_tour)) == pytest.approx(
        BEST_DRIVING + EXTRA_DAY_MIN * (BEST_DAYS - 1), abs=0.05
    )
    assert minutes.max() <= DAY_MIN + 1e-6
    assert [len(day) for day in days] == best_plan["stops_per_day"]
    assert [round(float(x), 2) for x in minutes] == best_plan["day_total_min"]


def test_the_best_plan_agrees_with_its_own_summary(best_plan, rebuilt):
    """The per-day figures of the file add up to the totals it reports."""
    _, days = rebuilt
    assert sum(best_plan["stops_per_day"]) == N_RESTAURANTS
    assert sum(best_plan["day_driving_min"]) == pytest.approx(BEST_DRIVING, abs=0.05)
    # the per-day kilometres are rounded to 100 m each, so their sum drifts from the total
    assert sum(best_plan["day_driving_km"]) == pytest.approx(BEST_KM, abs=0.5)
    assert best_plan["service_min"] == N_RESTAURANTS * SERVICE_MIN
    for stops, driving, total in zip(
        best_plan["stops_per_day"], best_plan["day_driving_min"], best_plan["day_total_min"], strict=True
    ):
        assert total == pytest.approx(driving + stops * SERVICE_MIN, abs=0.02)
    assert [list(day) for day in days] == [
        [_index(DATA)[label] for label in day] for day in best_plan["days_labels"]
    ]


def test_the_best_plan_hashes_to_the_row_that_found_it(best_plan, results):
    """The record keeps only the fingerprint of each tour: the plan must match its own row."""
    digest = hashlib.sha256(json.dumps(best_plan["tour"]).encode("utf-8")).hexdigest()[:16]
    row = next(row for row in results if row["name"] == best_plan["found_by"])
    assert row["tour_sha256"] == digest
    same = {row["name"] for row in results if row.get("tour_sha256") == digest}
    assert best_plan["found_by"] in same and len(same) >= 2  # the polish and the run it came from


def test_the_best_plan_drives_the_kilometres_it_claims(rebuilt, best_plan):
    """The kilometres come from the other committed matrix, so they are checked against it."""
    _, days = rebuilt
    distance = np.loadtxt(
        DATA / "madrid_burger_king_dist_km.csv", delimiter=",", skiprows=1, usecols=range(1, N_NODES + 1)
    )
    per_day = [float(sum(distance[a, b] for a, b in zip([0, *day], [*day, 0], strict=True))) for day in days]
    assert [round(km, 1) for km in per_day] == best_plan["day_driving_km"]
    assert sum(per_day) == pytest.approx(BEST_KM, abs=0.05)


# --------------------------------------------------------------------------- the bounds
def test_the_two_bound_files_agree_on_the_atsp(bounds, closure_bounds):
    for record in (bounds, closure_bounds):
        assert record["n"] == N_NODES
        assert record["atsp_proven_optimal"] is True
        assert record["atsp_gap"] == 0.0
        assert record["atsp_lower_bound_min"] == pytest.approx(record["atsp_incumbent_min"], abs=1e-6)
        assert record["atsp_lower_bound_min"] == pytest.approx(ATSP_BOUND, abs=0.05)
        assert len(record["atsp_tour"]) == N_NODES
        assert len(set(record["atsp_tour"])) == N_NODES
    assert bounds["triangle_violations"] == 738  # the OSRM minutes are not quite a metric
    assert closure_bounds["triangle_violations"] == 0  # the closure is, by construction
    assert closure_bounds["matrix"].startswith("metric closure")


def test_the_bound_is_below_the_best_plan(bounds, best_plan):
    """A bound above the plan would mean one of the two is wrong."""
    assert bounds["atsp_lower_bound_min"] < best_plan["driving_min"]
    assert bounds["days_lower_bound_from_driving"] <= best_plan["days"]
    assert bounds["days_lower_bound_from_stops"] <= bounds["days_lower_bound_from_driving"]


def test_the_day_bound_is_the_one_the_study_quotes(bounds):
    """``days >= ceil((5460 + 1078.3) / 480) = 14``: the reason 15 is not called optimal."""
    service = N_RESTAURANTS * SERVICE_MIN
    expected = -(-(service + bounds["atsp_lower_bound_min"]) // DAY_MIN)
    assert bounds["days_lower_bound_from_driving"] == expected == 14
    assert bounds["days_lower_bound_from_stops"] == -(-N_RESTAURANTS // bounds["max_stops_per_day"])


def test_no_recorded_configuration_beat_the_day_bound(results, bounds):
    reached = sorted({row["days"] for row in results})
    assert min(reached) == BEST_DAYS > bounds["days_lower_bound_from_driving"]
    assert sum(1 for row in results if row["days"] == BEST_DAYS) == 23


# --------------------------------------------------------------------------- the scripts
def test_report_prints_the_best_plan_at_the_top(best_plan):
    """The Markdown the documentation pastes: the first row of the table is the best plan."""
    done = _run("report.py")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| Configuration |"))
    first = [cell.strip() for cell in lines[header + 2].strip("|").split("|")]
    assert first[0] == best_plan["found_by"]
    assert first[3] == str(best_plan["days"])
    assert first[4] == f"{best_plan['driving_min']:.1f}"
    assert f"{best_plan['driving_min']:.1f} min of driving" in done.stdout
    assert "1078.3 min" in done.stdout  # the bound, in the gap section
    assert "none reaches 14" in done.stdout


def test_report_runs_in_every_shape(tmp_path):
    for extra in (["--format", "text"], ["--sort", "name"], ["--sort", "family"]):
        done = _run("report.py", *extra)
        assert done.returncode == 0, done.stderr
        assert "ClarkeWright" in done.stdout
    missing = tmp_path / "never-written.jsonl"
    done = _run("report.py", "--results", str(missing))
    assert done.returncode != 0 and "no record at" in done.stderr + done.stdout


def test_the_harness_still_knows_every_recorded_configuration(rows):
    """The check that keeps ``benchmark.py`` and the record in sync, both ways."""
    done = _run("benchmark.py", "--list")
    assert done.returncode == 0, done.stderr
    listed: dict[str, dict[str, str]] = {name: {} for name in BENCHMARK_ROUNDS.values()}
    for line in done.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        round_name, key, name = line.split(None, 2)
        listed[round_name][name.strip()] = key
    seen = {name: set() for name in listed}
    for row in rows:
        round_name = BENCHMARK_ROUNDS.get(row["round"])
        if round_name is None:  # the exact polish is exact_polish.py's, not the benchmark's
            assert row["round"] == "exact polish" and row["family"] == "hybrid"
            continue
        template = _template(row["name"])
        assert template in listed[round_name], f"{row['name']} is no longer in --list"
        key = listed[round_name][template]
        seen[round_name].add(key)
        if "key" in row:  # the long round records the key it was run with
            assert row["key"] == key, row["name"]
    for round_name, keys in listed.items():
        assert seen[round_name] == set(keys.values()), f"{round_name} lists a configuration never run"
    assert {len(keys) for keys in listed.values()} == {28, 9, 13}


def test_the_split_of_a_plan_is_the_fixed_one(polish, rebuilt, best_plan):
    """The bug the polish script exists to keep fixed: no empty leading or trailing day."""
    problem, days = rebuilt
    starts = [int(s) for s in problem.trip_starts(problem.to_index_tour(best_plan["tour"]))]
    assert len(starts) == BEST_DAYS + 1 and starts[0] == 1 and starts[-1] == N_NODES
    naive = np.split(problem.to_index_tour(best_plan["tour"])[1:], [s - 1 for s in starts])
    assert len(naive) == BEST_DAYS + 2 and not len(naive[0]) and not len(naive[-1])
    assert len(days) == BEST_DAYS and all(days)
    with pytest.raises(ValueError, match="are empty"):
        polish.check_partition([(1, 2), (), (3,)], 3)
    with pytest.raises(ValueError, match="exactly once"):
        polish.check_partition([(1, 2), (2,)], 3)


def _index(data_dir: Path) -> dict[str, int]:
    """Label -> matrix row of the committed points file."""
    lines = (data_dir / "madrid_burger_king.csv").read_text(encoding="utf-8").splitlines()[1:]
    return {line.split(",", 1)[0]: i for i, line in enumerate(lines)}


def _template(name: str) -> str:
    """A recorded name with its warm-start seed replaced by the placeholder ``--list`` prints."""
    start, bracket, rest = name.partition("[")
    return f"{start}[{{seed}}]" if bracket and rest.endswith("]") else name
