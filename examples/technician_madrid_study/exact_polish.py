"""Exact per-day polish of a finished plan, and exact exchanges of stops between two days.

Not a library solver: a hybrid built on top of one. The giant-tour metaheuristics decide which
restaurants share a day *and* in what order at the same time, on a tour of 183 nodes. A finished
plan, though, is fifteen tiny problems of at most 13 stops plus the office -- and 14 nodes is
`skroute.HeldKarp` territory, where the optimum is a matter of milliseconds. So:

1. **Re-solve every day exactly** with Held-Karp (asymmetric dynamic programming over subsets).
   Whatever the search left, each day comes back optimal for its own set of stops.
2. **Move stops between days**: every relocation of one stop to another day and every swap of
   two stops between two days, with both days re-solved exactly and accepted only if the total
   driving falls and both days still fit in 480 minutes (service included). Best improvement per
   pass, until nothing improves or ``--max-seconds`` runs out.
3. **Try to empty a day**: fewest days beats least driving, so the lightest day's stops are
   greedily relocated into the others; if all of them fit, the plan loses a day.

Applied to the 15-day plans of the campaign this changed nothing on the routes -- the days the
iterated local search produced were already exactly optimal -- and found 0.8 minutes on an
earlier plan by exchanging two stops. No day could be emptied. That is the strongest statement
the study can make about the plan short of proving the multi-day optimum, which nothing here does.

Reproducing
-----------
Seconds to a couple of minutes, depending on how much stage 2 finds.

    python exact_polish.py --plan results/best_plan.json                       # ~1 min
    python exact_polish.py --plan technician_madrid_timetable.csv --max-seconds 1800 \
        --results out.jsonl --name "Exact polish from [the delivered plan]"

Notes
-----
``--plan`` takes either a ``best_plan.json`` (or any JSON with a ``tour``), which is cut into
days by the problem's own optimal split, or a ``technician_madrid_timetable.csv`` written by
``examples/technician_madrid.py``, whose ``day`` column gives the days as driven.

`RoutingProblem.trip_starts` returns ``n_trips + 1`` positions -- the first is 1 and the last is
the sentinel ``len(tour)`` -- so a plan is cut on ``starts[1:-1]``. Cutting on all of them, as
the prototype of this script did, yields an empty leading and an empty trailing day: sixteen
"days" for a 15-day plan, and a day count one too high. Every day produced here is asserted
non-empty and the days are asserted to partition the 182 restaurants exactly once.

The instance loader and the path constants below are the same short section in the three scripts
of this directory, so each one runs on its own with nothing to install.

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org).
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skroute import HeldKarp, RoutingProblem

# --------------------------------------------------------------------------- the instance
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = REPO / "examples" / "data"
RESULTS_DIR = HERE / "results"
STEM = "madrid_burger_king"
SERVICE_MIN = 30.0  # minutes of maintenance per restaurant
DAY_MIN = 480.0  # the eight-hour working day
EXTRA_DAY_MIN = 480.0  # charge per extra day: fewest days first, then least driving
N_RESTAURANTS = 182

if hasattr(sys.stdout, "reconfigure"):  # a Windows console defaults to cp1252 and cannot print Δ or ©
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class Instance(NamedTuple):
    """The committed case: labels (the office first), ``(lat, lon)`` coordinates and minutes."""

    labels: list[str]
    coords: np.ndarray
    time: np.ndarray


def load_instance(data_dir: Path) -> Instance:
    """Read the three committed CSVs of ``data_dir`` (see ``examples/data/README.md``)."""
    with (data_dir / f"{STEM}.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels = [row["label"] for row in rows]
    coords = np.array([[float(row["lat"]), float(row["lon"])] for row in rows], dtype=float)
    time_matrix = np.loadtxt(
        data_dir / f"{STEM}_times_min.csv", delimiter=",", skiprows=1, usecols=range(1, len(labels) + 1)
    )
    if time_matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"{STEM}_times_min.csv is {time_matrix.shape}, expected {(len(labels),) * 2}")
    return Instance(labels=labels, coords=coords, time=time_matrix)


def make_problem(instance: Instance, extra_cost: float = EXTRA_DAY_MIN) -> RoutingProblem:
    """The multi-trip problem of the study: one trip is one working day."""
    return RoutingProblem(
        instance.time,
        labels=instance.labels,
        coords=instance.coords,
        time_matrix=instance.time,
        service_time=SERVICE_MIN,
        max_time_work=DAY_MIN,
        extra_cost=extra_cost,
        split="optimal",
    )


# --------------------------------------------------------------------------- reading a plan
Day = tuple[int, ...]  # the stops of one day, as matrix indices, in the order they are driven


def days_from_tour(problem: RoutingProblem, tour: Sequence[Any]) -> list[Day]:
    """Cut a giant tour into days with the problem's optimal split.

    ``trip_starts`` returns ``n_trips + 1`` positions, the last one the sentinel ``len(tour)``:
    the cuts are ``starts[1:-1]``. Splitting on every start instead adds an empty day at each
    end -- the bug this function exists to keep fixed.
    """
    index_tour = problem.to_index_tour(tour)
    starts = [int(s) for s in problem.trip_starts(index_tour)]
    body = index_tour[1:]  # the depot sits at position 0 and is not a stop
    cuts = [s - 1 for s in starts[1:-1]]
    days = [tuple(int(i) for i in part) for part in np.split(body, cuts)]
    check_partition(days, problem.n - 1)
    return days


def days_from_timetable(path: Path, labels: Sequence[str]) -> list[Day]:
    """The days of a ``technician_madrid_timetable.csv``, as driven (its ``day`` column)."""
    index = {label: i for i, label in enumerate(labels)}
    depot = labels[0]
    grouped: dict[int, list[int]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["label"] == depot:
                continue
            grouped.setdefault(int(row["day"]), []).append(index[row["label"]])
    days = [tuple(grouped[key]) for key in sorted(grouped)]
    check_partition(days, len(labels) - 1)
    return days


def check_partition(days: Sequence[Day], n_customers: int) -> None:
    """Every day non-empty, and the days visiting each restaurant exactly once."""
    empty = [k for k, day in enumerate(days, start=1) if not day]
    if empty:
        raise ValueError(f"days {empty} of the plan are empty: the plan was cut wrongly")
    visited = sorted(stop for day in days for stop in day)
    if visited != list(range(1, n_customers + 1)):
        raise ValueError(
            f"the plan visits {len(visited)} stops ({len(set(visited))} distinct), "
            f"expected each of the {n_customers} restaurants exactly once"
        )


def load_plan(path: Path, problem: RoutingProblem, labels: Sequence[str]) -> list[Day]:
    """The plan to polish: a timetable CSV as driven, or the tour of a JSON record."""
    if path.suffix.lower() == ".csv":
        return days_from_timetable(path, labels)
    record = json.loads(path.read_text(encoding="utf-8"))
    tour = record.get("tour")
    if not tour:
        raise SystemExit(f"{path} carries no tour")
    days = days_from_tour(problem, tour)
    recorded = record.get("days_labels")
    if recorded is not None:
        index = {label: i for i, label in enumerate(labels)}
        if [[index[label] for label in day] for day in recorded] != [list(day) for day in days]:
            raise ValueError(f"{path}: days_labels disagrees with the optimal split of its tour")
    return days


# --------------------------------------------------------------------------- the polish
class Polisher:
    """Exact day routes and exact inter-day exchanges on one instance, with a memoised solver."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.exact = functools.lru_cache(maxsize=None)(self._exact)

    def _exact(self, stops: Day) -> tuple[float, Day]:
        """Optimal closed route ``office -> stops -> office`` (Held-Karp, asymmetric)."""
        if not stops:
            return 0.0, ()
        if len(stops) == 1:
            return float(self.matrix[0, stops[0]] + self.matrix[stops[0], 0]), stops
        nodes = [0, *stops]
        sub = self.matrix[np.ix_(nodes, nodes)]
        solved = HeldKarp(max_nodes=20).fit(sub)
        order = tuple(nodes[i] for i in solved.tour_.tolist()[1:])  # drop the depot
        return float(solved.cost_), order

    def driving(self, day: Day) -> float:
        """Driving minutes of the optimal route of ``day``."""
        return self.exact(day)[0]

    def route(self, day: Day) -> Day:
        """The optimal order of ``day``."""
        return self.exact(day)[1]

    def duration(self, day: Day) -> float:
        """Optimal driving plus the services: what has to fit in the working day."""
        return self.driving(day) + SERVICE_MIN * len(day)

    def as_driven(self, day: Day) -> float:
        """Driving minutes of ``day`` in the order it is given, without re-solving it."""
        legs = [0, *day, 0]
        return float(sum(self.matrix[a, b] for a, b in pairwise(legs)))

    def total(self, plan: Sequence[Day]) -> float:
        """Driving of the whole plan with every day routed optimally."""
        return sum(self.driving(day) for day in plan)

    def best_exchange(self, plan: Sequence[Day], deadline: float) -> tuple[float, list[Day]] | None:
        """The best relocation or swap between two days, or ``None`` when nothing improves.

        Both days of a candidate move are re-solved exactly, so the gain reported is real and
        not an estimate from a giant-tour delta.
        """
        best_gain, best_plan = 1e-9, None
        for a, b in itertools.permutations(range(len(plan)), 2):
            day_a, day_b = plan[a], plan[b]
            before = self.driving(day_a) + self.driving(day_b)
            for stop in day_a:  # relocate one stop of a into b
                moved_a = tuple(x for x in day_a if x != stop)
                moved_b = (*day_b, stop)
                if self.duration(moved_b) > DAY_MIN:
                    continue
                gain = before - self.driving(moved_a) - self.driving(moved_b)
                if gain > best_gain:
                    best_gain, best_plan = gain, self._with(plan, a, b, moved_a, moved_b)
            if a < b:  # swap one stop of a with one of b (the pair is symmetric)
                for stop_a, stop_b in itertools.product(day_a, day_b):
                    swapped_a = tuple(stop_b if x == stop_a else x for x in day_a)
                    swapped_b = tuple(stop_a if x == stop_b else x for x in day_b)
                    if self.duration(swapped_a) > DAY_MIN or self.duration(swapped_b) > DAY_MIN:
                        continue
                    gain = before - self.driving(swapped_a) - self.driving(swapped_b)
                    if gain > best_gain:
                        best_gain, best_plan = gain, self._with(plan, a, b, swapped_a, swapped_b)
            if time.perf_counter() > deadline:
                break
        return None if best_plan is None else (best_gain, best_plan)

    def _with(self, plan: Sequence[Day], a: int, b: int, day_a: Day, day_b: Day) -> list[Day]:
        """``plan`` with days ``a`` and ``b`` replaced, re-routed, and an emptied day dropped."""
        out = list(plan)
        out[a], out[b] = self.route(day_a), self.route(day_b)
        return [day for day in out if day]

    def drop_a_day(self, plan: Sequence[Day]) -> list[Day] | None:
        """Try to empty one day by relocating its stops into the others, cheapest place first.

        The days are tried lightest first, and inside a day the stops furthest from the office
        first -- they are the ones that need the room. Greedy: a failure does not prove that no
        plan with one fewer day exists (see ``lower_bound.py`` for what is actually proved).
        """
        for victim in sorted(range(len(plan)), key=lambda k: len(plan[k])):
            others = [list(day) for k, day in enumerate(plan) if k != victim]
            for stop in sorted(plan[victim], key=lambda s: -self.matrix[0, s]):
                best: tuple[float, int] | None = None
                for k, day in enumerate(others):
                    grown = (*day, stop)
                    if self.duration(grown) > DAY_MIN:
                        continue
                    delta = self.driving(grown) - self.driving(tuple(day))
                    if best is None or delta < best[0]:
                        best = (delta, k)
                if best is None:
                    break
                others[best[1]].append(stop)
            else:
                return [self.route(tuple(day)) for day in others]
        return None


def polish(plan: Sequence[Day], polisher: Polisher, max_seconds: float) -> tuple[list[Day], dict[str, Any]]:
    """The three stages, printing what each one found; returns the plan and a small trace."""
    started = time.perf_counter()
    deadline = started + max_seconds
    as_given = sum(polisher.as_driven(day) for day in plan)
    current = [polisher.route(day) for day in plan]
    after_exact = polisher.total(current)
    print(
        f"{len(current)} days; driving as given {as_given:.1f} min -> "
        f"exact per-day routes {after_exact:.1f} min",
        flush=True,
    )
    over = [k for k, day in enumerate(current, start=1) if polisher.duration(day) > DAY_MIN + 1e-9]
    if over:
        raise ValueError(f"days {over} do not fit in {DAY_MIN:.0f} minutes after the exact routes")

    passes = 0
    while time.perf_counter() < deadline:
        found = polisher.best_exchange(current, deadline)
        if found is None:
            break
        passes += 1
        gain, current = found
        print(
            f"pass {passes}: -{gain:.1f} min -> {polisher.total(current):.1f} min, "
            f"{len(current)} days ({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
    after_exchanges = polisher.total(current)

    shorter = polisher.drop_a_day(current)
    if shorter is None:
        print("no day can be emptied by greedy relocation", flush=True)
    else:
        print(
            f"a day was emptied: {len(current)} -> {len(shorter)} days, "
            f"driving {polisher.total(shorter):.1f} min",
            flush=True,
        )
        current = shorter
    wall = time.perf_counter() - started
    trace = {
        "as_given_min": round(as_given, 1),
        "after_exact_days_min": round(after_exact, 1),
        "after_exchanges_min": round(after_exchanges, 1),
        "passes": passes,
        "wall_s": wall,
        "cache": polisher.exact.cache_info().currsize,
    }
    return current, trace


# --------------------------------------------------------------------------- the record
def result_row(
    name: str,
    plan: Sequence[Day],
    polisher: Polisher,
    trace: dict[str, Any],
    labels: Sequence[str],
    max_seconds: float,
    source: str,
) -> dict[str, Any]:
    """The row appended to the JSON Lines record, in the shape ``benchmark.py`` writes."""
    driving = round(polisher.total(plan), 1)
    tour = [str(labels[0])] + [str(labels[i]) for day in plan for i in day]
    return {
        "name": name,
        "family": "hybrid",
        "params": {
            "per_day": "Held-Karp exact",
            "moves": "relocate + swap between days, best improvement",
            "passes": trace["passes"],
        },
        "round": "exact polish",
        "phase": 3,
        "budget_s": max_seconds,
        "wall_s": round(trace["wall_s"], 1),
        "days": len(plan),
        "driving_min": driving,
        "objective": round(driving + EXTRA_DAY_MIN * (len(plan) - 1), 1),
        "stops_per_day": [len(day) for day in plan],
        "day_minutes": [round(polisher.duration(day)) for day in plan],
        "n_iter": trace["passes"],
        "stop_reason": "converged" if trace["wall_s"] < max_seconds else "time_limit",
        "fit_time_s": round(trace["wall_s"], 1),
        "tour_sha256": tour_hash(tour),
        "note": (
            f"from {source}: {trace['as_given_min']:.1f} -> "
            f"{trace['after_exact_days_min']:.1f} (exact days) -> {driving:.1f} min"
        ),
        "tour": tour,
    }


def tour_hash(tour: Sequence[str]) -> str:
    """The record's fingerprint of a plan: 16 hex characters of the sha256 of its JSON."""
    return hashlib.sha256(json.dumps(list(tour)).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    """The command line: which plan, how long, and where to append the result."""
    parser = argparse.ArgumentParser(
        description="Re-solve every day of a plan exactly and exchange stops between days.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=RESULTS_DIR / "best_plan.json",
        help="a best_plan.json (or any JSON with a tour), or a timetable CSV of the example",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1800.0,
        help="budget of the exchange stage (default 1800; the recorded polish took 56 s)",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"the CSVs (default {DEFAULT_DATA})")
    parser.add_argument("--name", help="name of the row in the record (default: from --plan)")
    parser.add_argument("--results", type=Path, help="JSON Lines record to append the result to")
    parser.add_argument("--keep-tours", action="store_true", help="also write the tour of the row")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Polish one plan, print what changed and optionally append the row to a record."""
    args = build_parser().parse_args(argv)
    instance = load_instance(args.data)
    problem = make_problem(instance)
    plan = load_plan(args.plan, problem, instance.labels)
    polisher = Polisher(instance.time)
    name = args.name or f"Exact per-day routes + inter-day exchanges (Held-Karp) from [{args.plan.name}]"
    polished, trace = polish(plan, polisher, args.max_seconds)
    row = result_row(name, polished, polisher, trace, instance.labels, args.max_seconds, args.plan.name)
    print(
        f"RESULT {row['days']} days, {row['driving_min']} min driving "
        f"({trace['as_given_min']} before), {trace['passes']} accepted pass(es), "
        f"{row['wall_s']:.0f} s; {trace['cache']} day routes solved exactly",
        flush=True,
    )
    if args.results is not None:
        if not args.keep_tours:
            row = {k: v for k, v in row.items() if k != "tour"}
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("appended to", args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
