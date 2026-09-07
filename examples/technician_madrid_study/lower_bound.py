"""Rigorous lower bounds for the technician instance: how far from optimal the best plan can be.

A heuristic that reaches 15 days on 59 configurations says nothing about 14 being impossible.
These are the two bounds that do say something, both proved on the committed matrices.

Bound on the driving
--------------------
The optimum of the plain closed ATSP over the 183 nodes is a lower bound on the driving of
**any** multi-day plan. A plan is one giant tour that passes through the office once per day;
removing those extra office visits shortcuts the tour, and shortcutting cannot lengthen a tour
when the travel times satisfy the triangle inequality -- so the plan drives at least as much as
the best plain tour. The ATSP is solved exactly with `skroute.MILP` (the
Dantzig-Fulkerson-Johnson programme with lazy subtour cuts, on HiGHS): with ``--time-limit``
large enough the bound is the proven optimum, and if the budget runs out ``lower_bound_`` (the
dual bound) is still valid, which is why it -- and never the incumbent -- is what is reported.

Road times violate the triangle inequality here and there (one-way streets, motorway ramps), so
the script counts the violating triples and offers ``--metric-closure``: Floyd-Warshall first,
which replaces every entry by the fastest way of getting there, satisfies the inequality by
construction, and can only *lower* the ATSP optimum. A bound proved on the closure is therefore
a bound for the original instance with no caveat left. On the committed matrix the two agree
(1078.3 minutes, 738 violating triples out of 6.1 M, 0.012 %), which is what removes the caveat.

Bounds on the number of days
----------------------------
1. Every day carries at most 480 minutes of service plus driving, and the whole plan needs
   ``182 * 30`` minutes of service and at least the ATSP bound of driving, so
   ``days >= ceil((5460 + driving_bound) / 480)`` -- 14 on the committed matrix.
2. Independently of the driving: a day of ``k`` stops needs at least
   ``k * 30 + (the cheapest leg out) + (the cheapest leg back) + (k - 1) * (the cheapest leg
   between two restaurants)``; the largest ``k`` that fits in 480 minutes caps the stops per day,
   so ``days >= ceil(182 / k)`` -- 13, weaker than the first.

Fourteen days is therefore not excluded by either bound, and no search ever found it: that is
the honest statement about the 15-day plan, not "optimal".

Reproducing
-----------
The recorded files are the truth; the MILP takes about nine and a half minutes each time.

    python lower_bound.py --out results/lower_bounds.json                    # ~10 min
    python lower_bound.py --metric-closure --out results/lower_bounds_metric_closure.json

Notes
-----
``atsp_cut_rounds`` is the number of MILP solves of the cut loop (`MILP.n_solves_`). The
committed files carry ``0`` there because the prototype read an attribute that `MILP` does not
expose; the cut rounds it really ran are visible in ``bound_trace``, the tail of the callback
log (iteration, dual bound, objective, cuts so far).

The instance loader and the path constants below are the same short section in the three scripts
of this directory, so each one runs on its own with nothing to install.

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import skroute
from skroute import MILP, RouteEvent

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
    time = np.loadtxt(
        data_dir / f"{STEM}_times_min.csv", delimiter=",", skiprows=1, usecols=range(1, len(labels) + 1)
    )
    if time.shape != (len(labels), len(labels)):
        raise ValueError(f"{STEM}_times_min.csv is {time.shape}, expected {(len(labels),) * 2}")
    return Instance(labels=labels, coords=coords, time=time)


# --------------------------------------------------------------------------- the bounds
def metric_closure(matrix: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Floyd-Warshall: the fastest route between every pair, plus what it changed.

    The closure satisfies the triangle inequality by construction and is entrywise no larger
    than the input, so the ATSP optimum on it is a valid -- possibly weaker -- lower bound for
    the original matrix, with no assumption about the road times left to make.
    """
    closed = matrix.copy()
    for k in range(len(closed)):
        np.minimum(closed, closed[:, [k]] + closed[[k], :], out=closed)
    changed = int((closed < matrix - 1e-9).sum())
    return closed, changed, float((matrix - closed).max())


def triangle_violations(matrix: np.ndarray) -> int:
    """Triples ``(i, j, k)`` with ``T[i, k] + T[k, j] < T[i, j]``: where shortcutting would cost."""
    total = 0
    for k in range(len(matrix)):
        total += int(np.sum(matrix[:, [k]] + matrix[[k], :] < matrix - 1e-9))
    return total


def max_stops_per_day(matrix: np.ndarray) -> int:
    """Largest ``k`` whose cheapest conceivable day of ``k`` stops still fits in the working day.

    The cheapest day of ``k`` stops needs the two cheapest office legs, ``k - 1`` cheapest legs
    between restaurants and ``k`` services -- a bound that ignores geometry entirely, and is
    therefore valid whatever the plan.
    """
    out = float(np.min(matrix[0, 1:]))
    back = float(np.min(matrix[1:, 0]))
    between = matrix[1:, 1:].copy()
    np.fill_diagonal(between, np.inf)
    hop = float(np.min(between))
    k = 0
    while (k + 1) * SERVICE_MIN + out + back + k * hop <= DAY_MIN:
        k += 1
    return k


def solve_atsp(matrix: np.ndarray, labels: Sequence[str], limit: float | None) -> dict[str, Any]:
    """The plain ATSP with `skroute.MILP`, keeping the tail of the bound trace of the cut loop."""
    trace: list[tuple[int, float | None, float | None, int | None]] = []

    def watch(event: RouteEvent) -> None:
        extra = event.extra
        trace.append((event.iteration, extra.get("lower_bound"), extra.get("objective"), extra.get("n_cuts")))

    milp = MILP(time_limit=limit, max_nodes=300)
    started = time.perf_counter()
    milp.fit(matrix, labels=list(labels), callback=watch)
    wall = time.perf_counter() - started
    return {
        "incumbent": float(milp.cost_),
        "bound": float(milp.lower_bound_),
        "gap": float(milp.gap_),
        "proven": bool(milp.is_optimal_),
        "wall_s": wall,
        "cut_rounds": int(milp.n_solves_),
        "n_cuts": int(milp.n_cuts_),
        "tour": [str(x) for x in milp.tour_.tolist()],
        "trace": trace[-10:],
    }


def bounds_report(instance: Instance, atsp: dict[str, Any], closure: bool) -> dict[str, Any]:
    """The JSON record of a bound run, in the shape of ``results/lower_bounds.json``."""
    service_total = N_RESTAURANTS * SERVICE_MIN
    bound = atsp["bound"]
    from_driving = math.ceil((service_total + bound) / DAY_MIN)
    stops = max_stops_per_day(instance.time)
    record: dict[str, Any] = {"n": len(instance.labels)}
    if closure:
        record["matrix"] = "metric closure (Floyd-Warshall) of the OSRM minutes"
    record.update(
        {
            "triangle_violations": triangle_violations(instance.time),
            "atsp_incumbent_min": atsp["incumbent"],
            "atsp_lower_bound_min": bound,
            "atsp_gap": atsp["gap"],
            "atsp_proven_optimal": atsp["proven"],
            "atsp_wall_s": round(atsp["wall_s"], 1),
            "atsp_cut_rounds": atsp["cut_rounds"],
            "atsp_tour": atsp["tour"],
            "days_lower_bound_from_driving": from_driving,
            "max_stops_per_day": stops,
            "days_lower_bound_from_stops": math.ceil(N_RESTAURANTS / stops),
            "bound_trace": atsp["trace"],
        }
    )
    return record


def print_report(record: dict[str, Any]) -> None:
    """What the study quotes: the triangle statistics, the driving bound and the two day bounds."""
    n = record["n"]
    violations = record["triangle_violations"]
    triples = n**3
    service_total = N_RESTAURANTS * SERVICE_MIN
    bound = record["atsp_lower_bound_min"]
    print(
        f"triangle inequality: {violations} violating triples of {triples} "
        f"({100 * violations / triples:.3f} %)"
    )
    print(
        f"ATSP over {n} nodes: incumbent {record['atsp_incumbent_min']:.1f} min, "
        f"lower bound {bound:.1f} min, gap {record['atsp_gap']:.6f}, "
        f"proven optimal {record['atsp_proven_optimal']}, "
        f"{record['atsp_wall_s']:.0f} s in {record['atsp_cut_rounds']} cut rounds"
    )
    print(
        f"days >= ceil(({service_total:.0f} service + {bound:.1f} driving) / {DAY_MIN:.0f}) = "
        f"{record['days_lower_bound_from_driving']}"
    )
    print(
        f"days >= ceil({N_RESTAURANTS} / {record['max_stops_per_day']} stops per day) = "
        f"{record['days_lower_bound_from_stops']}"
    )


# --------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    """The command line: a time limit, optionally the metric closure, and where to write."""
    parser = argparse.ArgumentParser(
        description="Prove the lower bounds of the technician instance (about ten minutes).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1500.0,
        metavar="SECONDS",
        help="budget of the MILP cut loop (default 1500; the recorded proof took 559 s)",
    )
    parser.add_argument(
        "--metric-closure",
        action="store_true",
        help="run Floyd-Warshall first, so the triangle inequality holds by construction",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"the CSVs (default {DEFAULT_DATA})")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("lower_bounds.json"),
        help="where to write the JSON record (default ./lower_bounds.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prove the bounds and write them, printing the numbers the study quotes."""
    args = build_parser().parse_args(argv)
    skroute.set_log_level("WARNING")
    instance = load_instance(args.data)
    if args.metric_closure:
        closed, changed, reduction = metric_closure(instance.time)
        print(f"metric closure: {changed} entries lowered, at most by {reduction:.2f} min")
        instance = instance._replace(time=closed)
    print(f"MILP on {len(instance.labels)} nodes, budget {args.time_limit:.0f} s ...", flush=True)
    atsp = solve_atsp(instance.time, instance.labels, args.time_limit)
    record = bounds_report(instance, atsp, closure=args.metric_closure)
    print_report(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
