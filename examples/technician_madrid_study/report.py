"""Read the recorded study and print it: the ranking, the budget comparison and the gap.

Offline and instant -- it reads ``results/results.jsonl`` and ``results/lower_bounds.json`` and
touches neither the matrices nor a solver. Three sections:

1. **The ranking**, every configuration by objective (fewest days first, then least driving), as
   a Markdown table meant to be pasted into the documentation. The columns are fixed and the
   minutes carry one decimal, exactly as recorded.
2. **Two minutes against an hour**, for the solver families that ran in both rounds: what the
   extra budget bought, in days and in driving minutes.
3. **The gap**, of the best plan against the proven lower bound on the driving and against the
   two lower bounds on the number of days -- the only honest way to say how good the plan is.

Usage
-----
    python report.py                                     # the committed record, Markdown
    python report.py --format text --sort name
    python report.py --results other.jsonl --bounds other_bounds.json

Notes
-----
A row whose recorded shape contradicts its recorded day count is marked in the ranking and
listed under the table: the campaign's day counter was off by one on the configuration that was
fitted on the symmetrised matrix, so its ``days`` and ``driving_min`` cannot both be believed
(its ``objective`` can -- see README.md, "What the record gets wrong").

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DAY_MIN = 480.0
SERVICE_TOTAL = 182 * 30.0
ROSTER = "roster (2 min)"
LONG = "long run (30-60 min)"
COLUMNS = ("Configuration", "Family", "Round", "Days", "Driving (min)", "Objective", "Wall (s)")
ALIGN = (":--", ":--", ":--", "--:", "--:", "--:", "--:")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """The record, one dict per line; a line that is not JSON is a corrupt record, not a warning."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number} is not JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def scored(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows that carry a result: a configuration that raised has no objective to rank."""
    return [row for row in rows if "objective" in row]


def suspect(row: dict[str, Any]) -> bool:
    """Whether the row's plan shape contradicts its day count (see the module docstring)."""
    stops = row.get("stops_per_day")
    return bool(stops) and len(stops) != row.get("days")


def sort_key(name: str) -> Any:
    """The ``--sort`` orders: by objective, by name or by family then objective."""
    if name == "name":
        return lambda row: row["name"]
    if name == "family":
        return lambda row: (row["family"], row["objective"])
    return lambda row: (row["objective"], row["wall_s"])


def cells(row: dict[str, Any], mark: str) -> tuple[str, ...]:
    """One line of the ranking: the numbers formatted exactly as the record spells them."""
    return (
        row["name"] + mark,
        row["family"],
        row["round"],
        f"{row['days']}",
        f"{row['driving_min']:.1f}",
        f"{row['objective']:.1f}",
        f"{row['wall_s']:.1f}",
    )


def markdown_table(header: Sequence[str], align: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    """A Markdown table; the column widths are not padded, the renderer does that."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(line) + " |" for line in body]
    return "\n".join(lines)


def _numeric(value: str) -> bool:
    """Whether a cell is a number, i.e. whether its column should be right-aligned."""
    try:
        float(value.replace("+", ""))
    except ValueError:
        return False
    return True


def text_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    """The same table as aligned plain text, for a terminal."""
    lines = [list(header), *[list(line) for line in body]]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    right = [all(_numeric(line[i]) for line in lines[1:]) for i in range(len(header))]
    out = []
    for number, line in enumerate(lines):
        cell = [
            value.rjust(widths[i]) if right[i] else value.ljust(widths[i]) for i, value in enumerate(line)
        ]
        out.append("  ".join(cell).rstrip())
        if number == 0:
            out.append("  ".join("-" * width for width in widths))
    return "\n".join(out)


def ranking(rows: Sequence[dict[str, Any]], order: str, markdown: bool) -> str:
    """Section 1: every configuration that produced a plan, best first."""
    ranked = sorted(scored(rows), key=sort_key(order))
    mark = " [*]" if markdown else " (*)"
    body = [cells(row, mark if suspect(row) else "") for row in ranked]
    table = markdown_table(COLUMNS, ALIGN, body) if markdown else text_table(COLUMNS, body)
    notes = [
        f"{mark.strip()} `{row['name']}` records {row['days']} days but a plan of "
        f"{len(row['stops_per_day'])} days; its objective ({row['objective']:.1f}) is the "
        'number to trust. See README.md, "What the record gets wrong".'
        for row in ranked
        if suspect(row)
    ]
    notes += [f"`{row['name']}` produced no plan -- {row['error']}" for row in rows if "error" in row]
    return "\n\n".join([table, *notes]) if notes else table


def budget_comparison(rows: Sequence[dict[str, Any]], markdown: bool) -> str:
    """Section 2: for each family in both rounds, the best of two minutes against the best hour."""
    header = (
        "Family",
        "Best in 2 min",
        "Days",
        "Driving (min)",
        "Best in 30-60 min",
        "Days",
        "Driving (min)",
        "Δ driving",
    )
    align = (":--", ":--", "--:", "--:", ":--", "--:", "--:", "--:")
    quick = _best_by_family(rows, ROSTER)
    slow = _best_by_family(rows, LONG)
    body = []
    for family in sorted(set(quick) & set(slow)):
        fast, long = quick[family], slow[family]
        body.append(
            (
                family,
                fast["name"],
                f"{fast['days']}",
                f"{fast['driving_min']:.1f}",
                long["name"],
                f"{long['days']}",
                f"{long['driving_min']:.1f}",
                f"{long['driving_min'] - fast['driving_min']:+.1f}",
            )
        )
    if not body:
        return "No family ran in both the roster and the long round."
    return markdown_table(header, align, body) if markdown else text_table(header, body)


def _best_by_family(rows: Sequence[dict[str, Any]], round_label: str) -> dict[str, dict[str, Any]]:
    """The best row of each family within one round."""
    best: dict[str, dict[str, Any]] = {}
    for row in scored(rows):
        if row["round"] != round_label:
            continue
        current = best.get(row["family"])
        if current is None or row["objective"] < current["objective"]:
            best[row["family"]] = row
    return best


def gap(rows: Sequence[dict[str, Any]], bounds: dict[str, Any], closure: dict[str, Any] | None) -> str:
    """Section 3: the best plan of the record against the proven bounds."""
    best = min(scored(rows), key=sort_key("objective"))
    bound = float(bounds["atsp_lower_bound_min"])
    days_bound = int(bounds["days_lower_bound_from_driving"])
    stops_bound = int(bounds["days_lower_bound_from_stops"])
    excess = best["driving_min"] - bound
    status = (
        "proven optimal" if bounds["atsp_proven_optimal"] else f"dual bound, gap {bounds['atsp_gap']:.4f}"
    )
    lines = [
        f"Best plan: **{best['days']} days, {best['driving_min']:.1f} min of driving** "
        f"({best['driving_min'] / 60:.1f} h), objective {best['objective']:.1f}, "
        f"found by `{best['name']}` ({best['round']}, {best['wall_s']:.0f} s).",
        "",
        f"- Driving: the plain ATSP over the {bounds['n']} nodes measures **{bound:.1f} min** "
        f"({status}, {bounds['atsp_wall_s']:.0f} s), a lower bound on the driving of any plan. "
        f"The best plan is {excess:+.1f} min above it ({100 * excess / bound:+.1f} %).",
        f"- Days: at least `ceil(({SERVICE_TOTAL:.0f} + {bound:.1f}) / {DAY_MIN:.0f})` = "
        f"**{days_bound}** from the day budget, and {stops_bound} from the stops-per-day cap "
        f"({bounds['max_stops_per_day']} stops). The plan uses {best['days']}: "
        f"{best['days'] - days_bound} above the bound.",
        f"- Triangle inequality: {bounds['triangle_violations']} violating triples of "
        f"{bounds['n'] ** 3} ({100 * bounds['triangle_violations'] / bounds['n'] ** 3:.3f} %), "
        "which is what the shortcutting argument needs.",
    ]
    if closure is not None:
        agree = abs(float(closure["atsp_lower_bound_min"]) - bound) <= 0.05
        lines.append(
            f"- On the metric closure of the matrix (triangle inequality by construction, "
            f"{closure['triangle_violations']} violations) the bound is "
            f"{float(closure['atsp_lower_bound_min']):.1f} min: "
            + ("the same, so the violations cost nothing." if agree else "lower, so it is the one to quote.")
        )
    reached = sorted({row["days"] for row in scored(rows)})
    fifteen = sum(1 for row in scored(rows) if row["days"] == min(reached))
    lines += [
        "",
        f"{fifteen} of {len(scored(rows))} configurations reach {min(reached)} days; "
        f"none reaches {days_bound}. The plan is not proved optimal -- nothing here proves a "
        "multi-day optimum -- it is the best of everything that was tried, and the exact polish "
        "cannot improve it.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line: which record, which bounds, how to sort and how to print."""
    parser = argparse.ArgumentParser(description="Print the recorded technician study.")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR / "results.jsonl",
        help=f"the JSON Lines record (default {RESULTS_DIR / 'results.jsonl'})",
    )
    parser.add_argument(
        "--bounds",
        type=Path,
        default=RESULTS_DIR / "lower_bounds.json",
        help="the lower bounds (default results/lower_bounds.json)",
    )
    parser.add_argument(
        "--closure-bounds",
        type=Path,
        default=RESULTS_DIR / "lower_bounds_metric_closure.json",
        help="the bounds proved on the metric closure; skipped when the file is absent",
    )
    parser.add_argument("--format", choices=("markdown", "text"), default="markdown")
    parser.add_argument("--sort", choices=("objective", "name", "family"), default="objective")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the three sections of the report."""
    args = build_parser().parse_args(argv)
    rows = load_rows(args.results)
    bounds = json.loads(args.bounds.read_text(encoding="utf-8"))
    closure = (
        json.loads(args.closure_bounds.read_text(encoding="utf-8")) if args.closure_bounds.is_file() else None
    )
    markdown = args.format == "markdown"
    heading = "## " if markdown else ""
    print(f"{heading}The ranking ({len(scored(rows))} configurations, sorted by {args.sort})\n")
    print(ranking(rows, args.sort, markdown))
    print(f"\n{heading}Two minutes against half an hour or an hour\n")
    print(budget_comparison(rows, markdown))
    print(f"\n{heading}How far from optimal\n")
    print(gap(rows, bounds, closure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
