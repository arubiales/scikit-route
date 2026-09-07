"""Turn a recorded plan into one self-contained, interactive HTML page.

The page shows the restaurants on a plain equirectangular map, one day at a time (or all days at
once), the timetable of the selected day, a Google Maps Directions link per leg, and — when the
study results are passed — the ranking of every solver configuration with the proven lower bound.
Everything is inlined: no network, no libraries, one file you can open or send by e-mail.

Examples
--------
Build the page from the plan recorded by the study::

    python examples/technician_madrid_study/plan_page.py --out plan.html

Build it from a plan you just solved yourself (the output directory of the example)::

    python examples/technician_madrid.py --out my_run
    python examples/technician_madrid_study/plan_page.py --timetable my_run/technician_madrid_timetable.csv \
        --days my_run/technician_madrid_days.csv --out my_plan.html

Notes
-----
The map is drawn as inline SVG with latitude and longitude scaled by ``cos(mean latitude)``, so
distances are honest at the scale of a city region; it is a schematic, not a street map, and the
lines between stops are straight. The route by road is one click away in Google Maps.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "examples" / "data"
RESULTS = HERE / "results"

# tab10, the palette skroute.viz uses for trips
PALETTE = (
    "#1f77b4",
    "#e07b39",
    "#2ca02c",
    "#c9312c",
    "#8e5bbd",
    "#8c564b",
    "#d84a9b",
    "#4c6a78",
    "#9aa11a",
    "#17a2b8",
    "#f2b134",
    "#2f8f6a",
    "#6b4ea3",
    "#b05d1c",
    "#3355aa",
    "#7a7a7a",
)
FAMILY_LABEL = {
    "construction": "construction",
    "local_search": "local search",
    "ils": "iterated local search",
    "ils-chained": "iterated local search (chained)",
    "sa-chained": "simulated annealing (chained)",
    "tabu-chained": "tabu search (chained)",
    "metaheuristic": "metaheuristic",
    "ensemble": "ensemble",
    "exact": "exact",
    "hybrid": "exact hybrid",
    "hunt-14": "14-day hunt",
}
STOP_LABEL = {
    "time_limit": "time limit",
    "max_iter": "iteration limit",
    "converged": "converged",
    "callback": "stopped by callback",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clock(minutes: float) -> str:
    total = math.floor(minutes + 0.5)
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def address_of(place: dict[str, str]) -> str:
    street = " ".join(x for x in (place.get("street", ""), place.get("housenumber", "")) if x)
    tail = " ".join(x for x in (place.get("postcode", ""), place.get("city", "")) if x)
    return ", ".join(x for x in (street, tail) if x)


def days_from_plan(
    plan: dict[str, Any], places: dict[str, dict[str, str]], start: float
) -> list[dict[str, Any]]:
    """Build the per-day timetable from ``best_plan.json`` and the committed matrices."""
    labels = [row["label"] for row in read_csv(DATA / "madrid_burger_king.csv")]
    n = len(labels)
    cols = range(1, n + 1)
    times = np.loadtxt(DATA / "madrid_burger_king_times_min.csv", delimiter=",", skiprows=1, usecols=cols)
    dist = np.loadtxt(DATA / "madrid_burger_king_dist_km.csv", delimiter=",", skiprows=1, usecols=cols)
    index = {label: i for i, label in enumerate(labels)}
    service = float(plan["instance"]["service_min"])
    out = []
    for day_number, day_labels in enumerate(plan["days_labels"], start=1):
        seq = [0, *(index[label] for label in day_labels), 0]
        at = start
        stops: list[dict[str, Any]] = [
            {
                "order": 0,
                "label": labels[0],
                "arrival": clock(at),
                "departure": clock(at),
                "travel": 0.0,
                "service": 0.0,
            }
        ]
        for order, (a, b) in enumerate(pairwise(seq), start=1):
            travel = float(times[a, b])
            at += travel
            last = b == 0
            stops.append(
                {
                    "order": 0 if last else order,
                    "label": labels[b],
                    "arrival": clock(at),
                    "departure": clock(at if last else at + service),
                    "travel": round(travel, 1),
                    "service": 0.0 if last else service,
                }
            )
            if not last:
                at += service
        driving = float(sum(times[a, b] for a, b in pairwise(seq)))
        km = float(sum(dist[a, b] for a, b in pairwise(seq)))
        out.append(
            {
                "day": day_number,
                "n_stops": len(day_labels),
                "driving": round(driving),
                "km": round(km, 1),
                "service": round(service * len(day_labels)),
                "total": round(driving + service * len(day_labels)),
                "back_at": stops[-1]["arrival"],
                "stops": stops,
            }
        )
    return out


def days_from_csvs(
    timetable_csv: Path, days_csv: Path, places: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """Build the per-day timetable from the CSV pair written by examples/technician_madrid.py."""
    rows = read_csv(timetable_csv)
    summary = {int(row["day"]): row for row in read_csv(days_csv)}
    out = []
    for day_number in sorted({int(row["day"]) for row in rows}):
        day_rows = sorted(
            (row for row in rows if int(row["day"]) == day_number), key=lambda row: int(row["order"])
        )
        stops = [
            {
                "order": int(row["order"]),
                "label": row["label"],
                "arrival": row["arrival"],
                "departure": row["departure"],
                "travel": round(float(row["travel_min"]), 1),
                "service": round(float(row["service_min"]), 1),
            }
            for row in day_rows
        ]
        info = summary[day_number]
        out.append(
            {
                "day": day_number,
                "n_stops": int(info["n_stops"]),
                "driving": round(float(info["driving_min"])),
                "km": round(float(info["driving_km"]), 1) if info.get("driving_km") else None,
                "service": round(float(info["service_min"])),
                "total": round(float(info["total_min"])),
                "back_at": info["back_at"],
                "stops": stops,
            }
        )
    return out


def google_urls(days: list[dict[str, Any]], places: dict[str, dict[str, str]]) -> None:
    """Attach the Google Maps Directions links of every day, straight from ``skroute.viz``.

    The library owns the leg splitting (Google takes nine intermediate stops per link, and
    consecutive legs share their boundary stop), so the page never reimplements it.
    """
    from skroute.viz import google_maps_urls

    labels = list(places)
    coords = np.array([[float(p["lat"]), float(p["lon"])] for p in places.values()], dtype=float)
    depot = labels[0]
    route: list[str] = []
    for day in days:
        route.append(depot)
        route.extend(stop["label"] for stop in day["stops"] if stop["label"] != depot)
    route.append(depot)
    per_day = google_maps_urls(route, coords, labels=labels)
    for day, urls in zip(days, per_day, strict=True):
        day["urls"] = urls


def study_section(results_path: Path | None, bounds_path: Path | None, best_name: str | None) -> str:
    """The ranking table of the study, or "" when the results were not passed."""
    if results_path is None or not results_path.exists():
        return ""
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    ranked = sorted(
        (row for row in rows if "objective" in row), key=lambda row: (row["objective"], row["wall_s"])
    )
    failed = [row for row in rows if "error" in row]
    bound_text = ""
    if bounds_path is not None and bounds_path.exists():
        bound = json.loads(bounds_path.read_text(encoding="utf-8"))
        days_lb = max(bound["days_lower_bound_from_driving"], bound["days_lower_bound_from_stops"])
        bound_text = (
            f'<p class="lede">Proven lower bound (exact solver, Dantzig&ndash;Fulkerson&ndash;Johnson cuts on HiGHS): '
            f"no plan can drive less than <b>{bound['atsp_lower_bound_min']:.0f} min</b> in total &mdash; that is the "
            f"optimal closed tour over every restaurant without returning to the office in between &mdash; so at least "
            f"<b>{days_lb} days</b> are needed once the {bound['n'] - 1} half-hour visits are counted.</p>"
        )
    body = []
    for row in ranked:
        mark = ' class="pick"' if best_name and row["name"] == best_name else ""
        wall = row["wall_s"]
        wall_text = (
            f"{wall:.1f} s"
            if wall < 60
            else (f"{wall / 60:.0f} min" if wall < 3600 else f"{wall / 3600:.1f} h")
        )
        stop = STOP_LABEL.get(row.get("stop_reason"), row.get("stop_reason") or "")
        body.append(
            f"<tr{mark}><td>{esc(row['name'])}</td><td>{FAMILY_LABEL.get(row['family'], row['family'])}</td>"
            f'<td class="n">{row["days"]}</td><td class="n">{row["driving_min"]:.0f} min</td>'
            f'<td class="n">{wall_text}</td><td>{stop}</td></tr>'
        )
    note = ""
    if failed:
        detail = "; ".join(f"{esc(row['name'])} ({esc(row['error'][:90])})" for row in failed)
        note = f'<p class="lede">Did not run: {detail}.</p>'
    return f"""
<section class="detail" id="study">
  <div class="title"><h2>Which algorithm found this</h2><span class="sum">{len(ranked)} configurations on the same
  instance, ranked by result (fewest days first, then least driving). The delivered plan is highlighted.</span></div>
  {bound_text}
  <div class="tablewrap"><table><thead><tr><th>Configuration</th><th>Family</th><th class="n">Days</th>
  <th class="n">Driving</th><th class="n">Budget</th><th>Stopped by</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
  {note}
</section>
"""


def esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build(days: list[dict[str, Any]], places: dict[str, dict[str, str]], study: str, title: str) -> str:
    office = places["office"]
    restaurants = [
        {
            "label": label,
            "name": place["name"],
            "lat": float(place["lat"]),
            "lon": float(place["lon"]),
            "city": place.get("city", ""),
            "address": address_of(place),
        }
        for label, place in places.items()
        if label != "office"
    ]
    for day in days:
        for stop in day["stops"]:
            place = places.get(stop["label"], {})
            stop["name"] = place.get("name", stop["label"])
            stop["address"] = address_of(place) if place else ""
            stop["lat"] = float(place["lat"]) if place else None
            stop["lon"] = float(place["lon"]) if place else None
    data = {
        "office": {
            "name": "Office",
            "address": address_of(office),
            "lat": float(office["lat"]),
            "lon": float(office["lon"]),
        },
        "points": restaurants,
        "days": days,
        "palette": list(PALETTE),
    }
    total_driving = sum(day["driving"] for day in days)
    total_km = sum(day["km"] or 0 for day in days)
    latest = max(days, key=lambda day: day["total"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.format(
        title=esc(title),
        n_days=len(days),
        n_stops=sum(day["n_stops"] for day in days),
        driving_h=f"{total_driving / 60:.1f}",
        total_km=f"{total_km:,.0f}",
        mean_stops=f"{sum(day['n_stops'] for day in days) / len(days):.1f}",
        latest_back=esc(latest["back_at"]),
        latest_day=latest["day"],
        office_address=esc(address_of(office)),
        study=study,
        payload=payload,
    )


PAGE = """<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap">
<style>
:root {{
  --paper: #f6f7f3; --panel: #ffffff; --ink: #1b222a; --muted: #5b6670; --line: #d9ddd5; --line-soft: #e9ece6;
  --accent: #0e6f7c; --accent-ink: #ffffff; --office: #b4332a; --faded: #c4cac4; --hover: #eef2ef;
  --shadow: 0 1px 2px rgba(20, 30, 40, .08), 0 8px 24px -16px rgba(20, 30, 40, .25);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #151a1f; --panel: #1c2329; --ink: #e4e8e4; --muted: #98a3a8; --line: #2c353c; --line-soft: #232b31;
    --accent: #5fc3cf; --accent-ink: #0d1a1c; --office: #e2635a; --faded: #3a444b; --hover: #232c33;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.6);
  }}
}}
:root[data-theme="dark"] {{
  --paper: #151a1f; --panel: #1c2329; --ink: #e4e8e4; --muted: #98a3a8; --line: #2c353c; --line-soft: #232b31;
  --accent: #5fc3cf; --accent-ink: #0d1a1c; --office: #e2635a; --faded: #3a444b; --hover: #232c33;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.6);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 "Barlow", "Helvetica Neue", Arial, sans-serif; }}
a {{ color: var(--accent); }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 24px 20px 48px; display: grid; gap: 20px; }}
header {{ display: grid; gap: 14px; }}
.eyebrow {{ font: 600 13px/1 "Barlow Condensed", sans-serif; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
h1 {{ margin: 0; font: 700 clamp(30px, 4vw, 44px)/1.05 "Barlow Condensed", sans-serif; text-wrap: balance; letter-spacing: -.01em; }}
.lede {{ margin: 0; max-width: 68ch; color: var(--muted); }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
.stat {{ padding: 12px 16px 12px 0; border-right: 1px solid var(--line); margin-right: 16px; }}
.stat:last-child {{ border-right: 0; }}
.stat b {{ display: block; font: 600 30px/1.1 "Barlow Condensed", sans-serif; font-variant-numeric: tabular-nums; }}
.stat span {{ color: var(--muted); font-size: 13px; }}
.board {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(280px, 2fr); gap: 20px; align-items: start; }}
@media (max-width: 860px) {{ .board {{ grid-template-columns: 1fr; }} }}
.mapcard {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; box-shadow: var(--shadow); padding: 12px; position: relative; }}
.mapcard svg {{ width: 100%; height: auto; display: block; }}
.maptools {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; justify-content: space-between; padding: 0 4px 10px; }}
.maptools .hint {{ color: var(--muted); font-size: 13px; }}
button.tool {{ font: 600 13px/1 "Barlow Condensed", sans-serif; letter-spacing: .06em; text-transform: uppercase; padding: 9px 12px; border-radius: 4px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }}
button.tool[aria-pressed="true"] {{ background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }}
button.tool:focus-visible, .daylist button:focus-visible, a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.tooltip {{ position: absolute; pointer-events: none; background: var(--ink); color: var(--paper); font-size: 13px; line-height: 1.35; padding: 8px 10px; border-radius: 4px; max-width: 260px; display: none; z-index: 2; }}
.tooltip b {{ display: block; font: 600 15px/1.2 "Barlow Condensed", sans-serif; }}
.daylist {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; box-shadow: var(--shadow); overflow: hidden; }}
.daylist h2 {{ margin: 0; padding: 12px 14px; font: 600 13px/1 "Barlow Condensed", sans-serif; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--line); }}
.daylist .head, .daylist button {{ display: grid; grid-template-columns: 14px 58px 1fr 1fr 1fr; gap: 10px; align-items: center; width: 100%; text-align: left; }}
.daylist .head {{ padding: 8px 14px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; border-bottom: 1px solid var(--line-soft); }}
.daylist button {{ padding: 9px 14px; border: 0; border-bottom: 1px solid var(--line-soft); background: transparent; color: var(--ink); font: inherit; font-variant-numeric: tabular-nums; cursor: pointer; }}
.daylist button:hover {{ background: var(--hover); }}
.daylist button[aria-pressed="true"] {{ background: var(--hover); box-shadow: inset 3px 0 0 var(--accent); }}
.daylist .sw {{ width: 12px; height: 12px; border-radius: 2px; }}
.daylist .d {{ font: 600 16px/1 "Barlow Condensed", sans-serif; }}
.daylist .r {{ text-align: right; }}
.detail {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; box-shadow: var(--shadow); padding: 16px 18px 18px; display: grid; gap: 14px; }}
.detail .title {{ display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: baseline; }}
.detail h2 {{ margin: 0; font: 700 28px/1 "Barlow Condensed", sans-serif; }}
.detail .sum {{ color: var(--muted); }}
.links {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.links a {{ display: inline-flex; align-items: center; gap: 8px; text-decoration: none; font: 600 14px/1 "Barlow Condensed", sans-serif; letter-spacing: .05em; text-transform: uppercase; padding: 10px 14px; border-radius: 4px; background: var(--accent); color: var(--accent-ink); }}
.links a svg {{ width: 14px; height: 14px; }}
.links .note {{ color: var(--muted); font-size: 13px; }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: top; white-space: nowrap; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }}
td.n, th.n {{ text-align: right; }}
td.name {{ white-space: normal; min-width: 220px; }}
td.name small {{ display: block; color: var(--muted); }}
td .badge {{ display: inline-block; min-width: 22px; text-align: center; border-radius: 11px; padding: 1px 6px; font: 600 12px/18px "Barlow Condensed", sans-serif; color: #fff; }}
tr.depot td {{ color: var(--muted); }}
tr.pick td {{ background: var(--hover); box-shadow: inset 3px 0 0 var(--accent); font-weight: 600; }}
footer {{ color: var(--muted); font-size: 13px; display: grid; gap: 6px; max-width: 84ch; }}
footer code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; background: var(--panel); border: 1px solid var(--line); padding: 1px 5px; border-radius: 3px; }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">scikit-route &middot; worked example &middot; OpenStreetMap data, OSRM driving times</div>
  <h1>{title}</h1>
  <p class="lede">Every day leaves the office at 08:00 and returns to it; each maintenance visit takes half an hour and
  no day exceeds eight hours. Pick a day to see its route, the timetable of every stop and the link that opens it in
  Google Maps. Office: {office_address}.</p>
  <div class="stats">
    <div class="stat"><b>{n_days}</b><span>days</span></div>
    <div class="stat"><b>{n_stops}</b><span>maintenance visits</span></div>
    <div class="stat"><b>{driving_h} h</b><span>driving in total ({total_km} km)</span></div>
    <div class="stat"><b>{mean_stops}</b><span>stops per day on average</span></div>
    <div class="stat"><b>{latest_back}</b><span>latest return (day {latest_day})</span></div>
  </div>
</header>

<section class="board">
  <div class="mapcard">
    <div class="maptools">
      <div class="hint" id="hint">Day 1 highlighted &middot; hover a stop</div>
      <button class="tool" id="all" aria-pressed="false">Show every day</button>
    </div>
    <svg id="map" viewBox="0 0 900 780" role="img" aria-label="Map of the restaurants and the route of the selected day"></svg>
    <div class="tooltip" id="tip"></div>
  </div>
  <aside class="daylist">
    <h2>Days</h2>
    <div class="head"><span></span><span>Day</span><span class="r">Stops</span><span class="r">Driving</span><span class="r">Back</span></div>
    <div id="days"></div>
  </aside>
</section>

<section class="detail" id="detail"></section>
{study}
<footer>
  <div>Built by <code>examples/technician_madrid_study/plan_page.py</code> from the plan recorded in
  <code>examples/technician_madrid_study/results/best_plan.json</code>. Reproduce the plan with
  <code>python examples/technician_madrid.py</code> and the comparison with
  <code>python examples/technician_madrid_study/benchmark.py</code>.</div>
  <div>Straight lines between stops; the road route is the Google Maps link. Travel times are static, captured on
  2026-09-05, with no lunch break and every restaurant assumed open on arrival. Data &copy; OpenStreetMap contributors
  (ODbL); routing by OSRM (router.project-osrm.org).</div>
</footer>
</div>

<script id="plan" type="application/json">{payload}</script>
<script>
(() => {{
  const plan = JSON.parse(document.getElementById("plan").textContent);
  const color = k => plan.palette[(k - 1) % plan.palette.length];
  const svg = document.getElementById("map"), NS = "http://www.w3.org/2000/svg";
  const W = 900, H = 780, PAD = 30;
  const pts = plan.points.concat([plan.office]);
  const lat0 = pts.reduce((a, p) => a + p.lat, 0) / pts.length;
  const kx = Math.cos(lat0 * Math.PI / 180);
  const xs = pts.map(p => p.lon * kx), ys = pts.map(p => p.lat);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const s = Math.min((W - 2 * PAD) / (maxX - minX), (H - 2 * PAD) / (maxY - minY));
  const ox = (W - s * (maxX - minX)) / 2, oy = (H - s * (maxY - minY)) / 2;
  const X = p => ox + (p.lon * kx - minX) * s, Y = p => H - (oy + (p.lat - minY) * s);
  const el = (tag, attrs, parent) => {{
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    (parent || svg).appendChild(node);
    return node;
  }};
  const byLabel = Object.fromEntries(plan.points.map(p => [p.label, p]));
  const dayOf = {{}};
  for (const day of plan.days) for (const stop of day.stops) if (stop.label !== "office") dayOf[stop.label] = day;
  const scale = el("g", {{}});
  const barPx = (10 / (111.32 * kx)) * kx * s;  // ten kilometres
  el("line", {{ x1: PAD, y1: H - 16, x2: PAD + barPx, y2: H - 16, stroke: "currentColor", "stroke-width": 1.5 }}, scale);
  el("text", {{ x: PAD + barPx / 2, y: H - 22, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }}, scale).textContent = "10 km";
  const gRoutes = el("g", {{}}), gPoints = el("g", {{}}), gMarks = el("g", {{}});
  const tip = document.getElementById("tip");
  let selected = 1, showAll = false;

  function esc(text) {{
    return String(text).replace(/[&<>"]/g, c => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }})[c]);
  }}
  function coordsOf(day) {{
    return day.stops.map(stop => stop.label === "office" ? plan.office : byLabel[stop.label]).filter(Boolean);
  }}
  function draw() {{
    gRoutes.innerHTML = ""; gPoints.innerHTML = ""; gMarks.innerHTML = "";
    for (const day of plan.days) {{
      if (!showAll && day.day !== selected) continue;
      const path = coordsOf(day).map((p, i) => (i ? "L" : "M") + X(p).toFixed(1) + " " + Y(p).toFixed(1)).join(" ");
      el("path", {{ d: path, fill: "none", stroke: color(day.day), "stroke-width": showAll ? 1.6 : 2.2, "stroke-linejoin": "round", opacity: showAll ? 0.85 : 0.95 }}, gRoutes);
    }}
    for (const p of plan.points) {{
      const day = dayOf[p.label];
      const active = showAll || (day && day.day === selected);
      const dot = el("circle", {{ cx: X(p), cy: Y(p), r: active ? 5 : 3.2, fill: active && day ? color(day.day) : "var(--faded)", stroke: "var(--panel)", "stroke-width": 1.2 }}, gPoints);
      const stop = day && day.stops.find(x => x.label === p.label);
      dot.addEventListener("mousemove", event => {{
        const box = svg.getBoundingClientRect();
        tip.style.display = "block";
        tip.style.left = (event.clientX - box.left + 14) + "px";
        tip.style.top = (event.clientY - box.top + 14) + "px";
        tip.innerHTML = "<b>" + esc(p.name) + "</b>" + (p.address ? esc(p.address) + "<br>" : "") +
          (day ? "day " + day.day + ", stop " + stop.order + " &middot; arrives " + stop.arrival + ", leaves " + stop.departure : "");
      }});
      dot.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});
    }}
    if (!showAll) {{
      const day = plan.days.find(d => d.day === selected);
      for (const stop of day.stops) {{
        if (stop.label === "office") continue;
        const p = byLabel[stop.label];
        if (!p) continue;
        el("circle", {{ cx: X(p), cy: Y(p), r: 9, fill: color(day.day), stroke: "var(--panel)", "stroke-width": 1.5 }}, gMarks);
        el("text", {{ x: X(p), y: Y(p) + 3.5, "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: "#fff", "font-family": "Barlow Condensed, sans-serif" }}, gMarks).textContent = stop.order;
      }}
    }}
    const o = plan.office, cx = X(o), cy = Y(o), star = [];
    for (let i = 0; i < 10; i++) {{
      const r = i % 2 ? 5 : 11, a = -Math.PI / 2 + i * Math.PI / 5;
      star.push((cx + r * Math.cos(a)).toFixed(1) + "," + (cy + r * Math.sin(a)).toFixed(1));
    }}
    el("polygon", {{ points: star.join(" "), fill: "var(--office)", stroke: "var(--panel)", "stroke-width": 1.2 }}, gMarks);
    el("text", {{ x: cx + 14, y: cy + 4, "font-size": 12, "font-weight": 600, fill: "var(--office)", "font-family": "Barlow Condensed, sans-serif" }}, gMarks).textContent = "office";
    document.getElementById("hint").textContent = showAll
      ? "Every day, one colour each"
      : "Day " + selected + " highlighted \u00b7 hover a stop";  // textContent: a real character, not an entity
  }}
  const list = document.getElementById("days");
  for (const day of plan.days) {{
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.day = day.day;
    button.setAttribute("aria-pressed", String(day.day === selected));
    button.innerHTML = '<span class="sw" style="background:' + color(day.day) + '"></span>' +
      '<span class="d">Day ' + day.day + '</span><span class="r">' + day.n_stops + '</span>' +
      '<span class="r">' + day.driving + ' min</span><span class="r">' + esc(day.back_at) + '</span>';
    button.addEventListener("click", () => {{
      selected = day.day; showAll = false;
      document.getElementById("all").setAttribute("aria-pressed", "false");
      update();
    }});
    list.appendChild(button);
  }}
  document.getElementById("all").addEventListener("click", event => {{
    showAll = !showAll;
    event.currentTarget.setAttribute("aria-pressed", String(showAll));
    draw();
  }});
  const pin = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z"/><circle cx="12" cy="10" r="2.5"/></svg>';
  function detail() {{
    const day = plan.days.find(d => d.day === selected);
    const legs = day.urls.map((url, i) => '<a href="' + esc(url) + '" target="_blank" rel="noopener">' + pin +
      (day.urls.length > 1 ? "Open in Google Maps &middot; leg " + (i + 1) + " of " + day.urls.length : "Open in Google Maps") + "</a>").join("");
    const note = day.urls.length > 1
      ? '<span class="note">Google Maps takes nine intermediate stops per link, so the second leg starts where the first ends.</span>'
      : "";
    const rows = day.stops.map(stop => {{
      const depot = stop.label === "office";
      const first = depot && stop === day.stops[0];
      const where = stop.address ? esc(stop.address) : "no address in OpenStreetMap";
      const link = stop.lat != null && !depot
        ? ' &middot; <a href="https://www.google.com/maps/search/?api=1&query=' + stop.lat.toFixed(6) + "%2C" + stop.lon.toFixed(6) + '" target="_blank" rel="noopener">see on Google Maps</a>'
        : "";
      const name = depot ? "Office" : esc(stop.name) + "<small>" + where + link + "</small>";
      const badge = depot ? (first ? "start" : "return") : '<span class="badge" style="background:' + color(day.day) + '">' + stop.order + "</span>";
      return '<tr class="' + (depot ? "depot" : "") + '"><td>' + badge + '</td><td class="name">' + name + "</td>" +
        '<td class="n">' + (first ? "" : stop.travel.toFixed(0) + " min") + "</td>" +
        '<td class="n">' + esc(stop.arrival) + "</td>" +
        '<td class="n">' + (depot ? "" : stop.service.toFixed(0) + " min") + "</td>" +
        '<td class="n">' + (depot && !first ? "" : esc(stop.departure)) + "</td></tr>";
    }}).join("");
    document.getElementById("detail").innerHTML =
      '<div class="title"><h2>Day ' + day.day + '</h2><span class="sum">' + day.n_stops + " visits &middot; " +
      day.driving + " min driving" + (day.km ? " (" + day.km.toFixed(0) + " km)" : "") + " &middot; " + day.service +
      " min of work on site &middot; back at the office at " + esc(day.back_at) + "</span></div>" +
      '<div class="links">' + legs + note + "</div>" +
      '<div class="tablewrap"><table><thead><tr><th>Stop</th><th>Restaurant</th><th class="n">Travel</th>' +
      '<th class="n">Arrival</th><th class="n">On site</th><th class="n">Departure</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
  }}
  function update() {{
    for (const button of list.querySelectorAll("button")) {{
      button.setAttribute("aria-pressed", String(Number(button.dataset.day) === selected));
    }}
    draw();
    detail();
  }}
  update();
}})();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plan", type=Path, default=RESULTS / "best_plan.json", help="recorded plan (best_plan.json)"
    )
    parser.add_argument(
        "--timetable",
        type=Path,
        help="timetable CSV written by examples/technician_madrid.py (instead of --plan)",
    )
    parser.add_argument(
        "--days", type=Path, help="days CSV written by examples/technician_madrid.py (with --timetable)"
    )
    parser.add_argument(
        "--results", type=Path, default=RESULTS / "results.jsonl", help="study results for the ranking table"
    )
    parser.add_argument(
        "--bounds",
        type=Path,
        default=RESULTS / "lower_bounds.json",
        help="lower bounds for the ranking table",
    )
    parser.add_argument("--no-study", action="store_true", help="leave the ranking table out")
    parser.add_argument("--start", default="08:00", help="departure time from the office (HH:MM)")
    parser.add_argument("--title", default="Every Burger King in Madrid, one technician", help="page title")
    parser.add_argument(
        "--out", type=Path, default=Path("technician_madrid_plan.html"), help="HTML file to write"
    )
    args = parser.parse_args(argv)

    if args.timetable and not args.days:
        parser.error("--timetable needs --days (both files are written side by side by the example)")
    places = {row["label"]: row for row in read_csv(DATA / "madrid_burger_king.csv")}
    if args.timetable:
        days = days_from_csvs(args.timetable, args.days, places)
        best_name = None
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        hours, minutes = (int(part) for part in args.start.split(":"))
        days = days_from_plan(plan, places, hours * 60 + minutes)
        best_name = plan.get("found_by")
    google_urls(days, places)
    study = "" if args.no_study else study_section(args.results, args.bounds, best_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(days, places, study, args.title), encoding="utf-8")
    driving = sum(day["driving"] for day in days)
    print(f"{args.out}: {len(days)} days, {sum(d['n_stops'] for d in days)} visits, {driving} min of driving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
