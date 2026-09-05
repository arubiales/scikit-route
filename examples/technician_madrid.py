"""Plan a maintenance technician's days over every Burger King of the Madrid region (D35).

    python examples/technician_madrid.py                          # committed data, a two-minute search
    python examples/technician_madrid.py --quick                  # a few seconds (the smoke test)
    python examples/technician_madrid.py --solver sa --time-limit 60 --live
    python examples/technician_madrid.py --refresh                # fetch OSM, Nominatim and OSRM again
    python examples/technician_madrid.py --refresh --provider google --google-key AIza...
    python examples/technician_madrid.py --google-key AIza...     # + a Google Maps page of the plan

The case, in the owner's words: *"imaginemos que tenemos que hacer los mantenimientos de los
Burger King de Madrid y nuestra oficina está en Leganés en Calle Ramón y Cajal 18. Un
mantenimiento medio tarda 30 min. Quiero ver todas las rutas como serían para cubrir todos los
Burger King por un técnico de mantenimiento de sistemas de alarma día tras día hasta que los
haga todos, lo más óptimo posible. Su jornada es de 8 horas."* — every Burger King of the
Comunidad de Madrid (OpenStreetMap, ``brand:wikidata=Q177054``), an office in Leganés, thirty
minutes per visit, eight-hour days, one technician, as few days and as little driving as
possible, on real road travel times.

The model: one trip of the multi-trip objective is one working day. The cost matrix ``X`` is
the driving time in minutes, so the objective is total driving; ``time_matrix`` is the same
matrix, ``service_time`` the minutes per visit, ``max_time_work`` the working day, and
``extra_cost`` — one day's budget in minutes by default — the charge per extra day, so a plan
with fewer days always wins before a plan with less driving. The giant tour is cut into days
with ``split="optimal"``.

The search: ``MultiStart(IteratedLocalSearch(local_search=("or_opt",), n_candidates=5))`` in
parallel processes, priced with the optimal split at every move, for 85 % of ``--time-limit``;
then a short polish of the winner with the full move set (2-opt and Or-opt). Relocations are
the move that repacks a day; on this instance the default move set (2-opt first) and a
greedy-split search followed by an optimal-split re-pricing both stop at 16 days within two
minutes, while Or-opt alone reaches 15 in about a third of the restarts.

The data of the run (the restaurants, the office and the OSRM matrices) are committed under
``examples/data/`` — see its README for the provenance — so the example runs offline;
``--refresh`` fetches everything again from the live services and rewrites the CSVs.

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skroute import (
    Genetic,
    Insertion,
    IteratedLocalSearch,
    MultiStart,
    NearestNeighbour,
    SimulatedAnnealing,
    TabuSearch,
)
from skroute.base import BaseRouter
from skroute.exceptions import InfeasibleProblemError
from skroute.metrics import Stop, timetable, timetable_summary
from skroute.utils import Bunch

log = logging.getLogger("skroute")

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
STEM = "madrid_burger_king"
PREFIX = "technician_madrid"

AREA = "Comunidad de Madrid"
BRAND_WIKIDATA = "Q177054"  # Burger King
AMENITY = "fast_food"
OFFICE_LABEL = "office"
OFFICE_ADDRESS = "Calle Ramón y Cajal 18, Leganés, Madrid, España"
OFFICE_NAME = "Oficina (Calle Ramón y Cajal 18, Leganés)"
# the addr:* columns of the office row: `geocode` returns a display name, not its parts
OFFICE_ADDR = {
    "city": "Leganés",
    "street": "Calle de Ramón y Cajal",
    "housenumber": "18",
    "postcode": "28916",
}
DUPLICATE_METRES = 60.0
ATTRIBUTION = "Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org)"
COLUMNS = ["label", "name", "lat", "lon", "city", "street", "housenumber", "postcode", "opening_hours"]
ADDR_TAGS = {
    "city": "addr:city",
    "street": "addr:street",
    "housenumber": "addr:housenumber",
    "postcode": "addr:postcode",
}

# fraction of --time-limit spent polishing the winner with the full move set (the rest is the search)
POLISH_SHARE = 0.15
# the search phase: relocations only, short candidate lists — the moves that repack a day, many of them
SEARCH_MOVES = ("or_opt",)
SEARCH_CANDIDATES = 5
QUICK_ITER = 10


# --------------------------------------------------------------------------- data
def load_data(data_dir: Path) -> Bunch:
    """The committed CSVs: ``labels``, ``names``, ``addresses``, ``coords``, ``time``, ``distance``.

    ``time`` is in minutes, ``distance`` in kilometres; the office is row 0.
    """
    points = data_dir / f"{STEM}.csv"
    with points.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or rows[0]["label"] != OFFICE_LABEL:
        raise SystemExit(
            f"{points}: the first row must be the office (label {OFFICE_LABEL!r}); run --refresh"
        )
    labels = [r["label"] for r in rows]
    coords = np.array([[float(r["lat"]), float(r["lon"])] for r in rows])
    time_ = _read_matrix(data_dir / f"{STEM}_times_min.csv", labels)
    distance = _read_matrix(data_dir / f"{STEM}_dist_km.csv", labels)
    return Bunch(
        labels=labels,
        names=[r["name"] for r in rows],
        addresses=[_address(r) for r in rows],
        coords=coords,
        time=time_,
        distance=distance,
        rows=rows,
    )


def _read_matrix(path: Path, labels: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    if header[1:] != labels or [r[0] for r in body] != labels:
        raise SystemExit(f"{path}: its labels do not match {STEM}.csv; run --refresh")
    return np.array([[float(v) for v in r[1:]] for r in body])


def _address(row: dict[str, str]) -> str:
    """``"<street> <housenumber>, <postcode> <city>"`` from the addr:* columns; ``""`` when unknown."""
    street = " ".join(p for p in (row.get("street", ""), row.get("housenumber", "")) if p)
    town = " ".join(p for p in (row.get("postcode", ""), row.get("city", "")) if p)
    return ", ".join(p for p in (street, town) if p)


def refresh_data(data_dir: Path, *, provider: str, api_key: str | None, limit: int | None = None) -> Bunch:
    """Fetch the restaurants (Overpass), the office (Nominatim), the matrices (OSRM/Google); rewrite the CSVs.

    ``limit`` keeps only the first restaurants — a quick check of the pipeline on a few points.
    """
    from skroute.preprocessing import fetch_pois, geocode, travel_time_matrix

    log.info("Fetching the Burger King restaurants of %s from OpenStreetMap (Overpass)...", AREA)
    pois = fetch_pois(AREA, amenity=AMENITY, wikidata=BRAND_WIKIDATA)
    keep = _drop_near_duplicates(pois.coords, DUPLICATE_METRES)
    log.info(
        "%d elements, %d kept after dropping near-duplicates within %.0f m",
        len(pois.labels),
        len(keep),
        DUPLICATE_METRES,
    )
    if limit is not None:
        keep = keep[:limit]
    log.info("Geocoding the office (%s)...", OFFICE_ADDRESS)
    geo_key = api_key if provider == "google" else None
    office = geocode(
        OFFICE_ADDRESS, provider="nominatim" if provider != "google" else "google", api_key=geo_key
    )
    log.info("Office at (%.6f, %.6f): %s", office.lat, office.lon, office.display_name)
    rows = [
        dict.fromkeys(COLUMNS, "")
        | OFFICE_ADDR
        | {"label": OFFICE_LABEL, "name": OFFICE_NAME, "lat": f"{office.lat:.6f}", "lon": f"{office.lon:.6f}"}
    ]
    for i in keep:
        tags = pois.tags[i]
        row = dict.fromkeys(COLUMNS, "")
        row.update(
            label=pois.labels[i],
            name=pois.names[i] or "Burger King",
            lat=f"{pois.coords[i, 0]:.6f}",
            lon=f"{pois.coords[i, 1]:.6f}",
            opening_hours=tags.get("opening_hours", ""),
        )
        for column, tag in ADDR_TAGS.items():
            row[column] = tags.get(tag, "")
        rows.append(row)
    coords = np.array([[float(r["lat"]), float(r["lon"])] for r in rows])
    log.info("Requesting the %d x %d travel-time matrix from %s...", len(rows), len(rows), provider)
    if provider == "google":
        res = travel_time_matrix(coords, provider="google", api_key=api_key, departure_time="now")
    else:
        res = travel_time_matrix(coords, provider="osrm")
    time_, distance = _fill_unroutable(
        np.asarray(res.time, dtype=float), np.asarray(res.distance, dtype=float) / 1000.0, coords
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    with (data_dir / f"{STEM}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_matrix(data_dir / f"{STEM}_times_min.csv", labels, time_, "%.2f")
    _write_matrix(data_dir / f"{STEM}_dist_km.csv", labels, distance, "%.3f")
    log.info("Data written to %s (%s)", data_dir, ATTRIBUTION)
    return load_data(data_dir)


def _drop_near_duplicates(coords: np.ndarray, metres: float) -> list[int]:
    """Row positions to keep: a point within ``metres`` of an earlier one is dropped (a way and its node)."""
    from skroute.preprocessing import haversine_matrix

    D = haversine_matrix(coords) * 1000.0
    return [i for i in range(len(coords)) if not (D[i, :i] < metres).any()]


def _fill_unroutable(
    time_: np.ndarray, distance: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the ``nan`` of each matrix on its own mask with the great-circle distance (30 km/h for minutes).

    An unroutable pair leaves ``nan`` in both matrices; a server that answers without a ``distances``
    table leaves the kilometres alone ``nan``. Either way both files must stay finite.
    """
    from skroute.preprocessing import haversine_matrix

    missing_time, missing_distance = np.isnan(time_), np.isnan(distance)
    if missing_time.any() or missing_distance.any():
        km = haversine_matrix(coords)
        log.warning(
            "%d unroutable times and %d missing distances filled with the great-circle distance "
            "(at 30 km/h for the times)",
            int(missing_time.sum()),
            int(missing_distance.sum()),
        )
        time_ = np.where(missing_time, km / 30.0 * 60.0, time_)
        distance = np.where(missing_distance, km, distance)
    return time_, distance


def _write_matrix(path: Path, labels: list[str], M: np.ndarray, fmt: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", *labels])
        for label, row in zip(labels, M, strict=True):
            writer.writerow([label, *(fmt % v for v in row)])


def display_names(data: Bunch) -> dict[str, str]:
    """``{label: name}`` for the maps: the brand plus the address when OpenStreetMap has one, else the OSM id.

    The office keeps its own name.
    """
    names = {}
    for label, name, address in zip(data.labels, data.names, data.addresses, strict=True):
        if label == OFFICE_LABEL:
            names[label] = name
        elif address:
            names[label] = f"{name} - {address}"
        else:
            names[label] = f"{name} ({label})"
    return names


# --------------------------------------------------------------------------- solving
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a technician's days over every Burger King of the Madrid region.",
        epilog=ATTRIBUTION,
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, metavar="DIR", help="directory of the CSVs"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="fetch the live services again and rewrite the CSVs"
    )
    parser.add_argument(
        "--provider", choices=["osrm", "google"], default="osrm", help="routing service for --refresh"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="with --refresh: keep only the first N restaurants (a quick check of the pipeline)",
    )
    parser.add_argument(
        "--google-key",
        default=None,
        help="Google Maps API key (default: GOOGLE_MAPS_API_KEY); also writes the Google Maps page",
    )
    parser.add_argument("--service", type=float, default=30.0, metavar="MIN", help="minutes per visit")
    parser.add_argument("--hours", type=float, default=8.0, help="working hours per day")
    parser.add_argument("--start", type=_hhmm, default="08:00", help="departure time from the office, HH:MM")
    parser.add_argument(
        "--solver",
        choices=["multistart", "ils", "sa", "tabu", "genetic"],
        default="multistart",
        help="search",
    )
    parser.add_argument("--seed", type=int, default=0, help="random_state")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="wall-clock budget of the whole search",
    )
    parser.add_argument(
        "--quick", action="store_true", help="a few iterations instead of the time budget (tests)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(f"./{PREFIX}_out"), metavar="DIR", help="output directory"
    )
    parser.add_argument("--live", action="store_true", help="watch the search with skroute.viz.LivePlot")
    parser.add_argument(
        "--record", metavar="GIF", default=None, help="record the search and save it as a GIF"
    )
    parser.add_argument(
        "--extra-day-cost",
        type=float,
        default=None,
        metavar="MIN",
        help="charge per extra day in minutes of driving (default: the day's budget, --hours * 60)",
    )
    parser.add_argument("--verbose", action="store_true", help="also log the solvers' own progress")
    return parser


def _hhmm(value: str) -> str:
    """argparse type of ``--start``: the rule of ``skroute.metrics.timetable`` (``HH:MM``, 24-hour clock)."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value)
    if match is None or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise argparse.ArgumentTypeError(f"must be HH:MM on the 24-hour clock, got {value!r}")
    return value.strip()


def _n_workers(n_restarts: int) -> int:
    """Processes ``MultiStart(n_jobs=-1)`` starts: joblib's count, which honours CPU quotas and affinity."""
    from joblib import cpu_count

    return max(1, min(n_restarts, cpu_count()))


def build_search(args: argparse.Namespace, seconds: float) -> BaseRouter:
    """The search-phase estimator of ``--solver`` under a ``seconds`` wall budget (``--quick``: a few steps).

    `solve` fits it under ``split="optimal"`` and then polishes its tour.
    """
    quick = args.quick
    seed = args.seed
    verbose = 1 if args.verbose else 0  # the solvers' own progress records, through the skroute logger
    ils_kw: dict[str, Any] = dict(
        n_iter=QUICK_ITER if quick else 10**6,
        patience=None,
        local_search=SEARCH_MOVES,
        n_candidates=SEARCH_CANDIDATES,
        random_state=seed,
        verbose=verbose,
    )
    if args.solver == "multistart":
        n_restarts = 2 if quick else 8
        # the restarts run in waves of one per CPU: the per-restart budget is the wall budget over the waves
        per_restart = None if quick else seconds / math.ceil(n_restarts / _n_workers(n_restarts))
        inner = IteratedLocalSearch(time_limit=per_restart, **ils_kw)
        # processes, not threads: the multi-trip kernels hold the GIL while pricing a move, so threads
        # would run the restarts one at a time
        n_jobs = None if (quick or args.live or args.record) else -1
        return MultiStart(
            inner,
            n_restarts=n_restarts,
            n_jobs=n_jobs,
            prefer="processes",
            random_state=seed,
            verbose=verbose,
        )
    limit = None if quick else seconds
    if args.solver == "ils":
        return IteratedLocalSearch(time_limit=limit, **ils_kw)
    if args.solver == "sa":
        return SimulatedAnnealing(
            alpha=0.9 if quick else 0.999, time_limit=limit, random_state=seed, verbose=verbose
        )
    if args.solver == "tabu":
        return TabuSearch(
            n_iter=QUICK_ITER if quick else 10**6,
            patience=None,
            time_limit=limit,
            random_state=seed,
            verbose=verbose,
        )
    return Genetic(
        pop_size=30 if quick else 100,
        n_generations=10 if quick else 10**6,
        patience=None,
        time_limit=limit,
        random_state=seed,
        verbose=verbose,
    )


def solve(args: argparse.Namespace, data: Bunch, callback: Any = None) -> tuple[BaseRouter, dict[str, Any]]:
    """Search with relocations under ``split="optimal"``, then polish the winner with the full move set.

    Returns the polished estimator (fitted with ``split="optimal"``, coordinates attached) and a dict of facts
    about the two phases for the report.
    """
    budget = args.hours * 60.0
    fit_kw: dict[str, Any] = dict(
        time_matrix=data.time,
        labels=data.labels,
        depot=OFFICE_LABEL,
        coords=data.coords,
        max_time_work=budget,
        extra_cost=budget if args.extra_day_cost is None else args.extra_day_cost,
        service_time=args.service,
    )
    search_seconds = args.time_limit * (1.0 - POLISH_SHARE)
    polish_seconds = args.time_limit * POLISH_SHARE
    search = build_search(args, search_seconds)
    log.info("Search phase: %r under split='optimal'", search)
    t0 = time.perf_counter()
    search.fit(data.time, split="optimal", callback=callback, **fit_kw)
    search_wall = time.perf_counter() - t0
    log.info(
        "  %d days, %.0f min of driving in %.0f s (%s)",
        search.n_trips_,
        float(search.trip_costs_.sum()),
        search_wall,
        _iterations(search),
    )
    # A short iterated search from the winner with the default moves (2-opt and Or-opt, longer candidate
    # lists): it starts from the winning tour and keeps the best tour seen, so it never returns a worse plan.
    polish = IteratedLocalSearch(
        init=search.tour_,
        n_iter=2 if args.quick else 10**6,
        patience=None,
        time_limit=None if args.quick else polish_seconds,
        random_state=args.seed,
        verbose=1 if args.verbose else 0,
    )
    log.info("Polish phase: IteratedLocalSearch (2-opt and Or-opt) from the winner under split='optimal'")
    t1 = time.perf_counter()
    polish.fit(data.time, split="optimal", **fit_kw)
    polish_wall = time.perf_counter() - t1
    log.info(
        "  %d days, %.0f min of driving in %.0f s (%s)",
        polish.n_trips_,
        float(polish.trip_costs_.sum()),
        polish_wall,
        _iterations(polish),
    )
    # what the search phase really did: only the iterated searches run with the relocation-only move set
    moves = "Or-opt relocations, " if args.solver in {"multistart", "ils"} else ""
    facts = {
        "search": f"{type(search).__name__} ({moves}optimal split)",
        "search_days": int(search.n_trips_),
        "search_driving": float(search.trip_costs_.sum()),
        "search_seconds": search_wall,
        "polish_seconds": polish_wall,
        "fit_kw": fit_kw,
    }
    return polish, facts


def _iterations(est: BaseRouter) -> str:
    n_iter = getattr(est, "n_iter_", None)
    if isinstance(est, MultiStart):
        inner = [getattr(e, "n_iter_", 0) for e in est.estimators_]
        return f"{len(inner)} restarts, {sum(inner)} iterations, stop {est.stop_reason_!r}"
    if n_iter is None:
        return "one pass"
    return f"{n_iter} iterations, stop {est.stop_reason_!r}"


def baselines(data: Bunch, fit_kw: dict[str, Any]) -> list[tuple[str, int, float]]:
    """``(name, days, driving minutes)`` of the two construction heuristics under the same objective."""
    import warnings

    out = []
    for est in (NearestNeighbour(), Insertion()):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", UserWarning
            )  # they ignore the budget during their search, by design
            est.fit(data.time, split="optimal", **fit_kw)
        out.append((type(est).__name__, int(est.n_trips_), float(est.trip_costs_.sum())))
    return out


# --------------------------------------------------------------------------- outputs
def write_timetable(path: Path, days: list[list[Stop]], data: Bunch) -> None:
    names = dict(zip(data.labels, data.names, strict=True))
    addresses = dict(zip(data.labels, data.addresses, strict=True))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["day", "order", "label", "name", "address", "arrival", "departure", "travel_min", "service_min"]
        )
        for day in days:
            for s in day:
                writer.writerow(
                    [
                        s.day,
                        s.order,
                        s.label,
                        names[s.label],
                        addresses[s.label],
                        s.arrival_time,
                        s.departure_time,
                        f"{s.travel:.2f}",
                        f"{s.service:.2f}",
                    ]
                )


def write_days(path: Path, summary: list[dict[str, Any]], km: Sequence[float]) -> None:
    """The totals per day; ``km`` is the road distance of each day (from the kilometres file)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["day", "n_stops", "driving_min", "driving_km", "service_min", "total_min", "back_at"]
        )
        for row, day_km in zip(summary, km, strict=True):
            writer.writerow(
                [
                    row["day"],
                    row["n_stops"],
                    f"{row['driving']:.2f}",
                    f"{day_km:.1f}",
                    f"{row['service']:.2f}",
                    f"{row['total']:.2f}",
                    row["back_at"],
                ]
            )


def trip_km(est: BaseRouter, data: Bunch) -> list[float]:
    """Kilometres driven on each day: the legs of every trip summed over the distance matrix."""
    index = {label: i for i, label in enumerate(data.labels)}
    out = []
    for trip in est.trips_:
        idx = [index[label] for label in trip.tolist()]
        out.append(float(sum(data.distance[a, b] for a, b in pairwise(idx))))
    return out


def write_google_urls(path: Path, urls: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("day\tleg\turl\n")
        for k, legs in enumerate(urls, start=1):
            for j, url in enumerate(legs, start=1):
                fh.write(f"{k}\t{j}/{len(legs)}\t{url}\n")


def _map_aspect(ax: Any, lat: float) -> None:
    ax.set_aspect(1.0 / math.cos(math.radians(lat)))


def _figure(figsize: tuple[float, float]) -> tuple[Any, Any, Any]:
    """``(fig, ax, tab20)`` from matplotlib's object-oriented API: no pyplot, so drawing the PNGs never
    touches the backend -- under ``--live`` the LivePlot window keeps its own and ``plt.show()`` holds it.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    return fig, fig.subplots(), colormaps["tab20"]


def plot_days(est: BaseRouter, data: Bunch, summary: list[dict[str, Any]], path: Path) -> None:
    """Every day in its own colour over the points (longitude as x, latitude as y)."""
    fig, ax, cmap = _figure((9, 8.5))
    lat, lon = data.coords[:, 0], data.coords[:, 1]
    index = {label: i for i, label in enumerate(data.labels)}
    ax.scatter(lon[1:], lat[1:], s=14, color="0.35", zorder=3)
    for k, trip in enumerate(est.trips_):
        idx = [index[label] for label in trip.tolist()]
        row = summary[k]
        ax.plot(
            lon[idx],
            lat[idx],
            "-",
            color=cmap(k % 20),
            lw=1.6,
            alpha=0.9,
            zorder=2,
            label=f"Day {k + 1}: {row['n_stops']} stops, {row['driving']:.0f} min driving",
        )
    ax.plot(lon[0], lat[0], marker="*", ms=16, color="black", zorder=4)
    ax.annotate("office", (lon[0], lat[0]), xytext=(6, -14), textcoords="offset points", fontsize=9)
    ax.set_title(
        f"{est.n_trips_} days for {len(data.labels) - 1} restaurants, "
        f"{float(est.trip_costs_.sum()) / 60:.1f} h of driving",
        fontsize=12,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    _map_aspect(ax, float(lat.mean()))
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.text(
        0.995, 0.005, ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.4"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_day(est: BaseRouter, data: Bunch, days: list[list[Stop]], k: int, path: Path) -> None:
    """One day zoomed in, its stops numbered in visiting order; the other restaurants faint."""
    fig, ax, cmap = _figure((8, 7))
    lat, lon = data.coords[:, 0], data.coords[:, 1]
    index = {label: i for i, label in enumerate(data.labels)}
    trip = est.trips_[k].tolist()
    idx = [index[label] for label in trip]
    ax.scatter(lon[1:], lat[1:], s=10, color="0.75", zorder=1)
    color = cmap(k % 20)
    ax.plot(lon[idx], lat[idx], "-", color=color, lw=2, zorder=2)
    stops = days[k][1:-1]
    for s in stops:
        i = index[s.label]
        ax.plot(lon[i], lat[i], "o", ms=13, color=color, mec="black", zorder=3)
        ax.text(lon[i], lat[i], str(s.order), ha="center", va="center", fontsize=7.5, color="white", zorder=4)
    ax.plot(lon[0], lat[0], marker="*", ms=18, color="black", zorder=4)
    ax.annotate("office", (lon[0], lat[0]), xytext=(7, -14), textcoords="offset points", fontsize=9)
    pad = 0.012
    ax.set_xlim(lon[idx].min() - pad, lon[idx].max() + pad)
    ax.set_ylim(lat[idx].min() - pad, lat[idx].max() + pad)
    row = timetable_summary([days[k]])[0]
    ax.set_title(
        f"Day {k + 1}: {row['n_stops']} stops, {row['driving']:.0f} min driving, "
        f"{days[k][0].departure_time} to {row['back_at']}",
        fontsize=12,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    _map_aspect(ax, float(lat[idx].mean()))
    ax.text(
        0.995, 0.005, ATTRIBUTION, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.4"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def write_outputs(est: BaseRouter, data: Bunch, args: argparse.Namespace, out: Path) -> list[dict[str, Any]]:
    """Every file of the plan under ``out``; returns the per-day summary."""
    from skroute.viz import google_maps_urls, to_kml

    out.mkdir(parents=True, exist_ok=True)
    days = timetable(est, start=args.start)
    summary = timetable_summary(days)
    names = display_names(data)
    trip_names = [f"Día {k}" for k in range(1, est.n_trips_ + 1)]

    write_timetable(out / f"{PREFIX}_timetable.csv", days, data)
    write_days(out / f"{PREFIX}_days.csv", summary, trip_km(est, data))
    to_kml(est, path=out / f"{PREFIX}.kml", names=names, depot_name=OFFICE_NAME, trip_names=trip_names)
    write_google_urls(out / f"{PREFIX}_google_urls.txt", google_maps_urls(est))
    try:
        from skroute.viz import plot_route_map

        fig = plot_route_map(est, names=names, trip_names=trip_names)
        fig.write_html(out / f"{PREFIX}_map.html", include_plotlyjs="cdn")
    except ImportError as exc:
        log.warning("Plotly map skipped: %s", exc)
    key = args.google_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        from skroute.viz import google_maps_html

        google_maps_html(
            est,
            path=out / f"{PREFIX}_google.html",
            api_key=key,
            names=names,
            trip_names=trip_names,
            title="Burger King maintenance plan, Madrid",
        )
    try:
        plot_days(est, data, summary, out / f"{PREFIX}_days.png")
        plot_day(est, data, days, 0, out / f"{PREFIX}_day1.png")
    except ImportError as exc:
        log.warning("PNG pictures skipped: %s", exc)
    return summary


def report(est: BaseRouter, summary: list[dict[str, Any]], facts: dict[str, Any], data: Bunch) -> None:
    n_visits = len(data.labels) - 1
    log.info("")
    log.info("%-4s %5s %10s %10s %8s", "Day", "Stops", "Driving", "Service", "Back at")
    for row in summary:
        log.info(
            "%-4d %5d %6.0f min %6.0f min %8s",
            row["day"],
            row["n_stops"],
            row["driving"],
            row["service"],
            row["back_at"],
        )
    driving = float(est.trip_costs_.sum())
    service = sum(r["service"] for r in summary)
    longest = max(summary, key=lambda r: r["total"])
    log.info("")
    log.info(
        "Totals: %d days, %.1f h of driving (%.0f km), %.1f h of service, %.1f stops per day; "
        "the longest day is day %d (%.0f min, back at %s)",
        est.n_trips_,
        driving / 60,
        sum(trip_km(est, data)),
        service / 60,
        n_visits / est.n_trips_,
        longest["day"],
        longest["total"],
        longest["back_at"],
    )
    fit_kw = facts["fit_kw"]
    lower = (
        math.ceil(n_visits * fit_kw["service_time"] / fit_kw["max_time_work"])
        if fit_kw["service_time"]
        else 1
    )
    log.info(
        "Lower bound: %d visits x %.0f min = %.1f h of service alone, at least %d days of %.0f min",
        n_visits,
        fit_kw["service_time"],
        n_visits * fit_kw["service_time"] / 60,
        lower,
        fit_kw["max_time_work"],
    )
    log.info(
        "Search: %s gave %d days / %.0f min in %.0f s; the polish with "
        "2-opt and Or-opt gives %d days / %.0f min in %.0f s",
        facts["search"],
        facts["search_days"],
        facts["search_driving"],
        facts["search_seconds"],
        est.n_trips_,
        driving,
        facts["polish_seconds"],
    )
    for name, days_, minutes in baselines(data, fit_kw):
        log.info(
            "Baseline %s: %d days, %.0f min of driving (%+.0f min against the plan)",
            name,
            days_,
            minutes,
            minutes - driving,
        )
    log.info("%s", ATTRIBUTION)


def make_callback(args: argparse.Namespace, data: Bunch) -> tuple[Any, Any]:
    """``(callback, recorder)`` for ``--live`` / ``--record``; ``(None, None)`` without them."""
    if not args.live and not args.record:
        return None, None
    from skroute.viz import LivePlot, Recorder

    xy = data.coords[:, ::-1]  # x = longitude, y = latitude
    if args.record:
        rec = Recorder(every=10)
        return rec, rec
    return LivePlot(xy, every=5, title="Burger King maintenance plan"), None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hours <= 0 or args.service < 0 or args.time_limit <= 0:
        parser.error("--hours and --time-limit must be positive, --service non-negative")
    if args.provider == "google" and not (args.google_key or os.environ.get("GOOGLE_MAPS_API_KEY")):
        parser.error("--provider google needs --google-key or GOOGLE_MAPS_API_KEY")
    if args.limit is not None and (args.limit < 1 or not args.refresh):
        parser.error("--limit N needs --refresh and N >= 1")
    # the report below is the console output; the solvers add their own records only under --verbose
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_dir = args.data
    try:
        if args.refresh:
            key = args.google_key or os.environ.get("GOOGLE_MAPS_API_KEY")
            data = refresh_data(data_dir, provider=args.provider, api_key=key, limit=args.limit)
        else:
            data = load_data(data_dir)
    except FileNotFoundError as exc:
        parser.error(f"{exc.filename} not found: pass --data DIR or run --refresh")
    n = len(data.labels)
    log.info(
        "%d restaurants and the office; driving times in minutes (%s), %.0f min per visit, %.0f-hour days",
        n - 1,
        "asymmetric" if not np.allclose(data.time, data.time.T) else "symmetric",
        args.service,
        args.hours,
    )
    callback, recorder = make_callback(args, data)
    try:
        est, facts = solve(args, data, callback=callback)
    except InfeasibleProblemError as exc:
        # a round trip plus the service does not fit in the day: say so in one line, not 182 labels
        message = str(exc)
        if len(message) > 300:
            message = message[:300] + "..."
        parser.error(
            f"the day is too short for some restaurants: {message} -- raise --hours or lower --service"
        )
    if recorder is not None:
        recorder.save(args.record, data.coords[:, ::-1], speed=10)
        log.info("Recording written to %s", args.record)
    out = args.out
    summary = write_outputs(est, data, args, out)
    report(est, summary, facts, data)
    log.info("Files written to %s", out.resolve())
    if args.live:
        import matplotlib.pyplot as plt

        if matplotlib_interactive():
            plt.show()
    return 0


def matplotlib_interactive() -> bool:
    import matplotlib

    return matplotlib.get_backend().lower() not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


if __name__ == "__main__":
    sys.exit(main())
