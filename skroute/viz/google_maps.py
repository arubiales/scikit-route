"""The plan on Google Maps (D34): Directions URLs, a KML for Google My Maps and Google Earth, and a
standalone Maps JavaScript page.

No optional extra is needed: this module imports neither matplotlib nor plotly (numpy and the core
of scikit-route only), so ``import skroute.viz.google_maps`` works without the ``viz`` extras.
Nothing here needs an API key except [`google_maps_html`][skroute.viz.google_maps_html], whose page
loads the Maps JavaScript API — the URLs open in any browser or in the Google Maps app, and the KML
is imported into Google My Maps (``Create a new map → Import``) or opened in Google Earth.

The three functions read the same inputs: a fitted estimator (``trips_`` and the coordinates of
the fit), a ``RouteEvent`` (its trips) or a plain label-space route — an open tour, a closed one
or a multi-trip route with the depot repeated between the days — with ``coords=`` as ``(n, 2)``
``(latitude, longitude)`` pairs in matrix row order and ``labels=`` naming those rows (``None``:
the route holds row positions). Every trip becomes a closed day ``depot → stops → depot``.
"""

from __future__ import annotations

import html
import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlencode

import numpy as np

from ..base import BaseRouter
from ..utils.validation import check_is_fitted, coerce_labels
from ._static import DEPOT_COLOR, closed_trips, coords_array, is_event, trips_from_array

__all__ = ["google_maps_html", "google_maps_urls", "to_kml"]

#: One colour per day, shared with ``plot_route_map`` (matplotlib's ``tab10`` cycle): blue, orange,
#: green, red, purple, brown, pink, grey, olive, cyan. Day ``k`` takes ``PALETTE[(k - 1) % 10]``: the
#: ten colours are distinct, an eleventh day repeats the first one's.
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
MODES = ("driving", "walking", "bicycling", "transit")
# The documented limit of the Maps URL scheme on the desktop site and in the Google Maps app; a link
# opened in a phone's browser keeps only three (``max_waypoints=3`` for those).
URL_MAX_WAYPOINTS = 9
JS_MAX_WAYPOINTS = 25  # the documented limit of a DirectionsService request
ENV_KEY = "GOOGLE_MAPS_API_KEY"
DIRECTIONS_URL = "https://www.google.com/maps/dir/"
JS_API_URL = "https://maps.googleapis.com/maps/api/js"
KML_NS = "http://www.opengis.net/kml/2.2"
NO_COORDS = "no coordinates: pass coords= as (n, 2) (latitude, longitude) pairs in matrix row order"


# --------------------------------------------------------------------------- the plan
@dataclass(frozen=True, eq=False)
class Plan:
    """What the three exports share: the closed index trips of a solution over validated coordinates.

    Plans compare and hash by identity (``eq=False``): the generated field-wise ``__eq__`` would
    compare the arrays and raise instead of answering.

    Attributes
    ----------
    latlon : ndarray of shape (n, 2)
        ``(latitude, longitude)`` in decimal degrees, matrix row order.
    trips : list of ndarray
        One closed trip per day, row positions ``[depot, stop, ..., stop, depot]``.
    depot : int
        Row of the depot.
    labels : ndarray of shape (n,)
        The label of each row (the positions themselves for a plain route without ``labels=``).
    time : ndarray of shape (n, n) or None
        The raw travel-time matrix of the fit, when the solution carries one.
    solver : str or None
        Class name of the solver, when the solution came from one.
    """

    latlon: np.ndarray
    trips: list[np.ndarray]
    depot: int
    labels: np.ndarray
    time: np.ndarray | None
    solver: str | None

    @property
    def n_trips(self) -> int:
        return len(self.trips)

    def stops(self, k: int) -> np.ndarray:
        """The rows visited on day ``k`` (0-based), depot excluded."""
        return self.trips[k][1:-1]

    def driving(self, k: int) -> float | None:
        """Travel time of day ``k`` from ``time`` (``None`` without a time matrix)."""
        if self.time is None:
            return None
        trip = self.trips[k]
        return float(self.time[trip[:-1], trip[1:]].sum())


def _check_latlon(xy: np.ndarray) -> np.ndarray:
    lat, lon = xy[:, 0], xy[:, 1]
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise ValueError(
            "coords must be (latitude, longitude) in decimal degrees: latitude within [-90, 90] and "
            "longitude within [-180, 180] (an (x, y) array needs its columns swapped)"
        )
    return xy


def _problem_latlon(problem: Any, coords: Any) -> np.ndarray:
    if coords is None:
        if problem.coords is None:
            raise ValueError(NO_COORDS + " (or fit with coords=)")
        coords = problem.coords
    return _check_latlon(coords_array(coords, problem.n))


def _index_trips(problem: Any, label_trips: list[np.ndarray]) -> list[np.ndarray]:
    """Closed label trips (``trips_``, ``RouteEvent.trips``) as closed row-position trips."""
    return [np.array([problem.index_of(x) for x in trip.tolist()], dtype=np.int64) for trip in label_trips]


def _routes(obj: Any, coords: Any = None, labels: Any = None) -> Plan:
    """Decode ``obj`` — a fitted estimator, a ``RouteEvent`` or a plain route — into a ``Plan``."""
    if isinstance(obj, BaseRouter):
        check_is_fitted(obj)
        if labels is not None:
            raise ValueError("labels= applies to a plain route; a fitted estimator carries its own labels")
        problem = obj.problem_
        return Plan(
            _problem_latlon(problem, coords),
            _index_trips(problem, obj.trips_),
            problem.depot,
            problem.labels,
            problem.time,
            type(obj).__name__,
        )
    if is_event(obj):
        if labels is not None:
            raise ValueError("labels= applies to a plain route; an event carries its problem's labels")
        ev_problem: Any = getattr(obj, "problem", None)
        if ev_problem is None:
            raise ValueError("the event carries no RoutingProblem; nothing to decode")
        tour = obj.best_tour if obj.best_tour is not None else obj.tour
        if tour is None:
            raise ValueError("the event carries no tour yet; nothing to export")
        return Plan(
            _problem_latlon(ev_problem, coords),
            closed_trips(ev_problem, tour),
            ev_problem.depot,
            ev_problem.labels,
            ev_problem.time,
            str(obj.solver),
        )
    if coords is None:
        raise ValueError(NO_COORDS)
    xy = _check_latlon(coords_array(coords))
    n = xy.shape[0]
    if labels is None:
        positions = obj
        node_labels = np.arange(n, dtype=np.int64)
    else:
        node_labels = coerce_labels(labels, n)
        index = {label: i for i, label in enumerate(node_labels.tolist())}
        try:
            positions = np.array([index[x] for x in np.asarray(obj).tolist()], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"the route names {exc.args[0]!r}, which is not one of labels=") from None
    trips = trips_from_array(positions)
    if not trips:
        raise ValueError("the route visits no node besides the depot; nothing to export")
    if any(int(t.min()) < 0 or int(t.max()) >= n for t in trips):
        raise ValueError("the route indexes rows beyond the end of coords (or negative rows)")
    stops, counts = np.unique(np.concatenate([t[1:-1] for t in trips]), return_counts=True)
    repeated = node_labels[stops[counts > 1]].tolist()
    if repeated:
        shown = ", ".join(repr(x) for x in repeated)
        raise ValueError(f"the route visits {shown} more than once; every stop appears once across the days")
    return Plan(xy, trips, int(trips[0][0]), node_labels, None, None)


def node_names(labels: np.ndarray, names: Any) -> list[str]:
    """One display name per row: ``names`` as a sequence in row order or a mapping from label to name
    (a label without an entry keeps its label); ``None`` names every row by its label."""
    text = [str(x) for x in labels.tolist()]
    if names is None:
        return text
    if isinstance(names, str | bytes):
        raise TypeError(
            "names must be a sequence of str (matrix row order) or a mapping {label: name}, not a single str"
        )
    if isinstance(names, Mapping):
        return [str(names.get(label, t)) for label, t in zip(labels.tolist(), text, strict=True)]
    given = [str(x) for x in names]
    if len(given) != len(text):
        raise ValueError(f"names has {len(given)} entries but there are {len(text)} nodes")
    return given


def trip_labels(trip_names: Any, n_trips: int) -> list[str]:
    """One title per day: ``trip_names`` (exactly ``n_trips`` entries) or ``"Day 1"``, ``"Day 2"``..."""
    if trip_names is None:
        return [f"Day {k + 1}" for k in range(n_trips)]
    if isinstance(trip_names, str | bytes):
        raise TypeError("trip_names must be a sequence of str, one per trip, not a single str")
    given = [str(x) for x in trip_names]
    if len(given) != n_trips:
        raise ValueError(f"trip_names has {len(given)} entries but the plan has {n_trips} trips")
    return given


def _hex_color(color: Any) -> str:
    """``"#rrggbb"`` from a CSS-style hex colour (``"#rrggbb"``, ``"rrggbb"`` or ``"#rgb"``)."""
    text = str(color).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(2 * c for c in text)
    if len(text) != 6 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"colors must be hex colours such as '#1f77b4'; got {color!r}")
    return "#" + text.lower()


def trip_colors(colors: Any, n_trips: int) -> list[str]:
    """One ``"#rrggbb"`` per day: ``colors`` (cycled when shorter) or the ``PALETTE`` (cycled past ten
    days)."""
    given = list(PALETTE) if colors is None else [_hex_color(c) for c in colors]
    if not given:
        raise ValueError("colors must hold at least one colour")
    return [given[k % len(given)] for k in range(n_trips)]


def _kml_color(hex_color: str) -> str:
    """KML's ``aabbggrr`` (opaque) of a ``"#rrggbb"`` colour."""
    rr, gg, bb = hex_color[1:3], hex_color[3:5], hex_color[5:7]
    return f"ff{bb}{gg}{rr}"


def _legs(trip: Sequence[int], max_waypoints: int) -> list[tuple[int, int]]:
    """Split a closed trip of ``m`` stops into legs of at most ``max_waypoints`` intermediate stops.

    Each leg is ``(start, end)``, positions into ``trip``: origin ``trip[start]``, waypoints
    ``trip[start + 1:end]``, destination ``trip[end]``; consecutive legs share their boundary
    stop, the first origin and the last destination are the depot.
    """
    if (
        not isinstance(max_waypoints, int | np.integer)
        or isinstance(max_waypoints, bool)
        or max_waypoints < 1
    ):
        raise ValueError(f"max_waypoints must be a positive integer; got {max_waypoints!r}")
    legs = []
    start, last = 0, len(trip) - 1
    while start < last:
        end = min(start + int(max_waypoints) + 1, last)
        legs.append((start, end))
        start = end
    return legs


def _latlon_text(latlon: np.ndarray, row: int) -> str:
    return f"{latlon[row, 0]:.6f},{latlon[row, 1]:.6f}"


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    return mode


def _check_path(path: Any, coords: Any, caller: str) -> Path:
    """``path`` as a ``Path``; when missing, a ``TypeError`` that spots a path put in the ``coords`` slot."""
    if path is None:
        if isinstance(coords, str | os.PathLike):
            raise TypeError(
                f"path is required and coords looks like a file path ({os.fspath(coords)!r}): pass it as "
                f"path= or as the third argument, {caller}(obj, None, {os.fspath(coords)!r})"
            )
        raise TypeError("path is required: where to write the file")
    return Path(os.fspath(path))


# --------------------------------------------------------------------------- URLs
def google_maps_urls(
    obj: Any, coords: Any = None, *, labels: Any = None, mode: str = "driving", max_waypoints: int = 9
) -> list[list[str]]:
    """Google Maps Directions URLs of a plan: one list of links per day, each link one leg.

    The Maps URL scheme (``https://www.google.com/maps/dir/?api=1&origin=...&destination=...
    &waypoints=...&travelmode=...``) takes at most nine waypoints between the origin and the
    destination, so a day with more stops is split into consecutive legs that share their
    boundary stop: the first leg leaves the depot, the last one returns to it, and every leg
    opens in a browser or in the Google Maps app with turn-by-turn directions. Coordinates are
    written as ``latitude,longitude`` with six decimals (about ten centimetres).

    Parameters
    ----------
    obj : fitted estimator, RouteEvent or route
        A fitted solver (its ``trips_`` and the coordinates of the fit), a progress event (its
        best tour, decoded with its problem) or a plain label-space route: an open tour, a closed
        one or a multi-trip route with the depot repeated between the days (``route_``); every
        occurrence of the depot — the first entry — starts a new day.
    coords : (n, 2) array-like, optional
        ``(latitude, longitude)`` in decimal degrees, matrix row order. Default: the coordinates
        the problem carries (``fit(..., coords=)``). Required for a plain route.
    labels : sequence of n hashables, optional
        For a plain route: the label of each row of ``coords``. ``None`` means the route holds
        row positions.
    mode : {"driving", "walking", "bicycling", "transit"}, default "driving"
        The ``travelmode`` of the links (Google ignores waypoints in transit mode).
    max_waypoints : int, default 9
        Intermediate stops per link. Nine is Google's documented limit for the URL scheme on the
        desktop site and in the Google Maps app; a link opened in a phone's *browser* keeps only
        three, so pass ``max_waypoints=3`` for links that will be opened there.

    Returns
    -------
    urls : list of list of str
        ``urls[k]`` are the links of day ``k`` in driving order; one link when the day has at
        most ``max_waypoints`` stops.

    Raises
    ------
    ValueError
        Without coordinates, for coordinates outside the latitude/longitude ranges or with the
        wrong number of rows, for ``labels=`` with a fitted estimator or an event, a route naming
        something that is not one of ``labels=`` (or indexing beyond ``coords``), a route that
        visits no stop or one more than once, an event without a tour yet, an unknown ``mode`` or a
        non-positive ``max_waypoints``.
    NotFittedError
        For an estimator that has not been fitted.

    Examples
    --------
    >>> from skroute import IteratedLocalSearch
    >>> from skroute.datasets import load_barcelona
    >>> from skroute.viz import google_maps_urls
    >>> bcn = load_barcelona()  # 19 places, coords are (latitude, longitude)
    >>> ils = IteratedLocalSearch(random_state=0).fit(bcn.cost, labels=bcn.labels, coords=bcn.coords)
    >>> urls = google_maps_urls(ils)
    >>> len(urls), len(urls[0])  # one day of 18 stops: two links of at most nine waypoints
    (1, 2)
    >>> urls[0][0].startswith("https://www.google.com/maps/dir/?api=1&origin=41.398568%2C2.167441")
    True
    >>> urls[0][1].endswith("&travelmode=driving")
    True

    A plain route over its own coordinates (row positions; the depot repeats between the days):

    >>> office = [40.331, -3.766]
    >>> places = [office, [40.42, -3.70], [40.45, -3.69], [40.30, -3.73]]
    >>> [len(day) for day in google_maps_urls([0, 2, 1, 0, 3, 0], places)]
    [1, 1]
    >>> google_maps_urls(["office", "b", "a"], places[:3], labels=["office", "a", "b"], mode="walking")[0]
    ['https://www.google.com/maps/dir/?api=1&origin=40.331000%2C-3.766000&destination=40.331000%2C-3.766000&waypoints=40.450000%2C-3.690000%7C40.420000%2C-3.700000&travelmode=walking']
    """
    _check_mode(mode)
    plan = _routes(obj, coords, labels)
    urls = []
    for trip in plan.trips:
        links = []
        for start, end in _legs(trip.tolist(), max_waypoints):
            params = [
                ("api", "1"),
                ("origin", _latlon_text(plan.latlon, int(trip[start]))),
                ("destination", _latlon_text(plan.latlon, int(trip[end]))),
            ]
            if end - start > 1:
                params.append(
                    ("waypoints", "|".join(_latlon_text(plan.latlon, int(i)) for i in trip[start + 1 : end]))
                )
            params.append(("travelmode", mode))
            links.append(DIRECTIONS_URL + "?" + urlencode(params))
        urls.append(links)
    return urls


# --------------------------------------------------------------------------- KML
def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrib: str) -> ET.Element:
    """A child element. Tags are unqualified: the root declares ``xmlns`` and the children inherit
    it, so nothing is registered in ElementTree's process-wide namespace map."""
    element = ET.SubElement(parent, tag, attrib)
    if text is not None:
        element.text = text
    return element


def _kml_point(placemark: ET.Element, latlon: np.ndarray, row: int) -> None:
    point = _sub(placemark, "Point")
    _sub(point, "coordinates", f"{latlon[row, 1]:.6f},{latlon[row, 0]:.6f},0")


def to_kml(
    obj: Any,
    coords: Any = None,
    path: str | os.PathLike[str] | None = None,
    *,
    labels: Any = None,
    names: Any = None,
    depot_name: str = "Depot",
    trip_names: Any = None,
    colors: Any = None,
) -> Path:
    """Write a plan as a KML 2.2 file: one folder per day with numbered stops and the day's line.

    The document holds the depot as one placemark, one ``Style`` per day (a 4-pixel line and
    pins in the day's colour) and one ``Folder`` per day named ``trip_names[k]`` (default
    ``"Day k"``) with a ``Placemark`` per stop — named ``"k.j <name>"``, described as
    ``"Day k, stop j of m"`` — and a ``LineString`` of the closed trip ``depot → stops → depot``.
    The line joins the stops as the crow flies; the page of
    [`google_maps_html`][skroute.viz.google_maps_html] draws the roads.

    **Google My Maps**: open https://www.google.com/maps/d/, ``Create a new map``, then in the
    untitled layer ``Import`` and choose the file — every folder becomes a group of numbered pins
    with its line, and the map opens on any phone signed into the account. **Google Earth**:
    ``File → Open`` (desktop) or ``Projects → Open → Import KML file`` (web).

    Parameters
    ----------
    obj : fitted estimator, RouteEvent or route
        As in [`google_maps_urls`][skroute.viz.google_maps_urls].
    coords : (n, 2) array-like, optional
        ``(latitude, longitude)`` in matrix row order; default: the coordinates of the fit.
    path : str or path-like
        Where to write the file (``.kml``). Required — it is positional after ``coords`` so
        ``to_kml(est, None, "plan.kml")`` and ``to_kml(est, path="plan.kml")`` both work.
    labels : sequence of n hashables, optional
        For a plain route: the label of each row of ``coords``.
    names : sequence of n str or mapping {label: str}, optional
        A display name per node (matrix row order, or by label); default: the labels.
    depot_name : str, default "Depot"
        Name of the depot's placemark.
    trip_names : sequence of str, optional
        One folder name per day; default ``"Day 1"``, ``"Day 2"``...
    colors : sequence of str, optional
        Hex colours (``"#rrggbb"``) per day, cycled when shorter; default: ``PALETTE`` — ten
        distinct colours, so from the eleventh day on they repeat (pass more for longer plans).

    Returns
    -------
    path : pathlib.Path
        The file written.

    Raises
    ------
    TypeError
        Without ``path`` (also when a path was given in the ``coords`` slot), or when ``names`` or
        ``trip_names`` is a single str rather than a sequence.
    ValueError
        For the inputs [`google_maps_urls`][skroute.viz.google_maps_urls] rejects (no or invalid
        coordinates, a route visiting a node twice or none...), ``names`` or ``trip_names`` of the
        wrong length, and ``colors`` empty or not hex.
    NotFittedError
        For an estimator that has not been fitted.

    Examples
    --------
    >>> import tempfile, xml.etree.ElementTree as ET
    >>> from pathlib import Path
    >>> from skroute import TwoOpt
    >>> from skroute.datasets import load_barcelona
    >>> from skroute.viz import to_kml
    >>> bcn = load_barcelona()
    >>> two = TwoOpt().fit(
    ...     bcn.cost, labels=bcn.labels, coords=bcn.coords, time_matrix=bcn.time, max_time_work=6.0
    ... )  # 6-hour days: two of them (TwoOpt is deterministic, so the figures below are exact)
    >>> out = Path(tempfile.mkdtemp())
    >>> kml = to_kml(two, path=out / "plan.kml", trip_names=["Monday", "Tuesday"])
    >>> root = ET.parse(kml).getroot()
    >>> ns = {"k": "http://www.opengis.net/kml/2.2"}
    >>> [f.findtext("k:name", namespaces=ns) for f in root.iterfind(".//k:Folder", ns)]
    ['Monday', 'Tuesday']
    >>> [len(f.findall("k:Placemark", ns)) - 1 for f in root.iterfind(".//k:Folder", ns)]  # stops per day
    [7, 11]
    >>> (
    ...     root.find(".//k:Folder/k:Placemark/k:name", ns).text,
    ...     root.find(".//k:Folder/k:Placemark/k:description", ns).text,
    ... )
    ('1.1 23', 'Day 1, stop 1 of 7')
    """
    out = _check_path(path, coords, "to_kml")
    plan = _routes(obj, coords, labels)
    text = node_names(plan.labels, names)
    days = trip_labels(trip_names, plan.n_trips)
    palette = trip_colors(colors, plan.n_trips)

    root = ET.Element("kml", {"xmlns": KML_NS})
    doc = _sub(root, "Document")
    title = f"{plan.solver} plan" if plan.solver else "Route plan"
    _sub(doc, "name", f"{title}: {plan.n_trips} {'day' if plan.n_trips == 1 else 'days'}")
    depot_style = _sub(doc, "Style", id="depot")
    icon = _sub(depot_style, "IconStyle")
    _sub(icon, "color", _kml_color(DEPOT_COLOR))
    _sub(icon, "scale", "1.3")
    _sub(_sub(icon, "Icon"), "href", "http://maps.google.com/mapfiles/kml/paddle/wht-stars.png")
    for k, color in enumerate(palette):  # KML 2.2 orders a Style's children: IconStyle before LineStyle
        style = _sub(doc, "Style", id=f"day{k + 1}")
        icon = _sub(style, "IconStyle")
        _sub(icon, "color", _kml_color(color))
        _sub(_sub(icon, "Icon"), "href", "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png")
        line = _sub(style, "LineStyle")
        _sub(line, "color", _kml_color(color))
        _sub(line, "width", "4")

    depot = _sub(doc, "Placemark")
    _sub(depot, "name", str(depot_name))
    _sub(depot, "description", f"Start and end of every day ({text[plan.depot]})")
    _sub(depot, "styleUrl", "#depot")
    _kml_point(depot, plan.latlon, plan.depot)

    for k, trip in enumerate(plan.trips):
        folder = _sub(doc, "Folder")
        _sub(folder, "name", days[k])
        stops = trip[1:-1].tolist()
        for j, row in enumerate(stops, start=1):
            mark = _sub(folder, "Placemark")
            _sub(mark, "name", f"{k + 1}.{j} {text[row]}")
            _sub(mark, "description", f"Day {k + 1}, stop {j} of {len(stops)}")
            _sub(mark, "styleUrl", f"#day{k + 1}")
            _kml_point(mark, plan.latlon, row)
        route = _sub(folder, "Placemark")
        _sub(route, "name", f"{days[k]} route")
        _sub(route, "description", f"Day {k + 1}: {len(stops)} stops, depot to depot")
        _sub(route, "styleUrl", f"#day{k + 1}")
        line_string = _sub(route, "LineString")
        _sub(line_string, "tessellate", "1")
        _sub(
            line_string,
            "coordinates",
            " ".join(f"{plan.latlon[i, 1]:.6f},{plan.latlon[i, 0]:.6f},0" for i in trip.tolist()),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


# --------------------------------------------------------------------------- HTML
_PAGE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  html, body { height: 100%; margin: 0; color: #222;
    font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #page { display: flex; height: 100%; }
  #map { flex: 1 1 auto; min-width: 0; }
  #panel { flex: 0 0 300px; overflow: auto; padding: 16px; border-left: 1px solid #ddd; background: #fafafa; }
  h1 { font-size: 17px; margin: 0 0 6px; }
  #totals { color: #555; margin: 0 0 14px; }
  #legend label { display: flex; align-items: center; gap: 8px; padding: 6px 0; cursor: pointer; }
  #legend .swatch { width: 14px; height: 14px; border-radius: 50%; flex: 0 0 14px; }
  #legend .meta { color: #666; font-size: 12px; margin-left: auto; white-space: nowrap; }
  #notes { color: #a33; font-size: 12px; margin-top: 12px; }
  #notes p { margin: 4px 0; }
  .credit { color: #888; font-size: 11px; margin-top: 18px; }
  @media (max-width: 720px) {
    #page { flex-direction: column; }
    #panel { flex: 0 0 auto; border-left: 0; border-top: 1px solid #ddd; }
    #map { min-height: 60vh; }
  }
</style>
</head>
<body>
<div id="page">
  <div id="map"></div>
  <div id="panel">
    <h1 id="heading"></h1>
    <p id="totals"></p>
    <div id="legend"></div>
    <div id="notes"></div>
    <p class="credit">Made with scikit-route. Roads by Google Maps Directions;
      straight lines where a request fails.</p>
  </div>
</div>
<script type="application/json" id="skroute-plan">$plan_json</script>
<script>
(function () {
  "use strict";
  var plan = JSON.parse(document.getElementById("skroute-plan").textContent);
  var MODES = {driving: "DRIVING", walking: "WALKING", bicycling: "BICYCLING", transit: "TRANSIT"};
  var map = null;
  var layers = [];

  function text(parent, tag, content, className) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    node.textContent = content;  // never innerHTML: the names come from the data
    parent.appendChild(node);
    return node;
  }

  function minutes(value) {
    return value === undefined || value === null ? "" : Math.round(value) + " min";
  }

  function note(message) { text(document.getElementById("notes"), "p", message); }

  // google.maps.Marker is deprecated since February 2024 in favour of AdvancedMarkerElement, but it
  // is still served and not scheduled for removal (the console notes it); advanced markers need a mapId.
  function marker(position, label, color, title, zIndex) {
    return new google.maps.Marker({
      position: position,
      title: title,
      zIndex: zIndex,
      label: {text: label, color: "#ffffff", fontSize: "11px", fontWeight: "bold"},
      icon: {path: google.maps.SymbolPath.CIRCLE, scale: 12, fillColor: color, fillOpacity: 1,
             strokeColor: "#ffffff", strokeWeight: 2}
    });
  }

  function setVisible(layer, visible) {
    layer.visible = visible;
    var target = visible ? map : null;
    layer.renderers.forEach(function (r) { r.setMap(target); });
    layer.polylines.forEach(function (p) { p.setMap(target); });
    layer.markers.forEach(function (m) { m.setMap(target); });
  }

  function fallback(layer, points, status) {
    // A dashed line, the Maps idiom: an invisible base stroke carrying repeated dash symbols.
    var line = new google.maps.Polyline({
      path: points, map: layer.visible ? map : null,
      strokeColor: layer.color, strokeOpacity: 0, strokeWeight: 3,
      icons: [{icon: {path: "M 0,-1 0,1", strokeColor: layer.color, strokeOpacity: 1, scale: 3},
               offset: "0", repeat: "14px"}]
    });
    layer.polylines.push(line);
    note(layer.name + ": one leg drawn as a dashed straight line, the directions request failed ("
         + status + ").");
  }

  // The Directions service throttles bursts from one client: the legs of every day queue up here,
  // MAX_IN_FLIGHT requests at a time, and a leg refused with OVER_QUERY_LIMIT is asked once more
  // after a pause before it falls back to a straight line.
  var MAX_IN_FLIGHT = 2;
  var RETRY_MS = 1000;
  var queue = [];
  var inFlight = 0;
  var directions = null;

  function pump() {
    while (inFlight < MAX_IN_FLIGHT && queue.length > 0) { request(queue.shift()); }
  }

  function request(job) {
    var slice = job.slice;
    inFlight += 1;
    directions.route({
      origin: slice[0],
      destination: slice[slice.length - 1],
      waypoints: slice.slice(1, -1).map(function (p) { return {location: p, stopover: true}; }),
      travelMode: google.maps.TravelMode[MODES[plan.mode] || "DRIVING"]
    }, function (result, status) {
      inFlight -= 1;
      if (status === "OK") {
        job.renderer.setDirections(result);
      } else if (status === "OVER_QUERY_LIMIT" && !job.retried) {
        job.retried = true;
        setTimeout(function () { queue.push(job); pump(); }, RETRY_MS);
      } else {
        fallback(job.layer, slice, status);
      }
      pump();
    });
  }

  function drawLeg(layer, points, leg) {
    var slice = points.slice(leg[0], leg[1] + 1);
    var renderer = new google.maps.DirectionsRenderer({
      map: layer.visible ? map : null, suppressMarkers: true, preserveViewport: true,
      polylineOptions: {strokeColor: layer.color, strokeOpacity: 0.85, strokeWeight: 5}
    });
    layer.renderers.push(renderer);
    queue.push({layer: layer, slice: slice, renderer: renderer, retried: false});
  }

  function legend(layer, trip) {
    var label = text(document.getElementById("legend"), "label", "");
    var box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", function () { setVisible(layer, box.checked); });
    label.appendChild(box);
    text(label, "span", "", "swatch").style.background = layer.color;
    text(label, "span", layer.name);
    var meta = trip.n_stops + (trip.n_stops === 1 ? " stop" : " stops");
    if (trip.driving_minutes !== undefined) { meta += " · " + minutes(trip.driving_minutes); }
    text(label, "span", meta, "meta");
  }

  window.initMap = function () {
    var depot = new google.maps.LatLng(plan.depot.lat, plan.depot.lon);
    var bounds = new google.maps.LatLngBounds(depot, depot);
    map = new google.maps.Map(document.getElementById("map"),
                              {center: depot, zoom: 11, mapTypeControl: false, streetViewControl: false});
    directions = new google.maps.DirectionsService();
    document.getElementById("heading").textContent = plan.title;
    var totals = plan.totals.n_trips + (plan.totals.n_trips === 1 ? " day, " : " days, ")
               + plan.totals.n_stops + " stops";
    if (plan.totals.driving_minutes !== undefined) {
      totals += ", " + minutes(plan.totals.driving_minutes) + " of driving";
    }
    document.getElementById("totals").textContent = totals;
    plan.trips.forEach(function (trip) {
      var layer = {name: trip.name, color: trip.color, visible: true,
                   renderers: [], polylines: [], markers: []};
      layers.push(layer);
      var points = [depot];
      trip.lat.forEach(function (lat, j) {
        var position = new google.maps.LatLng(lat, trip.lon[j]);
        bounds.extend(position);
        points.push(position);
        var title = trip.name + " · " + (j + 1) + ". " + trip.names[j];
        layer.markers.push(marker(position, String(j + 1), trip.color, title, 10 + j));
      });
      points.push(depot);
      layer.markers.forEach(function (m) { m.setMap(map); });
      trip.legs.forEach(function (leg) { drawLeg(layer, points, leg); });
      legend(layer, trip);
    });
    pump();
    new google.maps.Marker({
      position: depot, map: map, title: plan.depot.name, zIndex: 1000,
      icon: {path: "M 0,-24 6,-8 24,-8 10,2 15,20 0,9 -15,20 -10,2 -24,-8 -6,-8 z", scale: 0.7,
             fillColor: plan.depot.color, fillOpacity: 1, strokeColor: "#ffffff", strokeWeight: 1.5}
    });
    map.fitBounds(bounds, 40);
  };
})();
</script>
<script async src="$script_src"></script>
</body>
</html>
"""
)


def _plan_json(
    plan: Plan, text: list[str], days: list[str], palette: list[str], title: str, mode: str
) -> dict[str, Any]:
    """The plan as the JSON object embedded in the page (``<script type="application/json"
    id="skroute-plan">``): trips with ``lat``/``lon``/``names`` lists, the legs of at most 25
    waypoints as ``[start, end]`` positions into ``[depot, *stops, depot]``, colours and totals —
    ``driving_minutes`` only when the solution carries a time matrix (assumed in minutes)."""
    trips = []
    total_driving = 0.0
    for k, trip in enumerate(plan.trips):
        stops = trip[1:-1].tolist()
        entry: dict[str, Any] = {
            "name": days[k],
            "color": palette[k],
            "lat": [round(float(plan.latlon[i, 0]), 6) for i in stops],
            "lon": [round(float(plan.latlon[i, 1]), 6) for i in stops],
            "names": [text[i] for i in stops],
            "labels": [str(plan.labels[i]) for i in stops],
            "n_stops": len(stops),
            "legs": [list(leg) for leg in _legs(trip.tolist(), JS_MAX_WAYPOINTS)],
        }
        driving = plan.driving(k)
        if driving is not None:
            entry["driving_minutes"] = round(driving, 1)
            total_driving += driving
        trips.append(entry)
    totals: dict[str, Any] = {"n_trips": plan.n_trips, "n_stops": sum(t["n_stops"] for t in trips)}
    if plan.time is not None:
        totals["driving_minutes"] = round(total_driving, 1)
    return {
        "title": title,
        "mode": mode,
        "max_waypoints": JS_MAX_WAYPOINTS,
        "depot": {
            "lat": round(float(plan.latlon[plan.depot, 0]), 6),
            "lon": round(float(plan.latlon[plan.depot, 1]), 6),
            "name": text[plan.depot],
            "color": DEPOT_COLOR,
        },
        "trips": trips,
        "totals": totals,
    }


def google_maps_html(
    obj: Any,
    coords: Any = None,
    path: str | os.PathLike[str] | None = None,
    *,
    labels: Any = None,
    api_key: str | None = None,
    names: Any = None,
    trip_names: Any = None,
    title: str | None = None,
) -> Path:
    """Write a standalone page that draws the plan on Google Maps with real road directions.

    The page loads the Maps JavaScript API with ``api_key`` and, for every day, asks a
    ``DirectionsService`` for the roads — one request per leg of at most 25 waypoints, split like
    [`google_maps_urls`][skroute.viz.google_maps_urls] — drawn by a ``DirectionsRenderer`` in
    the day's colour with numbered markers at the stops and a star at the depot. A legend lists
    the days with a checkbox each (hiding a day hides its roads and markers), the stops and the
    driving minutes; a leg whose request fails (quota, an unroutable point) is drawn as a dashed
    straight line and reported under the legend. The plan itself is embedded as one JSON object
    (``<script type="application/json" id="skroute-plan">``), computed here: ``trips`` with
    ``lat``/``lon``/``names`` lists, ``legs``, ``color``, ``n_stops`` and — when the solution
    carries a time matrix, assumed in minutes as ``travel_time_matrix`` (D33) returns it
    — ``driving_minutes``; ``totals`` with ``n_trips``, ``n_stops`` and the same
    minutes. Directions requests are billed to the key's project.

    Parameters
    ----------
    obj : fitted estimator, RouteEvent or route
        As in [`google_maps_urls`][skroute.viz.google_maps_urls].
    coords : (n, 2) array-like, optional
        ``(latitude, longitude)`` in matrix row order; default: the coordinates of the fit.
    path : str or path-like
        Where to write the page (``.html``). Required — positional after ``coords``, so
        ``google_maps_html(est, None, "plan.html")`` and ``google_maps_html(est, path=...)`` both
        work.
    labels : sequence of n hashables, optional
        For a plain route: the label of each row of ``coords``.
    api_key : str, optional
        A Maps JavaScript API key (Directions API enabled); default: the ``GOOGLE_MAPS_API_KEY``
        environment variable. The key is written into the page — share the file accordingly.
    names : sequence of n str or mapping {label: str}, optional
        A display name per node (marker tooltips, the legend); default: the labels. A depot left
        unnamed (``None``, or a mapping without its label) reads ``"Depot (<label>)"``.
    trip_names : sequence of str, optional
        One name per day; default ``"Day 1"``, ``"Day 2"``...
    title : str, optional
        Page title; default ``"<Solver> plan"``.

    Returns
    -------
    path : pathlib.Path
        The file written.

    Raises
    ------
    TypeError
        Without ``path`` (also when a path was given in the ``coords`` slot), or when ``names`` or
        ``trip_names`` is a single str rather than a sequence.
    ValueError
        Without a key: ``"no Google Maps API key: pass api_key= or set GOOGLE_MAPS_API_KEY"``; for
        the inputs [`google_maps_urls`][skroute.viz.google_maps_urls] rejects (no or invalid
        coordinates, a route visiting a node twice or none...); ``names`` or ``trip_names`` of the
        wrong length.
    NotFittedError
        For an estimator that has not been fitted.

    Notes
    -----
    The days take the ten colours of ``PALETTE`` in turn, so from the eleventh day on they repeat.
    The Directions requests go out a couple at a time and a leg refused with ``OVER_QUERY_LIMIT`` is
    asked once more before it falls back to the dashed line. The markers are ``google.maps.Marker``,
    which Google deprecated in February 2024 in favour of ``AdvancedMarkerElement`` but keeps
    serving (the browser console notes it); a later release may switch.

    Examples
    --------
    >>> import json, re, tempfile
    >>> from pathlib import Path
    >>> from skroute import TwoOpt
    >>> from skroute.datasets import load_barcelona
    >>> from skroute.viz import google_maps_html
    >>> bcn = load_barcelona()
    >>> two = TwoOpt().fit(
    ...     bcn.cost, labels=bcn.labels, coords=bcn.coords, time_matrix=bcn.time * 60, max_time_work=360
    ... )  # the bundled times are hours: minutes for the page (TwoOpt is deterministic: exact figures)
    >>> out = Path(tempfile.mkdtemp())
    >>> page = google_maps_html(two, path=out / "plan.html", api_key="AIza-demo", title="Barcelona, two days")
    >>> text = page.read_text(encoding="utf-8")
    >>> text.count("AIza-demo"), 'src="https://maps.googleapis.com/maps/api/js?key=AIza-demo' in text
    (1, True)
    >>> plan = json.loads(re.search(r'id="skroute-plan">(.*?)</script>', text, re.S).group(1))
    >>> plan["totals"], [t["n_stops"] for t in plan["trips"]], plan["trips"][0]["legs"]
    ({'n_trips': 2, 'n_stops': 18, 'driving_minutes': 676.7}, [7, 11], [[0, 8]])
    """
    out = _check_path(path, coords, "google_maps_html")
    key = os.environ.get(ENV_KEY) if api_key is None else api_key
    if not key:
        raise ValueError(f"no Google Maps API key: pass api_key= or set {ENV_KEY}")
    plan = _routes(obj, coords, labels)
    text = node_names(plan.labels, names)
    days = trip_labels(trip_names, plan.n_trips)
    palette = trip_colors(None, plan.n_trips)
    depot_label = plan.labels.tolist()[plan.depot]
    if names is None or (isinstance(names, Mapping) and depot_label not in names):
        text[plan.depot] = f"Depot ({text[plan.depot]})"  # an unnamed depot says what it is
    heading = title if title is not None else (f"{plan.solver} plan" if plan.solver else "Route plan")
    data = _plan_json(plan, text, days, palette, heading, "driving")
    # JSON inside a <script> block: no "<", ">" or "&" may survive (a name could hold "</script>")
    plan_json = (
        json.dumps(data, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    src = JS_API_URL + "?" + urlencode([("key", key), ("callback", "initMap"), ("loading", "async")])
    page = _PAGE.substitute(title=html.escape(heading), plan_json=plan_json, script_src=html.escape(src))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
