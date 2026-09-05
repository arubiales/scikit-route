"""``skroute.viz.google_maps`` (D34): the plan as Directions URLs, a KML and a Maps JavaScript page.

Everything is checked by parsing what the functions produce -- the URLs with ``urllib.parse``, the
KML with ``xml.etree``, the page's JSON block -- without a browser, a network connection or an API
key of any worth. The module itself must import without matplotlib or plotly.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

import skroute
from skroute import BruteForce, IteratedLocalSearch, RoutingProblem
from skroute.base import RouteEvent
from skroute.datasets import load_barcelona
from skroute.viz import google_maps_html, google_maps_urls, to_kml
from skroute.viz.google_maps import ENV_KEY, JS_MAX_WAYPOINTS, PALETTE, Plan, _legs, _routes

NS = {"k": "http://www.opengis.net/kml/2.2"}
DIR = "https://www.google.com/maps/dir/"
SQUARE = np.array([[40.0, -3.0], [40.01, -3.0], [40.01, -2.99], [40.0, -2.99]])
C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
H4 = np.array([[0, 1, 2, 2], [1, 0, 1, 2], [2, 1, 0, 1], [2, 2, 1, 0]], dtype=float)


# --------------------------------------------------------------------------- helpers and fixtures
def latlon(xy, i):
    return f"{xy[i, 0]:.6f},{xy[i, 1]:.6f}"


def parse(url):
    """``(origin, [waypoints], destination, travelmode)`` of one Directions URL."""
    parts = urlparse(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == DIR
    q = parse_qs(parts.query, strict_parsing=True)
    assert q["api"] == ["1"] and len(q["origin"]) == 1 and len(q["destination"]) == 1
    waypoints = q["waypoints"][0].split("|") if "waypoints" in q else []
    return q["origin"][0], waypoints, q["destination"][0], q["travelmode"][0]


def stops_of(links):
    """The stop sequence a day's links describe, checking that consecutive legs share their boundary."""
    seq = []
    for k, link in enumerate(links):
        origin, waypoints, destination, _ = parse(link)
        if k:
            assert origin == parse(links[k - 1])[2]
        seq.extend(waypoints)
        if k < len(links) - 1:
            seq.append(destination)
    return seq


def plan_of(text):
    """The JSON object embedded in a page (exactly one ``skroute-plan`` block)."""
    blocks = re.findall(r'<script type="application/json" id="skroute-plan">(.*?)</script>', text, re.S)
    assert len(blocks) == 1
    return json.loads(blocks[0])


@pytest.fixture(scope="module")
def bcn():
    return load_barcelona()


@pytest.fixture(scope="module")
def one_day(bcn):
    return IteratedLocalSearch(n_iter=3, patience=None, random_state=0).fit(
        bcn.cost, labels=bcn.labels, coords=bcn.coords
    )


@pytest.fixture(scope="module")
def two_days(bcn):
    est = IteratedLocalSearch(n_iter=3, patience=None, random_state=0).fit(
        bcn.cost, labels=bcn.labels, coords=bcn.coords, time_matrix=bcn.time * 60, max_time_work=360.0
    )
    assert est.n_trips_ == 2
    return est


@pytest.fixture(scope="module")
def two_trips():
    """The 4-node example of the problem model: a budget of 4 h splits the tour into two trips."""
    X = {a: {b: C4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    T = {a: {b: H4[i, j] for j, b in enumerate("dabc")} for i, a in enumerate("dabc")}
    est = BruteForce().fit(X, time_matrix=T, depot="d", max_time_work=4.0, extra_cost=3.0, coords=SQUARE)
    assert est.n_trips_ == 2 and est.trips_[0].tolist() == ["d", "a", "b", "d"]
    return est


@pytest.fixture(scope="module")
def thirty():
    """A plain route of 30 stops (positions) around a depot at row 0."""
    rng = np.random.default_rng(30)
    xy = np.column_stack((40.3 + rng.random(31) * 0.4, -3.9 + rng.random(31) * 0.5))
    route = [0, *range(1, 31)]
    return route, xy


# --------------------------------------------------------------------------- import surface
def test_module_imports_without_matplotlib_or_plotly(two_trips):
    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None\n"
        "sys.modules['plotly'] = None\n"
        "import skroute.viz.google_maps as gm\n"
        "urls = gm.google_maps_urls([0, 1, 0, 2, 3], [[40, -3], [40.01, -3], [40.01, -2.99], [40, -2.99]])\n"
        "assert len(urls) == 2, urls\n"
        "assert 'matplotlib.pyplot' not in sys.modules and 'plotly.graph_objects' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(skroute.__file__).resolve().parents[1])}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert skroute.viz.google_maps.__all__ == ["google_maps_html", "google_maps_urls", "to_kml"]
    assert len(PALETTE) == len(set(PALETTE)) == 10 and all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in PALETTE)


# --------------------------------------------------------------------------- the shared plan
def test_routes_from_estimator_event_and_plain_route(two_trips, bcn, one_day):
    plan = _routes(two_trips)
    assert isinstance(plan, Plan) and plan.n_trips == 2 and plan.depot == 0
    assert [t.tolist() for t in plan.trips] == [[0, 1, 2, 0], [0, 3, 0]]
    assert plan.labels.tolist() == ["d", "a", "b", "c"] and plan.solver == "BruteForce"
    assert plan.time is not None and plan.driving(0) == 4.0 and plan.driving(1) == 4.0
    assert plan.stops(1).tolist() == [3]
    # the same solution as a RouteEvent
    problem = two_trips.problem_
    event = RouteEvent("BruteForce", "end", 0, 41.0, 41.0, two_trips.tour_, two_trips.tour_, problem, {})
    assert [t.tolist() for t in _routes(event).trips] == [[0, 1, 2, 0], [0, 3, 0]]
    # ... and as its label-space route with labels=, or as positions
    labels = problem.labels
    by_label = _routes(two_trips.route_, SQUARE, labels)
    assert [t.tolist() for t in by_label.trips] == [[0, 1, 2, 0], [0, 3, 0]]
    assert (
        by_label.labels.tolist() == ["d", "a", "b", "c"] and by_label.time is None and by_label.solver is None
    )
    by_position = _routes([0, 1, 2, 0, 3], SQUARE)
    assert [t.tolist() for t in by_position.trips] == [[0, 1, 2, 0], [0, 3, 0]]
    assert by_position.labels.tolist() == [0, 1, 2, 3]
    # an open tour is one closed trip; the depot is its first entry
    assert [t.tolist() for t in _routes([2, 0, 1], SQUARE[:3]).trips] == [[2, 0, 1, 2]]
    # a fitted estimator without coordinates takes coords=
    bare = IteratedLocalSearch(n_iter=1, patience=None, random_state=0).fit(bcn.cost, labels=bcn.labels)
    assert _routes(bare, bcn.coords).latlon.shape == (19, 2)
    assert _routes(one_day).driving(0) is None  # no time matrix in the fit


def test_routes_errors(two_trips, bcn):
    bare = IteratedLocalSearch(n_iter=1, patience=None, random_state=0).fit(bcn.cost, labels=bcn.labels)
    with pytest.raises(ValueError, match="no coordinates: pass coords="):
        google_maps_urls(bare)
    with pytest.raises(ValueError, match="no coordinates: pass coords="):
        google_maps_urls([0, 1, 2])
    with pytest.raises(ValueError, match=r"latitude within \[-90, 90\]"):
        google_maps_urls(two_trips, SQUARE * 10)
    with pytest.raises(ValueError, match=r"longitude within \[-180, 180\]"):
        google_maps_urls([0, 1], [[40.0, 200.0], [41.0, 2.0]])
    with pytest.raises(ValueError, match="coords has 3 rows but the problem has 4 nodes"):
        google_maps_urls(two_trips, SQUARE[:3])
    with pytest.raises(ValueError, match="labels= applies to a plain route"):
        google_maps_urls(two_trips, labels=list("dabc"))
    with pytest.raises(ValueError, match="route names 'x', which is not one of labels="):
        google_maps_urls(["d", "x"], SQUARE, labels=list("dabc"))
    with pytest.raises(ValueError, match="beyond the end of coords"):
        google_maps_urls([0, 7], SQUARE)
    with pytest.raises(ValueError, match="visits no node besides the depot"):
        google_maps_urls([0, 0], SQUARE)
    with pytest.raises(skroute.exceptions.NotFittedError):
        google_maps_urls(IteratedLocalSearch(), SQUARE)
    event = RouteEvent("SOM", "start", 0, np.nan, np.nan, None, None, two_trips.problem_, {})
    with pytest.raises(ValueError, match="no tour yet"):
        google_maps_urls(event)


def test_legs_split_at_the_boundary_stop():
    trip = list(range(12))  # depot 0, ten stops 1..10, depot 11 -- as positions into the trip
    assert _legs(trip, 9) == [(0, 10), (10, 11)]
    assert _legs(trip, 10) == [(0, 11)]
    assert _legs(trip, 25) == [(0, 11)]
    assert _legs(trip, 3) == [(0, 4), (4, 8), (8, 11)]
    assert _legs([0, 1, 0], 1) == [(0, 2)]
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="max_waypoints must be a positive integer"):
            _legs(trip, bad)


# --------------------------------------------------------------------------- URLs
def test_urls_one_link_per_short_day(two_trips):
    urls = google_maps_urls(two_trips)
    assert [len(day) for day in urls] == [1, 1]
    origin, waypoints, destination, mode = parse(urls[0][0])
    assert origin == destination == latlon(SQUARE, 0) == "40.000000,-3.000000"
    assert waypoints == [latlon(SQUARE, 1), latlon(SQUARE, 2)] and mode == "driving"
    assert parse(urls[1][0])[1] == [latlon(SQUARE, 3)]
    # the raw text: urlencode'd, six decimals, the documented parameters in order
    assert urls[0][0] == (
        DIR + "?api=1&origin=40.000000%2C-3.000000&destination=40.000000%2C-3.000000"
        "&waypoints=40.010000%2C-3.000000%7C40.010000%2C-2.990000&travelmode=driving"
    )
    # the same from the label-space route, from positions, and from an event
    labels = two_trips.problem_.labels
    assert google_maps_urls(two_trips.route_, SQUARE, labels=labels) == urls
    assert google_maps_urls([0, 1, 2, 0, 3, 0], SQUARE) == urls
    event = RouteEvent(
        "BruteForce", "end", 0, 41.0, 41.0, two_trips.tour_, two_trips.tour_, two_trips.problem_
    )
    assert google_maps_urls(event) == urls
    # explicit coordinates override the fit's
    moved = google_maps_urls(two_trips, SQUARE + 1.0)
    assert parse(moved[0][0])[0] == "41.000000,-2.000000"


def test_urls_split_long_days_into_shared_legs(thirty, one_day, bcn):
    route, xy = thirty
    (links,) = google_maps_urls(route, xy)
    assert len(links) == 4  # 30 stops: 9 + 1, 9 + 1, 9 + 1, 1 -> four legs
    assert parse(links[0])[0] == latlon(xy, 0) and parse(links[-1])[2] == latlon(xy, 0)
    assert stops_of(links) == [latlon(xy, i) for i in range(1, 31)]
    assert all(len(parse(link)[1]) <= 9 for link in links)
    (links25,) = google_maps_urls(route, xy, max_waypoints=25)
    assert len(links25) == 2 and stops_of(links25) == stops_of(links)
    assert len(parse(links25[0])[1]) == 25 and len(parse(links25[1])[1]) == 4
    (links3,) = google_maps_urls(route, xy, max_waypoints=3)
    assert len(links3) == 8 and stops_of(links3) == stops_of(links)
    # Barcelona's 18 stops: two links, the boundary shared, every stop once, all rows of the coords
    (day,) = google_maps_urls(one_day)
    assert len(day) == 2
    seq = stops_of(day)
    assert len(seq) == 18 and len(set(seq)) == 18
    assert set(seq) == {latlon(bcn.coords, i) for i in range(19)} - {latlon(bcn.coords, 0)}


def test_urls_modes_and_validation(two_trips):
    for mode in ("driving", "walking", "bicycling", "transit"):
        assert all(parse(link)[3] == mode for day in google_maps_urls(two_trips, mode=mode) for link in day)
    with pytest.raises(ValueError, match="mode must be one of"):
        google_maps_urls(two_trips, mode="flying")
    with pytest.raises(ValueError, match="max_waypoints must be a positive integer"):
        google_maps_urls(two_trips, max_waypoints=0)


# --------------------------------------------------------------------------- KML
def test_kml_document_structure(tmp_path, two_trips):
    out = to_kml(two_trips, path=tmp_path / "plan.kml")
    assert out == tmp_path / "plan.kml" and out.is_file()
    root = ET.parse(out).getroot()
    assert root.tag == "{http://www.opengis.net/kml/2.2}kml"
    doc = root.find("k:Document", NS)
    assert doc.findtext("k:name", namespaces=NS) == "BruteForce plan: 2 days"
    # one Style per day (plus the depot's): 4-pixel lines in distinct aabbggrr colours
    styles = {s.get("id"): s for s in doc.findall("k:Style", NS)}
    assert set(styles) == {"depot", "day1", "day2"}
    colours = [styles[f"day{k}"].findtext("k:LineStyle/k:color", namespaces=NS) for k in (1, 2)]
    assert colours == ["ffb4771f", "ff0e7fff"]  # PALETTE[0] = #1f77b4, PALETTE[1] = #ff7f0e
    assert styles["day1"].findtext("k:LineStyle/k:width", namespaces=NS) == "4"
    assert styles["day1"].findtext("k:IconStyle/k:color", namespaces=NS) == "ffb4771f"
    # the depot is one Placemark at Document level, lon,lat,0
    depots = doc.findall("k:Placemark", NS)
    assert len(depots) == 1 and depots[0].findtext("k:name", namespaces=NS) == "Depot"
    assert depots[0].findtext("k:styleUrl", namespaces=NS) == "#depot"
    assert depots[0].findtext("k:Point/k:coordinates", namespaces=NS) == "-3.000000,40.000000,0"
    # one Folder per day: numbered stops and the closed LineString
    folders = doc.findall("k:Folder", NS)
    assert [f.findtext("k:name", namespaces=NS) for f in folders] == ["Day 1", "Day 2"]
    day2 = folders[1].findall("k:Placemark", NS)
    assert [p.findtext("k:name", namespaces=NS) for p in day2] == ["2.1 c", "Day 2 route"]
    assert day2[0].findtext("k:description", namespaces=NS) == "Day 2, stop 1 of 1"
    assert day2[1].findtext("k:description", namespaces=NS) == "Day 2: 1 stops, depot to depot"
    assert [p.findtext("k:styleUrl", namespaces=NS) for p in day2] == ["#day2"] * 2
    assert day2[0].findtext("k:Point/k:coordinates", namespaces=NS) == "-2.990000,40.000000,0"
    line = day2[1].find("k:LineString", NS)
    assert line.findtext("k:tessellate", namespaces=NS) == "1"
    coords = line.findtext("k:coordinates", namespaces=NS).split()
    assert coords == ["-3.000000,40.000000,0", "-2.990000,40.000000,0", "-3.000000,40.000000,0"]
    day1 = folders[0].findall("k:Placemark", NS)
    assert [p.findtext("k:description", namespaces=NS) for p in day1[:2]] == [
        "Day 1, stop 1 of 2",
        "Day 1, stop 2 of 2",
    ]
    assert day1[2].findtext("k:LineString/k:coordinates", namespaces=NS).split() == [
        "-3.000000,40.000000,0",
        "-3.000000,40.010000,0",
        "-2.990000,40.010000,0",
        "-3.000000,40.000000,0",
    ]
    assert folders[0].findall("k:Placemark", NS)[0].findtext("k:name", namespaces=NS) == "1.1 a"
    assert out.read_text(encoding="utf-8").startswith("<?xml version='1.0' encoding='utf-8'?>")


def test_kml_names_trip_names_colors_and_escaping(tmp_path, two_trips, two_days, bcn):
    names = {"d": "Office <HQ>", "a": "Ana & co", "b": "Bea"}  # "c" keeps its label
    out = to_kml(
        two_trips,
        None,
        tmp_path / "named.kml",
        names=names,
        depot_name="Leganés office",
        trip_names=["Monday", "Tuesday"],
        colors=["#112233", "abcdef"],
    )
    doc = ET.parse(out).getroot().find("k:Document", NS)
    assert doc.find("k:Placemark/k:name", NS).text == "Leganés office"
    assert "Office <HQ>" in doc.find("k:Placemark/k:description", NS).text
    folders = doc.findall("k:Folder", NS)
    assert [f.findtext("k:name", namespaces=NS) for f in folders] == ["Monday", "Tuesday"]
    assert [p.findtext("k:name", namespaces=NS) for p in folders[0].findall("k:Placemark", NS)] == [
        "1.1 Ana & co",
        "1.2 Bea",
        "Monday route",
    ]
    assert [p.findtext("k:name", namespaces=NS) for p in folders[1].findall("k:Placemark", NS)] == [
        "2.1 c",
        "Tuesday route",
    ]
    styles = {s.get("id"): s for s in doc.findall("k:Style", NS)}
    assert styles["day1"].findtext("k:LineStyle/k:color", namespaces=NS) == "ff332211"
    assert styles["day2"].findtext("k:LineStyle/k:color", namespaces=NS) == "ffefcdab"
    text = out.read_text(encoding="utf-8")
    assert "Office <HQ>" not in text and "Ana &amp; co" in text  # markup escaped by ElementTree
    # a sequence of names in row order; a single colour is cycled; a nested directory is created
    seq = to_kml(
        two_trips, path=tmp_path / "deep" / "seq.kml", names=["O", "A", "B", "C"], colors=["#ff0000"]
    )
    doc = ET.parse(seq).getroot().find("k:Document", NS)
    assert doc.find("k:Folder/k:Placemark/k:name", NS).text == "1.1 A"
    assert {s.findtext("k:LineStyle/k:color", namespaces=NS) for s in doc.findall("k:Style", NS)} >= {
        "ff0000ff"
    }
    # the label-space route with labels= gives the same folders as the estimator
    labels = two_trips.problem_.labels
    again = to_kml(two_trips.route_, SQUARE, tmp_path / "route.kml", labels=labels)
    assert ET.parse(again).getroot().find("k:Document/k:name", NS).text == "Route plan: 2 days"
    assert len(ET.parse(again).getroot().findall(".//k:Folder", NS)) == 2
    # Barcelona, two days: 18 stops over two folders, every stop named by its label
    real = to_kml(two_days, path=tmp_path / "bcn.kml")
    folders = ET.parse(real).getroot().findall(".//k:Folder", NS)
    counts = [len(f.findall("k:Placemark", NS)) - 1 for f in folders]
    assert counts == [len(t) - 2 for t in two_days.trips_] and sum(counts) == 18
    with pytest.raises(ValueError, match="trip_names has 1 entries but the plan has 2 trips"):
        to_kml(two_trips, path=tmp_path / "bad.kml", trip_names=["Monday"])
    with pytest.raises(ValueError, match="names has 3 entries but there are 4 nodes"):
        to_kml(two_trips, path=tmp_path / "bad.kml", names=["a", "b", "c"])
    with pytest.raises(ValueError, match="colors must be hex colours"):
        to_kml(two_trips, path=tmp_path / "bad.kml", colors=["red"])
    with pytest.raises(ValueError, match="colors must hold at least one colour"):
        to_kml(two_trips, path=tmp_path / "bad.kml", colors=[])
    with pytest.raises(TypeError, match="path is required"):
        to_kml(two_trips)


# --------------------------------------------------------------------------- HTML
def test_html_page_embeds_the_plan_and_the_key_once(tmp_path, two_days, bcn):
    out = google_maps_html(two_days, path=tmp_path / "plan.html", api_key="AIzaSyDEMO-key_0123")
    assert out == tmp_path / "plan.html"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>") and text.count("AIzaSyDEMO-key_0123") == 1
    assert (
        '<script async src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDEMO-key_0123'
        '&amp;callback=initMap&amp;loading=async"></script>' in text
    )
    assert "window.initMap = function" in text
    for needle in (
        "DirectionsService",
        "DirectionsRenderer",
        "suppressMarkers: true",
        'box.type = "checkbox"',
        "google.maps.Polyline",
        'status === "OK"',
        "plan.depot.name",
    ):
        assert needle in text
    plan = plan_of(text)
    assert "AIzaSyDEMO" not in json.dumps(plan)
    assert plan["title"] == "IteratedLocalSearch plan" and plan["mode"] == "driving"
    assert plan["max_waypoints"] == JS_MAX_WAYPOINTS == 25
    assert "<title>IteratedLocalSearch plan</title>" in text
    # totals computed in Python; driving minutes because the fit had a time matrix (in minutes here)
    assert plan["totals"]["n_trips"] == 2 and plan["totals"]["n_stops"] == 18
    assert plan["totals"]["driving_minutes"] == pytest.approx(float(two_days.trip_times_.sum()), abs=0.1)
    assert [t["driving_minutes"] for t in plan["trips"]] == pytest.approx(
        two_days.trip_times_.tolist(), abs=0.1
    )
    # the depot and every stop, in driving order, with six-decimal coordinates
    depot_row = two_days.problem_.depot
    assert plan["depot"] == {
        "lat": round(float(bcn.coords[depot_row, 0]), 6),
        "lon": round(float(bcn.coords[depot_row, 1]), 6),
        "name": f"Depot ({bcn.depot})",
        "color": "#c0392b",
    }
    assert [t["name"] for t in plan["trips"]] == ["Day 1", "Day 2"]
    assert [t["color"] for t in plan["trips"]] == list(PALETTE[:2])
    for entry, trip in zip(plan["trips"], two_days.trips_, strict=True):
        rows = [two_days.problem_.index_of(x) for x in trip[1:-1].tolist()]
        assert entry["n_stops"] == len(rows) == len(entry["lat"]) == len(entry["lon"]) == len(entry["names"])
        assert entry["lat"] == [round(float(bcn.coords[i, 0]), 6) for i in rows]
        assert entry["lon"] == [round(float(bcn.coords[i, 1]), 6) for i in rows]
        assert entry["names"] == entry["labels"] == [str(x) for x in trip[1:-1].tolist()]
        assert entry["legs"] == [[0, len(rows) + 1]]  # at most 25 stops: one request per day


def test_html_legs_split_at_25_waypoints_and_no_minutes_without_a_time_matrix(tmp_path, thirty, one_day):
    route, xy = thirty
    plan = plan_of(google_maps_html(route, xy, tmp_path / "thirty.html", api_key="k").read_text())
    (trip,) = plan["trips"]
    assert trip["n_stops"] == 30 and trip["legs"] == [[0, 26], [26, 31]]  # 30 stops -> two requests
    assert plan["title"] == "Route plan" and plan["totals"] == {"n_trips": 1, "n_stops": 30}
    assert "driving_minutes" not in trip and plan["depot"]["name"] == "Depot (0)"
    assert trip["names"] == [str(i) for i in range(1, 31)]
    page = plan_of(google_maps_html(one_day, path=tmp_path / "bcn.html", api_key="k").read_text())
    assert "driving_minutes" not in page["totals"] and all("driving_minutes" not in t for t in page["trips"])
    assert page["trips"][0]["legs"] == [[0, 19]]


def test_html_escapes_user_text(tmp_path, two_trips):
    names = ["<script>alert(1)</script>", 'Ana & "Bea"', "</script><b>x</b>", "C"]
    out = google_maps_html(
        two_trips,
        path=tmp_path / "escaped.html",
        api_key="k",
        names=names,
        trip_names=["Mon <b>day</b>", "Tue"],
        title='Plan "<b>bold</b>" & co',
    )
    text = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text and "<b>x</b>" not in text and "<b>bold</b>" not in text
    assert "<title>Plan &quot;&lt;b&gt;bold&lt;/b&gt;&quot; &amp; co</title>" in text
    plan = plan_of(text)  # the JSON survives: "<", ">" and "&" are written as \\u escapes in the block
    assert plan["title"] == 'Plan "<b>bold</b>" & co'
    assert plan["depot"]["name"] == "<script>alert(1)</script>"
    assert [t["name"] for t in plan["trips"]] == ["Mon <b>day</b>", "Tue"]
    assert plan["trips"][0]["names"] == ['Ana & "Bea"', "</script><b>x</b>"]
    assert plan["trips"][1]["names"] == ["C"]
    assert text.count("<script") == 3 and text.count("</script>") == 3  # the JSON block, the code, the API
    block = re.search(r'id="skroute-plan">(.*?)</script>', text, re.S).group(1)
    assert "<" not in block and ">" not in block and "&" not in block


def test_html_key_from_the_environment(tmp_path, two_trips, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    with pytest.raises(ValueError, match="no Google Maps API key: pass api_key= or set GOOGLE_MAPS_API_KEY"):
        google_maps_html(two_trips, path=tmp_path / "nokey.html")
    with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
        google_maps_html(two_trips, path=tmp_path / "nokey.html", api_key="")
    monkeypatch.setenv(ENV_KEY, "env-key-42")
    text = google_maps_html(two_trips, path=tmp_path / "env.html").read_text()
    assert text.count("env-key-42") == 1 and "key=env-key-42&amp;callback=initMap" in text
    text = google_maps_html(two_trips, path=tmp_path / "given.html", api_key="given").read_text()
    assert "env-key-42" not in text and text.count("key=given") == 1
    with pytest.raises(TypeError, match="path is required"):
        google_maps_html(two_trips, api_key="k")
    # positional coords and path, as in the specification; a nested directory is created
    labels = two_trips.problem_.labels
    out = google_maps_html(
        two_trips.route_, SQUARE, tmp_path / "deep" / "route.html", labels=labels, api_key="k"
    )
    plan = plan_of(out.read_text())
    assert [t["n_stops"] for t in plan["trips"]] == [2, 1] and plan["depot"]["name"] == "Depot (d)"
    assert plan["trips"][0]["names"] == ["a", "b"] and plan["trips"][1]["names"] == ["c"]


def test_html_names_by_mapping_and_custom_titles(tmp_path, two_trips):
    out = google_maps_html(
        two_trips,
        path=tmp_path / "named.html",
        api_key="k",
        names={"d": "Office", "b": "Bea"},
        trip_names=["Monday", "Tuesday"],
        title="Two days",
    )
    plan = plan_of(out.read_text())
    assert plan["title"] == "Two days" and plan["depot"]["name"] == "Office"
    assert [t["name"] for t in plan["trips"]] == ["Monday", "Tuesday"]
    assert plan["trips"][0]["names"] == ["a", "Bea"] and plan["trips"][1]["names"] == ["c"]
    with pytest.raises(ValueError, match="trip_names has 3 entries but the plan has 2 trips"):
        google_maps_html(two_trips, path=tmp_path / "bad.html", api_key="k", trip_names=["a", "b", "c"])


def test_html_from_an_event_and_a_problem_without_coords(tmp_path, two_trips):
    problem = two_trips.problem_
    event = RouteEvent("BruteForce", "end", 0, 41.0, 41.0, two_trips.tour_, two_trips.tour_, problem)
    a = plan_of(google_maps_html(event, path=tmp_path / "event.html", api_key="k").read_text())
    b = plan_of(google_maps_html(two_trips, path=tmp_path / "est.html", api_key="k").read_text())
    assert a == b
    bare = RoutingProblem(C4, labels=list("dabc"), depot="d", time_matrix=H4, max_time_work=4.0)
    event = RouteEvent("BruteForce", "end", 0, 41.0, 41.0, two_trips.tour_, two_trips.tour_, bare)
    with pytest.raises(ValueError, match="no coordinates: pass coords="):
        google_maps_html(event, path=tmp_path / "bare.html", api_key="k")
    c = plan_of(google_maps_html(event, SQUARE, tmp_path / "bare.html", api_key="k").read_text())
    assert c["trips"] == b["trips"]
