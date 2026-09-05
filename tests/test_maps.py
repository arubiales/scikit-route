"""Tests of ``skroute.preprocessing.maps`` (D33): OSRM tables, Nominatim/Google geocoding, Overpass POIs.

Every test is offline: ``maps._urlopen`` is replaced by a fake that serves recorded JSON answers from
``tests/data/maps/`` (or synthetic ones computed from the request URL) and records the calls;
``maps._sleep`` / ``maps._monotonic`` are a fake clock, so back-off, pauses and the Nominatim throttle
are asserted without waiting. The ``test_live_*`` tests at the end carry the ``network`` marker
(deselected by default; the nightly runs them) and hit the real public servers on three points.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
import urllib.error
import urllib.parse
import warnings
from pathlib import Path

import numpy as np
import pytest

import skroute
from skroute.preprocessing import fetch_pois, geocode, maps, travel_time_matrix
from skroute.preprocessing.maps import MapServiceError
from skroute.utils import Bunch

DATA = Path(__file__).parent / "data" / "maps"

OFFICE = (40.3272, -3.7635)  # Leganés
SOL = (40.4168, -3.7038)  # Puerta del Sol, Madrid
ALCALA = (40.4819, -3.3635)  # Alcalá de Henares


def _load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class Script(list):
    """A scripted sequence of answers, consumed one per call (the retry scenarios)."""


class FakeOpener:
    """``urllib.request.urlopen`` stand-in: answers by URL regex, records every call.

    A route's answer is a dict/list (served as JSON), raw ``bytes``, an exception instance (raised),
    a `Script` of any of those, or a callable ``f(url)`` returning one of them.
    """

    def __init__(self) -> None:
        self.routes: list[tuple[str, object]] = []
        self.calls: list[dict] = []

    def add(self, pattern: str, answer) -> None:
        self.routes.append((pattern, answer))

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.calls.append(
            {
                "url": url,
                "query": dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)),
                "user_agent": request.get_header("User-agent"),
                "timeout": timeout,
                "data": request.data,
            }
        )
        for pattern, answer in self.routes:
            if re.search(pattern, url):
                result = answer(url) if callable(answer) else answer
                if isinstance(result, Script):
                    result = result.pop(0)
                if isinstance(result, BaseException):
                    raise result
                body = result if isinstance(result, bytes) else json.dumps(result).encode("utf-8")
                return io.BytesIO(body)
        raise AssertionError(f"unexpected request {url}")


def _http_error(url: str, code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, f"status {code}", None, io.BytesIO(body.encode("utf-8")))


@pytest.fixture
def opener(monkeypatch):
    """The fake opener and a fake clock: ``opener.sleeps`` records every sleep and advances ``opener.now``."""
    fake = FakeOpener()
    fake.now = 1000.0
    fake.sleeps = []

    def sleep(seconds):
        fake.sleeps.append(float(seconds))
        fake.now += float(seconds)

    monkeypatch.setattr(maps, "_urlopen", fake)
    monkeypatch.setattr(maps, "_sleep", sleep)
    monkeypatch.setattr(maps, "_monotonic", lambda: fake.now)
    monkeypatch.setattr(maps, "_last_nominatim_call", None)
    return fake


# --------------------------------------------------------------------------- OSRM tables


def _points(n: int) -> list[tuple[float, float]]:
    return [(40.0 + 0.1 * k, -3.0 - 0.05 * k) for k in range(n)]


def _index_of(points, lat: float, lon: float) -> int:
    for k, (plat, plon) in enumerate(points):
        if abs(plat - lat) < 1e-6 and abs(plon - lon) < 1e-6:
            return k
    raise AssertionError(f"unknown coordinate {lat},{lon}")


def _synthetic_osrm(points, *, unroutable=(), drop_distances=False):
    """OSRM table answers from the URL itself: duration(i, j) = 100 i + 10 j, distance = 1000 i + j."""

    def respond(url: str):
        path, _, query = url.partition("?")
        coords = re.split(r"/table/v1/[^/]+/", path)[1].split(";")
        global_ids = []
        for pair in coords:
            lon, lat = (float(v) for v in pair.split(","))  # OSRM order: lon,lat
            global_ids.append(_index_of(points, lat, lon))
        params = dict(urllib.parse.parse_qsl(query))
        sources = [global_ids[int(s)] for s in params["sources"].split(";")]
        destinations = [global_ids[int(d)] for d in params["destinations"].split(";")]
        durations = [
            [None if (i, j) in unroutable else (0 if i == j else 100.0 * i + 10.0 * j) for j in destinations]
            for i in sources
        ]
        answer = {"code": "Ok", "durations": durations}
        if not drop_distances:
            answer["distances"] = [
                [None if (i, j) in unroutable else (0 if i == j else 1000.0 * i + j) for j in destinations]
                for i in sources
            ]
        return answer

    return respond


def _expected_seconds(n: int):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    seconds = 100.0 * i + 10.0 * j
    np.fill_diagonal(seconds, 0.0)
    metres = 1000.0 * i + 1.0 * j
    np.fill_diagonal(metres, 0.0)
    return seconds, metres


def test_osrm_tiles_the_table_in_blocks_with_lon_lat_urls(opener):
    pts = _points(7)
    opener.add(r"/table/v1/driving/", _synthetic_osrm(pts))
    res = travel_time_matrix(pts, chunk_size=3, units="s")
    assert isinstance(res, Bunch) and set(res.keys()) == {"time", "distance", "units", "provider", "coords"}
    assert res.provider == "osrm" and res.units == {"time": "s", "distance": "m"}
    seconds, metres = _expected_seconds(7)
    np.testing.assert_array_equal(res.time, seconds)
    np.testing.assert_array_equal(res.distance, metres)
    assert res.time.dtype == np.float64 and res.distance.shape == (7, 7)
    np.testing.assert_array_equal(res.coords, np.asarray(pts))
    # ceil(7 / 3) = 3 blocks of nearly equal size (3, 2, 2) -> 3 x 3 = 9 requests, one pause between each
    assert len(opener.calls) == 9
    assert opener.sleeps == [1.0] * 8
    first = opener.calls[0]
    assert first["url"].startswith("https://router.project-osrm.org/table/v1/driving/")
    path = first["url"].split("/table/v1/driving/")[1].split("?")[0]
    assert path == "-3.000000,40.000000;-3.050000,40.100000;-3.100000,40.200000", "lon,lat with 6 decimals"
    assert first["url"].endswith("?sources=0;1;2&destinations=0;1;2&annotations=duration,distance")
    assert first["user_agent"] == maps.DEFAULT_USER_AGENT and first["timeout"] == 60.0
    assert first["data"] is None, "GET, never POST"
    second = opener.calls[1]  # rows block 0 (3 points) x cols block 1 (2 points): 5 coordinates
    assert second["url"].count(";") == 4 + 2 + 1
    assert second["query"] == {"sources": "0;1;2", "destinations": "3;4", "annotations": "duration,distance"}
    last = opener.calls[-1]
    assert last["query"] == {"sources": "0;1", "destinations": "0;1", "annotations": "duration,distance"}


@pytest.mark.parametrize(("n", "chunk", "n_requests"), [(6, 3, 4), (7, 4, 4), (2, 50, 1), (3, 1, 9)])
def test_osrm_request_count_follows_ceil_n_over_chunk_squared(opener, n, chunk, n_requests):
    pts = _points(n)
    opener.add(r"/table/v1/", _synthetic_osrm(pts))
    res = travel_time_matrix(pts, chunk_size=chunk, units="s", pause=0)
    assert len(opener.calls) == n_requests and opener.sleeps == []
    np.testing.assert_array_equal(res.time, _expected_seconds(n)[0])
    for call in opener.calls:
        assert call["url"].split("/table/v1/driving/")[1].split("?")[0].count(";") + 1 <= 2 * chunk


def test_osrm_unroutable_pairs_are_nan_with_one_warning_and_zero_diagonal(opener):
    pts = _points(4)
    opener.add(r"/table/v1/", _synthetic_osrm(pts, unroutable={(0, 3), (2, 1)}))
    with pytest.warns(RuntimeWarning, match=r"2 of 12 pairs could not be routed by osrm") as record:
        res = travel_time_matrix(pts, units="s")
    assert len(record) == 1
    assert np.isnan(res.time[0, 3]) and np.isnan(res.time[2, 1]) and np.isnan(res.time).sum() == 2
    assert np.isnan(res.distance[0, 3]) and np.isnan(res.distance).sum() == 2
    assert (np.diag(res.time) == 0).all() and (np.diag(res.distance) == 0).all()


def test_osrm_answer_without_distances_keeps_times(opener):
    pts = _points(3)
    opener.add(r"/table/v1/", _synthetic_osrm(pts, drop_distances=True))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = travel_time_matrix(pts, units="s")
    assert not record, "a missing 'distances' table is not an unroutable pair"
    np.testing.assert_array_equal(res.time, _expected_seconds(3)[0])
    off = ~np.eye(3, dtype=bool)
    assert np.isnan(res.distance[off]).all() and (np.diag(res.distance) == 0).all()


def test_osrm_recorded_answer_for_three_madrid_points(opener):
    opener.add(r"/table/v1/driving/", _load("osrm_table_madrid.json"))
    res = travel_time_matrix([OFFICE, SOL, ALCALA])
    assert len(opener.calls) == 1 and opener.sleeps == []
    assert res.units == {"time": "min", "distance": "m"}
    np.testing.assert_allclose(res.time[0, 1], 1428.6 / 60.0)
    np.testing.assert_allclose(res.time[1, 0], 1256.7 / 60.0)
    np.testing.assert_allclose(res.distance[0, 2], 47710.8)
    assert not np.array_equal(res.time, res.time.T), "roads are asymmetric; the matrix is kept as returned"
    assert np.isfinite(res.time).all() and (np.diag(res.time) == 0).all()
    hours = travel_time_matrix([OFFICE, SOL, ALCALA], units="h")
    np.testing.assert_allclose(hours.time, res.time / 60.0)
    assert hours.units["time"] == "h"


def test_travel_time_matrix_single_point_needs_no_request(opener):
    res = travel_time_matrix([OFFICE])
    assert opener.calls == [] and res.time.tolist() == [[0.0]] and res.distance.tolist() == [[0.0]]


def test_osrm_options_base_url_mode_user_agent_pause_timeout(opener):
    pts = _points(3)
    opener.add(r"^http://localhost:5000/table/v1/car/", _synthetic_osrm(pts))
    res = travel_time_matrix(
        pts, base_url="http://localhost:5000/", mode="car", user_agent="my-app/1.0", pause=0.25, timeout=5
    )
    assert res.time[0, 1] == pytest.approx(10.0 / 60.0)
    assert len(opener.calls) == 1 and opener.sleeps == []
    assert opener.calls[0]["user_agent"] == "my-app/1.0" and opener.calls[0]["timeout"] == 5.0
    opener.calls.clear()
    travel_time_matrix(pts, base_url="http://localhost:5000", mode="car", chunk_size=2, pause=0.25)
    assert len(opener.calls) == 4 and opener.sleeps == [0.25] * 3


def test_osrm_error_code_and_malformed_answers_raise(opener):
    pts = _points(2)
    opener.add(r"/table/v1/", {"code": "NoTable", "message": "No table found"})
    with pytest.raises(MapServiceError, match=r"OSRM answered 'NoTable' No table found") as info:
        travel_time_matrix(pts)
    assert info.value.status == 200 and "/table/v1/driving/" in info.value.url
    opener.routes.clear()
    opener.add(r"/table/v1/", {"code": "Ok", "durations": [[0, 1]]})  # 1 x 2 instead of 2 x 2
    with pytest.raises(MapServiceError, match=r"no 2 x 2 'durations' table"):
        travel_time_matrix(pts)
    opener.routes.clear()
    opener.add(r"/table/v1/", b"<html>Bad gateway</html>")
    with pytest.raises(MapServiceError, match=r"did not answer JSON") as info:
        travel_time_matrix(pts)
    assert info.value.body.startswith("<html>")


# --------------------------------------------------------------------------- HTTP layer: retries and errors


def test_get_json_retries_429_and_5xx_with_back_off_then_succeeds(opener):
    url = "https://example.org/table/v1/driving/x"
    opener.add(
        r"example\.org", Script([_http_error(url, 429, "slow down"), _http_error(url, 503), {"ok": True}])
    )
    assert maps._get_json(url, params={"a": 1}) == {"ok": True}
    assert len(opener.calls) == 3
    assert opener.sleeps == [0.5, 1.0], "back-off between attempts, none before the first"
    assert opener.calls[0]["url"] == url + "?a=1"


def test_get_json_gives_up_after_three_retries_with_status_and_body(opener):
    url = "https://example.org/x"
    opener.add(
        r"example\.org", Script([_http_error(url, 500, "boom " * 100) for _ in range(4)] + [{"never": 1}])
    )
    with pytest.raises(MapServiceError, match=r"HTTP 500 from https://example.org/x: boom") as info:
        maps._get_json(url)
    assert len(opener.calls) == 4 and opener.sleeps == [0.5, 1.0, 2.0]
    assert info.value.status == 500 and len(info.value.body) == 200 and info.value.url == url
    assert isinstance(info.value, RuntimeError)


def test_get_json_does_not_retry_other_4xx_and_retries_network_errors(opener):
    url = "https://example.org/x"
    opener.add(r"example\.org", Script([_http_error(url, 404, "nope"), {"never": 1}]))
    with pytest.raises(MapServiceError, match=r"HTTP 404") as info:
        maps._get_json(url)
    assert len(opener.calls) == 1 and opener.sleeps == [] and info.value.body == "nope"
    opener.calls.clear()
    opener.routes.clear()
    opener.add(
        r"example\.org",
        Script([urllib.error.URLError("connection refused"), TimeoutError("timed out"), {"k": 2}]),
    )
    assert maps._get_json(url) == {"k": 2}
    assert len(opener.calls) == 3 and opener.sleeps == [0.5, 1.0]
    opener.calls.clear()
    opener.routes.clear()
    opener.add(r"example\.org", Script([urllib.error.URLError("down")] * 4))
    with pytest.raises(MapServiceError, match=r"could not reach https://example.org/x: down") as info:
        maps._get_json(url, retries=3)
    assert info.value.status is None and len(opener.calls) == 4


def test_get_json_sets_user_agent_accept_and_redacts_keys(opener):
    opener.add(r"example\.org", Script([_http_error("https://example.org/g", 400, "bad key")]))
    with pytest.raises(MapServiceError) as info:
        maps._get_json(
            "https://example.org/g", params={"address": "x y", "key": "SECRET"}, headers={"X-A": "1"}
        )
    assert "SECRET" not in str(info.value) and "key=%2A%2A%2A" in info.value.url
    call = opener.calls[0]
    assert call["query"] == {"address": "x y", "key": "SECRET"}, "the request itself carries the key"
    assert call["user_agent"] == maps.DEFAULT_USER_AGENT
    assert maps._redact_url("https://e.org/p?a=1") == "https://e.org/p?a=1"
    assert maps._redact_url("https://e.org/p") == "https://e.org/p"


def test_default_user_agent_names_the_package_version_and_repository():
    assert (
        f"scikit-route/{skroute.__version__} (+https://github.com/arubiales/scikit-route)"
        == maps.DEFAULT_USER_AGENT
    )


def test_map_service_error_keeps_status_url_and_200_chars_of_body():
    err = MapServiceError("msg", status=429, url="u", body="x" * 500)
    assert (err.status, err.url, len(err.body), str(err)) == (429, "u", 200, "msg")
    assert MapServiceError("plain").status is None


# --------------------------------------------------------------------------- validation and providers


@pytest.mark.parametrize(
    ("coords", "match"),
    [
        ([1.0, 2.0], r"shape \(n, 2\)"),
        ([[1.0, 2.0, 3.0]], r"shape \(n, 2\)"),
        (np.empty((0, 2)), r"at least one point"),
        ([[40.0, np.nan]], r"finite"),
        ([[40.0, np.inf]], r"finite"),
        ([[91.0, 0.0]], r"latitudes .* \[-90, 90\].*\(lon, lat\)"),
        ([[40.0, -3.0], [-95.5, 2.0]], r"\[-90, 90\]"),
        ([[40.0, 181.0]], r"longitudes .* \[-180, 180\]"),
        ([[1.0], [2.0, 3.0]], r"\(n, 2\)"),
    ],
)
def test_travel_time_matrix_rejects_bad_coordinates(opener, coords, match):
    with pytest.raises(ValueError, match=match):
        travel_time_matrix(coords)
    assert opener.calls == []


def test_travel_time_matrix_rejects_bad_units_provider_chunk_pause_timeout(opener):
    pts = _points(2)
    with pytest.raises(ValueError, match=r"units must be one of \['s', 'min', 'h'\]; got 'days'"):
        travel_time_matrix(pts, units="days")
    with pytest.raises(ValueError, match=r"provider must be one of \['osrm', 'google'\]; got 'here'"):
        travel_time_matrix(pts, provider="here")
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match=r"chunk_size must be a positive integer"):
            travel_time_matrix(pts, chunk_size=bad)
    with pytest.raises(ValueError, match=r"pause must be a non-negative"):
        travel_time_matrix(pts, pause=-1)
    with pytest.raises(ValueError, match=r"timeout must be a positive"):
        travel_time_matrix(pts, timeout=0)
    with pytest.raises(ValueError, match=r"mode must be a non-empty string"):
        travel_time_matrix(pts, mode="")
    assert opener.calls == []


def test_travel_time_matrix_osrm_warns_when_departure_time_is_given(opener):
    pts = _points(2)
    opener.add(r"/table/v1/", _synthetic_osrm(pts))
    with pytest.warns(UserWarning, match=r"OSRM has no traffic model: departure_time is ignored"):
        res = travel_time_matrix(pts, departure_time="now")
    assert res.time[0, 1] == pytest.approx(10.0 / 60.0)


def _fake_googlemaps(monkeypatch, *, traffic=False):
    """Fake ``googlemaps`` module: duration = 60 s per 0.1 degree of latitude gap, +600 s in traffic."""
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, key):
            calls.append({"key": key})

        def distance_matrix(self, origins, destinations, **kwargs):
            calls.append({"origins": list(origins), "destinations": list(destinations), **kwargs})
            rows = []
            for o in origins:
                elements = []
                for d in destinations:
                    seconds = round(600.0 * abs(o[0] - d[0]), 6)
                    element = {
                        "status": "OK",
                        "distance": {"value": round(1000.0 * (abs(o[0] - d[0]) + abs(o[1] - d[1])), 3)},
                        "duration": {"value": seconds},
                    }
                    if traffic:
                        element["duration_in_traffic"] = {"value": seconds + 600.0}
                    elements.append(element)
                rows.append({"elements": elements})
            return {
                "status": "OK",
                "origin_addresses": ["a"] * len(origins),
                "destination_addresses": ["b"] * len(destinations),
                "rows": rows,
            }

    module = types.ModuleType("googlemaps")
    module.Client = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "googlemaps", module)
    return calls


def test_google_provider_converts_hours_to_the_requested_units(opener, monkeypatch):
    calls = _fake_googlemaps(monkeypatch)
    pts = _points(3)
    res = travel_time_matrix(pts, provider="google", api_key="KEY")
    assert opener.calls == [], "Google goes through the googlemaps client, not urllib"
    assert calls[0] == {"key": "KEY"}
    assert len(calls) == 2 and calls[1]["mode"] == "driving" and "departure_time" not in calls[1]
    assert res.provider == "google" and res.units == {"time": "min", "distance": "m"}
    assert res.time[0, 1] == pytest.approx(1.0) and res.time[0, 2] == pytest.approx(2.0)  # 60 s, 120 s
    assert res.distance[0, 1] == pytest.approx(150.0)
    assert (np.diag(res.time) == 0).all()
    seconds = travel_time_matrix(pts, provider="google", api_key="KEY", units="s", mode="walking")
    assert seconds.time[0, 2] == pytest.approx(120.0) and calls[-1]["mode"] == "walking"


def test_google_provider_forwards_departure_time_and_prefers_duration_in_traffic(monkeypatch):
    calls = _fake_googlemaps(monkeypatch, traffic=True)
    pts = _points(2)
    res = travel_time_matrix(pts, provider="google", api_key="KEY", departure_time="now")
    assert calls[-1]["departure_time"] == "now"
    assert res.time[0, 1] == pytest.approx(11.0), "660 s in traffic, not the 60 s free-flow duration"
    plain = travel_time_matrix(pts, provider="google", api_key="KEY")
    assert "departure_time" not in calls[-1] and plain.time[0, 1] == pytest.approx(11.0)


def test_google_provider_batches_with_chunk_size_and_validates(monkeypatch):
    calls = _fake_googlemaps(monkeypatch)
    pts = _points(5)
    travel_time_matrix(pts, provider="google", api_key="KEY", chunk_size=2)
    assert len(calls) == 1 + 9 and all(len(c["origins"]) <= 2 for c in calls[1:])
    with pytest.raises(ValueError, match=r"chunk_size must be at most 10 for this provider"):
        travel_time_matrix(pts, provider="google", api_key="KEY", chunk_size=11)
    with pytest.raises(ValueError, match=r"provider='google' needs api_key"):
        travel_time_matrix(pts, provider="google")
    with pytest.raises(
        ValueError, match=r"mode must be one of \['driving', 'walking', 'bicycling', 'transit'\]"
    ):
        travel_time_matrix(pts, provider="google", api_key="KEY", mode="car")


def test_google_distance_matrix_departure_time_is_backwards_compatible(monkeypatch):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    calls = _fake_googlemaps(monkeypatch, traffic=True)
    gdm = GoogleDistanceMatrix("KEY")
    assert gdm.departure_time is None
    res = gdm.fetch(_points(2))
    assert "departure_time" not in calls[-1] and res.units == {"distance": "m", "time": "h"}
    assert res.time[0, 1] == pytest.approx(660.0 / 3600.0), "duration_in_traffic is preferred when present"
    GoogleDistanceMatrix("KEY", departure_time=1_700_000_000).fetch(_points(2))
    assert calls[-1]["departure_time"] == 1_700_000_000


# --------------------------------------------------------------------------- geocoding


def test_geocode_nominatim_recorded_answer_and_url(opener):
    opener.add(r"nominatim\.openstreetmap\.org/search", _load("nominatim_leganes.json"))
    res = geocode("Calle Ramón y Cajal 18, Leganés")
    assert isinstance(res, Bunch) and set(res.keys()) == {"lat", "lon", "display_name", "raw"}
    assert (res.lat, res.lon) == (40.3295593, -3.7372695) and isinstance(res.lat, float)
    assert res.display_name.startswith("18, Calle de Ramón y Cajal") and res.raw["osm_type"] == "way"
    call = opener.calls[0]
    assert call["url"].startswith("https://nominatim.openstreetmap.org/search?")
    assert call["query"] == {"format": "jsonv2", "limit": "1", "q": "Calle Ramón y Cajal 18, Leganés"}
    assert call["user_agent"] == maps.DEFAULT_USER_AGENT and call["timeout"] == 10.0 and call["data"] is None
    assert opener.sleeps == [], "the first call never waits"


def test_geocode_nominatim_enforces_one_request_per_second(opener):
    opener.add(r"/search", _load("nominatim_leganes.json"))
    geocode("a")
    assert opener.sleeps == []
    opener.now += 0.3
    geocode("b")
    assert opener.sleeps == [pytest.approx(0.7)], "the second call waits the rest of the second"
    opener.now += 5.0
    geocode("c")
    assert len(opener.sleeps) == 1, "no wait once a second has passed"
    geocode("d")
    assert opener.sleeps[-1] == pytest.approx(1.0)
    assert len(opener.calls) == 4


def test_geocode_nominatim_empty_result_and_options(opener):
    opener.add(r"^https://geo\.example/search", [])
    with pytest.raises(ValueError, match=r"no result for 'Nowhere Street 0'"):
        geocode("Nowhere Street 0", base_url="https://geo.example/", user_agent="my-app/2", timeout=3)
    call = opener.calls[0]
    assert call["user_agent"] == "my-app/2" and call["timeout"] == 3.0
    opener.routes.clear()
    opener.add(r"/search", {"unexpected": True})
    with pytest.raises(MapServiceError, match=r"unexpected document"):
        geocode("x")
    opener.routes.clear()
    opener.add(r"/search", [{"display_name": "no coordinates"}])
    with pytest.raises(MapServiceError, match=r"without coordinates"):
        geocode("x")


def test_geocode_rejects_bad_query_and_provider(opener):
    for bad in ("", "   ", None, 3):
        with pytest.raises(ValueError, match=r"query must be a non-empty string"):
            geocode(bad)
    with pytest.raises(ValueError, match=r"provider must be one of \['nominatim', 'google'\]; got 'bing'"):
        geocode("x", provider="bing")
    with pytest.raises(ValueError, match=r"provider='google' needs api_key"):
        geocode("x", provider="google")
    assert opener.calls == []


def test_geocode_google_recorded_answer_statuses_and_key_redaction(opener):
    opener.add(r"maps\.googleapis\.com/maps/api/geocode/json", _load("google_geocode_leganes.json"))
    res = geocode("Calle Ramón y Cajal 18, Leganés", provider="google", api_key="SECRET")
    assert (res.lat, res.lon) == (40.3295593, -3.7372695)
    assert res.display_name == "C. de Ramón y Cajal, 18, 28916 Leganés, Madrid, Spain"
    assert res.raw["place_id"] == "ChIJ-example-place-id"
    call = opener.calls[0]
    assert call["query"] == {"address": "Calle Ramón y Cajal 18, Leganés", "key": "SECRET"}
    assert opener.sleeps == [], "Google is not throttled"
    opener.routes.clear()
    opener.add(r"geocode", {"status": "ZERO_RESULTS", "results": []})
    with pytest.raises(ValueError, match=r"no result for 'nowhere'"):
        geocode("nowhere", provider="google", api_key="SECRET")
    opener.routes.clear()
    opener.add(r"geocode", {"status": "REQUEST_DENIED", "error_message": "The provided API key is invalid."})
    with pytest.raises(MapServiceError, match=r"Google Geocoding answered 'REQUEST_DENIED' The provided"):
        geocode("x", provider="google", api_key="SECRET")
    opener.routes.clear()
    opener.add(
        r"^https://geo\.example/v1\?", Script([_http_error("https://geo.example/v1", 403, "forbidden")])
    )
    with pytest.raises(MapServiceError, match=r"HTTP 403") as info:
        geocode("x", provider="google", api_key="SECRET", base_url="https://geo.example/v1")
    assert "SECRET" not in str(info.value) and "SECRET" not in info.value.url


# --------------------------------------------------------------------------- points of interest


EXPECTED_QUERY = (
    "[out:json][timeout:90];\n"
    'area["boundary"="administrative"]["name"="Comunidad de Madrid"]->.a;\n'
    'nwr["amenity"="fast_food"]["brand"="Burger King"]["name"~"burger",i]'
    '["brand:wikidata"="Q177054"](area.a);\n'
    "out center;"
)


def test_fetch_pois_builds_the_overpass_query_and_gets_it(opener):
    opener.add(r"overpass-api\.de/api/interpreter", {"version": 0.6, "elements": []})
    with pytest.warns(UserWarning, match=r"no element matched in 'Comunidad de Madrid'"):
        res = fetch_pois(
            "Comunidad de Madrid", amenity="fast_food", brand="Burger King", name="burger", wikidata="Q177054"
        )
    call = opener.calls[0]
    assert call["url"].startswith("https://overpass-api.de/api/interpreter?data=") and call["data"] is None
    assert call["query"] == {"data": EXPECTED_QUERY}
    assert call["timeout"] == 90.0 and call["user_agent"] == maps.DEFAULT_USER_AGENT
    assert res.coords.shape == (0, 2) and res.coords.dtype == np.float64
    assert res.labels == [] and res.names == [] and res.addresses == [] and res.tags == []
    assert EXPECTED_QUERY in res.DESCR and "© OpenStreetMap contributors" in res.DESCR


def test_fetch_pois_single_filters_escaping_and_timeout(opener):
    opener.add(r"overpass", {"elements": []})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fetch_pois(
            'Sant "Cugat"', name=r"caf\é", timeout=25, base_url="https://overpass.example/api", user_agent="x"
        )
    call = opener.calls[0]
    assert call["url"].startswith("https://overpass.example/api?data=")
    assert call["query"]["data"] == (
        "[out:json][timeout:25];\n"
        'area["boundary"="administrative"]["name"="Sant \\"Cugat\\""]->.a;\n'
        'nwr["name"~"caf\\\\é",i](area.a);\n'
        "out center;"
    )
    assert call["timeout"] == 25.0 and call["user_agent"] == "x"
    assert (
        maps._overpass_query("A", timeout=1, wikidata="Q1").splitlines()[2]
        == 'nwr["brand:wikidata"="Q1"](area.a);'
    )
    assert maps._overpass_query("A", timeout=1, brand="B").splitlines()[2] == 'nwr["brand"="B"](area.a);'


def test_fetch_pois_requires_a_filter_an_area_and_a_known_provider(opener):
    with pytest.raises(ValueError, match=r"give at least one filter: brand=, name=, amenity= or wikidata="):
        fetch_pois("Leganés")
    with pytest.raises(ValueError, match=r"area must be a non-empty string"):
        fetch_pois("  ", brand="x")
    with pytest.raises(ValueError, match=r"provider must be one of \['overpass'\]; got 'osm'"):
        fetch_pois("Leganés", brand="x", provider="osm")
    assert opener.calls == []


def test_fetch_pois_parses_nodes_ways_relations_sorted_with_addresses(opener):
    opener.add(r"interpreter", _load("overpass_leganes.json"))
    with pytest.warns(RuntimeWarning, match=r"1 Overpass elements without coordinates were skipped"):
        res = fetch_pois("Leganés", amenity="fast_food", wikidata="Q177054")
    assert isinstance(res, Bunch)
    assert set(res.keys()) == {"coords", "labels", "names", "addresses", "tags", "DESCR"}
    assert res.labels == [
        "node/2613719490",
        "node/2631338026",
        "node/6723530125",
        "node/11083618706",
        "relation/15000001",
        "way/431072075",
        "way/1156068846",
    ], "sorted by type then id, whatever the server order"
    assert res.coords.shape == (7, 2) and res.coords.dtype == np.float64
    np.testing.assert_allclose(res.coords[0], [40.3365943, -3.7689667])  # a node: its own position
    np.testing.assert_allclose(res.coords[4], [40.3312, -3.7601])  # a relation: its centre
    assert (np.abs(res.coords[:, 0] - 40.33) < 0.1).all() and (np.abs(res.coords[:, 1] + 3.77) < 0.1).all()
    assert res.names == ["Burger King"] * 7, "the relation has no name tag: brand is the fallback"
    assert res.addresses == [
        "Avenida de la Universidad 12, 28911 Leganés",
        "Calle del Charco 4",
        "28914 Leganés",
        "",
        "",
        "",
        "",
    ]
    assert all(isinstance(t, dict) for t in res.tags) and res.tags[0]["addr:city"] == "Leganés"
    assert res.tags[5]["building"] == "yes" and res.tags[4]["type"] == "multipolygon"
    assert "7 OpenStreetMap elements inside the administrative area 'Leganés'" in res.DESCR
    assert res.DESCR.splitlines()[-1] == maps.OSM_ATTRIBUTION and "ODbL" in res.DESCR
    assert 'nwr["amenity"="fast_food"]["brand:wikidata"="Q177054"](area.a);' in res.DESCR
    assert maps._format_address({}) == "" and maps._format_address({"addr:housenumber": "5"}) == "5"


def test_fetch_pois_remark_and_malformed_answers_raise(opener):
    opener.add(
        r"interpreter", {"elements": [], "remark": "runtime error: Query timed out in 'query' at line 3"}
    )
    with pytest.raises(MapServiceError, match=r"Overpass reported a problem: runtime error: Query timed out"):
        fetch_pois("Leganés", brand="x")
    opener.routes.clear()
    opener.add(r"interpreter", {"version": 0.6})
    with pytest.raises(MapServiceError, match=r"unexpected document"):
        fetch_pois("Leganés", brand="x")
    opener.routes.clear()
    opener.calls.clear()
    opener.add(
        r"interpreter",
        Script([_http_error("https://overpass-api.de/api/interpreter", 406, "Not Acceptable")]),
    )
    with pytest.raises(MapServiceError, match=r"HTTP 406") as info:
        fetch_pois("Leganés", brand="x")
    assert info.value.status == 406 and len(opener.calls) == 1, "406 is not retried"


def test_fetch_pois_skips_elements_without_coordinates_or_ids(opener):
    elements = [
        {"type": "node", "id": 2, "lat": 1.0, "lon": 2.0, "tags": {"name": "b"}},
        {"type": "way", "id": 1, "tags": {"name": "no centre"}},
        {"type": "node", "lat": 1.0, "lon": 2.0},
        "garbage",
        {"type": "node", "id": 1, "lat": 3.0, "lon": 4.0},
    ]
    opener.add(r"interpreter", {"elements": elements})
    with pytest.warns(RuntimeWarning, match=r"3 Overpass elements without coordinates were skipped"):
        res = fetch_pois("A", brand="b")
    assert res.labels == ["node/1", "node/2"] and res.names == ["", "b"] and res.tags[0] == {}


def test_preprocessing_reexports_the_map_functions():
    from skroute import preprocessing

    for name in ("fetch_pois", "geocode", "travel_time_matrix"):
        assert name in preprocessing.__all__ and getattr(preprocessing, name) is getattr(maps, name)
    assert preprocessing.__all__ == sorted(preprocessing.__all__)
    assert maps.__all__ == ["MapServiceError", "fetch_pois", "geocode", "travel_time_matrix"]


# --------------------------------------------------------------------------- live services (-m network)


@pytest.mark.network
def test_live_osrm_three_points():
    res = travel_time_matrix([OFFICE, SOL, ALCALA], pause=0)
    assert res.time.shape == (3, 3) and np.isfinite(res.time).all() and (np.diag(res.time) == 0).all()
    assert 10 < res.time[0, 1] < 60 and 20 < res.time[0, 2] < 120, "minutes Leganés -> Sol / Alcalá"
    assert 10_000 < res.distance[0, 1] < 30_000


@pytest.mark.network
def test_live_nominatim_geocodes_the_office():
    res = geocode("Calle Ramón y Cajal 18, Leganés")
    assert abs(res.lat - 40.33) < 0.05 and abs(res.lon + 3.74) < 0.05 and "Leganés" in res.display_name


@pytest.mark.network
def test_live_overpass_finds_burger_kings_in_leganes():
    res = fetch_pois("Leganés", amenity="fast_food", wikidata="Q177054")
    assert 1 <= len(res.labels) <= 30 and res.coords.shape == (len(res.labels), 2)
    assert all(label.split("/")[0] in {"node", "way", "relation"} for label in res.labels)
    assert "© OpenStreetMap contributors" in res.DESCR
