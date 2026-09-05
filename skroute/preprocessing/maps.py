"""Real-world inputs from map services: travel-time matrices, geocoding and points of interest.

Three functions, all on the standard library (``urllib.request`` and ``json``), every one
honouring a ``timeout=`` and retrying a failed HTTP call up to three times with back-off
(0.5 s, 1 s, 2 s):

* `travel_time_matrix` -- square matrices of road travel times and distances between
  ``(lat, lon)`` points, from an OSRM server (default: the public demo server) or from
  the Google Distance Matrix API through
  `skroute.preprocessing.google.GoogleDistanceMatrix`;
* `geocode` -- one address or place name to ``(lat, lon)``, from Nominatim (default)
  or from the Google Geocoding API;
* `fetch_pois` -- every OpenStreetMap element matching some tags inside an
  administrative area, through the Overpass API, as coordinates plus names, addresses
  and tags.

The public services are shared resources with usage policies. Nominatim wants a
descriptive ``User-Agent`` and at most one request per second (both enforced here); the
OSRM demo server caps the size of a table request (hence the tiling) and expects a pause
between requests; Overpass limits the load per client. Data from OpenStreetMap is
licensed under the ODbL and must carry the attribution "© OpenStreetMap contributors"
wherever it is shown. Google's services need an API key and are billed to your account.

Tests never touch the network: they monkeypatch the module-level ``_urlopen`` (and
``_sleep`` / ``_monotonic``) and serve recorded JSON answers from ``tests/data/maps/``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from skroute._version import __version__
from skroute.utils import Bunch

from .google import MODES as GOOGLE_MODES
from .google import GoogleDistanceMatrix

__all__ = ["MapServiceError", "fetch_pois", "geocode", "travel_time_matrix"]

#: ``User-Agent`` sent when ``user_agent=None``: Nominatim's usage policy requires one that identifies
#: the application, and the other services appreciate it.
DEFAULT_USER_AGENT = f"scikit-route/{__version__} (+https://github.com/arubiales/scikit-route)"

#: Default servers. The OSRM one is the project's public demo (car profile only, limited table size).
OSRM_URL = "https://router.project-osrm.org"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
GOOGLE_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"

#: Attribution every OpenStreetMap-derived result must carry (ODbL).
OSM_ATTRIBUTION = (
    "Data © OpenStreetMap contributors, licensed under the Open Database License (ODbL): "
    "https://www.openstreetmap.org/copyright"
)

#: Points per block of an OSRM table request: a block request carries at most ``2 * chunk_size``
#: coordinates (100 with the default), within the demo server's limits.
DEFAULT_CHUNK_SIZE = 50

#: Nominatim's usage policy: at most one request per second.
NOMINATIM_MIN_INTERVAL = 1.0

_BACKOFF = (0.5, 1.0, 2.0)
_TIME_FACTORS = {"s": 1.0, "min": 1.0 / 60.0, "h": 1.0 / 3600.0}
_MATRIX_PROVIDERS = ("osrm", "google")
_GEOCODE_PROVIDERS = ("nominatim", "google")
_POI_PROVIDERS = ("overpass",)
_REDACTED_PARAMS = ("key",)

# Indirections the tests monkeypatch: the opener (recorded answers instead of the network), the clock
# and the sleeper (no real waiting for the back-off, the OSRM pause or the Nominatim throttle).
_urlopen = urllib.request.urlopen
_sleep = time.sleep
_monotonic = time.monotonic
_last_nominatim_call: float | None = None


class MapServiceError(RuntimeError):
    """A map service answered with an error, malformed content, or could not be reached.

    Raised after the retries of a failed HTTP call are exhausted (``429`` and ``5xx`` are
    retried, other statuses are not), when a service answers something that is not JSON,
    or when the JSON reports an error (``"code" != "Ok"`` from OSRM, a non-``OK`` status
    from Google, a ``remark`` from Overpass).

    Parameters
    ----------
    message : str
        Human-readable description.
    status : int or None, default=None
        HTTP status, ``None`` when the server could not be reached.
    url : str, default=""
        The request URL (API keys redacted).
    body : str, default=""
        The first 200 characters of the answer.

    Examples
    --------
    >>> from skroute.preprocessing.maps import MapServiceError
    >>> err = MapServiceError("HTTP 429 from https://example.org/table: rate limited", status=429)
    >>> err.status, str(err)
    (429, 'HTTP 429 from https://example.org/table: rate limited')
    """

    def __init__(self, message: str, *, status: int | None = None, url: str = "", body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url
        self.body = body[:200]


def _redact_url(url: str) -> str:
    """The URL with the values of credential parameters (``key=``) replaced by ``***``."""
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url
    pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    if not any(k in _REDACTED_PARAMS for k, _ in pairs):
        return url
    redacted = [(k, "***" if k in _REDACTED_PARAMS else v) for k, v in pairs]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(redacted)))


def _excerpt(data: bytes) -> str:
    """The first 200 characters of an answer body, for error messages."""
    return data[:800].decode("utf-8", errors="replace")[:200]


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return _excerpt(exc.read())
    except Exception:  # pragma: no cover - a body that cannot be read is not worth a second failure
        return ""


def _get_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    safe: str = "",
) -> Any:
    """GET ``url`` (with ``params`` URL-encoded) and decode the JSON answer, retrying with back-off.

    Retries ``HTTPError`` 429 and 5xx and connection failures (``URLError``, timeouts) up to
    ``retries`` times, sleeping 0.5 s, 1 s and 2 s between attempts; any other HTTP status,
    a non-JSON body and an exhausted retry budget raise `MapServiceError` with the status
    and the first 200 characters of the body. ``safe`` lists the characters ``params`` may
    keep unescaped (OSRM reads ``sources=0;1;2`` literally).
    """
    full_url = f"{url}?{urllib.parse.urlencode(dict(params), safe=safe)}" if params else url
    shown = _redact_url(full_url)
    request_headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(full_url, headers=request_headers)
    attempt = 0
    while True:
        try:
            with _urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            body = _http_error_body(exc)
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt >= retries:
                raise MapServiceError(
                    f"HTTP {exc.code} from {shown}: {body or exc.reason}",
                    status=exc.code,
                    url=shown,
                    body=body,
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                reason = getattr(exc, "reason", exc)
                raise MapServiceError(f"could not reach {shown}: {reason}", status=None, url=shown) from exc
        _sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
        attempt += 1
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise MapServiceError(
            f"{shown} did not answer JSON: {_excerpt(raw)!r}", status=200, url=shown, body=_excerpt(raw)
        ) from exc


def _validate_coords(coords: ArrayLike) -> np.ndarray:
    """``coords`` as a float64 ``(n, 2)`` array of ``(lat, lon)`` pairs, or ``ValueError``."""
    try:
        xy = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"coords must be an (n, 2) array-like of (lat, lon) pairs: {exc}") from exc
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2) -- one (lat, lon) pair per row; got {xy.shape}")
    if xy.shape[0] == 0:
        raise ValueError("coords must hold at least one point")
    if not np.isfinite(xy).all():
        raise ValueError("coords must be finite (no nan or inf)")
    if (np.abs(xy[:, 0]) > 90.0).any():
        raise ValueError(
            "latitudes (first column) must lie in [-90, 90]; coords are (lat, lon) pairs -- "
            "did you pass (lon, lat)?"
        )
    if (np.abs(xy[:, 1]) > 180.0).any():
        raise ValueError("longitudes (second column) must lie in [-180, 180]")
    return xy


def _time_factor(units: str) -> float:
    if units not in _TIME_FACTORS:
        raise ValueError(f"units must be one of {list(_TIME_FACTORS)}; got {units!r}")
    return _TIME_FACTORS[units]


def _check_provider(provider: str, allowed: tuple[str, ...]) -> None:
    if provider not in allowed:
        raise ValueError(f"provider must be one of {list(allowed)}; got {provider!r}")


def _check_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string; got {value!r}")
    return value.strip()


def _headers(user_agent: str | None) -> dict[str, str]:
    return {
        "User-Agent": _check_text(user_agent, "user_agent") if user_agent is not None else DEFAULT_USER_AGENT
    }


def _check_timeout(timeout: Any) -> float:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float, np.integer, np.floating))
        or timeout <= 0
    ):
        raise ValueError(f"timeout must be a positive number of seconds; got {timeout!r}")
    return float(timeout)


# --------------------------------------------------------------------------- travel-time matrices


def travel_time_matrix(
    coords: ArrayLike,
    *,
    provider: str = "osrm",
    mode: str = "driving",
    units: str = "min",
    api_key: str | None = None,
    departure_time: Any = None,
    base_url: str | None = None,
    chunk_size: int | None = None,
    user_agent: str | None = None,
    pause: float = 1.0,
    timeout: float = 60,
) -> Bunch:
    """Road travel times and distances between ``(lat, lon)`` points, from OSRM or Google.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        ``(latitude, longitude)`` pairs in decimal degrees (finite, latitudes in
        ``[-90, 90]``, longitudes in ``[-180, 180]``). Note the order: latitude first,
        as everywhere in scikit-route; the OSRM URLs are built ``lon,lat`` internally.
    provider : {"osrm", "google"}, default="osrm"
        ``"osrm"`` calls the table service of an OSRM server; ``"google"`` delegates to
        `skroute.preprocessing.google.GoogleDistanceMatrix` (needs ``api_key`` and the
        ``google`` extra; billed to your account).
    mode : str, default="driving"
        For OSRM, the profile segment of the URL (``/table/v1/<mode>/``): the public demo
        server only serves ``"driving"``; a self-hosted server names its profiles freely
        (``"car"``, ``"bike"``, ``"foot"``...). For Google, one of ``"driving"``,
        ``"walking"``, ``"bicycling"``, ``"transit"``.
    units : {"min", "s", "h"}, default="min"
        Unit of the returned ``time`` matrix. Distances are always in metres.
    api_key : str, optional
        Google Cloud key with the Distance Matrix API enabled; required by ``"google"``,
        ignored by ``"osrm"``.
    departure_time : "now", datetime or int, optional
        Google only: asks for ``duration_in_traffic`` (preferred over ``duration`` when
        the answer carries it). OSRM has no traffic model: a value is ignored with a
        ``UserWarning``.
    base_url : str, optional
        OSRM server, default ``https://router.project-osrm.org`` (the public demo:
        car profile, limited table size, one request per second is polite).
    chunk_size : int, optional
        OSRM: the points are split into ``ceil(n / chunk_size)`` blocks of nearly equal
        size and the table is requested block by block (``sources`` = one block,
        ``destinations`` = another), so a request never carries more than ``2 *
        chunk_size`` coordinates; default 50 (100 coordinates, the demo server's limit).
        Google: the ``batch_size`` of `GoogleDistanceMatrix` (``1..10``, default 10).
    user_agent : str, optional
        ``User-Agent`` header; default ``scikit-route/<version> (+repository URL)``.
    pause : float, default=1.0
        Seconds to wait between consecutive OSRM requests (``0`` for a server of yours).
    timeout : float, default=60
        Seconds to wait for each HTTP answer.

    Returns
    -------
    Bunch
        ``time`` and ``distance``: float64 ``(n, n)`` arrays (row = origin, column =
        destination; asymmetric when the roads are; diagonal ``0``); ``units``:
        ``{"time": units, "distance": "m"}``; ``provider``; ``coords``: the validated
        float64 ``(n, 2)`` array.

    Raises
    ------
    ValueError
        Bad coordinates, units, provider, chunk size, or a Google request without a key.
    MapServiceError
        The server could not be reached after the retries or answered an error.

    Warns
    -----
    RuntimeWarning
        Once, when some pairs could not be routed: they are ``nan`` in both matrices
        (complete them before solving; every solver needs finite matrices).

    Notes
    -----
    Each block request is ``GET {base_url}/table/v1/{mode}/{lon,lat;...}?sources=0;1;..&
    destinations=..&annotations=duration,distance``; the coordinate list is the row block
    followed by the column block (just the block for a diagonal one). ``n = 120`` points
    with the default ``chunk_size`` mean ``3 x 3 = 9`` requests and 8 pauses. A single
    point needs no request (a ``1 x 1`` table is just ``0``). Distances the server does
    not return stay ``nan`` in ``distance`` without making the pair unroutable.

    Examples
    --------
    >>> from skroute.preprocessing import travel_time_matrix
    >>> office = (40.3272, -3.7635)  # Leganés
    >>> sol = (40.4168, -3.7038)  # Puerta del Sol
    >>> res = travel_time_matrix([office, sol])  # doctest: +SKIP
    >>> res  # doctest: +SKIP
    Bunch(coords, distance, provider, time, units)
    >>> res.units, res.time.round(1).tolist()  # doctest: +SKIP
    ({'time': 'min', 'distance': 'm'}, [[0.0, 23.8], [20.9, 0.0]])
    >>> res.distance.round().tolist()  # doctest: +SKIP
    [[0.0, 16994.0], [14460.0, 0.0]]
    """
    xy = _validate_coords(coords)
    factor = _time_factor(units)
    _check_provider(provider, _MATRIX_PROVIDERS)
    if provider == "osrm":
        if departure_time is not None:
            warnings.warn(
                "OSRM has no traffic model: departure_time is ignored (use provider='google' for "
                "duration_in_traffic)",
                UserWarning,
                stacklevel=2,
            )
        seconds, metres = _osrm_table(
            xy,
            mode=mode,
            base_url=base_url,
            chunk_size=chunk_size,
            user_agent=user_agent,
            pause=pause,
            timeout=timeout,
        )
    else:
        seconds, metres = _google_table(
            xy, mode=mode, api_key=api_key, departure_time=departure_time, chunk_size=chunk_size
        )
    np.fill_diagonal(seconds, 0.0)
    np.fill_diagonal(metres, 0.0)
    unroutable = np.isnan(seconds)
    if unroutable.any():
        metres[unroutable] = np.nan
        n = xy.shape[0]
        warnings.warn(
            f"{int(unroutable.sum())} of {n * (n - 1)} pairs could not be routed by {provider} and are nan "
            "in time and distance; complete them before solving (every solver needs finite matrices)",
            RuntimeWarning,
            stacklevel=2,
        )
    return Bunch(
        time=seconds * factor,
        distance=metres,
        units={"time": units, "distance": "m"},
        provider=provider,
        coords=xy.copy(),
    )


def _check_chunk_size(chunk_size: Any, upper: int | None) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer; got {chunk_size!r}")
    if upper is not None and chunk_size > upper:
        raise ValueError(f"chunk_size must be at most {upper} for this provider; got {chunk_size!r}")
    return int(chunk_size)


def _osrm_table(
    xy: np.ndarray,
    *,
    mode: str,
    base_url: str | None,
    chunk_size: int | None,
    user_agent: str | None,
    pause: float,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Seconds and metres from an OSRM table service, block by block; ``nan`` where the server said null."""
    n = xy.shape[0]
    profile = urllib.parse.quote(_check_text(mode, "mode"), safe="")
    base = _check_text(base_url, "base_url").rstrip("/") if base_url is not None else OSRM_URL
    size = DEFAULT_CHUNK_SIZE if chunk_size is None else _check_chunk_size(chunk_size, None)
    if isinstance(pause, bool) or not isinstance(pause, (int, float, np.integer, np.floating)) or pause < 0:
        raise ValueError(f"pause must be a non-negative number of seconds; got {pause!r}")
    http_timeout = _check_timeout(timeout)
    headers = _headers(user_agent)
    seconds = np.full((n, n), np.nan, dtype=np.float64)
    metres = np.full((n, n), np.nan, dtype=np.float64)
    if n == 1:
        return seconds, metres  # the diagonal is forced to 0 by the caller; OSRM rejects a 1-point table
    blocks = [np.asarray(b, dtype=np.int64) for b in np.array_split(np.arange(n), -(-n // size))]
    first = True
    for rows in blocks:
        for cols in blocks:
            if not first and pause > 0:
                _sleep(float(pause))
            first = False
            diagonal = rows is cols
            index = rows if diagonal else np.concatenate([rows, cols])
            sources = range(len(rows))
            destinations = range(len(rows)) if diagonal else range(len(rows), len(index))
            path = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in xy[index])
            url = f"{base}/table/v1/{profile}/{path}"
            params = {
                "sources": ";".join(str(s) for s in sources),
                "destinations": ";".join(str(d) for d in destinations),
                "annotations": "duration,distance",
            }
            answer = _get_json(url, params=params, headers=headers, timeout=http_timeout, safe=";,")
            _fill_osrm_block(answer, seconds, metres, rows, cols, url)
    return seconds, metres


def _fill_osrm_block(
    answer: Any, seconds: np.ndarray, metres: np.ndarray, rows: np.ndarray, cols: np.ndarray, url: str
) -> None:
    if not isinstance(answer, dict) or answer.get("code") != "Ok":
        code = answer.get("code") if isinstance(answer, dict) else type(answer).__name__
        message = answer.get("message", "") if isinstance(answer, dict) else ""
        raise MapServiceError(
            f"OSRM answered {code!r} {message}".rstrip(), status=200, url=url, body=str(message)
        )
    durations = answer.get("durations")
    distances = answer.get("distances")
    if _as_table(durations, len(rows), len(cols)) is None:
        raise MapServiceError(
            f"OSRM answer has no {len(rows)} x {len(cols)} 'durations' table", status=200, url=url
        )
    _scatter(_as_table(durations, len(rows), len(cols)) or [], seconds, rows, cols)
    distance_table = _as_table(distances, len(rows), len(cols))
    if distance_table is not None:
        _scatter(distance_table, metres, rows, cols)


def _as_table(value: Any, n_rows: int, n_cols: int) -> list[list[Any]] | None:
    """``value`` when it is an ``n_rows x n_cols`` nested list, else ``None``."""
    if (
        isinstance(value, list)
        and len(value) == n_rows
        and all(isinstance(row, list) and len(row) == n_cols for row in value)
    ):
        return value
    return None


def _scatter(table: list[list[Any]], target: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
    """Write a block table into ``target`` at ``rows x cols``; ``None`` (unroutable) becomes ``nan``."""
    for r, row in enumerate(table):
        for c, value in enumerate(row):
            target[rows[r], cols[c]] = np.nan if value is None else float(value)


def _google_table(
    xy: np.ndarray, *, mode: str, api_key: str | None, departure_time: Any, chunk_size: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Seconds and metres through `GoogleDistanceMatrix` (hours -> seconds; ``nan`` where unroutable)."""
    if api_key is None or not str(api_key).strip():
        raise ValueError(
            "provider='google' needs api_key: a Google Cloud key with the Distance Matrix API enabled"
        )
    if mode not in GOOGLE_MODES:
        raise ValueError(f"mode must be one of {list(GOOGLE_MODES)} for provider='google'; got {mode!r}")
    batch = 10 if chunk_size is None else _check_chunk_size(chunk_size, 10)
    client = GoogleDistanceMatrix(str(api_key), mode, batch_size=batch, departure_time=departure_time)
    result = client.fetch(xy)
    return np.asarray(result.time, dtype=np.float64) * 3600.0, np.asarray(result.distance, dtype=np.float64)


# --------------------------------------------------------------------------- geocoding


def geocode(
    query: str,
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
    user_agent: str | None = None,
    timeout: float = 10,
    base_url: str | None = None,
) -> Bunch:
    """Coordinates of one address or place name, from Nominatim (OpenStreetMap) or Google.

    Parameters
    ----------
    query : str
        Free-text address or place, e.g. ``"Calle Ramón y Cajal 18, Leganés"``.
    provider : {"nominatim", "google"}, default="nominatim"
        ``"nominatim"`` uses the public OpenStreetMap geocoder (no key; usage policy:
        a descriptive ``User-Agent`` and at most one request per second -- both
        enforced here, the throttle by sleeping between consecutive calls);
        ``"google"`` uses the Geocoding API (needs ``api_key``; billed to your account).
    api_key : str, optional
        Google Cloud key with the Geocoding API enabled; required by ``"google"``.
    user_agent : str, optional
        ``User-Agent`` header; default ``scikit-route/<version> (+repository URL)``.
        For heavy use, Nominatim asks for one that identifies *your* application.
    timeout : float, default=10
        Seconds to wait for the HTTP answer.
    base_url : str, optional
        Server: default ``https://nominatim.openstreetmap.org`` (a self-hosted Nominatim
        has the same ``/search`` endpoint) or the Google Geocoding JSON endpoint.

    Returns
    -------
    Bunch
        ``lat`` and ``lon`` (floats, decimal degrees), ``display_name`` (the address
        the service resolved) and ``raw`` (the first result as returned by the
        service).

    Raises
    ------
    ValueError
        Empty query, unknown provider, missing key, or **no result** for the query
        (``"no result for ..."``).
    MapServiceError
        The server could not be reached after the retries or answered an error.

    Notes
    -----
    Nominatim is queried as ``GET {base_url}/search?format=jsonv2&limit=1&q=<query>``
    and the first hit is returned (Nominatim ranks by importance); Google as
    ``GET {base_url}?address=<query>&key=<api_key>`` with ``results[0].geometry.location``.
    Results from Nominatim are OpenStreetMap data (ODbL): keep the attribution
    "© OpenStreetMap contributors" when you show them.

    Examples
    --------
    >>> from skroute.preprocessing import geocode
    >>> office = geocode("Calle Ramón y Cajal 18, Leganés")  # doctest: +SKIP
    >>> office  # doctest: +SKIP
    Bunch(display_name, lat, lon, raw)
    >>> round(office.lat, 4), round(office.lon, 4)  # doctest: +SKIP
    (40.3296, -3.7373)
    >>> office.display_name  # doctest: +SKIP
    '18, Calle de Ramón y Cajal, ..., Leganés, Comunidad de Madrid, 28916, España'
    """
    text = _check_text(query, "query")
    _check_provider(provider, _GEOCODE_PROVIDERS)
    http_timeout = _check_timeout(timeout)
    headers = _headers(user_agent)
    if provider == "nominatim":
        return _geocode_nominatim(text, base_url=base_url, headers=headers, timeout=http_timeout)
    return _geocode_google(text, api_key=api_key, base_url=base_url, headers=headers, timeout=http_timeout)


def _throttle_nominatim() -> None:
    """Sleep until a second has passed since the previous Nominatim request, then stamp this one."""
    global _last_nominatim_call
    now = _monotonic()
    if _last_nominatim_call is not None:
        wait = NOMINATIM_MIN_INTERVAL - (now - _last_nominatim_call)
        if wait > 0:
            _sleep(wait)
            now = _monotonic()
    _last_nominatim_call = now


def _geocode_nominatim(query: str, *, base_url: str | None, headers: dict[str, str], timeout: float) -> Bunch:
    base = _check_text(base_url, "base_url").rstrip("/") if base_url is not None else NOMINATIM_URL
    _throttle_nominatim()
    answer = _get_json(
        f"{base}/search",
        params={"format": "jsonv2", "limit": 1, "q": query},
        headers=headers,
        timeout=timeout,
    )
    if not isinstance(answer, list):
        raise MapServiceError(f"Nominatim answered an unexpected document: {str(answer)[:200]!r}", status=200)
    if not answer:
        raise ValueError(f"no result for {query!r}")
    hit = answer[0]
    try:
        lat, lon = float(hit["lat"]), float(hit["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MapServiceError(
            f"Nominatim result without coordinates: {str(hit)[:200]!r}", status=200
        ) from exc
    return Bunch(lat=lat, lon=lon, display_name=str(hit.get("display_name", "")), raw=hit)


def _geocode_google(
    query: str, *, api_key: str | None, base_url: str | None, headers: dict[str, str], timeout: float
) -> Bunch:
    if api_key is None or not str(api_key).strip():
        raise ValueError("provider='google' needs api_key: a Google Cloud key with the Geocoding API enabled")
    base = _check_text(base_url, "base_url") if base_url is not None else GOOGLE_GEOCODING_URL
    answer = _get_json(base, params={"address": query, "key": str(api_key)}, headers=headers, timeout=timeout)
    status = answer.get("status") if isinstance(answer, dict) else None
    results = answer.get("results") if isinstance(answer, dict) else None
    if status == "ZERO_RESULTS" or (status == "OK" and not results):
        raise ValueError(f"no result for {query!r}")
    if status != "OK" or not isinstance(results, list):
        detail = str(answer.get("error_message", "")) if isinstance(answer, dict) else ""
        raise MapServiceError(
            f"Google Geocoding answered {status!r} {detail}".rstrip(), status=200, body=detail
        )
    hit = results[0]
    try:
        location = hit["geometry"]["location"]
        lat, lon = float(location["lat"]), float(location["lng"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MapServiceError(f"Google result without coordinates: {str(hit)[:200]!r}", status=200) from exc
    return Bunch(lat=lat, lon=lon, display_name=str(hit.get("formatted_address", "")), raw=hit)


# --------------------------------------------------------------------------- points of interest


def fetch_pois(
    area: str,
    *,
    brand: str | None = None,
    name: str | None = None,
    amenity: str | None = None,
    wikidata: str | None = None,
    provider: str = "overpass",
    base_url: str | None = None,
    timeout: float = 90,
    user_agent: str | None = None,
) -> Bunch:
    """Points of interest from OpenStreetMap: every element matching some tags inside an area.

    Parameters
    ----------
    area : str
        Name of an administrative area exactly as tagged in OpenStreetMap
        (``area["boundary"="administrative"]["name"=area]``): ``"Comunidad de Madrid"``,
        ``"Leganés"``, ``"Catalunya"``...
    brand : str, optional
        Exact ``brand`` tag (``"Burger King"``).
    name : str, optional
        Case-insensitive regular expression on the ``name`` tag (``"burger"``).
    amenity : str, optional
        Exact ``amenity`` tag (``"fast_food"``, ``"pharmacy"``, ``"townhall"``).
    wikidata : str, optional
        Exact ``brand:wikidata`` tag -- the most reliable brand filter, immune to
        spelling variants (``"Q177054"`` is Burger King).
    provider : {"overpass"}, default="overpass"
        Only the Overpass API is implemented.
    base_url : str, optional
        Overpass endpoint, default ``https://overpass-api.de/api/interpreter``.
    timeout : float, default=90
        Seconds: the ``[timeout:]`` of the query and the HTTP timeout.
    user_agent : str, optional
        ``User-Agent`` header; default ``scikit-route/<version> (+repository URL)``.

    Returns
    -------
    Bunch
        ``coords``: float64 ``(n, 2)`` array of ``(lat, lon)``; ``labels``: list of OSM
        ids ``"node/123"`` / ``"way/123"`` / ``"relation/123"``; ``names``: list of str
        (the ``name`` tag, else ``brand``, else ``""``); ``addresses``: list of str
        (``"<street> <housenumber>, <postcode> <city>"`` from the ``addr:*`` tags, ``""``
        when unknown); ``tags``: list of dicts (all the tags of each element);
        ``DESCR``: the query, the server, the count and the ODbL attribution. Elements
        are sorted by type and id -- i.e. by label -- so the order is deterministic.

    Raises
    ------
    ValueError
        Empty area, no filter given, or unknown provider.
    MapServiceError
        The server could not be reached after the retries, answered an error, or
        returned a ``remark`` (a query timeout or memory limit).

    Warns
    -----
    RuntimeWarning
        Elements without coordinates (a relation Overpass could not centre) are skipped.
    UserWarning
        Nothing matched: usually the area name does not match the OSM ``name`` tag
        of an administrative boundary.

    Notes
    -----
    One Overpass QL query, sent as ``GET {base_url}?data=<query>`` (the public server
    answers 406 to form-encoded POSTs)::

        [out:json][timeout:90];
        area["boundary"="administrative"]["name"="Comunidad de Madrid"]->.a;
        nwr["amenity"="fast_food"]["brand:wikidata"="Q177054"](area.a);
        out center;

    ``out center`` makes ways and relations (a restaurant mapped as a building) points at
    the centre of their bounding box; nodes keep their own position. Near-duplicates
    are **not** removed: a shop mapped both as a building way and as a node inside it
    appears twice. Drop them yourself when it matters, e.g. keep only nodes
    (``mask = [l.startswith("node/") for l in res.labels]``) or merge elements closer
    than a few metres with `skroute.preprocessing.haversine_matrix`. The data is
    OpenStreetMap's (ODbL): show "© OpenStreetMap contributors" with it.

    Examples
    --------
    >>> from skroute.preprocessing import fetch_pois
    >>> bk = fetch_pois("Leganés", amenity="fast_food", wikidata="Q177054")  # doctest: +SKIP
    >>> bk  # doctest: +SKIP
    Bunch(DESCR, addresses, coords, labels, names, tags)
    >>> bk.coords.shape, bk.labels[:2], bk.names[0]  # doctest: +SKIP
    ((6, 2), ['node/2613719490', 'node/2631338026'], 'Burger King')
    >>> print(bk.DESCR.splitlines()[-1])  # doctest: +SKIP
    Data © OpenStreetMap contributors, licensed under the Open Database License (ODbL): https://www.openstreetmap.org/copyright
    """
    area_name = _check_text(area, "area")
    _check_provider(provider, _POI_PROVIDERS)
    filters = {"amenity": amenity, "brand": brand, "name": name, "wikidata": wikidata}
    if all(value is None for value in filters.values()):
        raise ValueError("give at least one filter: brand=, name=, amenity= or wikidata=")
    http_timeout = _check_timeout(timeout)
    headers = _headers(user_agent)
    base = _check_text(base_url, "base_url") if base_url is not None else OVERPASS_URL
    query = _overpass_query(area_name, timeout=int(http_timeout), **filters)
    answer = _get_json(base, params={"data": query}, headers=headers, timeout=http_timeout)
    if not isinstance(answer, dict) or not isinstance(answer.get("elements"), list):
        raise MapServiceError(f"Overpass answered an unexpected document: {str(answer)[:200]!r}", status=200)
    if answer.get("remark"):
        remark = str(answer["remark"])
        raise MapServiceError(f"Overpass reported a problem: {remark}", status=200, url=base, body=remark)
    records, skipped = _parse_overpass_elements(answer["elements"])
    if skipped:
        warnings.warn(
            f"{skipped} Overpass elements without coordinates were skipped", RuntimeWarning, stacklevel=2
        )
    if not records:
        warnings.warn(
            f"no element matched in {area_name!r}: check that the area is the exact OSM name of an "
            "administrative boundary and that the filters are OSM tag values",
            UserWarning,
            stacklevel=2,
        )
    coords = np.array([[lat, lon] for _, _, lat, lon, _ in records], dtype=np.float64).reshape(-1, 2)
    labels = [f"{kind}/{osm_id}" for kind, osm_id, _, _, _ in records]
    tags = [element_tags for _, _, _, _, element_tags in records]
    names = [str(t.get("name") or t.get("brand") or "") for t in tags]
    addresses = [_format_address(t) for t in tags]
    descr = (
        f"{len(records)} OpenStreetMap elements inside the administrative area {area_name!r}, "
        f"fetched from {base} with the Overpass QL query:\n\n{query}\n\n"
        "Labels are OSM ids (node/123, way/123, relation/123); ways and relations are placed at the "
        "centre of their bounding box ('out center'). Near-duplicates (a building way and the shop node "
        "inside it) are kept.\n\n"
        f"{OSM_ATTRIBUTION}"
    )
    return Bunch(coords=coords, labels=labels, names=names, addresses=addresses, tags=tags, DESCR=descr)


def _ql_string(value: Any) -> str:
    """``value`` as a double-quoted Overpass QL string literal."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _overpass_query(
    area: str,
    *,
    timeout: int,
    amenity: str | None = None,
    brand: str | None = None,
    name: str | None = None,
    wikidata: str | None = None,
) -> str:
    filters = []
    if amenity is not None:
        filters.append(f'["amenity"={_ql_string(amenity)}]')
    if brand is not None:
        filters.append(f'["brand"={_ql_string(brand)}]')
    if name is not None:
        filters.append(f'["name"~{_ql_string(name)},i]')
    if wikidata is not None:
        filters.append(f'["brand:wikidata"={_ql_string(wikidata)}]')
    return (
        f"[out:json][timeout:{timeout}];\n"
        f'area["boundary"="administrative"]["name"={_ql_string(area)}]->.a;\n'
        f"nwr{''.join(filters)}(area.a);\n"
        "out center;"
    )


def _parse_overpass_elements(
    elements: list[Any],
) -> tuple[list[tuple[str, int, float, float, dict[str, str]]], int]:
    """``(type, id, lat, lon, tags)`` per located element, sorted by type and id; plus the skipped count."""
    records: list[tuple[str, int, float, float, dict[str, str]]] = []
    skipped = 0
    for element in elements:
        if not isinstance(element, dict):
            skipped += 1
            continue
        kind = str(element.get("type", ""))
        if kind == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:
            centre = element.get("center") or {}
            lat, lon = centre.get("lat"), centre.get("lon")
        osm_id = element.get("id")
        if lat is None or lon is None or osm_id is None or not kind:
            skipped += 1
            continue
        raw_tags = element.get("tags") or {}
        tags = {str(k): str(v) for k, v in raw_tags.items()} if isinstance(raw_tags, dict) else {}
        records.append((kind, int(osm_id), float(lat), float(lon), tags))
    records.sort(key=lambda r: (r[0], r[1]))
    return records, skipped


def _format_address(tags: Mapping[str, str]) -> str:
    """``"<street> <housenumber>, <postcode> <city>"`` from the ``addr:*`` tags; ``""`` when none is known."""
    street = " ".join(p for p in (tags.get("addr:street", ""), tags.get("addr:housenumber", "")) if p)
    city = " ".join(p for p in (tags.get("addr:postcode", ""), tags.get("addr:city", "")) if p)
    return ", ".join(p for p in (street, city) if p)
