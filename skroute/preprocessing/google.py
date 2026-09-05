"""Distance and duration matrices from the Google Distance Matrix API (optional ``googlemaps`` extra).

Using the API costs money: every request is billed to the account behind ``api_key``
(see https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing).
scikit-route has no relationship with Google; this module only spares you the
plumbing.
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from skroute.utils import Bunch
from skroute.utils.validation import coerce_labels

from ._convert import _as_label_list

__all__ = ["CostScraper", "GoogleDistanceMatrix"]

_log = logging.getLogger("skroute")

#: The Distance Matrix API accepts at most 100 elements (origins x destinations) per request.
_MAX_ELEMENTS_PER_REQUEST = 100

MODES = ("driving", "walking", "bicycling", "transit")


class GoogleDistanceMatrix:
    """Fetch road distances and travel times between coordinates from the Google Distance Matrix API.

    Parameters
    ----------
    api_key : str
        Credential of a Google Cloud project with the Distance Matrix API enabled.
    mode : {"driving", "walking", "bicycling", "transit"}, default="driving"
        Travel mode passed to the API.
    batch_size : int, default=10
        Origins and destinations are sent ``batch_size x batch_size`` per request
        (the API caps a request at 100 elements, hence ``1 <= batch_size <= 10``).
        The 1.0 ``CostScraper`` issued one request per pair -- 18 336 requests for
        the 192 Qatar nodes; with ``batch_size=10`` that is 400 requests.
    departure_time : "now", datetime or int, optional
        Forwarded to the API when given (``"now"``, a ``datetime`` or a Unix
        timestamp): the answer then carries ``duration_in_traffic``, which `fetch`
        prefers over ``duration`` for the ``time`` matrix. Traffic-aware requests are
        billed at the higher "Advanced" rate; ``mode`` must be ``"driving"`` for Google
        to honour it.

    Attributes
    ----------
    addresses_ : list of str
        After `fetch`, the address Google resolved for each node (from the
        ``origin_addresses`` of the responses; ``""`` when unresolved).
    n_requests_ : int
        After `fetch`, the number of requests issued.

    Notes
    -----
    ``googlemaps`` is imported lazily in ``__init__``; install it with
    ``pip install scikit-route[google]``. Progress is logged to the ``skroute`` logger
    at INFO (one record per request); enable it with
    ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.
    Elements the API cannot route (``status != "OK"``, or an ``"OK"`` element without
    a ``distance`` or ``duration`` value) become ``nan`` and are logged at WARNING;
    complete them before solving (every solver needs finite matrices).

    Examples
    --------
    >>> from skroute.preprocessing.google import GoogleDistanceMatrix
    >>> gdm = GoogleDistanceMatrix("<your key>")  # doctest: +SKIP
    >>> res = gdm.fetch([(41.3874, 2.1686), (41.5518, 2.2473)], labels=[10000007, 1])  # doctest: +SKIP
    >>> res.distance.shape, res.units  # doctest: +SKIP
    ((2, 2), {'distance': 'm', 'time': 'h'})
    """

    def __init__(
        self, api_key: str, mode: str = "driving", *, batch_size: int = 10, departure_time: Any = None
    ) -> None:
        try:
            import googlemaps
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("googlemaps is required: pip install scikit-route[google]") from exc
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}; got {mode!r}")
        if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool):
            raise ValueError(f"batch_size must be an integer in [1, 10]; got {batch_size!r}")
        if not 1 <= batch_size <= int(_MAX_ELEMENTS_PER_REQUEST**0.5):
            raise ValueError(f"batch_size must be an integer in [1, 10]; got {batch_size!r}")
        self.api_key = api_key
        self.mode = mode
        self.batch_size = int(batch_size)
        self.departure_time = departure_time
        self._client = googlemaps.Client(key=api_key)

    def fetch(self, coords: ArrayLike, labels: ArrayLike | None = None) -> Bunch:
        """Request the full ``(n, n)`` distance and duration matrices.

        Parameters
        ----------
        coords : array-like of shape (n, 2)
            ``(latitude, longitude)`` in decimal degrees.
        labels : sequence of length n, optional
            Unique node labels for the returned ``labels`` field; defaults to ``0..n-1``.

        Returns
        -------
        Bunch
            ``distance`` (metres) and ``time`` (hours; ``duration_in_traffic`` when the
            answer carries it, else ``duration``) as ``float64 (n, n)`` arrays,
            ``labels`` (``int64`` when every label is an integer, ``object`` otherwise
            -- the rule of `skroute.utils.validation.coerce_labels`) and
            ``units == {"distance": "m", "time": "h"}``. The matrices are directional
            (Google's durations are not symmetric).
        """
        xy = np.asarray(coords, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2); got {xy.shape}")
        n = xy.shape[0]
        if labels is None:
            lab = np.arange(n, dtype=np.int64)
        else:
            items = _as_label_list(labels)
            if len(items) != n:
                raise ValueError(f"labels must have length {n}; got {len(items)}")
            lab = coerce_labels(items, n)

        points = [(float(lat), float(lon)) for lat, lon in xy]
        distance = np.full((n, n), np.nan, dtype=np.float64)
        time = np.full((n, n), np.nan, dtype=np.float64)
        addresses = [""] * n
        bs = self.batch_size
        request_kwargs: dict[str, Any] = {"mode": self.mode}
        if self.departure_time is not None:
            request_kwargs["departure_time"] = self.departure_time
        starts = list(range(0, n, bs))
        total = len(starts) ** 2
        done = 0
        for i0 in starts:
            i1 = min(i0 + bs, n)
            for j0 in starts:
                j1 = min(j0 + bs, n)
                done += 1
                _log.info(
                    "GoogleDistanceMatrix: request %d/%d (%d x %d elements)", done, total, i1 - i0, j1 - j0
                )
                response = self._client.distance_matrix(points[i0:i1], points[j0:j1], **request_kwargs)
                self._fill(response, distance, time, addresses, i0, i1, j0, j1)
        self.addresses_ = addresses
        self.n_requests_ = done
        if np.isnan(distance).any():
            _log.warning(
                "GoogleDistanceMatrix: %d of %d elements could not be routed and are nan",
                int(np.isnan(distance).sum()),
                n * n,
            )
        return Bunch(distance=distance, time=time, labels=lab, units={"distance": "m", "time": "h"})

    @staticmethod
    def _fill(
        response: dict[str, Any],
        distance: np.ndarray,
        time: np.ndarray,
        addresses: list[str],
        i0: int,
        i1: int,
        j0: int,
        j1: int,
    ) -> None:
        if response.get("status", "OK") != "OK":
            _log.warning(
                "GoogleDistanceMatrix: response status %r for block [%d:%d, %d:%d]",
                response.get("status"),
                i0,
                i1,
                j0,
                j1,
            )
            return
        rows = response.get("rows", [])
        for k, origin_address in enumerate(response.get("origin_addresses", [])):
            if i0 + k < i1 and not addresses[i0 + k]:
                addresses[i0 + k] = str(origin_address or "")
        for r, row in enumerate(rows[: i1 - i0]):
            for c, element in enumerate(row.get("elements", [])[: j1 - j0]):
                if element.get("status") != "OK":
                    continue
                metres = _element_value(element, "distance")
                seconds = _element_value(element, "duration_in_traffic")  # present with departure_time
                if seconds is None:
                    seconds = _element_value(element, "duration")
                if metres is None or seconds is None:
                    # An "OK" element without both values is unroutable for us: a KeyError here would
                    # abort fetch() after the quota of the previous requests has been spent.
                    _log.warning(
                        "GoogleDistanceMatrix: element [%d, %d] has status OK but no distance/duration value",
                        i0 + r,
                        j0 + c,
                    )
                    continue
                distance[i0 + r, j0 + c] = metres
                time[i0 + r, j0 + c] = seconds / 3600.0


def _element_value(element: dict[str, Any], key: str) -> float | None:
    """``element[key]["value"]`` as a float, or ``None`` when absent or not a number."""
    field = element.get(key)
    value = field.get("value") if isinstance(field, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


class CostScraper:
    """Deprecated 1.0 interface over `GoogleDistanceMatrix`; removed in 3.0.

    Parameters
    ----------
    api : str
        API key.
    nodes : list of (id, latitude, longitude)
        One tuple per node, as in 1.0.
    mode : str, default="driving"
        Travel mode.

    Notes
    -----
    Emits ``DeprecationWarning`` on construction. ``scrap()`` returns the ``Bunch`` of
    `GoogleDistanceMatrix.fetch`; ``pandas()`` builds the 1.0 long table (one
    row per unordered pair, ``meters``/``seconds`` taken from the ``origin -> destination``
    direction); ``to_pickle()`` raises ``NotImplementedError``.
    """

    def __init__(self, api: str, nodes: list[tuple[Any, ...]], mode: str = "driving") -> None:
        warnings.warn(
            "CostScraper is deprecated since 2.0 and will be removed in 3.0; "
            "use GoogleDistanceMatrix(api_key).fetch(coords, labels)",
            DeprecationWarning,
            stacklevel=2,
        )
        self.labels = [n[0] for n in nodes]
        self.coords = [tuple(n[1:]) for n in nodes]
        self.mode = mode
        self._client = GoogleDistanceMatrix(api, mode)
        self.result_: Bunch | None = None

    def scrap(self) -> Bunch:
        """Fetch the matrices (billed to your account); the ``Bunch`` of `GoogleDistanceMatrix.fetch`.

        Stored under ``result_`` as well.
        """
        self.result_ = self._client.fetch(self.coords, labels=self.labels)
        return self.result_

    def pandas(self) -> Any:
        """The 1.0 long table as a ``DataFrame`` (requires pandas; calls `scrap` first if needed)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("pandas is required for pandas(): pip install scikit-route[pandas]") from exc
        res = self.result_ if self.result_ is not None else self.scrap()
        addresses = getattr(self._client, "addresses_", [""] * len(self.labels))
        records = []
        for i, j in combinations(range(len(self.labels)), 2):
            records.append(
                (
                    self.labels[i],
                    *self.coords[i],
                    addresses[i],
                    self.labels[j],
                    *self.coords[j],
                    addresses[j],
                    float(res.distance[i, j]),
                    float(res.time[i, j]) * 3600.0,
                )
            )
        columns = [
            "id_origin",
            "lat_origin",
            "lon_origin",
            "address_origin",
            "id_destinity",
            "lat_destinity",
            "lon_destinity",
            "address_destinity",
            "meters",
            "seconds",
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    def to_pickle(self, filename: str | None = None) -> None:
        """Removed in 2.0."""
        raise NotImplementedError("to_pickle was removed in 2.0; use pandas().to_pickle(...)")
