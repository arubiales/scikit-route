"""Dense distance matrices from coordinates: Euclidean, Manhattan, TSPLIB 95 and haversine."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "distance_matrix",
    "euclidean_matrix",
    "haversine_matrix",
    "tsplib_nint",
]

#: Mean Earth radius (IUGG), in kilometres, used by the ``"haversine"`` metric.
EARTH_RADIUS_KM = 6371.0088

#: TSPLIB 95 constants of the ``GEO`` edge-weight type (kept as printed in the standard).
_TSPLIB_PI = 3.141592
_TSPLIB_RRR = 6378.388

#: Above this many nodes a dense ``float64`` matrix exceeds ~3.2 GB and a warning is emitted.
_LARGE_N = 20_000

METRICS = (
    "euclidean",
    "manhattan",
    "tsplib_euc_2d",
    "tsplib_ceil_2d",
    "tsplib_man_2d",
    "tsplib_att",
    "tsplib_geo",
    "haversine",
)


def tsplib_nint(x: ArrayLike) -> NDArray[np.float64] | float:
    """Round to the nearest integer the TSPLIB 95 way: ``nint(x) = floor(x + 0.5)``.

    Half-integers round *up* (``2.5 -> 3.0``), unlike :func:`numpy.rint`, which rounds
    half to even (``2.5 -> 2.0``). The difference is not academic: with ``np.rint`` the
    Waterloo instance ``qa194`` evaluates its published optimal tour to 9351 instead of
    9352 (SPEC D15).

    Parameters
    ----------
    x : array-like of float
        Values to round. A scalar returns a Python ``float``; an array returns an
        array of the same shape.

    Returns
    -------
    ndarray of float64 or float
        ``floor(x + 0.5)`` as floating-point integers (the dtype stays ``float64`` so
        the result can be stored straight into a cost matrix).

    Examples
    --------
    >>> from skroute.preprocessing import tsplib_nint
    >>> tsplib_nint(2.5), tsplib_nint(-2.5), tsplib_nint(2.49)
    (3.0, -2.0, 2.0)
    >>> tsplib_nint([0.5, 1.5, 2.5]).tolist()
    [1.0, 2.0, 3.0]
    """
    out = np.floor(np.asarray(x, dtype=np.float64) + 0.5)
    if out.ndim == 0:
        return float(out)
    return out


def _check_coords(coords: ArrayLike) -> NDArray[np.float64]:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2); got {xy.shape}")
    if xy.shape[0] == 0:
        raise ValueError("coords is empty: at least one node is required")
    if not np.all(np.isfinite(xy)):
        raise ValueError("coords must be finite (no NaN or inf)")
    return np.ascontiguousarray(xy)


def _geo_radians(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """TSPLIB ``GEO``: ``DDD.MM`` degrees-and-minutes to radians with ``PI = 3.141592``.

    The integer part is taken by truncation (``deg = (int) x``, as in Concorde and the
    published optima -- ``ulysses16.opt.tour`` evaluates to 6859 this way and to 6917
    with ``nint``), not by ``nint`` as the TSPLIB 95 booklet literally prints.
    """
    deg = np.trunc(values)
    minutes = values - deg
    return _TSPLIB_PI * (deg + 5.0 * minutes / 3.0) / 180.0


def _block_euclidean(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def _block_manhattan(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.abs(a[:, None, 0] - b[None, :, 0]) + np.abs(a[:, None, 1] - b[None, :, 1])


def _block_tsplib_euc_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.floor(_block_euclidean(a, b) + 0.5)


def _block_tsplib_ceil_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.ceil(_block_euclidean(a, b))


def _block_tsplib_man_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.floor(_block_manhattan(a, b) + 0.5)


def _block_tsplib_att(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    r = np.sqrt((dx * dx + dy * dy) / 10.0)
    t = np.floor(r + 0.5)
    return np.where(t < r, t + 1.0, t)


def _block_tsplib_geo(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    # ``a`` and ``b`` are already in TSPLIB radians, columns (lat, lon).
    q1 = np.cos(a[:, None, 1] - b[None, :, 1])
    q2 = np.cos(a[:, None, 0] - b[None, :, 0])
    q3 = np.cos(a[:, None, 0] + b[None, :, 0])
    arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    np.clip(arg, -1.0, 1.0, out=arg)
    return np.trunc(_TSPLIB_RRR * np.arccos(arg) + 1.0)


def _block_haversine(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    # ``a`` and ``b`` are in radians, columns (lat, lon).
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    h = np.sin(dlat / 2.0) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2.0) ** 2
    np.clip(h, 0.0, 1.0, out=h)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


_BLOCK_FUNCTIONS: dict[str, Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]] = {
    "euclidean": _block_euclidean,
    "manhattan": _block_manhattan,
    "tsplib_euc_2d": _block_tsplib_euc_2d,
    "tsplib_ceil_2d": _block_tsplib_ceil_2d,
    "tsplib_man_2d": _block_tsplib_man_2d,
    "tsplib_att": _block_tsplib_att,
    "tsplib_geo": _block_tsplib_geo,
    "haversine": _block_haversine,
}


def distance_matrix(
    coords: ArrayLike,
    metric: str = "euclidean",
    *,
    block_size: int = 2048,
) -> NDArray[np.float64]:
    """Dense ``(n, n)`` distance matrix of a set of points.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        One row per node. For ``"euclidean"``, ``"manhattan"`` and the planar TSPLIB
        metrics the columns are ``(x, y)`` exactly as in a TSPLIB ``NODE_COORD_SECTION``.
        For ``"tsplib_geo"`` the columns are ``(latitude, longitude)`` in the TSPLIB
        ``DDD.MM`` degrees-and-minutes notation; for ``"haversine"`` they are decimal
        degrees ``(latitude, longitude)``.
    metric : {"euclidean", "manhattan", "tsplib_euc_2d", "tsplib_ceil_2d", \
"tsplib_man_2d", "tsplib_att", "tsplib_geo", "haversine"}, default="euclidean"
        Distance definition; the TSPLIB metrics reproduce the TSPLIB 95 edge-weight
        types ``EUC_2D``, ``CEIL_2D``, ``MAN_2D``, ``ATT`` and ``GEO`` exactly (see Notes).
    block_size : int, default=2048
        Number of rows computed per block. Bounds the peak memory of the temporaries
        to a few ``block_size x n`` arrays instead of ``n x n``; the result itself is
        always the full ``(n, n)`` matrix.

    Returns
    -------
    ndarray of shape (n, n), dtype float64
        C-contiguous, symmetric, with a zero diagonal.

    Warns
    -----
    UserWarning
        When ``n > 20_000``: the dense matrix alone needs more than 3.2 GB.

    Notes
    -----
    Every TSPLIB metric follows the TSPLIB 95 definitions with
    ``nint(x) = floor(x + 0.5)`` (:func:`tsplib_nint`, never :func:`numpy.rint`):

    * ``tsplib_euc_2d``: ``nint(sqrt(dx**2 + dy**2))``
    * ``tsplib_ceil_2d``: ``ceil(sqrt(dx**2 + dy**2))``
    * ``tsplib_man_2d``: ``nint(|dx| + |dy|)``
    * ``tsplib_att``: ``r = sqrt((dx**2 + dy**2) / 10); t = nint(r); d = t + 1 if t < r else t``
    * ``tsplib_geo``: each ``DDD.MM`` coordinate is converted with ``deg = int(x); m = x
      - deg; rad = PI * (deg + 5 m / 3) / 180`` where ``PI = 3.141592`` (``int`` is
      truncation, the convention of Concorde and of the published optima: the TSPLIB
      booklet prints ``nint`` there, which evaluates ``ulysses16.opt.tour`` to 6917
      instead of 6859); then with ``RRR = 6378.388``, ``q1 = cos(lon_i - lon_j)``,
      ``q2 = cos(lat_i - lat_j)``, ``q3 = cos(lat_i + lat_j)`` the distance is
      ``int(RRR * acos(0.5 * ((1 + q1) q2 - (1 - q1) q3)) + 1)`` -- truncation plus
      one, not ``nint``.

    ``haversine`` uses the great-circle formula on a sphere of radius 6371.0088 km and
    returns kilometres. The diagonal is set to zero for every metric (``GEO`` would
    otherwise give 1 there); no solver ever reads it.

    Complexity is O(n**2) time and memory for the result.

    References
    ----------
    G. Reinelt, *TSPLIB 95*, Universitaet Heidelberg, 1995, section 2.1.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.preprocessing import distance_matrix
    >>> xy = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    >>> distance_matrix(xy).tolist()
    [[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]]
    >>> distance_matrix(xy, metric="manhattan")[0].tolist()
    [0.0, 7.0, 14.0]
    >>> distance_matrix([[0.0, 0.0], [1.0, 1.0]], metric="tsplib_euc_2d")[0, 1]   # nint(1.414...)
    1.0
    >>> distance_matrix([[0.0, 0.0], [1.0, 1.0]], metric="tsplib_ceil_2d")[0, 1]
    2.0
    """
    if metric not in _BLOCK_FUNCTIONS:
        raise ValueError(f"metric must be one of {list(METRICS)}; got {metric!r}")
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError(f"block_size must be a positive integer; got {block_size!r}")
    xy = _check_coords(coords)
    n = xy.shape[0]
    if n > _LARGE_N:
        gib = n * n * 8 / 1024**3
        warnings.warn(
            f"building a dense {n} x {n} float64 matrix ({gib:.1f} GiB); scikit-route 2.0 solves "
            "only dense matrices, consider subsampling (e.g. load_tsp(name, n_nodes=...))",
            UserWarning,
            stacklevel=2,
        )

    if metric == "tsplib_geo":
        xy = _geo_radians(xy)
    elif metric == "haversine":
        xy = np.radians(xy)

    block = _BLOCK_FUNCTIONS[metric]
    out = np.empty((n, n), dtype=np.float64)
    step = int(block_size)
    for start in range(0, n, step):
        stop = min(start + step, n)
        out[start:stop, :] = block(xy[start:stop], xy)
    np.fill_diagonal(out, 0.0)
    return out


def euclidean_matrix(coords: ArrayLike) -> NDArray[np.float64]:
    """Plain Euclidean distance matrix; shorthand for ``distance_matrix(coords, "euclidean")``.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Planar ``(x, y)`` coordinates.

    Returns
    -------
    ndarray of shape (n, n), dtype float64

    Examples
    --------
    >>> from skroute.preprocessing import euclidean_matrix
    >>> euclidean_matrix([[0.0, 0.0], [0.0, 2.0]]).tolist()
    [[0.0, 2.0], [2.0, 0.0]]
    """
    return distance_matrix(coords, "euclidean")


def haversine_matrix(latlon: ArrayLike) -> NDArray[np.float64]:
    """Great-circle distances in kilometres; shorthand for ``distance_matrix(latlon, "haversine")``.

    Parameters
    ----------
    latlon : array-like of shape (n, 2)
        ``(latitude, longitude)`` in decimal degrees.

    Returns
    -------
    ndarray of shape (n, n), dtype float64
        Kilometres on a sphere of radius 6371.0088 km.

    Examples
    --------
    >>> from skroute.preprocessing import haversine_matrix
    >>> madrid, barcelona = (40.4168, -3.7038), (41.3874, 2.1686)
    >>> D = haversine_matrix([madrid, barcelona])
    >>> bool(500 < D[0, 1] < 510)  # about 505 km
    True
    """
    return distance_matrix(latlon, "haversine")
