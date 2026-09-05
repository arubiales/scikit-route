"""Tests of ``skroute.preprocessing``: metrics, pivots, dict conversions, normalisation, Google client."""

from __future__ import annotations

import logging
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import preprocessing
from skroute.datasets import read_tsplib, read_tsplib_tour
from skroute.preprocessing import (
    _distances,
    distance_matrix,
    euclidean_matrix,
    from_dict_of_dicts,
    haversine_matrix,
    normalize_coords,
    pairs_to_matrix,
    to_dict_of_dicts,
    tsplib_nint,
)
from skroute.utils import Bunch

DATA = Path(__file__).parent / "data"
METRICS = [
    "euclidean",
    "manhattan",
    "tsplib_euc_2d",
    "tsplib_ceil_2d",
    "tsplib_man_2d",
    "tsplib_att",
    "tsplib_geo",
    "haversine",
]
MADRID, BARCELONA = (40.4168, -3.7038), (41.3874, 2.1686)


def _closed_cost(C: np.ndarray, ids: np.ndarray) -> float:
    idx = np.asarray(ids) - 1
    return float(sum(C[idx[k], idx[(k + 1) % len(idx)]] for k in range(len(idx))))


def _coords_for(metric: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if metric in ("tsplib_geo", "haversine"):
        return np.column_stack([rng.uniform(-80, 80, n), rng.uniform(-179, 179, n)])
    return rng.uniform(-1000, 1000, (n, 2))


# --------------------------------------------------------------------------- distance_matrix: generic


@pytest.mark.parametrize("metric", METRICS)
def test_distance_matrix_shape_symmetry_and_diagonal(metric):
    xy = _coords_for(metric, 37, seed=1)
    D = distance_matrix(xy, metric)
    assert D.shape == (37, 37) and D.dtype == np.float64 and D.flags.c_contiguous
    assert np.all(np.isfinite(D)) and np.all(D >= 0)
    assert np.array_equal(D, D.T)
    assert np.all(np.diagonal(D) == 0.0)
    if metric.startswith("tsplib"):
        assert np.array_equal(D, np.floor(D)), "TSPLIB metrics are integer-valued"


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("block_size", [1, 3, 7, 5000])
def test_block_size_does_not_change_the_result(metric, block_size):
    xy = _coords_for(metric, 23, seed=2)
    np.testing.assert_array_equal(
        distance_matrix(xy, metric, block_size=block_size), distance_matrix(xy, metric)
    )


def test_distance_matrix_accepts_lists_int_arrays_and_fortran_order():
    pts = [[0, 0], [3, 4], [6, 8]]
    expected = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]])
    np.testing.assert_array_equal(distance_matrix(pts), expected)
    np.testing.assert_array_equal(distance_matrix(np.array(pts, dtype=np.int32)), expected)
    np.testing.assert_array_equal(distance_matrix(np.asfortranarray(np.array(pts, dtype=float))), expected)
    assert distance_matrix([[1.0, 1.0]]).tolist() == [[0.0]]


@pytest.mark.parametrize(
    ("coords", "match"),
    [
        ([[0.0, 0.0, 0.0]], r"shape \(n, 2\)"),
        ([1.0, 2.0], r"shape \(n, 2\)"),
        (np.empty((0, 2)), "empty"),
        ([[0.0, np.nan]], "finite"),
        ([[np.inf, 0.0]], "finite"),
    ],
)
def test_distance_matrix_rejects_bad_coordinates(coords, match):
    with pytest.raises(ValueError, match=match):
        distance_matrix(coords)


def test_distance_matrix_rejects_unknown_metric_and_bad_block_size():
    with pytest.raises(ValueError, match=r"metric must be one of .*'haversine'.*; got 'cosine'"):
        distance_matrix([[0.0, 0.0], [1.0, 1.0]], metric="cosine")
    for bad in (0, -3, 2.5, True, "8"):
        with pytest.raises(ValueError, match="block_size must be a positive integer"):
            distance_matrix([[0.0, 0.0], [1.0, 1.0]], block_size=bad)


def test_warns_above_the_dense_ceiling(monkeypatch):
    monkeypatch.setattr(_distances, "_LARGE_N", 5)
    xy = _coords_for("euclidean", 6, seed=3)
    with pytest.warns(UserWarning, match=r"dense 6 x 6 float64 matrix \(0\.0 GB\)") as record:
        D = distance_matrix(xy)
    assert D.shape == (6, 6)
    assert record[0].filename == __file__, "attributed to the caller"
    # through the shorthands the warning must still name the caller, not skroute/preprocessing/_distances.py
    with pytest.warns(UserWarning) as record:
        euclidean_matrix(xy)
    assert record[0].filename == __file__
    with pytest.warns(UserWarning) as record:
        haversine_matrix(_coords_for("haversine", 6, seed=3))
    assert record[0].filename == __file__


def test_size_warning_quotes_decimal_gigabytes(monkeypatch):
    # 20 001 nodes: 20001**2 * 8 bytes = 3.2 GB (decimal), the unit of every size quoted in the package
    monkeypatch.setattr(_distances, "_LARGE_N", 100)
    xy = _coords_for("euclidean", 101, seed=5)
    with pytest.warns(UserWarning, match=r"\(0\.0 GB\)"):
        distance_matrix(xy)
    assert f"{20001 * 20001 * 8 / 1e9:.1f}" == "3.2"


def test_no_warning_below_the_ceiling(recwarn):
    distance_matrix(_coords_for("euclidean", 50, seed=4))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# --------------------------------------------------------------------------- distance_matrix: metrics


def test_euclidean_and_manhattan_known_values():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(distance_matrix(xy), [[0, 5, 10], [5, 0, 5], [10, 5, 0]])
    np.testing.assert_array_equal(distance_matrix(xy, "manhattan"), [[0, 7, 14], [7, 0, 7], [14, 7, 0]])
    np.testing.assert_array_equal(euclidean_matrix(xy), distance_matrix(xy, "euclidean"))
    assert distance_matrix([[0.0, 0.0], [1.0, 1.0]])[0, 1] == pytest.approx(math.sqrt(2.0))


def test_tsplib_euc_2d_rounds_half_up_where_rint_rounds_half_to_even():
    xy = np.array(
        [[0.0, 0.0], [1.5, 2.0], [0.0, -2.5], [-3.0, 4.0]]
    )  # distances 2.5, 2.5 and 5 from the origin
    D = distance_matrix(xy, "tsplib_euc_2d")
    assert D[0].tolist() == [0.0, 3.0, 3.0, 5.0]
    plain = distance_matrix(xy, "euclidean")
    assert np.rint(plain)[0].tolist() == [0.0, 2.0, 2.0, 5.0], "np.rint would evaluate the half-integers down"
    np.testing.assert_array_equal(D, np.floor(plain + 0.5))
    np.testing.assert_array_equal(D, tsplib_nint(plain))


def test_tsplib_ceil_2d_and_man_2d():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 4.0], [1.2, 1.3]])
    ceil = distance_matrix(xy, "tsplib_ceil_2d")
    assert ceil[0].tolist() == [0.0, 2.0, 5.0, 2.0]
    man = distance_matrix(xy, "tsplib_man_2d")
    assert man[0].tolist() == [0.0, 2.0, 7.0, 3.0], "nint(|dx| + |dy|): 2.5 rounds to 3"
    assert distance_matrix(xy, "manhattan")[0, 3] == pytest.approx(2.5)


def test_tsplib_att_hand_computed_pairs():
    # r = sqrt(d2 / 10); t = nint(r); d = t + 1 if t < r else t
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0], [30.0, 40.0], [10.0, 10.0]])
    D = distance_matrix(xy, "tsplib_att")
    # sqrt(10) = 3.162 -> t = 3 < r -> 4 ; sqrt(40) = 6.325 -> 6 < r -> 7 ;
    # sqrt(250) = 15.81 -> 16 ; sqrt(20) = 4.47 -> 4 < r -> 5
    assert D[0].tolist() == [0.0, 4.0, 7.0, 16.0, 5.0]


def _geo_reference(a: tuple[float, float], b: tuple[float, float]) -> int:
    """TSPLIB 95 GEO as Concorde evaluates it (integer part by truncation), written independently."""
    PI, RRR = 3.141592, 6378.388

    def rad(x: float) -> float:
        deg = int(x)
        return PI * (deg + 5.0 * (x - deg) / 3.0) / 180.0

    lat_i, lon_i, lat_j, lon_j = rad(a[0]), rad(a[1]), rad(b[0]), rad(b[1])
    q1 = math.cos(lon_i - lon_j)
    q2 = math.cos(lat_i - lat_j)
    q3 = math.cos(lat_i + lat_j)
    return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


def test_tsplib_geo_matches_a_hand_computation():
    b = read_tsplib(DATA / "geo_small.tsp")
    D = distance_matrix(b.coords, "tsplib_geo")
    pts = [tuple(p) for p in b.coords.tolist()]
    for i in range(4):
        for j in range(4):
            if i != j:
                assert D[i, j] == _geo_reference(pts[i], pts[j])
    assert D[0, 1] == 509.0, "Ithaca -> Troy in ulysses16, 509 km"
    # a negative longitude (west of Greenwich) and an exact half-degree, where nint and truncation differ
    p, q = (36.08, -5.21), (37.51, 15.17)
    assert distance_matrix([p, q], "tsplib_geo")[0, 1] == _geo_reference(p, q)


def test_geo_is_truncation_plus_one_not_nint():
    PI, RRR = 3.141592, 6378.388

    def raw_geo(a, b):  # the un-rounded RRR * acos(...) of the TSPLIB formula
        def rad(x):
            deg = int(x)
            return PI * (deg + 5.0 * (x - deg) / 3.0) / 180.0

        q1 = math.cos(rad(a[1]) - rad(b[1]))
        q2 = math.cos(rad(a[0]) - rad(b[0]))
        q3 = math.cos(rad(a[0]) + rad(b[0]))
        return RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3))

    origin = (10.00, 10.00)
    checked_below_half = 0
    for minutes in range(1, 60):
        other = (10.00, 10.0 + minutes / 100.0)
        raw = raw_geo(origin, other)
        d = distance_matrix([origin, other], "tsplib_geo")[0, 1]
        assert d == math.floor(raw + 1.0), "int(RRR * acos(...) + 1.0): truncation plus one"
        if raw - math.floor(raw) < 0.5:
            assert d == math.floor(raw + 0.5) + 1, "one more than nint would give"
            checked_below_half += 1
    assert checked_below_half > 10


def test_ulysses16_optimal_tour_evaluates_to_6859_under_geo():
    b = read_tsplib(DATA / "ulysses16.tsp")
    tour = read_tsplib_tour(DATA / "ulysses16.opt.tour")
    assert _closed_cost(distance_matrix(b.coords, "tsplib_geo"), tour) == 6859.0


def test_att48_optimal_tour_evaluates_to_10628_under_att():
    b = read_tsplib(DATA / "att48.tsp")
    tour = read_tsplib_tour(DATA / "att48.opt.tour")
    assert _closed_cost(distance_matrix(b.coords, "tsplib_att"), tour) == 10628.0


def test_haversine_known_distances():
    D = haversine_matrix([MADRID, BARCELONA])
    assert D[0, 1] == pytest.approx(505.0, rel=0.01), "Madrid - Barcelona is about 505 km"
    np.testing.assert_array_equal(D, distance_matrix([MADRID, BARCELONA], "haversine"))
    one_degree = distance_matrix([[0.0, 0.0], [0.0, 1.0]], "haversine")[0, 1]
    assert one_degree == pytest.approx(2 * math.pi * 6371.0088 / 360.0, rel=1e-9)
    antipodes = distance_matrix([[0.0, 0.0], [0.0, 180.0]], "haversine")[0, 1]
    assert antipodes == pytest.approx(math.pi * 6371.0088, rel=1e-9)
    assert distance_matrix([[51.5, -0.1], [51.5, -0.1]], "haversine")[0, 1] == 0.0
    assert _distances.EARTH_RADIUS_KM == 6371.0088


# --------------------------------------------------------------------------- tsplib_nint


def test_tsplib_nint_scalars_and_arrays():
    assert tsplib_nint(2.5) == 3.0 and isinstance(tsplib_nint(2.5), float)
    assert tsplib_nint(-2.5) == -2.0
    assert tsplib_nint(2.49) == 2.0 and tsplib_nint(-0.5) == 0.0 and tsplib_nint(0.49999) == 0.0
    out = tsplib_nint([[0.5, 1.5], [2.5, -1.5]])
    assert isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, -1.0]]
    assert np.rint(2.5) == 2.0, "the numpy default rounds half to even"


@settings(derandomize=True, deadline=None, max_examples=200)
@given(st.integers(min_value=-10_000, max_value=10_000))
def test_tsplib_nint_rounds_every_half_integer_up(k):
    x = k + 0.5
    assert tsplib_nint(x) == k + 1
    assert tsplib_nint(x) == math.floor(x + 0.5)


@settings(derandomize=True, deadline=None, max_examples=200)
@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_tsplib_nint_is_floor_of_x_plus_half(x):
    assert tsplib_nint(x) == math.floor(x + 0.5)


# --------------------------------------------------------------------------- pairs_to_matrix


def test_pairs_to_matrix_first_appearance_order_and_mirroring():
    M, labels = pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0])
    assert labels.tolist() == [1, 2, 3] and labels.dtype == np.int64
    assert M.dtype == np.float64 and M.shape == (3, 3)
    assert M.tolist() == [[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]]
    # order of the rows is irrelevant (the 1.0 dfcolumn_to_dict depended on it)
    M2, labels2 = pairs_to_matrix([2, 1, 1], [3, 3, 2], [4.0, 9.0, 5.0], labels=labels)
    np.testing.assert_array_equal(M2, M)
    np.testing.assert_array_equal(labels2, labels)
    M3, labels3 = pairs_to_matrix([2, 1, 1], [3, 3, 2], [4.0, 9.0, 5.0])
    assert labels3.tolist() == [2, 3, 1], "first appearance scans origin then destination"
    np.testing.assert_array_equal(M3, M[np.ix_([1, 2, 0], [1, 2, 0])])


def test_pairs_to_matrix_directional_and_both_directions_given():
    M, labels = pairs_to_matrix(["a", "b"], ["b", "a"], [1.0, 2.0], symmetric=False)
    assert labels.tolist() == ["a", "b"]
    assert M.tolist() == [[0.0, 1.0], [2.0, 0.0]]
    M, _ = pairs_to_matrix(["a", "b"], ["b", "a"], [1.0, 2.0], symmetric=True)
    assert M.tolist() == [[0.0, 1.0], [2.0, 0.0]], "entries given in both directions are kept as given"


def test_pairs_to_matrix_diagonal_defaults_to_zero_and_explicit_diagonal_is_kept():
    M, _ = pairs_to_matrix([1, 1, 2, 3], [1, 2, 3, 3], [7.0, 5.0, 4.0, 9.0], fill=0.0)
    assert M[0, 0] == 7.0 and M[1, 1] == 0.0 and M[2, 2] == 9.0


def test_pairs_to_matrix_missing_entries():
    with pytest.raises(ValueError, match=r"2 entries are missing .* \(1, 3\); pass fill="):
        pairs_to_matrix([1, 2], [2, 3], [1.0, 1.0])
    with pytest.raises(ValueError, match="or symmetric=True to mirror"):
        pairs_to_matrix([1, 1, 2], [2, 3, 3], [1.0, 1.0, 1.0], symmetric=False)
    M, _ = pairs_to_matrix([1, 2], [2, 3], [1.0, 1.0], fill=np.inf)
    assert M[0, 2] == np.inf and M[2, 0] == np.inf and M[0, 1] == 1.0 and M[1, 0] == 1.0
    M, _ = pairs_to_matrix([1, 1, 2], [2, 3, 3], [1.0, 2.0, 3.0], symmetric=False, fill=-1.0)
    assert M.tolist() == [[0.0, 1.0, 2.0], [-1.0, 0.0, 3.0], [-1.0, -1.0, 0.0]]


def test_pairs_to_matrix_given_labels():
    M, labels = pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0], labels=[3, 2, 1])
    assert labels.tolist() == [3, 2, 1]
    assert M.tolist() == [[0.0, 4.0, 9.0], [4.0, 0.0, 5.0], [9.0, 5.0, 0.0]]
    M, labels = pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0], labels=[1, 2, 3, 4], fill=0.0)
    assert M.shape == (4, 4) and labels.tolist() == [1, 2, 3, 4] and M[3].tolist() == [0.0] * 4
    with pytest.raises(ValueError, match="label 3 is not in labels"):
        pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0], labels=[1, 2])
    with pytest.raises(ValueError, match="labels must be unique"):
        pairs_to_matrix([1], [2], [1.0], labels=[1, 2, 2])


def test_pairs_to_matrix_errors_and_edge_cases():
    with pytest.raises(ValueError, match="same length; got 2, 1 and 2"):
        pairs_to_matrix([1, 2], [2], [1.0, 2.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        pairs_to_matrix([[1, 2]], [[2, 3]], [1.0])
    M, labels = pairs_to_matrix([1, 1], [2, 2], [1.0, 2.0])
    assert M[0, 1] == 2.0, "the last duplicate wins"
    M, labels = pairs_to_matrix([5], [5], [3.0])
    assert M.tolist() == [[3.0]] and labels.tolist() == [5]
    M, labels = pairs_to_matrix(np.array([1, 1, 2]), np.array([2, 3, 3]), np.array([5, 9, 4]))
    assert labels.dtype == np.int64 and M.dtype == np.float64


def test_pairs_to_matrix_accepts_dataframe_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"o": [1, 1, 2], "d": [2, 3, 3], "v": [5.0, 9.0, 4.0]})
    M, labels = pairs_to_matrix(df["o"], df["d"], df["v"])
    assert labels.tolist() == [1, 2, 3] and labels.dtype == np.int64
    assert M.tolist() == [[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]]
    mixed = pd.DataFrame({"o": [1, "a"], "d": ["a", 1], "v": [1.0, 2.0]})
    _, labels = pairs_to_matrix(mixed["o"], mixed["d"], mixed["v"])
    assert labels.dtype == object and labels.tolist() == [1, "a"]


def test_pairs_to_matrix_mixed_labels_keep_their_python_types():
    # np.asarray([1, "a"]) gives ['1', 'a']: a user passing depot=1 afterwards would never find it.
    M, labels = pairs_to_matrix([1, "a"], ["a", 1], [1.0, 2.0])
    assert labels.dtype == object and labels.tolist() == [1, "a"]
    assert isinstance(labels[0], int) and labels[0] == 1
    assert M.tolist() == [[0.0, 1.0], [2.0, 0.0]]
    # one column mixed and the other not: the 1 of both columns must be the same label
    M, labels = pairs_to_matrix(["depot", "depot", 1], [1, 2, 2], [1.0, 2.0, 3.0])
    assert labels.tolist() == ["depot", 1, 2] and M.shape == (3, 3)
    assert M.tolist() == [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    # ints next to a float are not promoted to float either
    _, labels = pairs_to_matrix([1, 2.5], [2.5, 1], [1.0, 2.0])
    assert labels.dtype == object and labels.tolist() == [1, 2.5] and isinstance(labels[0], int)
    # tuple labels stay one element each (never expanded into a 2-D array)
    _, labels = pairs_to_matrix([(0, 0), (0, 0)], [(1, 1), (2, 2)], [1.0, 2.0], fill=0.0)
    assert labels.shape == (3,) and labels.tolist() == [(0, 0), (1, 1), (2, 2)]
    # explicit labels= follow the same rule
    _, labels = pairs_to_matrix([1, 2], [2, 1], [1.0, 2.0], labels=[2, 1, "x"], fill=0.0)
    assert labels.dtype == object and labels.tolist() == [2, 1, "x"]


def test_pairs_to_matrix_label_dtype_is_int64_or_object_like_coerce_labels():
    from skroute.utils.validation import coerce_labels

    _, labels = pairs_to_matrix(["a", "b", "a"], ["b", "c", "c"], [1.0, 2.0, 3.0])
    assert labels.dtype == object, "strings are object, never '<U1'"
    np.testing.assert_array_equal(labels, coerce_labels(["a", "b", "c"], 3))
    i32 = np.array([1, 2], dtype=np.int32)
    _, labels = pairs_to_matrix(i32, i32[::-1], [1.0, 2.0])
    assert labels.dtype == np.int64, "int64 whatever the platform's default integer"
    _, labels = pairs_to_matrix([np.int32(7), np.int64(8)], [8, 7], [1.0, 2.0])
    assert labels.dtype == np.int64 and labels.tolist() == [7, 8]
    _, labels = pairs_to_matrix([True], [False], [1.0])
    assert labels.dtype == object, "bools are not integer labels"
    with pytest.raises(ValueError, match="one-dimensional"):
        pairs_to_matrix(1, 2, 3.0)


# --------------------------------------------------------------------------- dict of dicts


def test_dict_of_dicts_round_trip_with_labels():
    rng = np.random.default_rng(0)
    C = rng.random((5, 5))
    np.fill_diagonal(C, 0.0)
    labels = ["a", "b", "c", "d", "e"]
    d = to_dict_of_dicts(C, labels)
    assert list(d) == labels and all(list(row) == labels for row in d.values())
    assert all(isinstance(v, float) for row in d.values() for v in row.values())
    M, back = from_dict_of_dicts(d)
    np.testing.assert_array_equal(M, C)
    assert back.tolist() == labels
    d0 = to_dict_of_dicts(C)
    assert list(d0) == [0, 1, 2, 3, 4]
    M0, back0 = from_dict_of_dicts(d0)
    np.testing.assert_array_equal(M0, C)
    assert back0.tolist() == [0, 1, 2, 3, 4]


def test_from_dict_of_dicts_missing_diagonal_defaults_to_zero_and_errors():
    M, labels = from_dict_of_dicts({1: {2: 5.0}, 2: {1: 6.0}})
    assert M.tolist() == [[0.0, 5.0], [6.0, 0.0]] and labels.tolist() == [1, 2]
    with pytest.raises(ValueError, match="empty"):
        from_dict_of_dicts({})
    with pytest.raises(ValueError, match=r"inner key 3 .* is not an outer key"):
        from_dict_of_dicts({1: {2: 1.0, 3: 1.0}, 2: {1: 1.0}})
    with pytest.raises(ValueError, match=r"row 2 lacks the entries \[1\]"):
        from_dict_of_dicts({1: {2: 1.0}, 2: {}})
    with pytest.raises(ValueError, match="is not a mapping"):
        from_dict_of_dicts({1: [0.0]})


def test_to_dict_of_dicts_errors():
    with pytest.raises(ValueError, match="square"):
        to_dict_of_dicts([[0.0, 1.0]])
    with pytest.raises(ValueError, match="labels has 1 entries but the matrix has 2 rows"):
        to_dict_of_dicts([[0.0, 1.0], [1.0, 0.0]], labels=["a"])
    with pytest.raises(ValueError, match="labels must be unique"):
        to_dict_of_dicts([[0.0, 1.0], [1.0, 0.0]], labels=["a", "a"])  # would silently collapse to one row


def test_from_dict_of_dicts_labels_follow_the_label_rule():
    M, labels = from_dict_of_dicts({1: {"a": 2.0}, "a": {1: 3.0}})
    assert labels.dtype == object and labels.tolist() == [1, "a"] and isinstance(labels[0], int)
    assert M.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    _, labels = from_dict_of_dicts({(1, 2): {(3, 4): 1.0}, (3, 4): {(1, 2): 2.0}})
    assert labels.shape == (2,) and labels.tolist() == [(1, 2), (3, 4)], "tuple keys are one label each"
    _, labels = from_dict_of_dicts({"a": {"b": 1.0}, "b": {"a": 1.0}})
    assert labels.dtype == object
    _, labels = from_dict_of_dicts({np.int32(1): {2: 1.0}, 2: {np.int32(1): 1.0}})
    assert labels.dtype == np.int64 and labels.tolist() == [1, 2]


@settings(derandomize=True, deadline=None, max_examples=50)
@given(n=st.integers(min_value=1, max_value=8), seed=st.integers(min_value=0, max_value=1000))
def test_dict_of_dicts_round_trip_property(n, seed):
    C = np.random.default_rng(seed).random((n, n))
    M, labels = from_dict_of_dicts(to_dict_of_dicts(C))
    np.testing.assert_array_equal(M, C)
    assert labels.tolist() == list(range(n))


# --------------------------------------------------------------------------- normalize_coords


def test_normalize_coords_preserves_aspect_ratio():
    xy = np.array([[10.0, 10.0], [30.0, 20.0], [20.0, 15.0]])
    out = normalize_coords(xy)
    assert out.dtype == np.float64 and out.shape == (3, 2)
    assert out.tolist() == [[0.0, 0.0], [1.0, 0.5], [0.5, 0.25]]
    assert out.min() == 0.0 and out.max() == 1.0
    span = out.max(axis=0) - out.min(axis=0)
    assert span[0] / span[1] == pytest.approx(20.0 / 10.0), "the ratio of the ranges is preserved"
    D_before, D_after = distance_matrix(xy), distance_matrix(out)
    np.testing.assert_allclose(D_after * 20.0, D_before, rtol=1e-12)


def test_normalize_coords_edge_cases():
    assert normalize_coords([[3.0, 4.0], [3.0, 4.0]]).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert normalize_coords([[3.0, 4.0]]).tolist() == [[0.0, 0.0]]
    tall = normalize_coords([[0.0, 0.0], [0.0, 4.0], [1.0, 2.0]])
    assert tall.tolist() == [[0.0, 0.0], [0.0, 1.0], [0.25, 0.5]]
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        normalize_coords([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="finite"):
        normalize_coords([[1.0, np.nan], [0.0, 0.0]])
    with pytest.raises(ValueError, match="coords is empty"):
        normalize_coords(np.empty((0, 2)))  # the same message as distance_matrix, not numpy's


def test_public_names_are_exported():
    expected = {
        "distance_matrix",
        "euclidean_matrix",
        "haversine_matrix",
        "tsplib_nint",
        "pairs_to_matrix",
        "to_dict_of_dicts",
        "from_dict_of_dicts",
        "normalize_coords",
        "fetch_pois",
        "geocode",
        "travel_time_matrix",
    }
    assert set(preprocessing.__all__) == expected
    for name in expected:
        assert getattr(preprocessing, name).__doc__


# --------------------------------------------------------------------------- Google client (mocked)


def _fake_googlemaps(monkeypatch, *, fail_pairs=(), bad_status=False, missing_fields=None):
    """Install a fake ``googlemaps`` module whose Client answers with deterministic distances.

    ``missing_fields`` maps an ``(origin, destination)`` pair to the element fields to
    drop from an otherwise ``"OK"`` element (a malformed response).
    """
    calls: list[dict] = []
    missing_fields = missing_fields or {}

    class FakeClient:
        def __init__(self, key):
            self.key = key

        def distance_matrix(self, origins, destinations, mode):
            calls.append({"origins": list(origins), "destinations": list(destinations), "mode": mode})
            if bad_status:
                return {"status": "OVER_QUERY_LIMIT", "rows": []}
            rows = []
            for o in origins:
                elements = []
                for d in destinations:
                    if (o, d) in fail_pairs:
                        elements.append({"status": "ZERO_RESULTS"})
                        continue
                    metres = round(
                        1000.0 * (abs(o[0] - d[0]) + 2.0 * abs(o[1] - d[1]))
                    )  # asymmetric on purpose
                    seconds = metres / 10.0 + (36.0 if o[0] < d[0] else 0.0)
                    element = {"status": "OK", "distance": {"value": metres}, "duration": {"value": seconds}}
                    for field in missing_fields.get((o, d), ()):
                        del element[field]
                    elements.append(element)
                rows.append({"elements": elements})
            return {
                "status": "OK",
                "origin_addresses": [f"addr {o[0]:.2f},{o[1]:.2f}" for o in origins],
                "destination_addresses": [f"addr {d[0]:.2f},{d[1]:.2f}" for d in destinations],
                "rows": rows,
            }

    module = types.ModuleType("googlemaps")
    module.Client = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "googlemaps", module)
    return calls


def _points(n: int) -> list[tuple[float, float]]:
    return [(40.0 + 0.1 * k, -3.0 + 0.05 * k) for k in range(n)]


def _expected(points):
    n = len(points)
    dist = np.zeros((n, n))
    time = np.zeros((n, n))
    for i, o in enumerate(points):
        for j, d in enumerate(points):
            metres = round(1000.0 * (abs(o[0] - d[0]) + 2.0 * abs(o[1] - d[1])))
            dist[i, j] = metres
            time[i, j] = (metres / 10.0 + (36.0 if o[0] < d[0] else 0.0)) / 3600.0
    return dist, time


def test_google_fetch_batches_requests_and_returns_metres_and_hours(monkeypatch, caplog):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    calls = _fake_googlemaps(monkeypatch)
    pts = _points(23)
    gdm = GoogleDistanceMatrix("KEY", batch_size=10)
    assert (gdm.api_key, gdm.mode, gdm.batch_size) == ("KEY", "driving", 10)
    with caplog.at_level(logging.INFO, logger="skroute"):
        res = gdm.fetch(pts, labels=list(range(100, 123)))
    assert isinstance(res, Bunch)
    assert set(res.keys()) == {"distance", "time", "labels", "units"}
    assert res.units == {"distance": "m", "time": "h"}
    assert res.labels.tolist() == list(range(100, 123))
    dist, time = _expected(pts)
    assert res.distance.shape == (23, 23) and res.distance.dtype == np.float64
    np.testing.assert_array_equal(res.distance, dist)
    np.testing.assert_allclose(res.time, time, rtol=1e-12)
    assert not np.array_equal(res.time, res.time.T), "directional matrices are kept as returned"
    assert len(calls) == 9 == gdm.n_requests_, "ceil(23 / 10) ** 2 requests, not one per pair"
    assert all(len(c["origins"]) <= 10 and len(c["destinations"]) <= 10 for c in calls)
    assert all(len(c["origins"]) * len(c["destinations"]) <= 100 for c in calls)
    assert all(c["mode"] == "driving" for c in calls)
    assert calls[0]["origins"] == pts[:10] and calls[0]["destinations"] == pts[:10]
    assert calls[-1]["origins"] == pts[20:] and calls[-1]["destinations"] == pts[20:]
    assert len(gdm.addresses_) == 23 and gdm.addresses_[0] == "addr 40.00,-3.00"
    records = [r for r in caplog.records if r.name == "skroute" and "request" in r.getMessage()]
    assert len(records) == 9 and records[-1].getMessage().startswith("GoogleDistanceMatrix: request 9/9")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_google_fetch_defaults_labels_mode_and_small_batches(monkeypatch):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    calls = _fake_googlemaps(monkeypatch)
    pts = _points(5)
    res = GoogleDistanceMatrix("KEY", "walking", batch_size=2).fetch(np.asarray(pts))
    assert res.labels.tolist() == [0, 1, 2, 3, 4]
    assert len(calls) == 9 and all(c["mode"] == "walking" for c in calls)
    np.testing.assert_array_equal(res.distance, _expected(pts)[0])
    single = GoogleDistanceMatrix("KEY", batch_size=1).fetch(pts[:3])
    np.testing.assert_array_equal(single.distance, _expected(pts[:3])[0])
    assert len(calls) == 9 + 9


def test_google_unroutable_elements_become_nan_and_are_logged(monkeypatch, caplog):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    pts = _points(4)
    _fake_googlemaps(monkeypatch, fail_pairs={(pts[0], pts[3])})
    with caplog.at_level(logging.WARNING, logger="skroute"):
        res = GoogleDistanceMatrix("KEY").fetch(pts)
    assert np.isnan(res.distance[0, 3]) and np.isnan(res.time[0, 3])
    assert np.isnan(res.distance).sum() == 1 and np.isfinite(res.distance[3, 0])
    assert any("1 of 16 elements could not be routed" in r.getMessage() for r in caplog.records)


def test_google_ok_element_without_a_value_is_unroutable_not_a_crash(monkeypatch, caplog):
    from skroute.preprocessing.google import GoogleDistanceMatrix, _element_value

    pts = _points(3)
    _fake_googlemaps(
        monkeypatch, missing_fields={(pts[0], pts[2]): ("duration",), (pts[1], pts[0]): ("distance",)}
    )
    with caplog.at_level(logging.WARNING, logger="skroute"):
        res = GoogleDistanceMatrix("KEY").fetch(pts)  # a KeyError here would waste the quota already spent
    assert np.isnan(res.distance[0, 2]) and np.isnan(res.time[0, 2])
    assert np.isnan(res.distance[1, 0]) and np.isnan(res.time[1, 0])
    assert np.isnan(res.distance).sum() == 2 and np.isnan(res.time).sum() == 2
    dist, _ = _expected(pts)
    ok = ~np.isnan(res.distance)
    np.testing.assert_array_equal(res.distance[ok], dist[ok])
    assert sum("status OK but no distance/duration value" in r.getMessage() for r in caplog.records) == 2
    assert any("2 of 9 elements could not be routed" in r.getMessage() for r in caplog.records)
    assert _element_value({"distance": {"value": 5}}, "distance") == 5.0
    assert _element_value({"distance": 5}, "distance") is None, "a non-dict field is malformed"
    assert _element_value({"distance": {"value": "5"}}, "distance") is None
    assert _element_value({"distance": {"value": True}}, "distance") is None
    assert _element_value({}, "duration") is None


def test_google_fetch_labels_follow_the_label_rule(monkeypatch):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    _fake_googlemaps(monkeypatch)
    pts = _points(3)
    gdm = GoogleDistanceMatrix("KEY")
    assert gdm.fetch(pts).labels.dtype == np.int64, "the default 0..n-1 is int64 on every platform"
    assert gdm.fetch(pts, labels=np.array([5, 6, 7], dtype=np.int32)).labels.dtype == np.int64
    mixed = gdm.fetch(pts, labels=[1, "a", (2, 3)]).labels
    assert mixed.dtype == object and mixed.tolist() == [1, "a", (2, 3)] and isinstance(mixed[0], int)
    assert gdm.fetch(pts, labels=["a", "b", "c"]).labels.dtype == object
    with pytest.raises(ValueError, match="unique"):
        gdm.fetch(pts, labels=[1, 1, 2])


def test_google_bad_response_status_leaves_the_block_nan(monkeypatch, caplog):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    _fake_googlemaps(monkeypatch, bad_status=True)
    with caplog.at_level(logging.WARNING, logger="skroute"):
        res = GoogleDistanceMatrix("KEY").fetch(_points(3))
    assert np.isnan(res.distance).all() and np.isnan(res.time).all()
    assert any("OVER_QUERY_LIMIT" in r.getMessage() for r in caplog.records)


def test_google_constructor_validation(monkeypatch):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    _fake_googlemaps(monkeypatch)
    with pytest.raises(ValueError, match="mode must be one of"):
        GoogleDistanceMatrix("KEY", mode="flying")
    for bad in (0, 11, -1, 2.5, True):
        with pytest.raises(ValueError, match=r"batch_size must be an integer in \[1, 10\]"):
            GoogleDistanceMatrix("KEY", batch_size=bad)
    gdm = GoogleDistanceMatrix("KEY")
    with pytest.raises(ValueError, match=r"coords must have shape \(n, 2\)"):
        gdm.fetch([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="labels must have length 2"):
        gdm.fetch([[1.0, 2.0], [3.0, 4.0]], labels=[1])


def test_google_import_error_message_without_googlemaps(monkeypatch):
    from skroute.preprocessing.google import GoogleDistanceMatrix

    monkeypatch.setitem(sys.modules, "googlemaps", None)
    with pytest.raises(ImportError, match=r"^googlemaps is required: pip install scikit-route\[google\]$"):
        GoogleDistanceMatrix("KEY")


def test_cost_scraper_is_a_deprecated_wrapper(monkeypatch):
    from skroute.preprocessing.google import CostScraper, GoogleDistanceMatrix

    calls = _fake_googlemaps(monkeypatch)
    pts = _points(3)
    nodes = [(10 + k, lat, lon) for k, (lat, lon) in enumerate(pts)]
    with pytest.warns(DeprecationWarning, match="CostScraper is deprecated since 2.0") as record:
        scraper = CostScraper("KEY", nodes, mode="bicycling")
    assert record[0].filename == __file__
    assert scraper.labels == [10, 11, 12] and scraper.coords == pts
    assert scraper.result_ is None and not calls, "nothing is requested before scrap()"
    res = scraper.scrap()
    assert scraper.result_ is res and isinstance(res, Bunch)
    assert res.labels.tolist() == [10, 11, 12] and all(c["mode"] == "bicycling" for c in calls)
    np.testing.assert_array_equal(res.distance, _expected(pts)[0])
    np.testing.assert_array_equal(GoogleDistanceMatrix("KEY", "bicycling").fetch(pts).distance, res.distance)
    with pytest.raises(
        NotImplementedError, match=r"^to_pickle was removed in 2.0; use pandas\(\).to_pickle\(...\)$"
    ):
        scraper.to_pickle("x.pkl")
    pd = pytest.importorskip("pandas")
    frame = scraper.pandas()
    assert isinstance(frame, pd.DataFrame) and frame.shape == (3, 10)
    assert frame.columns.tolist() == [
        "id_origin", "lat_origin", "lon_origin", "address_origin", "id_destinity", "lat_destinity",
        "lon_destinity", "address_destinity", "meters", "seconds",
    ]  # fmt: skip
    assert frame["id_origin"].tolist() == [10, 10, 11] and frame["id_destinity"].tolist() == [11, 12, 12]
    assert frame["meters"].tolist() == [res.distance[0, 1], res.distance[0, 2], res.distance[1, 2]]
    assert frame["seconds"].tolist() == pytest.approx(
        [res.time[0, 1] * 3600, res.time[0, 2] * 3600, res.time[1, 2] * 3600]
    )
    assert frame["address_origin"].tolist()[0].startswith("addr ")


def test_cost_scraper_pandas_fetches_when_needed(monkeypatch):
    from skroute.preprocessing.google import CostScraper

    pytest.importorskip("pandas")
    calls = _fake_googlemaps(monkeypatch)
    with pytest.warns(DeprecationWarning):
        scraper = CostScraper("KEY", [(1, 40.0, -3.0), (2, 40.1, -3.1)])
    frame = scraper.pandas()
    assert frame.shape == (1, 10) and len(calls) == 1 and scraper.result_ is not None
