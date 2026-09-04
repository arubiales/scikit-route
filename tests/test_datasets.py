"""Tests of ``skroute.datasets``: the TSPLIB reader, the 27 Waterloo loaders and the five cost loaders."""

from __future__ import annotations

import csv
import inspect
import io
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skroute import datasets
from skroute.datasets import TSPBunch, _loaders, list_tsp, load_tsp, read_tsplib, read_tsplib_tour
from skroute.preprocessing import distance_matrix
from skroute.utils import Bunch

DATA = Path(__file__).parent / "data"

# The explicit 4-node matrix of tests/data/explicit_matrix.tsp (optimal tour 1-2-3-4, cost 22).
EXPLICIT4 = np.array(
    [[0.0, 5.0, 9.0, 10.0], [5.0, 0.0, 4.0, 8.0], [9.0, 4.0, 0.0, 3.0], [10.0, 8.0, 3.0, 0.0]]
)


def _closed_cost(C: np.ndarray, ids: np.ndarray) -> float:
    idx = np.asarray(ids) - 1
    return float(sum(C[idx[k], idx[(k + 1) % len(idx)]] for k in range(len(idx))))


def _euc_text(coords, *, sep=" : ", eof=True, name="gen", labels=None, dimension=True) -> str:
    n = len(coords)
    lines = [f"NAME{sep}{name}", f"TYPE{sep}TSP"]
    if dimension:
        lines.append(f"DIMENSION{sep}{n}")
    lines += [f"EDGE_WEIGHT_TYPE{sep}EUC_2D", "NODE_COORD_SECTION"]
    ids = range(1, n + 1) if labels is None else labels
    lines += [f"{i} {float(x)!r} {float(y)!r}" for i, (x, y) in zip(ids, coords, strict=True)]
    if eof:
        lines.append("EOF")
    return "\n".join(lines) + "\n"


def _explicit_text(C: np.ndarray, fmt: str, *, sep=": ") -> str:
    n = C.shape[0]
    if fmt == "FULL_MATRIX":
        rows = [" ".join(str(int(v)) for v in C[i]) for i in range(n)]
    elif fmt == "UPPER_ROW":
        rows = [" ".join(str(int(v)) for v in C[i, i + 1 :]) for i in range(n - 1)]
    elif fmt == "LOWER_ROW":
        rows = [" ".join(str(int(v)) for v in C[i, :i]) for i in range(1, n)]
    elif fmt == "UPPER_DIAG_ROW":
        rows = [" ".join(str(int(v)) for v in C[i, i:]) for i in range(n)]
    elif fmt == "LOWER_DIAG_ROW":
        rows = [" ".join(str(int(v)) for v in C[i, : i + 1]) for i in range(n)]
    else:  # pragma: no cover
        raise AssertionError(fmt)
    lines = [
        f"NAME{sep}m{n}",
        f"TYPE{sep}TSP",
        f"DIMENSION{sep}{n}",
        f"EDGE_WEIGHT_TYPE{sep}EXPLICIT",
        f"EDGE_WEIGHT_FORMAT{sep}{fmt}",
        "EDGE_WEIGHT_SECTION",
        *rows,
        "EOF",
    ]
    return "\n".join(lines) + "\n"


def _symmetric_int_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 100, size=(n, n)).astype(np.float64)
    C = np.triu(A, 1)
    return C + C.T


# --------------------------------------------------------------------------- reader: fixtures


def test_explicit_full_matrix_fixture_with_crlf():
    raw = (DATA / "explicit_matrix.tsp").read_bytes()
    assert b"\r\n" in raw, "the fixture must keep its CRLF line endings"
    b = read_tsplib(DATA / "explicit_matrix.tsp")
    assert isinstance(b, Bunch)
    assert b.name == "explicit4"
    assert b.type == "TSP"
    assert b.dimension == 4
    assert b.edge_weight_type == "EXPLICIT"
    assert b.edge_weight_format == "FULL_MATRIX"
    assert "FULL_MATRIX" in b.comment
    assert b.coords is None and b.display_coords is None
    assert b.cost.dtype == np.float64
    np.testing.assert_array_equal(b.cost, EXPLICIT4)
    assert b.labels.dtype == np.int64
    assert b.labels.tolist() == [1, 2, 3, 4]


def test_upper_row_fixture_matches_full_matrix_and_keeps_display_data():
    b = read_tsplib(DATA / "explicit_upper_row.tsp")
    assert b.edge_weight_format == "UPPER_ROW"
    np.testing.assert_array_equal(b.cost, EXPLICIT4)
    assert b.comment.count("\n") == 1, "two COMMENT lines are joined with a newline"
    np.testing.assert_array_equal(b.display_coords, [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [9.0, 12.0]])
    assert b.labels.tolist() == [1, 2, 3, 4]


def test_geo_fixture_keeps_raw_coordinates_and_is_a_block_of_ulysses16():
    b = read_tsplib(DATA / "geo_small.tsp")
    assert b.edge_weight_type == "GEO" and b.edge_weight_format is None
    assert b.cost is None
    np.testing.assert_array_equal(b.coords, [[38.24, 20.42], [39.57, 26.15], [40.56, 25.32], [36.26, 23.12]])
    full = distance_matrix(read_tsplib(DATA / "ulysses16.tsp").coords, metric="tsplib_geo")
    np.testing.assert_array_equal(distance_matrix(b.coords, metric="tsplib_geo"), full[:4, :4])


def test_tiny_tour_costs_22_on_the_explicit_fixture():
    tour = read_tsplib_tour(DATA / "tiny.tour")
    assert tour.dtype == np.int64
    assert tour.tolist() == [1, 2, 3, 4]
    assert _closed_cost(read_tsplib(DATA / "explicit_matrix.tsp").cost, tour) == 22.0


def test_tour_written_on_a_single_line():
    tour = read_tsplib_tour(DATA / "ulysses16.opt.tour")
    assert tour.shape == (16,)
    assert sorted(tour.tolist()) == list(range(1, 17))
    assert tour[0] == 1


def test_att48_fixture_reads_as_att_coordinates():
    b = read_tsplib(DATA / "att48.tsp")
    assert (b.edge_weight_type, b.dimension, b.coords.shape) == ("ATT", 48, (48, 2))
    assert read_tsplib_tour(DATA / "att48.opt.tour").shape == (48,)


# --------------------------------------------------------------------------- reader: generated files


@pytest.mark.parametrize("sep", [" : ", ": ", ":"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("eof", [True, False])
def test_euc_2d_round_trip_tolerates_keyword_spelling_crlf_and_missing_eof(tmp_path, sep, newline, eof):
    coords = np.array([[0.5, 1.25], [10.0, -3.0], [7.75, 7.75]])
    text = _euc_text(coords, sep=sep, eof=eof).replace("\n", newline)
    path = tmp_path / "gen.tsp"
    path.write_bytes(text.encode("latin-1"))
    b = read_tsplib(path)
    assert b.name == "gen"
    assert b.dimension == 3 and b.edge_weight_type == "EUC_2D"
    np.testing.assert_array_equal(b.coords, coords)
    assert b.coords.dtype == np.float64 and b.coords.flags.c_contiguous
    assert b.labels.tolist() == [1, 2, 3]


@pytest.mark.parametrize("fmt", ["FULL_MATRIX", "UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW"])
def test_explicit_formats_round_trip(tmp_path, fmt):
    C = _symmetric_int_matrix(6, seed=sum(map(ord, fmt)))  # a stable seed per format (hash() is per-process)
    path = tmp_path / f"{fmt}.tsp"
    path.write_text(_explicit_text(C, fmt), encoding="latin-1")
    b = read_tsplib(path)
    assert b.edge_weight_format == fmt
    np.testing.assert_array_equal(b.cost, C)
    assert b.labels.tolist() == list(range(1, 7))


def test_full_matrix_may_be_asymmetric():
    C = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
    b = read_tsplib(io.StringIO(_explicit_text(C, "FULL_MATRIX").replace("TYPE: TSP", "TYPE: ATSP")))
    assert b.type == "ATSP"
    np.testing.assert_array_equal(b.cost, C)


def test_reader_accepts_str_path_pathlib_text_and_binary_file_objects(tmp_path):
    text = _euc_text([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "f.tsp"
    path.write_text(text, encoding="latin-1")
    expected = read_tsplib(path).coords
    for source in (str(path), io.StringIO(text), io.BytesIO(text.encode("latin-1"))):
        np.testing.assert_array_equal(read_tsplib(source).coords, expected)
    with path.open(encoding="latin-1") as fh:
        np.testing.assert_array_equal(read_tsplib(fh).coords, expected)


def test_reader_tolerates_lowercase_keys_indentation_and_section_colon():
    text = (
        "  name : lower\n"
        "type: tsp\n"
        "\n"
        "dimension : 2\n"
        "edge_weight_type: EUC_2D\n"
        "NODE_COORD_SECTION:\n"
        "   1   0   0\n"
        "   2   3   4\n"
    )
    b = read_tsplib(io.StringIO(text))
    assert (b.name, b.type, b.dimension) == ("lower", "TSP", 2)
    np.testing.assert_array_equal(b.coords, [[0.0, 0.0], [3.0, 4.0]])


def test_reader_drops_a_utf8_bom_and_decodes_utf8_or_latin1(tmp_path):
    text = "NAME: bom\nCOMMENT: Cádiz\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\nEOF\n"
    for source in (io.BytesIO(("﻿" + text).encode("utf-8")), io.StringIO("﻿" + text)):
        b = read_tsplib(source)
        assert b.name == "bom" and b.comment == "Cádiz", "the BOM used to swallow NAME as 'ï»¿NAME'"
    path = tmp_path / "bom.tsp"
    path.write_bytes(("﻿" + text).encode("utf-8"))
    assert read_tsplib(path).name == "bom"
    with path.open(encoding="utf-8") as fh:  # a text handle that yields the BOM as a character
        assert read_tsplib(fh).name == "bom"
    assert read_tsplib(io.BytesIO(text.encode("latin-1"))).comment == "Cádiz", "latin-1 files still decode"
    first_line_ewt = "﻿EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n"
    assert read_tsplib(io.BytesIO(first_line_ewt.encode("utf-8"))).edge_weight_type == "EUC_2D"


def test_type_keeps_only_its_first_word():
    text = "TYPE: TSP (M.~Hofmeister)\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\nEOF\n"
    assert read_tsplib(io.StringIO(text)).type == "TSP"
    assert read_tsplib(io.StringIO(text.replace("TSP (M.~Hofmeister)", ""))).type == "TSP"


def test_data_on_the_section_keyword_line_is_kept():
    b = read_tsplib(io.StringIO("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION 1 0 0\n2 1 1\nEOF\n"))
    assert b.dimension == 2 and b.coords.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    b = read_tsplib(io.StringIO("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION : 1 0 0\n2 1 1\nEOF\n"))
    assert b.coords.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert read_tsplib_tour(io.StringIO("TYPE: TOUR\nTOUR_SECTION 1 2 3 -1\nEOF\n")).tolist() == [1, 2, 3]


def test_a_keyword_line_inside_a_section_does_not_close_it():
    text = "DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
    text += "DISPLAY_DATA_TYPE: NO_DISPLAY\n1 0 0\n2 1 1\n"
    assert read_tsplib(io.StringIO(text)).coords.tolist() == [[0.0, 0.0], [1.0, 1.0]]


def test_reader_defaults_for_absent_optional_keywords():
    text = "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n7 1 1\n9 2 2\nEOF\n"
    b = read_tsplib(io.StringIO(text))
    assert b.name is None and b.comment is None and b.type == "TSP"
    assert b.dimension == 2, "DIMENSION is derived from the section when absent"
    assert b.labels.tolist() == [7, 9], "ids are kept exactly as written"


@pytest.mark.parametrize("ewt", ["EUC_3D", "MAX_2D", "XRAY1", "SPECIAL"])
def test_unsupported_edge_weight_type_message(ewt):
    text = f"NAME: x\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: {ewt}\nNODE_COORD_SECTION\n1 0 0\n2 1 1\nEOF\n"
    with pytest.raises(ValueError, match=rf"^EDGE_WEIGHT_TYPE {ewt} is not supported in this version$"):
        read_tsplib(io.StringIO(text))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("NAME: x\nDIMENSION: 2\nNODE_COORD_SECTION\n1 0 0\n2 1 1\n", "EDGE_WEIGHT_TYPE is missing"),
        ("DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nEOF\n", "needs a NODE_COORD_SECTION"),
        ("DIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 1 1\n", "DIMENSION is 3"),
        ("DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 1\n", "triples"),
        ("DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 a 1\n", "is not a number"),
        ("DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1.5 0 0\n2 1 1\n", "is not an integer"),
        ("DIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n1 0 0\n2 1 1\n", "outside any section"),
        (
            "DIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n",
            "needs an EDGE_WEIGHT_SECTION",
        ),
        (
            "DIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FUNCTION\n"
            "EDGE_WEIGHT_SECTION\n0 1 1 0\n",
            "EDGE_WEIGHT_FORMAT FUNCTION is not supported",
        ),
        (
            "EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n0 1 1 0\n",
            "DIMENSION is required",
        ),
        (
            "DIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: UPPER_ROW\n"
            "EDGE_WEIGHT_SECTION\n1 2\n",
            "needs 3 numbers; got 2",
        ),
        (
            "DIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\nDISPLAY_DATA_SECTION\n1 0 0\nNODE_COORD_SECTION\n"
            "1 0 0\n2 1 1\n3 2 2\n",
            "DISPLAY_DATA_SECTION has 1 nodes but DIMENSION is 3",
        ),
        (
            "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n1 3 4\n",
            "NODE_COORD_SECTION repeats node id 1",
        ),
        (
            "DIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
            "DISPLAY_DATA_SECTION\n1 0 0\n1 1 1\nEDGE_WEIGHT_SECTION\n0 1 1 0\n",
            "DISPLAY_DATA_SECTION repeats node id 1",
        ),
        (
            "DIMENSION: 0\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
            "EDGE_WEIGHT_SECTION\n",
            "DIMENSION must be a positive integer; got 0",
        ),
        (
            "DIMENSION: -1\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: UPPER_ROW\n"
            "EDGE_WEIGHT_SECTION\n1\n",
            "DIMENSION must be a positive integer; got -1",
        ),
        (
            "DIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_SECTION\n0 1 1 0\n",
            "EDGE_WEIGHT_FORMAT is required with EDGE_WEIGHT_TYPE EXPLICIT",
        ),
    ],
)
def test_reader_errors(text, match):
    with pytest.raises(ValueError, match=match):
        read_tsplib(io.StringIO(text))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("NAME: t\nTYPE: TOUR\nDIMENSION: 3\nEOF\n", "no TOUR_SECTION"),
        ("TYPE: TOUR\nTOUR_SECTION\n-1\nEOF\n", "holds no node ids"),
        ("TYPE: TOUR\nDIMENSION: 4\nTOUR_SECTION\n1 2 3\n-1\n", "3 nodes but DIMENSION is 4"),
        ("TYPE: TOUR\nTOUR_SECTION\n1 x 3\n-1\n", "is not a number"),
    ],
)
def test_tour_reader_errors(text, match):
    with pytest.raises(ValueError, match=match):
        read_tsplib_tour(io.StringIO(text))


def test_tour_reader_stops_at_first_minus_one_and_ignores_dimension_when_absent():
    tour = read_tsplib_tour(io.StringIO("TOUR_SECTION\n3\n1\n2\n-1\n5 6\n-1\nEOF\n"))
    assert tour.tolist() == [3, 1, 2]


_finite = st.floats(allow_nan=False, allow_infinity=False, width=64)


@settings(derandomize=True, deadline=None, max_examples=60)
@given(st.lists(st.tuples(_finite, _finite), min_size=1, max_size=12))
def test_euc_2d_round_trip_property(points):
    coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    b = read_tsplib(io.StringIO(_euc_text(coords)))
    assert b.dimension == len(points)
    np.testing.assert_array_equal(b.coords, coords)
    assert b.labels.tolist() == list(range(1, len(points) + 1))


@settings(derandomize=True, deadline=None, max_examples=60)
@given(
    n=st.integers(min_value=2, max_value=9),
    seed=st.integers(min_value=0, max_value=10_000),
    fmt=st.sampled_from(["FULL_MATRIX", "UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW"]),
)
def test_explicit_round_trip_property(n, seed, fmt):
    C = _symmetric_int_matrix(n, seed)
    np.testing.assert_array_equal(read_tsplib(io.StringIO(_explicit_text(C, fmt))).cost, C)


# --------------------------------------------------------------------------- Waterloo loaders

# (wrapper, file name, n, published optimum) -- the exact 1.0 wrapper names.
WRAPPERS: list[tuple[Callable[..., TSPBunch], str, int, int]] = [
    (datasets.load_sahara, "wi29", 29, 27603),
    (datasets.load_djibouti, "dj38", 38, 6656),
    (datasets.load_qatar, "qa194", 194, 9352),
    (datasets.load_uruguay, "uy734", 734, 79114),
    (datasets.load_zimbabwe, "zi929", 929, 95345),
    (datasets.load_luxembourg, "lu980", 980, 11340),
    (datasets.load_rwanda, "rw1621", 1621, 26051),
    (datasets.load_oman, "mu1979", 1979, 86891),
    (datasets.load_nicaragua, "nu3496", 3496, 96132),
    (datasets.load_canada, "ca4663", 4663, 1290319),
    (datasets.load_tanzania, "tz6117", 6117, 394718),
    (datasets.load_egypt, "eg7146", 7146, 172386),
    (datasets.load_yemen, "ym7663", 7663, 238314),
    (datasets.load_panama, "pm8079", 8079, 114855),
    (datasets.load_ireland, "ei8246", 8246, 206171),
    (datasets.load_argentina, "ar9152", 9152, 837479),
    (datasets.load_japan, "ja9847", 9847, 491924),
    (datasets.load_greece, "gr9882", 9882, 300899),
    (datasets.load_kazakhstan, "kz9976", 9976, 1061881),
    (datasets.load_finland, "fi10639", 10639, 520527),
    (datasets.load_morocco, "mo14185", 14185, 427377),
    (datasets.load_honduras, "ho14473", 14473, 177092),
    (datasets.load_italy, "it16862", 16862, 557315),
    (datasets.load_vietnam, "vm22775", 22775, 569288),
    (datasets.load_sweden, "sw24978", 24978, 855597),
    (datasets.load_burma, "bm33708", 33708, 959289),
    (datasets.load_china, "ch71009", 71009, 4566506),
]
BIG_FOUR = {"vm22775", "sw24978", "bm33708", "ch71009"}
TSP_FIELDS = {"name", "coords", "labels", "depot", "edge_weight_type", "optimal_tour_length", "DESCR"}


@pytest.mark.parametrize(("loader", "name", "n", "optimum"), WRAPPERS, ids=[w[1] for w in WRAPPERS])
def test_country_wrapper_loads_the_whole_instance(loader, name, n, optimum):
    b = loader()
    assert isinstance(b, TSPBunch) and isinstance(b, Bunch)
    assert set(b.keys()) == TSP_FIELDS, "distance_matrix is a method, not a key"
    assert b.name == name
    assert b.coords.shape == (n, 2) and b.coords.dtype == np.float64
    assert np.all(np.isfinite(b.coords))
    assert b.labels.dtype == np.int64
    np.testing.assert_array_equal(b.labels, np.arange(1, n + 1))
    assert b.depot == 1 and isinstance(b.depot, int)
    assert b.edge_weight_type == "EUC_2D"
    assert b.optimal_tour_length == optimum
    assert name in b.DESCR and str(n) in b.DESCR
    assert ("cannot be solved whole" in b.DESCR) is (name in BIG_FOUR)
    assert ("n_nodes=5000" in b.DESCR) is (name in BIG_FOUR)
    assert b.DESCR == load_tsp(name).DESCR


def test_list_tsp_has_the_27_names_smallest_first():
    names = list_tsp()
    assert len(names) == 27 and len(set(names)) == 27
    assert names == [w[1] for w in WRAPPERS]
    sizes = [int("".join(ch for ch in nm if ch.isdigit())) for nm in names]
    assert sizes == sorted(sizes)
    assert list_tsp() is not names, "a fresh list every call"


def test_optimal_tour_length_is_the_waterloo_status_table():
    # https://www.math.uwaterloo.ca/tsp/world/summary.html (2022-07-31): 25 proven optima, two open.
    assert load_tsp("kz9976").optimal_tour_length == 1061881, "the proven optimum, not the 2001 tour"
    assert load_tsp("ch71009").optimal_tour_length == 4566506, "Waterloo's current best-known tour"
    for name in ("bm33708", "ch71009"):
        b = load_tsp(name)
        assert "best-known" in b.DESCR and "not proven optimal" in b.DESCR
        assert _loaders._INSTANCES[name].gap in b.DESCR
    assert _loaders._INSTANCES["bm33708"].gap == "0.031 %"
    assert _loaders._INSTANCES["ch71009"].gap == "0.024 %"
    assert all(v.gap is None for k, v in _loaders._INSTANCES.items() if k not in ("bm33708", "ch71009"))
    for name in ("wi29", "kz9976"):
        b = load_tsp(name)
        assert "proven optimal tour length" in b.DESCR and "not proven" not in b.DESCR
    assert "published optima" not in load_tsp("wi29").DESCR


def test_load_tsp_unknown_name():
    with pytest.raises(ValueError, match="unknown instance 'xx1'"):
        load_tsp("xx1")


def test_wrapper_forwards_keyword_arguments():
    a = datasets.load_qatar(n_nodes=15, random_state=3)
    b = load_tsp("qa194", n_nodes=15, random_state=3)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.coords.shape == (15, 2) and a.optimal_tour_length is None


def test_subsample_is_deterministic_keeps_the_first_node_and_the_file_order():
    full = load_tsp("qa194")
    a = load_tsp("qa194", n_nodes=40)
    b = load_tsp("qa194", n_nodes=40)  # random_state defaults to 2019
    c = load_tsp("qa194", n_nodes=40, random_state=2019)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.labels, c.labels)
    np.testing.assert_array_equal(a.coords, b.coords)
    assert a.coords.shape == (40, 2) and a.labels.shape == (40,)
    assert a.labels[0] == 1 and a.depot == 1
    assert np.all(np.diff(a.labels) > 0), "file order is preserved"
    assert len(set(a.labels.tolist())) == 40, "without replacement"
    np.testing.assert_array_equal(a.coords, full.coords[a.labels - 1])
    assert a.optimal_tour_length is None
    assert a.name == "qa194" and a.edge_weight_type == "EUC_2D"
    assert "Subsample" in a.DESCR and "40 of the 194" in a.DESCR
    other = load_tsp("qa194", n_nodes=40, random_state=7)
    assert not np.array_equal(a.labels, other.labels)
    assert a.coords.flags.c_contiguous


def test_subsample_edge_sizes():
    whole = load_tsp("wi29", n_nodes=29)
    assert whole.optimal_tour_length == 27603 and whole.coords.shape == (29, 2)
    one = load_tsp("wi29", n_nodes=1)
    assert one.labels.tolist() == [1] and one.optimal_tour_length is None
    assert load_tsp("wi29", n_nodes=np.int64(5)).coords.shape == (5, 2)


@pytest.mark.parametrize("bad", [0, 30, -1, 2.5, True, "10"])
def test_subsample_rejects_invalid_sizes(bad):
    with pytest.raises(ValueError, match="n_nodes"):
        load_tsp("wi29", n_nodes=bad)


@pytest.mark.parametrize(
    ("mode", "expected_n"), [("small", 10), ("medium", 6), ("big", 29)], ids=["small", "medium", "big"]
)
def test_mode_is_deprecated_and_maps_to_n_nodes(mode, expected_n):
    with pytest.warns(DeprecationWarning, match="mode= is deprecated since 2.0") as record:
        b = load_tsp("wi29", mode=mode)
    assert b.coords.shape == (expected_n, 2)
    assert (b.optimal_tour_length == 27603) is (mode == "big")
    assert record[0].filename == __file__, "the warning points at the caller, not at skroute"


def test_mode_deprecation_through_a_country_wrapper_points_at_the_caller():
    with pytest.warns(DeprecationWarning) as record:
        b = datasets.load_qatar(mode="medium")
    assert b.coords.shape == (round(0.2 * 194), 2)
    assert record[0].filename == __file__


def test_mode_may_be_positional_as_in_1_0():
    with pytest.warns(DeprecationWarning, match="mode= is deprecated since 2.0") as record:
        b = datasets.load_qatar("small")  # 1.0: def load_qatar(mode="big")
    assert b.coords.shape == (10, 2) and b.optimal_tour_length is None
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        assert datasets.load_sahara("big").optimal_tour_length == 27603
    for loader, *_ in WRAPPERS:
        parameters = inspect.signature(loader).parameters
        assert list(parameters) == ["mode", "kwargs"] and parameters["mode"].default is None


def test_mode_errors():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="mode must be"):
        load_tsp("wi29", mode="huge")
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        load_tsp("wi29", mode="small", n_nodes=5)


@pytest.mark.parametrize("name", ["wi29", "dj38"])
def test_distance_matrix_is_integer_symmetric_cached(name):
    b = load_tsp(name)
    C = b.distance_matrix()
    n = b.coords.shape[0]
    assert C.shape == (n, n) and C.dtype == np.float64 and C.flags.c_contiguous
    assert np.array_equal(C, np.floor(C)), "EUC_2D distances are integers"
    assert np.array_equal(C, C.T)
    assert np.all(np.diagonal(C) == 0.0)
    assert np.all(C[~np.eye(n, dtype=bool)] > 0)
    assert b.distance_matrix() is C, "cached: the same object every call"
    assert "_distance_matrix" not in b and "_distance_matrix" not in list(b.keys())
    np.testing.assert_array_equal(C, distance_matrix(b.coords, metric="tsplib_euc_2d"))
    # nearest-neighbour tour from the depot: a plain integer cost, never better than the optimum.
    visited = [0]
    while len(visited) < n:
        row = C[visited[-1]].copy()
        row[visited] = np.inf
        visited.append(int(row.argmin()))
    cost = float(sum(C[visited[k], visited[(k + 1) % n]] for k in range(n)))
    assert cost.is_integer()
    assert cost >= b.optimal_tour_length


def test_qa194_has_two_half_integer_pairs_where_rint_would_differ():
    b = load_tsp("qa194")
    C = b.distance_matrix()
    d = np.sqrt(((b.coords[:, None, :] - b.coords[None, :, :]) ** 2).sum(-1))
    rint = np.rint(d)
    np.fill_diagonal(rint, 0.0)
    differs = rint != C
    assert int(differs.sum()) == 4, "two unordered pairs (D15)"
    assert np.all(d[differs] * 2 == np.floor(d[differs] * 2)), "they are exact half-integers"
    assert np.all(C[differs] == rint[differs] + 1), "nint rounds them up, rint (half-to-even) down"


def test_distance_matrix_refuses_large_instances_unless_forced(monkeypatch):
    monkeypatch.setattr(_loaders, "_MAX_DENSE_N", 10)
    b = load_tsp("wi29")
    with pytest.raises(ValueError, match=r"wi29 has 29 nodes.*n_nodes=.*force=True"):
        b.distance_matrix()
    with pytest.raises(ValueError):
        b.distance_matrix(force=False)
    C = b.distance_matrix(force=True)
    assert C.shape == (29, 29)
    assert b.distance_matrix() is C, "once built, the cache serves it without force"
    small = load_tsp("wi29", n_nodes=10)
    assert small.distance_matrix().shape == (10, 10)


def test_distance_matrix_follows_the_edge_weight_type_of_the_bunch():
    geo = read_tsplib(DATA / "geo_small.tsp")
    b = TSPBunch(name="geo_small", coords=geo.coords, labels=geo.labels, depot=1, edge_weight_type="GEO")
    np.testing.assert_array_equal(b.distance_matrix(), distance_matrix(geo.coords, metric="tsplib_geo"))
    xy = [[0.0, 0.0], [1.0, 1.0]]
    lower = TSPBunch(name="x", coords=xy, labels=[1, 2], depot=1, edge_weight_type="ceil_2d")
    assert lower.distance_matrix()[0, 1] == 2.0, "case-insensitive: ceil(sqrt(2))"


def test_distance_matrix_rejects_an_unknown_edge_weight_type():
    b = TSPBunch(name="x", coords=[[0.0, 0.0], [3.0, 4.0]], labels=[1, 2], depot=1, edge_weight_type="EUC_3D")
    with pytest.raises(ValueError, match=r"edge_weight_type 'EUC_3D' has no tsplib_\* metric"):
        b.distance_matrix()  # used to fall back silently to the planar EUC_2D metric
    b = TSPBunch(
        name="x", coords=[[0.0, 0.0], [3.0, 4.0]], labels=[1, 2], depot=1, edge_weight_type="EXPLICIT"
    )
    with pytest.raises(ValueError, match="has no tsplib_"):
        b.distance_matrix()


def test_large_instance_sizes_are_quoted_in_decimal_gigabytes(monkeypatch):
    b = load_tsp("ch71009")
    assert "about 40 GB" in b.DESCR, "71009**2 * 8 bytes = 40.3e9: the figure of the SPEC and the docs"
    assert "GiB" not in b.DESCR
    with pytest.raises(ValueError, match=r"needs 40\.3 GB"):
        b.distance_matrix()
    monkeypatch.setattr(_loaders, "_MAX_DENSE_N", 10)
    with pytest.raises(ValueError, match=r"needs 0\.0 GB"):
        load_tsp("wi29").distance_matrix()


def test_size_warning_through_tspbunch_points_at_the_caller(monkeypatch):
    from skroute.preprocessing import _distances

    monkeypatch.setattr(_loaders, "_MAX_DENSE_N", 10)
    monkeypatch.setattr(_distances, "_LARGE_N", 10)
    b = load_tsp("wi29")
    with pytest.warns(UserWarning, match="dense 29 x 29") as record:
        b.distance_matrix(force=True)
    assert record[0].filename == __file__, "not skroute/datasets/_loaders.py"


def test_tspbunch_behaves_like_a_bunch():
    b = load_tsp("wi29")
    assert repr(b).startswith("TSPBunch(") and "coords" in repr(b)
    assert b["coords"] is b.coords
    b.note = "x"
    assert b["note"] == "x" and "note" in list(b.keys())
    del b.note
    assert "note" not in b
    with pytest.raises(AttributeError):
        _ = b.missing
    assert "distance_matrix" in dir(b) and callable(b.distance_matrix)
    assert b.distance_matrix.__doc__ is not None


# --------------------------------------------------------------------------- cost loaders

SPANISH_COLUMNS = [
    "id_origin", "id_destinity", "lat_origin", "lon_origin", "lat_destiniy", "lon_destinity", "cluster",
    "origin", "destinity", "meters", "secs", "hours", "kilometers", "cost",
]  # fmt: skip
QATAR_COLUMNS = [
    "id_origin", "lat_origin", "lon_origin", "address_origin", "id_destinity", "lat_destinity",
    "lon_destinity", "address_destinity", "meters", "seconds",
]  # fmt: skip

# (loader, csv file, n, depot label, cost unit, rows in the long table)
COST_LOADERS: list[tuple[Callable[..., Bunch], str, int, int, str, int]] = [
    (datasets.load_alicante_murcia, "alicante_murcia", 8, 10000002, "EUR", 36),
    (datasets.load_barcelona, "barcelona", 19, 10000007, "EUR", 190),
    (datasets.load_madrid, "madrid", 18, 10000016, "EUR", 171),
    (datasets.load_valencia, "valencia", 14, 10000022, "EUR", 105),
    (datasets.load_qatar_costs, "qatar_costs", 192, 1, "km", 18336),
]
COST_IDS = [c[1] for c in COST_LOADERS]
COST_FIELDS = {"cost", "time", "distance", "coords", "labels", "depot", "units", "DESCR", "frame"}


def _read_rows(file: str) -> list[dict[str, str]]:
    path = Path(_loaders.__file__).parent / "_data" / "costs" / f"{file}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize(("loader", "file", "n", "depot", "cost_unit", "rows"), COST_LOADERS, ids=COST_IDS)
def test_cost_loader_matrices(loader, file, n, depot, cost_unit, rows):
    b = loader()
    assert isinstance(b, Bunch) and not isinstance(b, TSPBunch)
    assert set(b.keys()) == COST_FIELDS
    for key in ("cost", "time", "distance"):
        M = b[key]
        assert isinstance(M, np.ndarray) and M.shape == (n, n) and M.dtype == np.float64
        assert np.all(np.isfinite(M)) and np.all(M >= 0)
        assert np.array_equal(M, M.T), f"{key} must be symmetric"
        assert np.all(np.diagonal(M) == 0.0)
    if cost_unit == "EUR":
        off = ~np.eye(n, dtype=bool)
        assert np.all(b.cost[off] > 0) and np.all(b.time[off] > 0) and np.all(b.distance[off] > 0)
    assert b.labels.dtype == np.int64 and b.labels.shape == (n,)
    assert len(set(b.labels.tolist())) == n
    assert b.depot == depot and isinstance(b.depot, int) and b.labels[0] == depot
    assert b.coords.shape == (n, 2) and b.coords.dtype == np.float64 and np.all(np.isfinite(b.coords))
    assert b.units == {"cost": cost_unit, "time": "h", "distance": "m"}
    assert b.frame is None
    assert (
        isinstance(b.DESCR, str) and str(depot) in b.DESCR and file.replace("_costs", "") in b.DESCR.lower()
    )
    lat, lon = b.coords[:, 0], b.coords[:, 1]
    if file == "qatar_costs":
        assert np.all((lat > 24) & (lat < 27)) and np.all((lon > 50) & (lon < 52))
        assert len(set(range(1, 194)) - set(b.labels.tolist())) == 1, "ids 1..193 with one absent"
    else:
        assert np.all((lat > 36) & (lat < 44)) and np.all((lon > -10) & (lon < 5))


@pytest.mark.parametrize(("loader", "file", "n", "depot", "cost_unit", "rows"), COST_LOADERS, ids=COST_IDS)
def test_cost_loader_values_match_the_csv(loader, file, n, depot, cost_unit, rows):
    b = loader()
    table = _read_rows(file)
    assert len(table) == rows
    index = {int(lab): k for k, lab in enumerate(b.labels.tolist())}
    order: dict[int, None] = {}
    for row in table:
        o, d = int(float(row["id_origin"])), int(float(row["id_destinity"]))
        order.setdefault(o, None)
        order.setdefault(d, None)
        i, j = index[o], index[d]
        metres = float(row["meters"])
        assert b.distance[i, j] == metres and b.distance[j, i] == metres
        if cost_unit == "EUR":
            assert b.cost[i, j] == float(row["cost"])
            assert b.time[i, j] == float(row["hours"])
            lat_d = float(row["lat_destiniy"])
        else:
            assert b.cost[i, j] == metres / 1000.0
            assert b.time[i, j] == float(row["seconds"]) / 3600.0
            lat_d = float(row["lat_destinity"])
        assert tuple(b.coords[i]) == (float(row["lat_origin"]), float(row["lon_origin"]))
        assert tuple(b.coords[j]) == (lat_d, float(row["lon_destinity"]))
    assert b.labels.tolist() == list(order), "labels in first-appearance order"
    if cost_unit == "EUR":
        assert (b.time == 0).sum() == n, "an hours entry for every pair"
        # time is the `hours` column, NOT secs / 3600: the tables add a fixed 7-minute stop per leg.
        secs = {(int(float(r["id_origin"])), int(float(r["id_destinity"]))): float(r["secs"]) for r in table}
        for (o, d), s in secs.items():
            if o != d:
                assert b.time[index[o], index[d]] * 3600.0 - s == pytest.approx(420.0, abs=1e-6)


def test_qatar_costs_document_their_one_zero_pair():
    q = datasets.load_qatar_costs()
    off = ~np.eye(192, dtype=bool)
    i, j = int(np.flatnonzero(q.labels == 104)[0]), int(np.flatnonzero(q.labels == 111)[0])
    for M in (q.cost, q.time, q.distance):
        zeros = sorted(map(tuple, np.argwhere((M == 0) & off).tolist()))
        assert zeros == sorted([(i, j), (j, i)]), "exactly the pair recorded as 0 m / 0 s in the table"
    assert "(104, 111)" in q.DESCR and "4 km" in q.DESCR
    assert "(104, 111)" in datasets.load_qatar_costs.__doc__


def test_cost_loaders_are_independent_objects():
    a, b = datasets.load_barcelona(), datasets.load_barcelona()
    assert a.cost is not b.cost and np.array_equal(a.cost, b.cost)
    a.cost[0, 1] = -1.0
    assert datasets.load_barcelona().cost[0, 1] > 0


@pytest.mark.parametrize(("loader", "file", "n", "depot", "cost_unit", "rows"), COST_LOADERS, ids=COST_IDS)
def test_as_frame_returns_labelled_dataframes(loader, file, n, depot, cost_unit, rows):
    pd = pytest.importorskip("pandas")
    plain = loader()
    b = loader(as_frame=True)
    for key in ("cost", "time", "distance"):
        df = b[key]
        assert isinstance(df, pd.DataFrame) and df.shape == (n, n)
        assert df.index.tolist() == plain.labels.tolist() == df.columns.tolist()
        np.testing.assert_array_equal(df.to_numpy(), plain[key])
    assert b.cost.loc[depot, depot] == 0.0
    assert isinstance(b.frame, pd.DataFrame)
    columns = QATAR_COLUMNS if cost_unit == "km" else SPANISH_COLUMNS
    assert b.frame.shape == (rows, len(columns)) and b.frame.columns.tolist() == columns
    assert b.frame["id_origin"].dtype.kind == "i" and b.frame["id_destinity"].dtype.kind == "i"
    assert b.frame["meters"].dtype.kind == "f"
    np.testing.assert_array_equal(b.labels, plain.labels)
    np.testing.assert_array_equal(b.coords, plain.coords)
    assert b.depot == depot and b.units == plain.units


def test_as_frame_without_pandas_raises_the_documented_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(
        ImportError, match=r"^pandas is required for as_frame=True: pip install scikit-route\[pandas\]$"
    ):
        datasets.load_valencia(as_frame=True)
    assert datasets.load_valencia().cost.shape == (14, 14), "the default path never imports pandas"


def test_load_costs_qatar_is_a_deprecated_alias_of_the_qatar_table():
    with pytest.warns(DeprecationWarning, match="load_costs_qatar is deprecated since 2.0") as record:
        b = datasets.load_costs_qatar()
    assert record[0].filename == __file__
    q = datasets.load_qatar_costs()
    assert b.cost.shape == (192, 192) and b.depot == 1 and b.units["cost"] == "km"
    np.testing.assert_array_equal(b.cost, q.cost)
    pytest.importorskip("pandas")
    with pytest.warns(DeprecationWarning):
        assert datasets.load_costs_qatar(as_frame=True).frame.shape == (18336, 10)


def test_public_names_are_exported():
    assert set(datasets.__all__) >= {"load_tsp", "list_tsp", "read_tsplib", "read_tsplib_tour", "TSPBunch"}
    assert {w[0].__name__ for w in WRAPPERS} <= set(datasets.__all__)
    assert {c[0].__name__ for c in COST_LOADERS} | {"load_costs_qatar"} <= set(datasets.__all__)
    for name in datasets.__all__:
        obj = getattr(datasets, name)
        assert obj.__doc__, f"{name} lacks a docstring"
