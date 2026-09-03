"""Pure-Python readers for the TSPLIB 95 ``.tsp`` and ``.tour`` formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from skroute.utils import Bunch

__all__ = ["read_tsplib", "read_tsplib_tour"]

#: Edge-weight types read from a ``NODE_COORD_SECTION`` (coordinates are kept raw; the conversion
#: to distances happens in :func:`skroute.preprocessing.distance_matrix`).
COORD_TYPES = ("EUC_2D", "CEIL_2D", "MAN_2D", "ATT", "GEO")

#: ``EDGE_WEIGHT_FORMAT`` values accepted for ``EDGE_WEIGHT_TYPE: EXPLICIT``.
EXPLICIT_FORMATS = ("FULL_MATRIX", "UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW")

_SECTIONS = frozenset(
    {
        "NODE_COORD_SECTION",
        "EDGE_WEIGHT_SECTION",
        "DISPLAY_DATA_SECTION",
        "TOUR_SECTION",
        "DEPOT_SECTION",
        "DEMAND_SECTION",
        "EDGE_DATA_SECTION",
        "FIXED_EDGES_SECTION",
    }
)

_BOM = "﻿"


def _decode(data: bytes) -> str:
    """UTF-8 first (dropping a BOM), latin-1 as the fallback.

    TSPLIB files are ASCII; the accented comments of a few are latin-1, and files
    re-saved by Windows editors carry a UTF-8 BOM that, decoded as latin-1, would
    glue ``ï»¿`` to the first keyword and silently lose ``NAME``.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _read_text(path_or_file: str | os.PathLike[str] | IO[str] | IO[bytes]) -> str:
    if hasattr(path_or_file, "read"):
        data = path_or_file.read()
        text = _decode(data) if isinstance(data, bytes) else str(data)
    else:
        text = _decode(Path(path_or_file).read_bytes())
    return text.lstrip(_BOM)  # a text file object opened as plain "utf-8" yields the BOM as a character


def _parse(text: str) -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    """Split a TSPLIB file into specification entries, comment lines and section tokens.

    Tolerates ``KEY : value`` and ``KEY: value``, CRLF line endings, indentation,
    a trailing colon after a section keyword, data tokens on the same line as the
    section keyword (``TOUR_SECTION 1 2 3 -1``), a specification line between a
    section's header and its data, and a missing ``EOF``.
    """
    spec: dict[str, str] = {}
    comments: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        first = parts[0].rstrip(":").upper()
        if first == "EOF":
            break
        if first in _SECTIONS:
            current = first
            tokens = sections.setdefault(current, [])
            if len(parts) > 1:  # data written on the keyword line
                tokens.extend(parts[1].lstrip(":").split())
            continue
        if ":" in line and first[:1].isalpha():
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if key == "COMMENT":
                comments.append(value)
            else:
                spec[key] = value
            continue  # ``current`` is kept: a keyword line does not close the section around it
        if current is None:
            raise ValueError(f"cannot parse line {raw!r}: data outside any section")
        sections[current].extend(line.split())
    return spec, comments, sections


def _to_float(token: str, what: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"{what}: {token!r} is not a number") from None


def _to_int(token: str, what: str) -> int:
    value = _to_float(token, what)
    if value != int(value):
        raise ValueError(f"{what}: {token!r} is not an integer")
    return int(value)


def _coord_section(
    tokens: list[str], dimension: int | None, what: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if not tokens:
        raise ValueError(f"{what} is empty")
    if len(tokens) % 3 != 0:
        raise ValueError(f"{what} must hold 'id x y' triples; got {len(tokens)} numbers")
    n = len(tokens) // 3
    if dimension is not None and n != dimension:
        raise ValueError(f"{what} has {n} nodes but DIMENSION is {dimension}")
    ids = np.fromiter((_to_int(tokens[3 * k], what) for k in range(n)), dtype=np.int64, count=n)
    unique, counts = np.unique(ids, return_counts=True)
    if unique.shape[0] != n:
        raise ValueError(f"{what} repeats node id {int(unique[counts > 1][0])}")
    xy = np.empty((n, 2), dtype=np.float64)
    for k in range(n):
        xy[k, 0] = _to_float(tokens[3 * k + 1], what)
        xy[k, 1] = _to_float(tokens[3 * k + 2], what)
    return ids, xy


def _explicit_matrix(tokens: list[str], n: int, fmt: str) -> NDArray[np.float64]:
    values = np.fromiter(
        (_to_float(t, "EDGE_WEIGHT_SECTION") for t in tokens), dtype=np.float64, count=len(tokens)
    )
    expected = {
        "FULL_MATRIX": n * n,
        "UPPER_ROW": n * (n - 1) // 2,
        "LOWER_ROW": n * (n - 1) // 2,
        "UPPER_DIAG_ROW": n * (n + 1) // 2,
        "LOWER_DIAG_ROW": n * (n + 1) // 2,
    }[fmt]
    if values.shape[0] != expected:
        raise ValueError(
            f"EDGE_WEIGHT_SECTION in format {fmt} for DIMENSION {n} needs {expected} numbers; "
            f"got {values.shape[0]}"
        )
    if fmt == "FULL_MATRIX":
        return values.reshape(n, n)
    cost = np.zeros((n, n), dtype=np.float64)
    if fmt == "UPPER_ROW":
        i, j = np.triu_indices(n, k=1)
    elif fmt == "LOWER_ROW":
        i, j = np.tril_indices(n, k=-1)
    elif fmt == "UPPER_DIAG_ROW":
        i, j = np.triu_indices(n, k=0)
    else:  # LOWER_DIAG_ROW
        i, j = np.tril_indices(n, k=0)
    cost[i, j] = values
    cost[j, i] = values
    return cost


def read_tsplib(path_or_file: str | os.PathLike[str] | IO[str] | IO[bytes]) -> Bunch:
    """Read a TSPLIB 95 ``.tsp`` file (coordinates or an explicit matrix).

    Parameters
    ----------
    path_or_file : path-like or file object
        Path of the file, or an open text/binary file object.

    Returns
    -------
    Bunch
        With the fields

        - ``name``, ``comment`` (``COMMENT`` lines joined with newlines; ``None`` when absent),
          ``type`` (``"TSP"``, ``"ATSP"``, ...: the first word of the ``TYPE`` entry, so a
          trailing author note is dropped), ``dimension`` (``int``),
          ``edge_weight_type``, ``edge_weight_format`` (``None`` unless ``EXPLICIT``);
        - ``coords``: ``float64 (n, 2)`` for the coordinate types ``EUC_2D``, ``CEIL_2D``,
          ``MAN_2D``, ``ATT`` and ``GEO`` (kept raw -- ``GEO`` stays in ``DDD.MM``
          notation; convert with :func:`skroute.preprocessing.distance_matrix` and the
          matching ``tsplib_*`` metric), otherwise ``None``;
        - ``cost``: ``float64 (n, n)`` for ``EXPLICIT`` in the formats ``FULL_MATRIX``,
          ``UPPER_ROW``, ``LOWER_ROW``, ``UPPER_DIAG_ROW`` and ``LOWER_DIAG_ROW``
          (triangular formats are mirrored), otherwise ``None``;
        - ``display_coords``: ``float64 (n, 2)`` from a ``DISPLAY_DATA_SECTION``, else ``None``;
        - ``labels``: ``int64 (n,)`` node ids exactly as written in the file (1-based in
          every TSPLIB instance), in file order; they must be unique.

    Raises
    ------
    ValueError
        For an unsupported ``EDGE_WEIGHT_TYPE`` (``"EDGE_WEIGHT_TYPE EUC_3D is not
        supported in this version"``), a missing or unsupported ``EDGE_WEIGHT_FORMAT``
        with ``EXPLICIT``, a ``DIMENSION`` below 1 or disagreeing with a section's size,
        a repeated node id, or a malformed line.

    Notes
    -----
    Tolerant of ``KEY : value`` and ``KEY: value`` (``dj38.tsp`` mixes both), CRLF
    line endings, indented data lines, data tokens on the section keyword's own line,
    a UTF-8 BOM, UTF-8 or latin-1 text, and a missing ``EOF``. Sections this reader
    does not use (``DEMAND_SECTION``, ``FIXED_EDGES_SECTION``, ...) are skipped;
    check ``type`` when reading anything other than a plain TSP. Pure Python: no
    pandas, no regular expressions.

    References
    ----------
    G. Reinelt, *TSPLIB 95*, Universitaet Heidelberg, 1995.

    Examples
    --------
    >>> import io
    >>> from skroute.datasets import read_tsplib
    >>> text = "NAME: tiny\\nTYPE: TSP\\nDIMENSION: 3\\nEDGE_WEIGHT_TYPE: EUC_2D\\n" \\
    ...        "NODE_COORD_SECTION\\n1 0 0\\n2 3 4\\n3 6 8\\nEOF\\n"
    >>> b = read_tsplib(io.StringIO(text))
    >>> b.name, b.dimension, b.edge_weight_type, b.coords.shape, b.labels.tolist()
    ('tiny', 3, 'EUC_2D', (3, 2), [1, 2, 3])
    >>> text = "NAME: m3\\nTYPE: TSP\\nDIMENSION: 3\\nEDGE_WEIGHT_TYPE: EXPLICIT\\n" \\
    ...        "EDGE_WEIGHT_FORMAT: UPPER_ROW\\nEDGE_WEIGHT_SECTION\\n5 9\\n4\\nEOF\\n"
    >>> read_tsplib(io.StringIO(text)).cost.tolist()
    [[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]]
    """
    spec, comments, sections = _parse(_read_text(path_or_file))

    dimension: int | None = _to_int(spec["DIMENSION"], "DIMENSION") if "DIMENSION" in spec else None
    if dimension is not None and dimension < 1:
        raise ValueError(f"DIMENSION must be a positive integer; got {dimension}")
    ewt = spec.get("EDGE_WEIGHT_TYPE", "").upper() or None
    fmt = spec.get("EDGE_WEIGHT_FORMAT", "").upper() or None

    coords: NDArray[np.float64] | None = None
    cost: NDArray[np.float64] | None = None
    labels: NDArray[np.int64] | None = None
    display: NDArray[np.float64] | None = None

    if "DISPLAY_DATA_SECTION" in sections:
        labels, display = _coord_section(sections["DISPLAY_DATA_SECTION"], dimension, "DISPLAY_DATA_SECTION")

    if ewt is None:
        raise ValueError("EDGE_WEIGHT_TYPE is missing")
    if ewt in COORD_TYPES:
        if "NODE_COORD_SECTION" not in sections:
            raise ValueError(f"EDGE_WEIGHT_TYPE {ewt} needs a NODE_COORD_SECTION")
        labels, coords = _coord_section(sections["NODE_COORD_SECTION"], dimension, "NODE_COORD_SECTION")
        dimension = coords.shape[0]
    elif ewt == "EXPLICIT":
        if "EDGE_WEIGHT_SECTION" not in sections:
            raise ValueError("EDGE_WEIGHT_TYPE EXPLICIT needs an EDGE_WEIGHT_SECTION")
        if fmt is None:
            raise ValueError("EDGE_WEIGHT_FORMAT is required with EDGE_WEIGHT_TYPE EXPLICIT")
        if fmt not in EXPLICIT_FORMATS:
            raise ValueError(f"EDGE_WEIGHT_FORMAT {fmt} is not supported in this version")
        if dimension is None:
            raise ValueError("DIMENSION is required with EDGE_WEIGHT_TYPE EXPLICIT")
        cost = _explicit_matrix(sections["EDGE_WEIGHT_SECTION"], dimension, fmt)
        if labels is None:
            labels = np.arange(1, dimension + 1, dtype=np.int64)
    else:
        raise ValueError(f"EDGE_WEIGHT_TYPE {ewt} is not supported in this version")

    return Bunch(
        name=spec.get("NAME"),
        comment="\n".join(comments) if comments else None,
        type=(spec.get("TYPE") or "TSP").split()[0].upper(),
        dimension=int(dimension),
        edge_weight_type=ewt,
        edge_weight_format=fmt if ewt == "EXPLICIT" else None,
        coords=coords,
        cost=cost,
        display_coords=display,
        labels=labels,
    )


def read_tsplib_tour(path_or_file: str | os.PathLike[str] | IO[str] | IO[bytes]) -> NDArray[np.int64]:
    """Read a TSPLIB 95 ``.tour`` file and return the node ids of its first tour.

    Parameters
    ----------
    path_or_file : path-like or file object
        Path of the file, or an open text/binary file object.

    Returns
    -------
    ndarray of int64, shape (n,)
        The ids of the ``TOUR_SECTION`` in order (1-based, exactly as written), up to
        the ``-1`` terminator; the closing return to the first node is implicit. Ids
        written on the ``TOUR_SECTION`` line itself count.

    Raises
    ------
    ValueError
        If the file has no ``TOUR_SECTION``, the tour is empty, or its length
        disagrees with a ``DIMENSION`` entry.

    Examples
    --------
    >>> import io
    >>> from skroute.datasets import read_tsplib_tour
    >>> text = "NAME: tiny.tour\\nTYPE: TOUR\\nDIMENSION: 3\\nTOUR_SECTION\\n1\\n3\\n2\\n-1\\nEOF\\n"
    >>> read_tsplib_tour(io.StringIO(text)).tolist()
    [1, 3, 2]
    """
    spec, _, sections = _parse(_read_text(path_or_file))
    if "TOUR_SECTION" not in sections:
        raise ValueError("the file has no TOUR_SECTION")
    ids: list[int] = []
    for token in sections["TOUR_SECTION"]:
        value = _to_int(token, "TOUR_SECTION")
        if value == -1:
            break
        ids.append(value)
    if not ids:
        raise ValueError("TOUR_SECTION holds no node ids")
    if "DIMENSION" in spec:
        dimension = _to_int(spec["DIMENSION"], "DIMENSION")
        if dimension != len(ids):
            raise ValueError(f"the tour has {len(ids)} nodes but DIMENSION is {dimension}")
    return np.asarray(ids, dtype=np.int64)
