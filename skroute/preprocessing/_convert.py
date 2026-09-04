"""Conversions between long tables, dict-of-dicts and dense matrices; coordinate normalisation."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skroute.utils.validation import coerce_labels

from ._distances import _check_coords

__all__ = [
    "from_dict_of_dicts",
    "normalize_coords",
    "pairs_to_matrix",
    "to_dict_of_dicts",
]


def _as_label_list(values: Any) -> list[Any]:
    """Sequence / Series / 1-D array -> list of hashable Python scalars, each kept as given.

    A plain sequence is never passed through ``np.asarray``: numpy would promote mixed
    labels (``[1, "a"]`` -> ``["1", "a"]``, ``[1, 2.0]`` -> ``[1.0, 2.0]``) and the ``1``
    would no longer match the ``depot=1`` the user passes later. numpy scalars are
    unboxed to Python scalars; arrays and pandas objects go through ``tolist()``, which
    does the same (an ``object`` array keeps its elements).
    """
    if isinstance(values, np.ndarray):
        arr = values
    elif hasattr(values, "to_numpy"):  # pandas Series / Index (``object`` dtype when mixed)
        arr = np.asarray(values.to_numpy())
    else:
        try:
            items = [x.item() if isinstance(x, np.generic) else x for x in values]
        except TypeError:
            raise ValueError(f"expected a one-dimensional sequence of labels; got {values!r}") from None
        if any(isinstance(x, (list, np.ndarray)) for x in items):
            raise ValueError("expected a one-dimensional sequence of labels; got nested sequences")
        return items
    if arr.ndim != 1:
        raise ValueError(f"expected a one-dimensional sequence of labels; got shape {arr.shape}")
    return arr.tolist()


def pairs_to_matrix(
    origin: ArrayLike,
    destination: ArrayLike,
    value: ArrayLike,
    *,
    symmetric: bool = True,
    labels: ArrayLike | None = None,
    fill: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Pivot a long table of ``(origin, destination, value)`` rows into a dense matrix.

    This is the 2.0 replacement of ``dfcolumn_to_dict``, whose result depended on the
    row order of the table. Here every row is placed by its labels, so the order of
    the rows is irrelevant.

    Parameters
    ----------
    origin, destination : sequence of hashable, length m
        Node labels of each row (lists, arrays or DataFrame columns of equal length).
    value : sequence of float, length m
        The quantity of each row (cost, seconds, metres, ...).
    symmetric : bool, default=True
        When ``True`` an entry ``(i, j)`` that is absent from the table takes the
        value of ``(j, i)`` when that one is present. Entries given in both
        directions are kept as given (the result is then only symmetric if the
        data is).
    labels : sequence of hashable, optional
        Order of the rows/columns of the matrix. Defaults to the order of first
        appearance while scanning the rows (origin, then destination). Every
        origin and destination must be in ``labels``.
    fill : float, optional
        Value for the entries that are neither in the table nor (with
        ``symmetric=True``) mirrored. Without it, such an entry raises
        ``ValueError``. The diagonal is exempt: a missing ``(i, i)`` is 0.

    Returns
    -------
    matrix : ndarray of shape (n, n), dtype float64
        Dense matrix in the order of ``labels``.
    labels : ndarray of shape (n,), dtype int64 or object
        The labels, in the order used for the matrix and exactly as given (a label
        ``1`` next to a label ``"a"`` stays the integer ``1``): ``int64`` when every
        label is an integer, ``object`` otherwise -- the rule of every label array
        in scikit-route (`skroute.utils.validation.coerce_labels`).

    Raises
    ------
    ValueError
        If the three sequences differ in length, a label is missing from ``labels``,
        or an off-diagonal entry is missing without ``fill``.

    Notes
    -----
    When the same ``(origin, destination)`` pair appears several times the last
    row wins.

    Examples
    --------
    >>> from skroute.preprocessing import pairs_to_matrix
    >>> M, labels = pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0])
    >>> labels.tolist()
    [1, 2, 3]
    >>> M.tolist()
    [[0.0, 5.0, 9.0], [5.0, 0.0, 4.0], [9.0, 4.0, 0.0]]
    >>> M, _ = pairs_to_matrix(["a", "b"], ["b", "a"], [1.0, 2.0], symmetric=False)
    >>> M.tolist()
    [[0.0, 1.0], [2.0, 0.0]]
    """
    orig = _as_label_list(origin)
    dest = _as_label_list(destination)
    vals = np.asarray(value, dtype=np.float64).ravel()
    if not (len(orig) == len(dest) == vals.shape[0]):
        raise ValueError(
            "origin, destination and value must have the same length; "
            f"got {len(orig)}, {len(dest)} and {vals.shape[0]}"
        )

    if labels is None:
        order: dict[Any, None] = {}
        for o, d in zip(orig, dest, strict=True):
            order.setdefault(o, None)
            order.setdefault(d, None)
        label_list = list(order)
    else:
        label_list = _as_label_list(labels)
        if len(set(label_list)) != len(label_list):
            raise ValueError("labels must be unique")
    index = {lab: k for k, lab in enumerate(label_list)}
    n = len(label_list)

    matrix = np.zeros((n, n), dtype=np.float64)
    seen = np.zeros((n, n), dtype=bool)
    for o, d, v in zip(orig, dest, vals, strict=True):
        try:
            i, j = index[o], index[d]
        except KeyError as exc:
            raise ValueError(f"label {exc.args[0]!r} is not in labels") from None
        matrix[i, j] = v
        seen[i, j] = True

    if symmetric:
        mirror = seen.T & ~seen
        matrix[mirror] = matrix.T[mirror]
        seen |= mirror

    np.fill_diagonal(seen, True)  # a missing (i, i) defaults to 0
    missing = ~seen
    if missing.any():
        if fill is None:
            i, j = np.argwhere(missing)[0]
            raise ValueError(
                f"{int(missing.sum())} entries are missing from the table, e.g. "
                f"({label_list[i]!r}, {label_list[j]!r}); pass fill= to complete them"
                + ("" if symmetric else " or symmetric=True to mirror the reverse direction")
            )
        matrix[missing] = float(fill)

    return matrix, coerce_labels(label_list, n)


def to_dict_of_dicts(matrix: ArrayLike, labels: ArrayLike | None = None) -> dict[Any, dict[Any, float]]:
    """Dense matrix -> ``{label_i: {label_j: value}}`` (the legacy dict-of-dicts input format).

    Parameters
    ----------
    matrix : array-like of shape (n, n)
        Square matrix.
    labels : sequence of hashable of length n, optional
        Keys of the dictionaries, unique; defaults to ``0..n-1``.

    Returns
    -------
    dict
        Outer and inner keys in the order of ``labels``; values are Python floats.

    Raises
    ------
    ValueError
        If the matrix is not square, or ``labels`` has the wrong length or repeats a
        label (two equal keys would silently collapse two rows into one).

    Examples
    --------
    >>> from skroute.preprocessing import to_dict_of_dicts
    >>> to_dict_of_dicts([[0, 5], [5, 0]], labels=["a", "b"])
    {'a': {'a': 0.0, 'b': 5.0}, 'b': {'a': 5.0, 'b': 0.0}}
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square; got shape {m.shape}")
    n = m.shape[0]
    keys = list(range(n)) if labels is None else _as_label_list(labels)
    if len(keys) != n:
        raise ValueError(f"labels has {len(keys)} entries but the matrix has {n} rows")
    if len(set(keys)) != n:
        raise ValueError("labels must be unique")
    return {ki: {kj: float(m[i, j]) for j, kj in enumerate(keys)} for i, ki in enumerate(keys)}


def from_dict_of_dicts(
    d: Mapping[Hashable, Mapping[Hashable, float]],
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """``{label_i: {label_j: value}}`` -> ``(matrix, labels)`` in the order of the outer keys.

    Parameters
    ----------
    d : mapping of mapping
        Outer keys are the node labels; each inner mapping gives the values of that
        row. A missing diagonal entry defaults to 0 (1.0's ``matrix_to_dict``
        dropped zeros); any other missing entry raises ``ValueError``, and so does
        an inner key that is not an outer key.

    Returns
    -------
    matrix : ndarray of shape (n, n), dtype float64
    labels : ndarray of shape (n,), dtype int64 or object
        The outer keys as given: ``int64`` when all are integers, ``object``
        otherwise (strings, tuples, mixed types), as `pairs_to_matrix`.

    Examples
    --------
    >>> from skroute.preprocessing import from_dict_of_dicts
    >>> M, labels = from_dict_of_dicts({1: {2: 5.0}, 2: {1: 5.0}})
    >>> M.tolist(), labels.tolist()
    ([[0.0, 5.0], [5.0, 0.0]], [1, 2])
    """
    keys = list(d)
    if not keys:
        raise ValueError("the dict-of-dicts is empty")
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    matrix = np.zeros((n, n), dtype=np.float64)
    for ki, row in d.items():
        i = index[ki]
        if not isinstance(row, Mapping):
            raise ValueError(f"the value of key {ki!r} is not a mapping")
        for kj, v in row.items():
            if kj not in index:
                raise ValueError(f"inner key {kj!r} (row {ki!r}) is not an outer key")
            matrix[i, index[kj]] = float(v)
        missing = [kj for kj in keys if kj not in row and kj != ki]
        if missing:
            raise ValueError(f"row {ki!r} lacks the entries {missing[:5]!r}")
    return matrix, coerce_labels(keys, n)


def normalize_coords(coords: ArrayLike) -> NDArray[np.float64]:
    """Aspect-preserving min-max scaling of planar coordinates into the unit square.

    Both axes are shifted to start at 0 and divided by the *same* factor, the
    larger of the two ranges, so the longer side of the bounding box spans
    ``[0, 1]`` and the shorter one ``[0, r]`` with ``r <= 1``. Distances are
    scaled uniformly and the geometry of the instance is preserved (this is what
    `SOM` feeds its ring with).

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Finite coordinates, at least one row.

    Returns
    -------
    ndarray of shape (n, 2), dtype float64
        Coordinates in ``[0, 1]**2``; all zeros when every point coincides.

    Raises
    ------
    ValueError
        For a shape other than ``(n, 2)``, an empty input or a non-finite
        coordinate -- the same checks and messages as `distance_matrix`.

    Examples
    --------
    >>> from skroute.preprocessing import normalize_coords
    >>> normalize_coords([[10.0, 10.0], [30.0, 20.0]]).tolist()
    [[0.0, 0.0], [1.0, 0.5]]
    """
    xy = _check_coords(coords)
    lo = xy.min(axis=0)
    span = float((xy.max(axis=0) - lo).max())
    if span == 0.0:
        return np.zeros_like(xy)
    return (xy - lo) / span
