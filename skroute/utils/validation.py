"""Input coercion and state checks shared by every solver.

``coerce_matrix`` and ``coerce_labels`` implement the exact coercion contract of
SPEC §3.3 (ndarray, DataFrame duck-typed, dict-of-dicts); ``check_random_state``
and ``check_is_fitted`` are the two helpers of §3.4.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from ..exceptions import NotFittedError

__all__ = ["check_is_fitted", "check_random_state", "coerce_labels", "coerce_matrix"]


def coerce_matrix(M: Any, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Coerce a cost or time matrix to a C-contiguous float64 array and extract its labels.

    A float64, C-contiguous ndarray is returned as is (no copy): ``RoutingProblem`` keeps such an
    input as a view of the caller's array (see its Notes on aliasing); every other input is converted.

    Parameters
    ----------
    M : (n, n) array-like, DataFrame or dict-of-dicts
        Rows are origins, columns destinations. A DataFrame is recognised by duck
        typing (``to_numpy``, ``index``, ``columns``) and must carry the same labels
        in ``index`` and ``columns``, in the same order. A dict-of-dicts
        ``{i: {j: cost}}`` (the 1.0 input) uses its key order as labels.
    name : str
        Name used in error messages (``"X"`` or ``"time_matrix"``).

    Returns
    -------
    arr : ndarray of shape (n, n), float64, C-contiguous
        The coerced matrix.
    labels : ndarray of shape (n,) or None
        Labels carried by ``M`` (DataFrame index, dict keys) coerced with
        ``coerce_labels``; ``None`` for a plain array.

    Raises
    ------
    ValueError
        If the matrix is not square and 2-D, contains NaN or infinite values, if a
        dict-of-dicts is not square or if a DataFrame's index and columns differ.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.utils.validation import coerce_matrix
    >>> arr, labels = coerce_matrix([[0, 1], [1, 0]], "X")
    >>> arr.dtype.name, arr.flags["C_CONTIGUOUS"], labels
    ('float64', True, None)
    >>> arr, labels = coerce_matrix({"a": {"a": 0, "b": 2}, "b": {"a": 2, "b": 0}}, "X")
    >>> labels.tolist()
    ['a', 'b']
    """
    if isinstance(M, dict):  # legacy dict-of-dicts
        labels = list(M)
        try:
            arr = np.array([[M[i][j] for j in labels] for i in labels], dtype=np.float64)
        except KeyError as e:
            raise ValueError(f"{name}: dict-of-dicts is not square, missing key {e}") from None
        lab: np.ndarray | None = coerce_labels(labels, len(labels))
    elif hasattr(M, "to_numpy") and hasattr(M, "index") and hasattr(M, "columns"):  # DataFrame, duck-typed
        if list(M.index) != list(M.columns):
            raise ValueError(f"{name}: index and columns must hold the same labels in the same order")
        arr, lab = np.ascontiguousarray(M.to_numpy(dtype=np.float64)), coerce_labels(M.index, len(M.index))
    else:
        arr, lab = np.ascontiguousarray(np.asarray(M, dtype=np.float64)), None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, lab


def coerce_labels(seq: Any, n: int) -> np.ndarray:
    """Coerce a sequence of ``n`` unique hashables to a 1-D label array.

    The label dtype is ALWAYS ``int64`` or ``object``, whatever the input path
    (ndarray + ``labels=``, DataFrame index, dict keys), so ``tour_``, ``labels_``
    and ``depot_`` compare equal across paths and ``numpy.array_equal`` never
    mixes kinds. Integer-like labels (numpy or Python ints, never bool) become
    ``int64``; anything else (strings, mixed, tuples) becomes ``object``
    (``np.asarray(["a", "b"])`` would give ``'<U1'`` and a DataFrame index
    ``'object'`` — hence the rule). The items themselves are Python scalars: a numpy
    string array yields ``str`` labels, never ``numpy.str_``.

    Parameters
    ----------
    seq : sequence of hashables
        The labels, in matrix row order.
    n : int
        Expected number of labels.

    Returns
    -------
    labels : ndarray of shape (n,), dtype int64 or object

    Raises
    ------
    ValueError
        If ``seq`` does not hold exactly ``n`` unique items.
    TypeError
        If an item is unhashable (raised by ``set``).

    Examples
    --------
    >>> from skroute.utils.validation import coerce_labels
    >>> coerce_labels([3, 1, 2], 3).dtype.name
    'int64'
    >>> coerce_labels(["a", "b", "c"], 3).dtype.name
    'object'
    >>> coerce_labels([1, "b", (2, 3)], 3).dtype.name
    'object'
    """
    # numpy scalars become Python scalars (a ``'<U1'`` array would otherwise fill the object array
    # with ``numpy.str_``), so every label is a plain ``str``/``int``/tuple whatever the input path
    items = [
        x.item() if isinstance(x, np.generic) else x
        for x in (seq.tolist() if isinstance(seq, np.ndarray) else seq)
    ]
    if len(items) != n or len(set(items)) != n:
        raise ValueError(f"labels must be {n} unique hashables")
    if all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in items):
        return np.array(items, dtype=np.int64)
    out = np.empty(n, dtype=object)
    out[:] = items  # element-wise, so tuple labels are not expanded into a 2-D array
    return out


def check_random_state(seed: Any) -> np.random.Generator:
    """Turn ``seed`` into a ``numpy.random.Generator``.

    Parameters
    ----------
    seed : None, int or numpy.random.Generator
        ``None`` or an integer are passed to ``numpy.random.default_rng``; a
        ``Generator`` is returned as is (and is therefore advanced by the fit that
        uses it). The legacy ``RandomState`` is not accepted.

    Returns
    -------
    rng : numpy.random.Generator

    Raises
    ------
    TypeError
        If ``seed`` is of any other kind (``bool`` included).

    Examples
    --------
    >>> from skroute.utils import check_random_state
    >>> rng = check_random_state(0)
    >>> int(rng.integers(0, 10)) == int(check_random_state(0).integers(0, 10))
    True
    >>> check_random_state(rng) is rng
    True
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or (isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_))):
        return np.random.default_rng(None if seed is None else int(seed))
    raise TypeError("random_state must be None, an int or a numpy.random.Generator")


def check_is_fitted(estimator: Any) -> None:
    """Raise [`NotFittedError`][skroute.exceptions.NotFittedError] unless ``estimator`` has been fitted.

    An estimator is fitted when it carries the ``cost_`` attribute, which
    [`fit`][skroute.base.BaseRouter.fit] sets last.

    Parameters
    ----------
    estimator : BaseRouter
        The estimator to check.

    Raises
    ------
    NotFittedError
        With the message ``"This <Name> instance is not fitted yet. Call 'fit' first."``.

    Examples
    --------
    >>> from skroute.base import BaseRouter
    >>> from skroute.utils import check_is_fitted
    >>> class Identity(BaseRouter):
    ...     def _solve(self, problem, rng):
    ...         return problem.to_index_tour(problem.labels)
    >>> check_is_fitted(Identity())
    Traceback (most recent call last):
        ...
    skroute.exceptions.NotFittedError: This Identity instance is not fitted yet. Call 'fit' first.
    >>> check_is_fitted(Identity().fit([[0, 1, 2], [1, 0, 1], [2, 1, 0]]))
    """
    if not hasattr(estimator, "cost_"):
        name = type(estimator).__name__
        raise NotFittedError(f"This {name} instance is not fitted yet. Call 'fit' first.")
