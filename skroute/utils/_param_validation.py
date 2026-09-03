"""Declarative validation of estimator hyper-parameters (SPEC §3.4, D23).

A solver declares ``_parameter_constraints = {"alpha": [Interval(float, 0.0, 1.0, closed="neither")],
"init": [Options(str, {"nearest_neighbour", "random"}), "array-like"], ...}``; ``BaseRouter.fit``
calls ``validate_parameter_constraints`` before solving. A list of constraints means "any of".

Accepted constraint spellings
-----------------------------
- ``Interval`` — a number in a range, ``type`` in ``{int, float, Integral, Real}``.
- ``Options`` — one of a set of values of a given type.
- ``"array-like"`` — anything with ``__len__``/``shape`` and ``__getitem__`` that is not a string or a dict.
- ``"random_state"`` — ``None``, an int or a ``numpy.random.Generator``.
- ``"boolean"`` — a Python or numpy bool.
- ``"verbose"`` — a non-negative int or a bool.
- ``None`` — the value ``None``.
- ``callable`` — a callable.
- a Python type — an instance of it (``isinstance``).

numpy scalars are accepted wherever Python numbers are (``np.float64`` for float, ``np.integer`` for
int); ``bool`` is never an int.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

__all__ = ["Interval", "Options", "validate_parameter_constraints"]


def _is_bool(val: Any) -> bool:
    return isinstance(val, (bool, np.bool_))


def _is_int(val: Any) -> bool:
    return isinstance(val, Integral) and not _is_bool(val)


def _is_real(val: Any) -> bool:
    return isinstance(val, Real) and not _is_bool(val)


def _type_name(t: Any) -> str:
    if t is Integral or t is int:
        return "int"
    if t is Real or t is float:
        return "float"
    return getattr(t, "__qualname__", None) or getattr(t, "__name__", None) or str(t)


class _Constraint:
    """Base of every constraint object: ``is_satisfied_by`` and a human-readable ``__str__``."""

    def is_satisfied_by(self, val: Any) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"


class Interval(_Constraint):
    """Constraint representing a typed interval.

    Parameters
    ----------
    type : {int, float, numbers.Integral, numbers.Real}
        Kind of number accepted. ``int``/``Integral`` accept Python and numpy
        integers (never bool); ``float``/``Real`` accept any real number (ints
        included, NaN excluded).
    low : number or None
        Lower bound; ``None`` means unbounded below.
    high : number or None
        Upper bound; ``None`` means unbounded above.
    closed : {"both", "left", "right", "neither"}, default "both"
        Which bounds are inclusive.

    Examples
    --------
    >>> from skroute.utils._param_validation import Interval
    >>> c = Interval(float, 0.0, 1.0, closed="neither")
    >>> str(c)
    'a float in the range (0.0, 1.0)'
    >>> c.is_satisfied_by(0.5), c.is_satisfied_by(1.0), c.is_satisfied_by(True)
    (True, False, False)
    >>> Interval(int, 1, None, closed="left").is_satisfied_by(3)
    True
    """

    def __init__(self, type: Any, low: Any, high: Any, closed: str = "both") -> None:
        if type not in (int, float, Integral, Real):
            raise ValueError("type must be int, float, numbers.Integral or numbers.Real")
        if closed not in ("both", "left", "right", "neither"):
            raise ValueError("closed must be 'both', 'left', 'right' or 'neither'")
        if low is not None and high is not None and low > high:
            raise ValueError("low must be <= high")
        self.type = type
        self.low = low
        self.high = high
        self.closed = closed

    def _right_type(self, val: Any) -> bool:
        if self.type in (int, Integral):
            return _is_int(val)
        return _is_real(val) and not math.isnan(val)

    def is_satisfied_by(self, val: Any) -> bool:
        if not self._right_type(val):
            return False
        if self.low is not None:
            if self.closed in ("both", "left"):
                if val < self.low:
                    return False
            elif val <= self.low:
                return False
        if self.high is not None:
            if self.closed in ("both", "right"):
                if val > self.high:
                    return False
            elif val >= self.high:
                return False
        return True

    def __str__(self) -> str:
        lb = "[" if self.closed in ("both", "left") else "("
        rb = "]" if self.closed in ("both", "right") else ")"
        low = "-inf" if self.low is None else repr(self.low)
        high = "inf" if self.high is None else repr(self.high)
        article = "an" if _type_name(self.type) == "int" else "a"
        return f"{article} {_type_name(self.type)} in the range {lb}{low}, {high}{rb}"


class Options(_Constraint):
    """Constraint representing a finite set of accepted values of one type.

    Parameters
    ----------
    type : type
        The values must be instances of this type (``str`` most of the time).
    options : set
        The accepted values.

    Examples
    --------
    >>> from skroute.utils._param_validation import Options
    >>> c = Options(str, {"greedy", "optimal"})
    >>> str(c)
    "a str among {'greedy', 'optimal'}"
    >>> c.is_satisfied_by("greedy"), c.is_satisfied_by("both")
    (True, False)
    """

    def __init__(self, type: Any, options: Iterable[Any]) -> None:
        self.type = type
        self.options = set(options)

    def is_satisfied_by(self, val: Any) -> bool:
        try:
            return isinstance(val, self.type) and val in self.options
        except TypeError:  # unhashable value
            return False

    def __str__(self) -> str:
        opts = ", ".join(sorted(repr(o) for o in self.options))
        return f"a {_type_name(self.type)} among {{{opts}}}"


class _ArrayLike(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        if isinstance(val, (str, bytes, Mapping)):
            return False
        return (hasattr(val, "__len__") or hasattr(val, "shape")) and hasattr(val, "__getitem__")

    def __str__(self) -> str:
        return "an array-like"


class _RandomState(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        return val is None or _is_int(val) or isinstance(val, np.random.Generator)

    def __str__(self) -> str:
        return "None, an int or a numpy.random.Generator"


class _Boolean(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        return _is_bool(val)

    def __str__(self) -> str:
        return "a boolean"


class _Verbose(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        return _is_bool(val) or (_is_int(val) and val >= 0)

    def __str__(self) -> str:
        return "an int >= 0 or a boolean"


class _NoneConstraint(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        return val is None

    def __str__(self) -> str:
        return "None"


class _Callable(_Constraint):
    def is_satisfied_by(self, val: Any) -> bool:
        return callable(val)

    def __str__(self) -> str:
        return "a callable"


class _InstanceOf(_Constraint):
    def __init__(self, type: type) -> None:
        self.type = type

    def is_satisfied_by(self, val: Any) -> bool:
        if self.type in (int, Integral):
            return _is_int(val)
        if self.type in (float, Real):
            return _is_real(val)
        if self.type is bool:
            return _is_bool(val)
        return isinstance(val, self.type)

    def __str__(self) -> str:
        name = _type_name(self.type)
        article = "an" if name[:1].lower() in "aeiou" else "a"
        return f"{article} {name}"


_STRING_CONSTRAINTS: dict[str, _Constraint] = {
    "array-like": _ArrayLike(),
    "random_state": _RandomState(),
    "boolean": _Boolean(),
    "verbose": _Verbose(),
}


def make_constraint(constraint: Any) -> _Constraint:
    """Turn one constraint spelling into a constraint object."""
    if isinstance(constraint, _Constraint):
        return constraint
    if constraint is None:
        return _NoneConstraint()
    if constraint is callable:
        return _Callable()
    if isinstance(constraint, str):
        try:
            return _STRING_CONSTRAINTS[constraint]
        except KeyError:
            raise ValueError(f"Unknown constraint {constraint!r}") from None
    if isinstance(constraint, type):
        return _InstanceOf(constraint)
    raise ValueError(f"Unknown constraint {constraint!r}")


def validate_parameter_constraints(
    constraints: Mapping[str, Any], params: Mapping[str, Any], caller_name: str
) -> None:
    """Validate hyper-parameters against their declared constraints.

    Parameters
    ----------
    constraints : dict of {str: constraint or list of constraints}
        The ``_parameter_constraints`` of the estimator. A list means "any of".
        Parameters without an entry are not validated.
    params : dict of {str: object}
        The estimator's parameters, typically ``est.get_params(deep=False)``.
    caller_name : str
        Name of the estimator, used in the error message.

    Raises
    ------
    ValueError
        With sklearn's message format: ``The 'alpha' parameter of SimulatedAnnealing
        must be a float in the range (0.0, 1.0). Got 1.5 instead.``

    Examples
    --------
    >>> from skroute.utils._param_validation import Interval, validate_parameter_constraints
    >>> constraints = {"alpha": [Interval(float, 0.0, 1.0, closed="neither")]}
    >>> validate_parameter_constraints(constraints, {"alpha": 0.5}, "SimulatedAnnealing")
    >>> validate_parameter_constraints(constraints, {"alpha": 1.5}, "SA")
    Traceback (most recent call last):
        ...
    ValueError: The 'alpha' parameter of SA must be a float in the range (0.0, 1.0). Got 1.5 instead.
    """
    for name, spec in constraints.items():
        if name not in params:
            continue
        value = params[name]
        alternatives: Sequence[Any] = spec if isinstance(spec, (list, tuple)) else [spec]
        objs = [make_constraint(c) for c in alternatives]
        if any(c.is_satisfied_by(value) for c in objs):
            continue
        texts = [str(c) for c in objs]
        if len(texts) == 1:
            wanted = texts[0]
        else:
            wanted = ", ".join(texts[:-1]) + f" or {texts[-1]}"
        raise ValueError(f"The {name!r} parameter of {caller_name} must be {wanted}. Got {value!r} instead.")
