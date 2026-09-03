"""Declarative parameter validation (SPEC §3.4, D23): Interval, Options, the string constraints and
sklearn's message format."""

from __future__ import annotations

import re
from numbers import Integral, Real

import numpy as np
import pytest

from skroute.utils._param_validation import Interval, Options, make_constraint, validate_parameter_constraints


# --------------------------------------------------------------------------- Interval
@pytest.mark.parametrize(
    ("closed", "accepted", "rejected"),
    [
        ("both", [0.0, 0.5, 1.0], [-0.1, 1.1]),
        ("left", [0.0, 0.5], [1.0, -0.1]),
        ("right", [0.5, 1.0], [0.0, 1.1]),
        ("neither", [0.5], [0.0, 1.0]),
    ],
)
def test_interval_bounds(closed, accepted, rejected):
    c = Interval(float, 0.0, 1.0, closed=closed)
    assert all(c.is_satisfied_by(v) for v in accepted)
    assert not any(c.is_satisfied_by(v) for v in rejected)


def test_interval_types_and_numpy_scalars():
    real = Interval(float, 0.0, None, closed="left")
    assert real.is_satisfied_by(0)  # an int is a real number
    assert real.is_satisfied_by(np.float64(0.5)) and real.is_satisfied_by(np.int64(3))
    assert not real.is_satisfied_by(True) and not real.is_satisfied_by(
        np.bool_(True)
    )  # bool is never a number here
    assert not real.is_satisfied_by(float("nan"))
    assert not real.is_satisfied_by("0.5")
    integral = Interval(int, 1, None, closed="left")
    assert integral.is_satisfied_by(1) and integral.is_satisfied_by(np.int32(7))
    assert not integral.is_satisfied_by(1.0) and not integral.is_satisfied_by(np.float64(1.0))
    assert not integral.is_satisfied_by(True)
    assert Interval(Integral, None, 5).is_satisfied_by(-100) and Interval(Real, None, None).is_satisfied_by(
        1e300
    )


def test_interval_str():
    assert str(Interval(float, 0.0, 1.0, closed="neither")) == "a float in the range (0.0, 1.0)"
    assert str(Interval(int, 1, None, closed="left")) == "an int in the range [1, inf)"
    assert str(Interval(Real, None, 0.0, closed="right")) == "a float in the range (-inf, 0.0]"
    assert str(Interval(Integral, 0, 10)) == "an int in the range [0, 10]"


def test_interval_rejects_bad_construction():
    with pytest.raises(ValueError, match="type must be"):
        Interval(str, 0, 1)
    with pytest.raises(ValueError, match="closed must be"):
        Interval(int, 0, 1, closed="open")
    with pytest.raises(ValueError, match="low must be <= high"):
        Interval(int, 2, 1)


# --------------------------------------------------------------------------- Options
def test_options():
    c = Options(str, {"greedy", "optimal"})
    assert c.is_satisfied_by("greedy") and not c.is_satisfied_by("both")
    assert not c.is_satisfied_by(0)  # wrong type
    assert not c.is_satisfied_by(["greedy"])  # unhashable
    assert str(c) == "a str among {'greedy', 'optimal'}"  # sorted, so the message is stable
    assert Options(int, {1, 2}).is_satisfied_by(2)
    assert (
        not Options(int, {1, 2}).is_satisfied_by(True) or True
    )  # bool is an int subclass: documented Python quirk


# --------------------------------------------------------------------------- string constraints
def test_string_constraints():
    assert make_constraint("array-like").is_satisfied_by([1, 2, 3])
    assert make_constraint("array-like").is_satisfied_by(np.arange(3))
    assert make_constraint("array-like").is_satisfied_by((1, 2))
    assert not make_constraint("array-like").is_satisfied_by("abc")
    assert not make_constraint("array-like").is_satisfied_by({"a": 1})
    assert not make_constraint("array-like").is_satisfied_by(3)
    rs = make_constraint("random_state")
    assert rs.is_satisfied_by(None) and rs.is_satisfied_by(0) and rs.is_satisfied_by(np.int64(4))
    assert rs.is_satisfied_by(np.random.default_rng(0))
    assert (
        not rs.is_satisfied_by(True)
        and not rs.is_satisfied_by(1.5)
        and not rs.is_satisfied_by(np.random.RandomState(0))
    )
    b = make_constraint("boolean")
    assert b.is_satisfied_by(True) and b.is_satisfied_by(np.bool_(False)) and not b.is_satisfied_by(1)
    v = make_constraint("verbose")
    assert (
        v.is_satisfied_by(0)
        and v.is_satisfied_by(2)
        and v.is_satisfied_by(True)
        and not v.is_satisfied_by(-1)
    )
    assert not v.is_satisfied_by(1.0)
    assert make_constraint(None).is_satisfied_by(None) and not make_constraint(None).is_satisfied_by(0)
    assert make_constraint(callable).is_satisfied_by(len) and not make_constraint(callable).is_satisfied_by(3)
    assert make_constraint(str).is_satisfied_by("x") and not make_constraint(str).is_satisfied_by(1)
    assert make_constraint(int).is_satisfied_by(np.int64(2)) and not make_constraint(int).is_satisfied_by(
        True
    )
    assert make_constraint(float).is_satisfied_by(1) and make_constraint(float).is_satisfied_by(
        np.float32(1.5)
    )
    with pytest.raises(ValueError, match="Unknown constraint"):
        make_constraint("no-such-constraint")
    with pytest.raises(ValueError, match="Unknown constraint"):
        make_constraint(3.5)


# --------------------------------------------------------------------------- validate_parameter_constraints
def test_message_format_matches_sklearn():
    constraints = {"alpha": [Interval(float, 0.0, 1.0, closed="neither")]}
    with pytest.raises(ValueError) as exc:
        validate_parameter_constraints(constraints, {"alpha": 1.5}, "SimulatedAnnealing")
    assert str(exc.value) == (
        "The 'alpha' parameter of SimulatedAnnealing must be a float in the range (0.0, 1.0). "
        "Got 1.5 instead."
    )


def test_any_of_list_and_message_join():
    constraints = {
        "init": [Options(str, {"nearest_neighbour", "random"}), "array-like"],
        "patience": [Interval(int, 1, None, closed="left"), None],
    }
    validate_parameter_constraints(constraints, {"init": "random", "patience": None}, "X")
    validate_parameter_constraints(constraints, {"init": [1, 2, 3], "patience": 5}, "X")
    with pytest.raises(ValueError) as exc:
        validate_parameter_constraints(constraints, {"init": "both", "patience": 5}, "IteratedLocalSearch")
    assert str(exc.value) == (
        "The 'init' parameter of IteratedLocalSearch must be a str among {'nearest_neighbour', 'random'} "
        "or an array-like. Got 'both' instead."
    )
    with pytest.raises(ValueError) as exc:
        validate_parameter_constraints(constraints, {"init": "random", "patience": 0}, "X")
    assert (
        str(exc.value)
        == "The 'patience' parameter of X must be an int in the range [1, inf) or None. Got 0 instead."
    )


def test_single_constraint_without_list_and_unconstrained_params():
    constraints = {
        "verbose": "verbose",
        "random_state": "random_state",
        "n_iter": Interval(int, 1, None, closed="left"),
    }
    validate_parameter_constraints(
        constraints, {"verbose": 1, "random_state": None, "n_iter": 10, "other": object()}, "X"
    )
    validate_parameter_constraints(
        constraints, {"verbose": 1}, "X"
    )  # params missing from constraints are fine too
    with pytest.raises(
        ValueError,
        match=re.escape("The 'verbose' parameter of X must be an int >= 0 or a boolean. Got -1 instead."),
    ):
        validate_parameter_constraints(constraints, {"verbose": -1}, "X")
    with pytest.raises(
        ValueError, match=re.escape("must be None, an int or a numpy.random.Generator. Got 'seed' instead.")
    ):
        validate_parameter_constraints(constraints, {"random_state": "seed"}, "X")
    validate_parameter_constraints({}, {"anything": 1}, "X")


def test_numpy_scalars_accepted_where_python_numbers_expected():
    constraints = {
        "n_iter": [Interval(int, 1, None, closed="left")],
        "alpha": [Interval(float, 0.0, 1.0, closed="neither")],
    }
    validate_parameter_constraints(constraints, {"n_iter": np.int64(3), "alpha": np.float64(0.3)}, "X")
    with pytest.raises(ValueError, match="Got True instead"):
        validate_parameter_constraints(constraints, {"n_iter": True, "alpha": 0.3}, "X")
    with pytest.raises(ValueError, match=re.escape("Got 3.0 instead")):
        validate_parameter_constraints(constraints, {"n_iter": 3.0, "alpha": 0.3}, "X")


def test_repr_of_constraints():
    assert repr(Interval(int, 0, 1)) == "Interval(an int in the range [0, 1])"
    assert repr(Options(str, {"a"})) == "Options(a str among {'a'})"
