"""scikit-route: route optimisation with a scikit-learn flavoured API.

Every solver is an estimator: ``__init__`` stores the knobs, ``fit(X, ...)`` returns
``self`` and the results live in trailing-underscore attributes (``tour_``, ``route_``,
``trips_``, ``cost_``...). Plain TSP is ``est.fit(C)``; the multi-trip objective is
``est.fit(C, time_matrix=T, max_time_work=8, extra_cost=12.83, people=2)``.

The public names are exported lazily (PEP 562): importing ``skroute`` does not import
any solver until it is first used.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import TYPE_CHECKING, Any

from ._version import __version__

# ---------------------------------------------------------------------------------------------
# REGISTRY of the public surface (SPEC §3.4 "Top-level surface", D18, D27): the ONE place where
# a public name is mapped to the module that defines it.
#
# MAINTAINED BY THE LEAD: one line per public name. Solver work packages ask the lead to add
# their line when their package lands; a P1 deferral PR (D26) removes it. all_solvers()
# derives its roster from this dict and imports every solver module EAGERLY: a registered
# module that is missing or broken raises ImportError. Never swallow that error — a silently
# shorter roster would hide a packaging bug from the test battery, which is parametrised over
# all_solvers().
# ---------------------------------------------------------------------------------------------
_EXPORTS: dict[str, str] = {
    # spine
    "RoutingProblem": "skroute.problem",
    "BaseRouter": "skroute.base",
    "RouterTags": "skroute.base",
    "clone": "skroute.base",
    "is_router": "skroute.base",
    "check_router": "skroute.utils.estimator_checks",
    # SOLVER ENTRIES — appended by each solver work package when it lands (D29), one line per
    # public name, grouped under its package comment, at the END of its group:
    # exact (D18)
    # construction
    # local search
    # metaheuristics
    "SimulatedAnnealing": "skroute.metaheuristics",
    "TabuSearch": "skroute.metaheuristics",
    # ensemble
}

# The five solver subpackages of D18: a registered name defined in one of them is a solver.
_SOLVER_MODULES: frozenset[str] = frozenset(
    {
        "skroute.exact",
        "skroute.construction",
        "skroute.local_search",
        "skroute.metaheuristics",
        "skroute.ensemble",
    }
)
# Solvers that cannot be instantiated without arguments, hence never returned by all_solvers()
# (D27): MultiStart wraps another estimator and is covered by tests/test_ensemble.py.
_NEEDS_ARGUMENTS: frozenset[str] = frozenset({"MultiStart"})

__all__ = ["__version__", "all_solvers", "set_log_level", *sorted(_EXPORTS)]

if TYPE_CHECKING:
    # Eager imports so mypy and mkdocstrings see real types (neither sees __getattr__). The
    # redundant ``X as X`` spelling marks an explicit re-export (PEP 484), so linters and type
    # checkers do not report the names as unused.
    from .base import BaseRouter as BaseRouter
    from .base import RouterTags as RouterTags
    from .base import clone as clone
    from .base import is_router as is_router
    from .problem import RoutingProblem as RoutingProblem
    from .utils.estimator_checks import check_router as check_router

    # SOLVER IMPORTS — appended by each solver work package (same names as its registry lines).
    # isort: split
    from .metaheuristics import SimulatedAnnealing as SimulatedAnnealing
    from .metaheuristics import TabuSearch as TabuSearch


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a public name on first access and cache it on the module."""
    try:
        module_path = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'skroute' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_path), name)
    setattr(sys.modules[__name__], name, value)
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def all_solvers() -> list[type[BaseRouter]]:
    """Every solver class that can be instantiated with no arguments, sorted by name (D27).

    The roster is derived from the registry at the top of ``skroute/__init__.py``: every
    registered name defined in one of the five solver subpackages (``exact``,
    ``construction``, ``local_search``, ``metaheuristics``, ``ensemble``) except
    ``MultiStart``, which needs an estimator. Imports are eager: a registered module that
    is missing or does not define its name raises ``ImportError`` — the roster is the
    parametrisation of the whole test battery and must never shrink silently.

    Returns
    -------
    solvers : list of type
        Subclasses of :class:`~skroute.base.BaseRouter`, sorted by ``__name__``.

    Raises
    ------
    ImportError
        When a registered solver module cannot be imported or lacks the registered name.

    Examples
    --------
    >>> from skroute import all_solvers
    >>> solvers = all_solvers()  # doctest: +SKIP
    >>> [s.__name__ for s in solvers][:3]  # doctest: +SKIP
    ['AntColony', 'BruteForce', 'ClarkeWright']
    """
    classes: list[type[BaseRouter]] = []
    for name, module_path in _EXPORTS.items():
        if module_path not in _SOLVER_MODULES or name in _NEEDS_ARGUMENTS:
            continue
        module = importlib.import_module(module_path)  # ImportError propagates on purpose
        try:
            classes.append(getattr(module, name))
        except AttributeError:
            raise ImportError(
                f"{module_path} does not define {name!r}: fix the registry in skroute/__init__.py"
            ) from None
    return sorted(classes, key=lambda cls: cls.__name__)


# ---------------------------------------------------------------------------------------------
# Logging (D24): no printing anywhere; ``verbose`` routes to the "skroute" logger. Python's
# last-resort handler shows only WARNING and above, so a NullHandler is attached here and
# ``set_log_level`` gives users a one-liner to see INFO records.
# ---------------------------------------------------------------------------------------------
_log = logging.getLogger("skroute")
_log.addHandler(logging.NullHandler())


def set_log_level(level: int | str) -> None:
    """Set the level of the ``skroute`` logger and make its records visible.

    Sets ``logging.getLogger("skroute").setLevel(level)``; if the logger's only handler is
    the ``NullHandler`` attached at import, adds a ``StreamHandler`` to stderr with the
    format ``"%(name)s %(levelname)s %(message)s"``. Equivalent to
    ``logging.basicConfig(level=logging.INFO)`` for users who do not configure logging.

    Parameters
    ----------
    level : int or str
        A logging level (``logging.INFO``, ``"INFO"``, ``"DEBUG"``...).

    Examples
    --------
    >>> import logging, skroute
    >>> skroute.set_log_level("INFO")
    >>> logging.getLogger("skroute").level == logging.INFO
    True
    """
    _log.setLevel(level)
    if all(isinstance(h, logging.NullHandler) for h in _log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        _log.addHandler(handler)
