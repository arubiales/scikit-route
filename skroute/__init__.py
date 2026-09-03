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
from typing import TYPE_CHECKING

from ._version import __version__

# ---------------------------------------------------------------------------------------------
# Registry of the public surface (SPEC §3.4 "Top-level surface", D18, D27).
#
# MAINTAINED BY THE LEAD: one line per public name, mapping it to the module that defines it.
# Solver work packages ask the lead to add their line when their package lands. all_solvers()
# imports every registered solver module EAGERLY and must fail loudly (ImportError) when a
# registered module is missing or broken — never swallow the error: a silently shorter roster
# would hide a packaging bug from the test battery that is parametrised over all_solvers().
# ---------------------------------------------------------------------------------------------
_EXPORTS: dict[str, str] = {
    # spine
    "RoutingProblem": "skroute.problem",
    "BaseRouter": "skroute.base",
    "RouterTags": "skroute.base",
    "clone": "skroute.base",
    "is_router": "skroute.base",
    "check_router": "skroute.utils.estimator_checks",
    # exact (D18)
    "BruteForce": "skroute.exact",
    "HeldKarp": "skroute.exact",
    "MILP": "skroute.exact",
    # construction
    "NearestNeighbour": "skroute.construction",
    "Insertion": "skroute.construction",
    "ClarkeWright": "skroute.construction",
    "NRBS": "skroute.construction",
    # local search
    "TwoOpt": "skroute.local_search",
    "OrOpt": "skroute.local_search",
    "LocalSearch": "skroute.local_search",
    "IteratedLocalSearch": "skroute.local_search",
    # metaheuristics
    "SimulatedAnnealing": "skroute.metaheuristics",
    "TabuSearch": "skroute.metaheuristics",
    "Genetic": "skroute.metaheuristics",
    "AntColony": "skroute.metaheuristics",
    "SOM": "skroute.metaheuristics",
    # ensemble
    "MultiStart": "skroute.ensemble",
    "EnsembleGenetic": "skroute.ensemble",
    "EnsembleSimulatedAnnealing": "skroute.ensemble",
}

# Solver classes that all_solvers() returns (D27): every no-argument-constructible solver of D18
# including the two Ensemble wrappers; MultiStart needs an estimator and is never returned.
_SOLVERS: tuple[str, ...] = (
    "BruteForce",
    "HeldKarp",
    "MILP",
    "NearestNeighbour",
    "Insertion",
    "ClarkeWright",
    "NRBS",
    "TwoOpt",
    "OrOpt",
    "LocalSearch",
    "IteratedLocalSearch",
    "SimulatedAnnealing",
    "TabuSearch",
    "Genetic",
    "AntColony",
    "SOM",
    "EnsembleGenetic",
    "EnsembleSimulatedAnnealing",
)

__all__ = [
    "__version__",
    "all_solvers",
    "set_log_level",
    *sorted(_EXPORTS),
]

if TYPE_CHECKING:  # eager imports so mypy and mkdocstrings see real types (they do not see __getattr__)
    from .base import BaseRouter, RouterTags, clone, is_router
    from .construction import NRBS, ClarkeWright, Insertion, NearestNeighbour
    from .ensemble import EnsembleGenetic, EnsembleSimulatedAnnealing, MultiStart
    from .exact import MILP, BruteForce, HeldKarp
    from .local_search import IteratedLocalSearch, LocalSearch, OrOpt, TwoOpt
    from .metaheuristics import SOM, AntColony, Genetic, SimulatedAnnealing, TabuSearch
    from .problem import RoutingProblem
    from .utils.estimator_checks import check_router


def __getattr__(name: str) -> object:
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


def all_solvers() -> list[type]:
    """Every solver class that can be instantiated with no arguments, sorted by name (D27).

    Returns the solvers of D18 plus ``EnsembleGenetic`` and ``EnsembleSimulatedAnnealing``;
    ``MultiStart`` is never returned (it needs an estimator). Imports are eager and a
    registered module that is missing raises ``ImportError`` — the roster is the
    parametrisation of the whole test battery and must never shrink silently.

    Returns
    -------
    solvers : list of type
        Subclasses of :class:`~skroute.base.BaseRouter`, sorted by ``__name__``.

    Examples
    --------
    >>> from skroute import all_solvers
    >>> solvers = all_solvers()  # doctest: +SKIP
    >>> [s.__name__ for s in solvers][:3]  # doctest: +SKIP
    ['AntColony', 'BruteForce', 'ClarkeWright']
    """
    classes = [getattr(importlib.import_module(_EXPORTS[name]), name) for name in _SOLVERS]
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
