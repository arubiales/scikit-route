"""Metaheuristics: population- and trajectory-based solvers that accept the multi-trip objective.

Every class here is stochastic (``random_state``), iterative (``history_``, ``n_iter_``,
``stop_reason_``) and, unless its docstring says otherwise, budget-aware: the search itself
prices candidate tours with the problem's own decoder (SPEC §4.4).

Each solver work package appends its import line and its ``__all__`` entries at the END of
the two lists below, one per line, so merges between packages are trivial (D29).
"""

from ._ant_colony import AntColony
from ._genetic import Genetic

__all__ = [
    "AntColony",
    "Genetic",
]
