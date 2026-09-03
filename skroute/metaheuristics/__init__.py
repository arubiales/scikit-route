"""Metaheuristics: population-free and population-based searches over the giant tour (SPEC §4.4).

Every class here is a stochastic, iterative, budget-aware :class:`~skroute.base.BaseRouter`:
it consumes ``random_state`` (D10: all randomness is pre-drawn in Python and handed to
``nogil`` kernels as arrays), records ``history_``/``n_iter_``/``stop_reason_`` and sees the
multi-trip objective during its search. Solver packages append their exports at the end of
the import list and of ``__all__``, one line each (D29).
"""

from ._simulated_annealing import SimulatedAnnealing

# from ._tabu_search import TabuSearch

__all__ = [
    "SimulatedAnnealing",
    "TabuSearch",
]
