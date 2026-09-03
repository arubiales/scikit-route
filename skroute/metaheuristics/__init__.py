"""Metaheuristics: population-free and population-based searches over the giant tour (SPEC §4.4).

Every class here is a stochastic, iterative :class:`~skroute.base.BaseRouter`: it consumes
``random_state`` (D10: all randomness is pre-drawn in Python and handed to ``nogil`` kernels
as arrays) and records ``history_``/``n_iter_``/``stop_reason_``. All but :class:`SOM` see the
multi-trip objective during their search. Solver packages append their exports at the end of
the import list and of ``__all__``, one line each (D29).
"""

from ._ant_colony import AntColony
from ._genetic import Genetic
from ._simulated_annealing import SimulatedAnnealing
from ._som import SOM
from ._tabu_search import TabuSearch

__all__ = [
    "SOM",
    "AntColony",
    "Genetic",
    "SimulatedAnnealing",
    "TabuSearch",
]
