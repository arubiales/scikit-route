"""Local search: the 2-opt and Or-opt descents, their alternation and iterated local search.

Every solver here is budget-aware (the multi-trip objective steers the search) and accepts
asymmetric matrices. The three descents are deterministic; ``IteratedLocalSearch`` is the
recommended default solver of the library.
"""

from ._iterated import IteratedLocalSearch
from ._local_search import LocalSearch
from ._or_opt import OrOpt
from ._two_opt import TwoOpt

__all__ = ["IteratedLocalSearch", "LocalSearch", "OrOpt", "TwoOpt"]
