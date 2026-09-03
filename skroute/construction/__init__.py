"""Construction heuristics: deterministic tours built from scratch in O(n²)-O(n² log n) (SPEC §4.2).

Four solvers, all ``kind="construction"`` and deterministic (no ``random_state``):

* :class:`NearestNeighbour` — greedy walk from the depot over the core kernel.
* :class:`Insertion` — farthest, cheapest or nearest insertion (``strategy=``), direction-aware.
* :class:`ClarkeWright` — parallel savings; the only solver that refuses asymmetric matrices and
  the only construction heuristic that sees the multi-trip budget while it builds.
* :class:`NRBS` — the 2020 Node Ranking Based on Stats heuristic, ported faithfully.

They are the usual starting points of the local searches and metaheuristics (``init=``), and
honest baselines: the tolerance table of the test-suite records their measured gaps.
"""

from ._clarke_wright import ClarkeWright
from ._insertion import Insertion
from ._nearest_neighbour import NearestNeighbour
from ._nrbs import NRBS

__all__ = ["NRBS", "ClarkeWright", "Insertion", "NearestNeighbour"]
