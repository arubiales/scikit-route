"""Exact solvers: they certify the optimum and set ``is_optimal_`` (SPEC §4.1).

- :class:`BruteForce` — exhaustive enumeration, exact for the multi-trip objective too (n <= 11).
- :class:`HeldKarp` — bitmask dynamic programming, plain TSP and ATSP (n <= 20).
- :class:`MILP` — Dantzig-Fulkerson-Johnson with lazy subtour cuts on HiGHS (n <= 300).
"""

from ._brute_force import BruteForce
from ._held_karp import HeldKarp
from ._milp import MILP

__all__ = ["MILP", "BruteForce", "HeldKarp"]
