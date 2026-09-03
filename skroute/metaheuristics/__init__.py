"""Metaheuristics: population-, trajectory- and network-based solvers of SPEC §4.4 (D18).

Append-only registry of the package (D29): one import per line, one ``__all__`` entry per line,
each work package adds its own at the end.
"""

from ._som import SOM

__all__ = [
    "SOM",
]
