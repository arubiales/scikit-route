"""Bundled routing instances and the TSPLIB reader.

Two families ship with scikit-route:

* the 27 **Waterloo national TSP instances** (`load_tsp(name)` and one
  `load_<country>()` wrapper each), coordinates with the tour length published
  by the University of Waterloo under the TSPLIB ``EUC_2D`` metric -- the proven
  optimum for 25 of them, the best-known tour for ``bm33708`` and ``ch71009``;
* five **road-cost tables** (`load_alicante_murcia`, `load_barcelona`,
  `load_madrid`, `load_valencia`, `load_qatar_costs`) with cost, time and
  distance matrices for the multi-trip objective.

Loaders return `Bunch` objects whose matrices are plain
arrays: pass ``labels=b.labels`` (and ``depot=b.depot``) to ``fit``.
`read_tsplib` reads TSPLIB 95 ``.tsp`` files with the edge-weight types
``EUC_2D``, ``CEIL_2D``, ``MAN_2D``, ``ATT``, ``GEO`` and ``EXPLICIT`` (row
formats); `read_tsplib_tour` reads ``.tour`` files.
"""

from ._loaders import (
    TSPBunch,
    list_tsp,
    load_alicante_murcia,
    load_argentina,
    load_barcelona,
    load_burma,
    load_canada,
    load_china,
    load_costs_qatar,
    load_djibouti,
    load_egypt,
    load_finland,
    load_greece,
    load_honduras,
    load_ireland,
    load_italy,
    load_japan,
    load_kazakhstan,
    load_luxembourg,
    load_madrid,
    load_morocco,
    load_nicaragua,
    load_oman,
    load_panama,
    load_qatar,
    load_qatar_costs,
    load_rwanda,
    load_sahara,
    load_sweden,
    load_tanzania,
    load_tsp,
    load_uruguay,
    load_valencia,
    load_vietnam,
    load_yemen,
    load_zimbabwe,
)
from ._tsplib import read_tsplib, read_tsplib_tour

__all__ = [
    "TSPBunch",
    "list_tsp",
    "load_alicante_murcia",
    "load_argentina",
    "load_barcelona",
    "load_burma",
    "load_canada",
    "load_china",
    "load_costs_qatar",
    "load_djibouti",
    "load_egypt",
    "load_finland",
    "load_greece",
    "load_honduras",
    "load_ireland",
    "load_italy",
    "load_japan",
    "load_kazakhstan",
    "load_luxembourg",
    "load_madrid",
    "load_morocco",
    "load_nicaragua",
    "load_oman",
    "load_panama",
    "load_qatar",
    "load_qatar_costs",
    "load_rwanda",
    "load_sahara",
    "load_sweden",
    "load_tanzania",
    "load_tsp",
    "load_uruguay",
    "load_valencia",
    "load_vietnam",
    "load_yemen",
    "load_zimbabwe",
    "read_tsplib",
    "read_tsplib_tour",
]
