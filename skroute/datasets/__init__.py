"""Bundled routing instances and the TSPLIB reader.

Two families ship with scikit-route:

* the 27 **Waterloo national TSP instances** (`load_tsp(name)` and one
  `load_<country>()` wrapper each), coordinates with a published optimum under
  the TSPLIB ``EUC_2D`` metric;
* five **road-cost tables** (`load_alicante_murcia`, `load_barcelona`,
  `load_madrid`, `load_valencia`, `load_qatar_costs`) with cost, time and
  distance matrices for the multi-trip objective.

Loaders return :class:`~skroute.utils.Bunch` objects whose matrices are plain
arrays: pass ``labels=b.labels`` (and ``depot=b.depot``) to ``fit``.
:func:`read_tsplib` and :func:`read_tsplib_tour` read any TSPLIB 95 file.
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
