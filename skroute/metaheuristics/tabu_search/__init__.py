"""Deprecated 1.0 import path ``skroute.metaheuristics.tabu_search`` (removed in 3.0).

Import ``TabuSearch`` from ``skroute.metaheuristics`` instead (SPEC §4.6).
"""

import warnings

from .. import TabuSearch

warnings.warn(
    "skroute.metaheuristics.tabu_search is deprecated since 2.0 and will be removed in 3.0; "
    "import TabuSearch from skroute.metaheuristics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TabuSearch"]
