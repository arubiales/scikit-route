"""Deprecated 1.0 import path ``skroute.metaheuristics.som`` (removed in 3.0).

Import ``SOM`` from ``skroute.metaheuristics`` instead (SPEC §4.6).
"""

import warnings

from .. import SOM

warnings.warn(
    "skroute.metaheuristics.som is deprecated since 2.0 and will be removed in 3.0; "
    "import SOM from skroute.metaheuristics",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SOM"]
