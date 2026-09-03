"""Deprecated 1.0 import path ``skroute.metaheuristics.genetics`` (removed in 3.0).

``Genetic`` lives in ``skroute.metaheuristics`` and ``EnsembleGenetic`` in ``skroute.ensemble`` (SPEC §4.6).
"""

import warnings

from ...ensemble import EnsembleGenetic
from .. import Genetic

warnings.warn(
    "skroute.metaheuristics.genetics is deprecated since 2.0 and will be removed in 3.0; "
    "import Genetic from skroute.metaheuristics and EnsembleGenetic from skroute.ensemble",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Genetic", "EnsembleGenetic"]  # noqa: RUF022 - the 1.0 __all__, verbatim
