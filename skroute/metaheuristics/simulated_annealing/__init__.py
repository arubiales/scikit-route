"""Deprecated 1.0 import path ``skroute.metaheuristics.simulated_annealing`` (removed in 3.0).

``SimulatedAnnealing`` lives in ``skroute.metaheuristics`` and ``EnsembleSimulatedAnnealing`` in
``skroute.ensemble`` (SPEC §4.6). 1.0's ``__all__`` misspelt the first name (``SimmulatedAnnealing``);
this shim exports the real one.
"""

import warnings

from ...ensemble import EnsembleSimulatedAnnealing
from .. import SimulatedAnnealing

warnings.warn(
    "skroute.metaheuristics.simulated_annealing is deprecated since 2.0 and will be removed in 3.0; "
    "import SimulatedAnnealing from skroute.metaheuristics and "
    "EnsembleSimulatedAnnealing from skroute.ensemble",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SimulatedAnnealing", "EnsembleSimulatedAnnealing"]  # noqa: RUF022 - the 1.0 order (typo fixed)
