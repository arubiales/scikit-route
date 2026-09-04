"""Ensembles: independent restarts of a stochastic solver, in parallel, keeping the best (SPEC §4.5).

`MultiStart` wraps any stochastic `BaseRouter`; it is not returned by
`skroute.all_solvers` because it needs an estimator (D27). `EnsembleGenetic` and
`EnsembleSimulatedAnnealing` are the 1.0 ensembles as explicit-parameter wrappers over
``MultiStart`` (kept until 3.0; new code should use ``MultiStart`` directly).
"""

from ._legacy import EnsembleGenetic, EnsembleSimulatedAnnealing
from ._multistart import MultiStart

__all__ = ["MultiStart", "EnsembleGenetic", "EnsembleSimulatedAnnealing"]  # noqa: RUF022 - general tool first
