"""Deprecated 1.0 import path ``skroute.heuristics.NRBS`` (removed in 3.0): use ``skroute.construction``."""

import warnings

from ...construction import NRBS

warnings.warn(
    "skroute.heuristics.NRBS is deprecated since 2.0 and will be removed in 3.0; "
    "import NRBS from skroute.construction",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NRBS"]
