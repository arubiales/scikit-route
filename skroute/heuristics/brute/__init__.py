"""Deprecated 1.0 import path ``skroute.heuristics.brute`` (removed in 3.0): use ``skroute.exact``."""

import warnings

from ...exact import BruteForce

warnings.warn(
    "skroute.heuristics.brute is deprecated since 2.0 and will be removed in 3.0; "
    "import BruteForce from skroute.exact",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BruteForce"]
