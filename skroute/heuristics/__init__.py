"""Deprecated 1.0 import path ``skroute.heuristics`` (removed in 3.0).

Exposes the ``brute`` and ``NRBS`` subpackages of 1.0.0a2 (SPEC §4.6). Import
:class:`~skroute.exact.BruteForce` from ``skroute.exact`` and :class:`~skroute.construction.NRBS`
from ``skroute.construction`` instead.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType

warnings.warn(
    "skroute.heuristics is deprecated since 2.0 and will be removed in 3.0; "
    "import BruteForce from skroute.exact and NRBS from skroute.construction",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NRBS", "brute"]


def __getattr__(name: str) -> ModuleType:
    """Load the legacy subpackages on first access (each one warns on its own)."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
