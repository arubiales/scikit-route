"""Compiled core of scikit-route (Cython). See ``_routing.pxd`` for the frozen contract."""

try:
    from . import _routing
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scikit-route's compiled core is missing. Install a wheel (pip install scikit-route) "
        "or build from source with a C compiler: pip install -e ."
    ) from e

__all__ = ["_routing"]
