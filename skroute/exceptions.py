"""Exceptions raised by scikit-route."""

__all__ = ["InfeasibleProblemError", "NotFittedError"]


class NotFittedError(ValueError, AttributeError):
    """Raised by :func:`skroute.utils.check_is_fitted` when a fitted attribute is
    accessed before :meth:`fit` was called."""


class InfeasibleProblemError(ValueError):
    """Raised when a node cannot be served in a single trip within ``max_time_work``."""
