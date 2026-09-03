"""Shared helpers: ``Bunch``, input validation, the warm-start helper and the estimator checks.

- :class:`Bunch` — dict with attribute access (the return type of the dataset loaders).
- :func:`check_random_state` — ``None``/int/``Generator`` -> :class:`numpy.random.Generator`.
- :func:`check_is_fitted` — raises :class:`~skroute.exceptions.NotFittedError` before ``fit``.
- :func:`initial_tour` — builds a solver's starting tour from its ``init`` parameter.
- :func:`~skroute.utils.estimator_checks.check_router` — the structural test battery
  (in :mod:`skroute.utils.estimator_checks`; also exported as ``skroute.check_router``).
"""

from ._bunch import Bunch
from ._init_tour import initial_tour
from .validation import check_is_fitted, check_random_state

__all__ = ["Bunch", "check_is_fitted", "check_random_state", "initial_tour"]
