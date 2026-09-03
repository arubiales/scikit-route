"""Shared helpers: ``Bunch``, input validation, the warm-start helper and the estimator checks.

- [`Bunch`][skroute.utils.Bunch] — dict with attribute access (the return type of the dataset loaders).
- [`check_random_state`][skroute.utils.check_random_state] — ``None``/int/``Generator`` ->
  ``numpy.random.Generator``.
- [`check_is_fitted`][skroute.utils.check_is_fitted] — raises
  [`NotFittedError`][skroute.exceptions.NotFittedError] before ``fit``.
- [`initial_tour`][skroute.utils.initial_tour] — builds a solver's starting tour from its ``init`` parameter.
- [`check_router`][skroute.utils.estimator_checks.check_router] — the structural test battery
  (in ``skroute.utils.estimator_checks``; also exported as ``skroute.check_router``).
"""

from ._bunch import Bunch
from ._init_tour import initial_tour
from .validation import check_is_fitted, check_random_state

__all__ = ["Bunch", "check_is_fitted", "check_random_state", "initial_tour"]
