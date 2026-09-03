# Utilities

Small helpers shared by the solvers, the dataset loaders and the test-suite:
[`Bunch`](#skroute.utils.Bunch) (the return type of the loaders),
[`check_random_state`](#skroute.utils.check_random_state) and
[`check_is_fitted`](#skroute.utils.check_is_fitted) (the two estimator helpers of the base
class), [`initial_tour`](#skroute.utils.initial_tour) (the warm-start helper behind every
`init=` parameter) and [`check_router`](#skroute.utils.estimator_checks.check_router), the
structural test battery a new solver must pass before it is merged
(`tests/test_common.py` runs it over `skroute.all_solvers()`).

::: skroute.utils

::: skroute.utils.estimator_checks.check_router
