"""``MultiStart``: independent restarts of one stochastic solver in parallel, keeping the best (SPEC §4.5)."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ..base import BaseRouter, RouteEvent, RouterTags, clone
from ..problem import RoutingProblem
from ..utils._param_validation import Interval, Options

__all__ = ["MultiStart"]

log = logging.getLogger("skroute")

# joblib semantics: None = one job (unless a ``joblib.parallel_config`` context says otherwise), a
# positive int = that many workers, a negative int = all CPUs but ``-n_jobs - 1``; zero is invalid.
_N_JOBS = [None, Interval(Integral, 1, None, closed="left"), Interval(Integral, None, -1, closed="right")]


def _fit_one(estimator: BaseRouter, problem: RoutingProblem) -> BaseRouter:
    """Fit one restart on the shared problem and return it (joblib returns results in input order)."""
    return estimator.fit(problem)


def _forwarding(
    owner: BaseRouter, callback: Callable[[RouteEvent], Any], restart: int
) -> Callable[[RouteEvent], Any]:
    """The user's callback with ``extra["restart"] = restart`` added to every inner event (D30).

    A ``True`` answer stops the running restart (the inner solver sees the ``True``) and marks the
    owner so that no further restart is launched.
    """

    def forward(event: RouteEvent) -> Any:
        result = callback(dataclasses.replace(event, extra={**event.extra, "restart": restart}))
        if isinstance(result, (bool, np.bool_)) and bool(result):
            owner._stop_requested = True
            return True
        return result

    return forward


class MultiStart(BaseRouter):
    """Run a stochastic solver from ``n_restarts`` independent seeds and keep the best tour.

    Every restart is a :func:`~skroute.clone` of ``estimator`` whose ``random_state`` is a
    :class:`numpy.random.Generator` seeded from a child of one
    :class:`numpy.random.SeedSequence`, so the restarts are statistically independent and the
    whole run is reproducible from ``random_state`` alone. The restarts are fitted on the
    **shared** :class:`~skroute.RoutingProblem` through :class:`joblib.Parallel`; the result
    (which restart wins and its tour) is identical for any ``n_jobs`` and backend, because the
    seeds are assigned by restart index before anything runs.

    Parameters
    ----------
    estimator : BaseRouter
        An unfitted **stochastic** solver (one with a ``random_state`` parameter). A
        deterministic estimator is refused at fit time with ``ValueError``: restarting it would
        return the same tour ``n_restarts`` times.
    n_restarts : int >= 1, default 10
        Number of independent restarts (outer iterations).
    n_jobs : int or None, default None
        Workers for :class:`joblib.Parallel`: ``None`` runs the restarts one after another
        (unless an enclosing ``joblib.parallel_config`` says otherwise), ``-1`` uses every
        CPU, a positive int that many workers. Never changes the result.
    prefer : {"threads", "processes"} or None, default "threads"
        joblib backend hint. Threads are the default because the solver kernels release the
        GIL and a large cost matrix (900 MB at n = 10 639) must not be pickled once per
        worker; they give a near-linear speed-up for ``SimulatedAnnealing``,
        ``IteratedLocalSearch`` and ``TabuSearch`` and little for the Python-heavy ``Genetic``
        and ``SOM``, for which ``prefer="processes"`` is one keyword away. ``None`` lets
        joblib choose.
    random_state : int, numpy.random.Generator or None, default None
        Seed of the restart seeds: one integer is drawn from it and spawns ``n_restarts``
        child ``SeedSequence`` objects. The same seed on the same machine gives bit-identical
        results whatever ``n_jobs``; a passed ``Generator`` is advanced by the fit.
    verbose : int, default 0
        ``0`` is silent; ``1`` logs every ``max(1, n_restarts // 10)`` finished restarts and a
        summary at INFO; ``2`` logs every restart. Records go to the ``skroute`` logger at
        INFO; enable them with ``logging.basicConfig(level=logging.INFO)`` or
        ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    estimators_ : list of BaseRouter
        The ``n_restarts`` fitted clones, in restart order.
    costs_ : ndarray of shape (n_restarts,), float64
        ``cost_`` of every restart.
    best_index_ : int
        Index of the winning restart (the lowest index on a tie).
    best_estimator_ : BaseRouter
        ``estimators_[best_index_]``; its ``tour_`` is the tour returned.
    history_ : ndarray of shape (n_iter_,), float64
        Copied from ``best_estimator_`` when the estimator is iterative.
    n_iter_ : int
        Copied from ``best_estimator_`` when the estimator is iterative.
    stop_reason_ : str
        Copied from ``best_estimator_`` when the estimator is iterative; ``"callback"`` when the
        ``callback`` of ``fit`` stopped the ensemble (``estimators_`` and ``costs_`` then hold only
        the restarts that ran).

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_`` and
    the other fitted attributes shared by every solver.

    Notes
    -----
    **Tags** delegate to the estimator's (``budget_aware``, ``requires_symmetric``,
    ``requires_coords``, ``iterative``, ``max_nodes``...) with ``kind="ensemble"`` and
    ``stochastic=True``: a ``MultiStart`` of a budget-aware solver is budget-aware, a
    ``MultiStart`` of ``SOM`` still needs coordinates.

    **Seeding (D10, D17).** ``seed = rng.integers(2**63 - 1)``; restart ``k`` receives
    ``np.random.default_rng(np.random.SeedSequence(seed).spawn(n_restarts)[k])``. The base
    class recomputes ``cost_`` from the winning tour (D2) and copies nothing else from the
    inner solvers except the three iterative attributes.

    **Parameter protocol.** ``get_params(deep=True)`` exposes the inner knobs as
    ``estimator__<name>``; ``set_params(estimator__alpha=0.99)`` reaches them; ``clone`` copies
    the estimator; ``repr`` prints it.

    **Callback events (D30).** ``MultiStart`` emits ``"start"`` (no tour, ``extra["n_restarts"]``)
    and ``"end"`` under its own name. When ``n_jobs`` is ``None`` or ``1`` the restarts run one
    after another and every event of every inner fit is forwarded to the callback under the inner
    class's name with ``extra["restart"]`` (the restart index); a ``True`` answer stops the running
    restart and no further restart is launched. Parallel runs (any other ``n_jobs``) forward
    nothing: the drawing libraries behind the usual callbacks are not thread-safe.

    **Supports:** whatever the estimator supports; stochastic; iterative iff the estimator is.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute import MultiStart, SimulatedAnnealing
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
    >>> ms = MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=0)
    >>> ms = ms.fit(wi.distance_matrix(), labels=wi.labels)
    >>> ms.cost_ / wi.optimal_tour_length < 1.03  # the fast-tier tolerance of the wrapped solver
    True
    >>> len(ms.estimators_), ms.costs_.shape, ms.cost_ == float(ms.costs_.min())
    (4, (4,), True)
    >>> ms.best_estimator_ is ms.estimators_[ms.best_index_]
    True
    >>> int(ms.route_[0]) == int(ms.route_[-1]) == int(ms.depot_) == 1
    True
    >>> ms.n_iter_ == len(ms.history_) and ms.stop_reason_ == "converged"
    True

    The result does not depend on the number of workers:

    >>> C = wi.distance_matrix()
    >>> a = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=1, random_state=0).fit(C)
    >>> b = MultiStart(SimulatedAnnealing(), n_restarts=4, n_jobs=2, random_state=0).fit(C)
    >>> bool(np.array_equal(a.tour_, b.tour_)) and a.best_index_ == b.best_index_
    True

    The inner knobs are reachable with the ``estimator__`` prefix:

    >>> ms.set_params(estimator__alpha=0.99).estimator.alpha
    0.99
    >>> ms
    MultiStart(estimator=SimulatedAnnealing(alpha=0.99), n_restarts=4, random_state=0)
    """

    _parameter_constraints: dict[str, Any] = {
        "estimator": [BaseRouter],
        "n_restarts": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": _N_JOBS,
        "prefer": [Options(str, {"threads", "processes"}), None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    estimators_: list[BaseRouter]
    costs_: np.ndarray
    best_index_: int
    best_estimator_: BaseRouter

    def __init__(
        self,
        estimator: BaseRouter,
        n_restarts: int = 10,
        n_jobs: int | None = None,
        prefer: str | None = "threads",
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs
        self.prefer = prefer
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        """The estimator's tags with ``kind="ensemble"`` and ``stochastic=True``."""
        inner = self.estimator._get_tags() if isinstance(self.estimator, BaseRouter) else RouterTags()
        return dataclasses.replace(inner, kind="ensemble", stochastic=True)

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        assert rng is not None  # stochastic tag: the base class always hands a Generator
        estimator = self.estimator
        if not (estimator._get_tags().stochastic and "random_state" in estimator._get_param_names()):
            raise ValueError("MultiStart needs a stochastic estimator (one with random_state)")
        n_restarts = int(self.n_restarts)
        seed = int(rng.integers(0, 2**63 - 1))
        seeds = np.random.SeedSequence(seed).spawn(n_restarts)
        restarts = [clone(estimator).set_params(random_state=np.random.default_rng(s)) for s in seeds]
        self._emit("start", 0, None, np.nan, n_restarts=n_restarts)  # D30
        callback = self._callback
        fitted: list[BaseRouter]
        if callback is not None and self.n_jobs in (None, 1):
            # D30: sequential restarts forward the callback, each event tagged with its restart index; a
            # True answer stops the running restart and the launch of the next ones. Parallel runs never
            # forward: the drawing libraries behind the usual callbacks are not thread-safe.
            fitted = []
            for k, est in enumerate(restarts):
                fitted.append(est.fit(problem, callback=_forwarding(self, callback, k)))
                if self._stop_requested:
                    break
        else:
            fitted = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
                delayed(_fit_one)(est, problem) for est in restarts
            )
        costs = np.array([est.cost_ for est in fitted], dtype=np.float64)
        best_index = int(np.argmin(costs))  # first occurrence: ties go to the lowest index
        best = fitted[best_index]
        if self.verbose:
            every = max(1, n_restarts // 10) if self.verbose == 1 else 1
            for k, est in enumerate(fitted, start=1):
                if k % every == 0:
                    log.info("MultiStart restart %d/%d: cost=%.6f", k, n_restarts, est.cost_)
            log.info(
                "MultiStart: best restart %d of %d with cost=%.6f (worst %.6f)",
                best_index,
                n_restarts,
                costs[best_index],
                costs.max(),
            )
        self.estimators_ = fitted
        self.costs_ = costs
        self.best_index_ = best_index
        self.best_estimator_ = best
        for attr in ("history_", "n_iter_", "stop_reason_"):
            if hasattr(best, attr):
                setattr(self, attr, getattr(best, attr))
        if self._stop_requested and hasattr(self, "stop_reason_"):
            self.stop_reason_ = "callback"  # D30: the ensemble itself was cut short
        return problem.to_index_tour(best.tour_)
