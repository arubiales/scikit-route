"""Self-organising map for the travelling salesman problem (SPEC §4.4, R8, D28): numpy only.

A one-dimensional Kohonen ring of ``n_units`` neurons lives in the plane of the (aspect-preserving,
min-max normalised) city coordinates. Every *sample* presents one random city: the closest neuron
(the winner) and its ring neighbours, weighted by a Gaussian of the wrapped ring distance, move
towards the city; the learning rate and the neighbourhood radius decay after every sample. Samples
are grouped in *epochs* — the outer iterations of the iterative contract — and after every epoch the
ring is decoded into a tour (cities ordered by the index of their winning neuron, ties by city index,
rotated to the depot) and priced with the problem's own objective, so ``history_`` is the best-so-far
cost per epoch and the returned tour is the best epoch's one.
"""

from __future__ import annotations

import logging
from numbers import Integral, Real

import numpy as np

from ..base import BaseRouter, RouterTags
from ..preprocessing import normalize_coords
from ..problem import RoutingProblem
from ..utils._param_validation import Interval

__all__ = ["SOM"]

log = logging.getLogger("skroute")

_LR_FLOOR = 1e-3  # learning_rate below this at the end of an epoch -> "converged"
_RADIUS_FLOOR = 1.0  # radius below one ring unit at the end of an epoch -> "converged"
# Floor of the Gaussian's width (ring positions). At sigma = 1e-3 every neuron other than the winner
# already gets exp(-1 / 2e-6) == 0.0 exactly in double precision, so no finite result changes; without
# it a radius below ~1e-154 (legal: radius_decay=0.5, radius=1e-200, n_iter=1e6 with radius_decay=0.96)
# makes `radius * radius` underflow to 0.0, the winner's Gaussian 0/0 = NaN, the weights NaN, and every
# city then wins the first NaN neuron: the run returns the index-order tour with only numpy warnings.
_SIGMA_MIN = 1e-3
_DECODE_BLOCK = 1 << 22  # entries of the (cities, units) distance block built while decoding


def _ring_to_tour(winners: np.ndarray, depot: int) -> np.ndarray:
    """Decode winner indices into an index tour: cities by winner, ties by city index, depot first.

    Parameters
    ----------
    winners : ndarray of shape (n,), int
        Index of the winning neuron of every city.
    depot : int
        Index of the depot.

    Returns
    -------
    tour : ndarray of shape (n,), int64
        A permutation of ``range(n)`` with ``tour[0] == depot``.
    """
    winners = np.asarray(winners, dtype=np.int64)
    n = winners.shape[0]
    order = np.lexsort((np.arange(n), winners))  # primary key: winner; secondary: city index
    k = int(np.flatnonzero(order == depot)[0])
    return np.ascontiguousarray(np.roll(order, -k), dtype=np.int64)


def _winners(xy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Closest neuron of every city (first index on ties), computed in blocks of bounded memory."""
    n, m = xy.shape[0], weights.shape[0]
    out = np.empty(n, dtype=np.int64)
    rows = max(1, _DECODE_BLOCK // m)
    wx, wy = weights[:, 0][None, :], weights[:, 1][None, :]
    for a in range(0, n, rows):
        b = min(n, a + rows)
        d2 = (xy[a:b, 0][:, None] - wx) ** 2 + (xy[a:b, 1][:, None] - wy) ** 2
        out[a:b] = np.argmin(d2, axis=1)
    return out


class SOM(BaseRouter):
    """Self-organising map (Kohonen ring) for the TSP over planar coordinates.

    A ring of ``n_units`` neurons is initialised at random inside the bounding box of the
    normalised coordinates. Each *sample* presents one random city: the nearest neuron wins,
    and every neuron moves towards the city by ``learning_rate`` times a Gaussian of its
    wrapped ring distance to the winner (standard deviation ``radius``); then
    ``learning_rate *= lr_decay`` and ``radius *= radius_decay``. Samples are grouped in
    **epochs** of ``max(1, n_iter // 100)`` samples — the outer iterations — and after every
    epoch the ring is decoded into a tour and priced with the problem's objective.

    The solver **needs coordinates** (``fit(X, coords=...)``, tag ``requires_coords``): the
    map is trained on the geometry only; the cost matrix ``X`` prices the decoded tours and
    may be anything (asymmetric, road costs...).

    Parameters
    ----------
    n_units : int or None, default=None
        Neurons on the ring. ``None`` means ``8 * n`` (n cities). More neurons give the ring
        more freedom to separate close cities at a proportional cost per sample.
    learning_rate : float in (0, 1], default=0.8
        Initial step of the update towards the presented city; decays by ``lr_decay`` after
        every sample.
    lr_decay : float in (0, 1], default=0.99997
        Multiplicative decay of ``learning_rate`` per sample. ``1.0`` disables the decay.
    radius : float or None, default=None
        Initial standard deviation (in ring positions) of the Gaussian neighbourhood around
        the winner. ``None`` means ``n_units / 10``.
    radius_decay : float in (0, 1], default=0.9997
        Multiplicative decay of ``radius`` per sample. ``1.0`` disables the decay.
    n_iter : int, default=100_000
        Maximum number of samples (single-city presentations). They run in epochs of
        ``max(1, n_iter // 100)`` samples — the outer iterations — so a run has about 100
        epochs (between 100 and 199) for ``n_iter >= 200`` and one epoch per sample below.
    random_state : int, numpy.random.Generator or None, default=None
        Seed of the initial ring and of the city drawn at every sample. All randomness is
        pre-drawn per epoch from ``numpy.random.default_rng(random_state)`` (D10); the same
        seed on the same machine gives bit-identical results.
    verbose : int, default=0
        ``0`` is silent; ``1`` logs every ``max(1, n_epochs // 10)`` epochs plus the stop;
        ``2`` logs every epoch. Records go to the ``skroute`` logger at INFO; enable them
        with ``logging.basicConfig(level=logging.INFO)`` or ``skroute.set_log_level("INFO")``.

    Attributes
    ----------
    history_ : ndarray of shape (n_iter_,), float64
        Best-so-far cost after every epoch (monotone non-increasing; never the cost of the
        current ring, whose trace is not monotone — R8).
    n_iter_ : int
        Epochs actually run (``== len(history_)``).
    n_samples_ : int
        Samples actually drawn (``<= n_iter``).
    stop_reason_ : {"converged", "max_iter", "callback"}
        ``"converged"`` when, at the end of an epoch, ``radius < 1`` or
        ``learning_rate < 1e-3``; ``"max_iter"`` after ``n_iter`` samples; ``"callback"`` when
        the ``callback`` of ``fit`` returned ``True``. This solver has no ``time_limit`` or
        ``patience`` parameter and never emits those values.

    See :class:`~skroute.base.BaseRouter` for ``tour_``, ``route_``, ``trips_``, ``cost_``
    and the other fitted attributes every solver shares.

    Notes
    -----
    **Algorithm** (Angéniol et al., 1988; Kohonen, 1982). Coordinates are rescaled with
    :func:`~skroute.preprocessing.normalize_coords` (the longer side of the bounding box spans
    ``[0, 1]``, the aspect ratio is preserved, so ``radius`` and ``learning_rate`` mean the same
    on every instance). Per sample: ``winner = argmin_j ||w_j - x||``,
    ``g_j = exp(-d(j, winner)^2 / (2 radius^2))`` with ``d`` the wrapped ring distance, and
    ``w_j += learning_rate * g_j * (x - w_j)``. Decoding maps every city to its closest neuron,
    orders the cities by neuron index (ties by city index) and rotates the sequence to the depot.
    The Gaussian's width is floored at ``1e-3`` ring positions (``radius`` itself keeps decaying
    and drives the stop rule): below that every neuron but the winner already gets ``g_j == 0``
    exactly in double precision, so the floor changes no finite result and only keeps extreme
    ``radius``/``radius_decay`` values (``radius_decay=0.5``, ``radius=1e-200``) from underflowing
    ``radius**2`` to ``0.0`` and turning the winner's update into ``0/0``.

    **Complexity.** Every sample is O(``n_units``) vectorised numpy work; decoding an epoch is
    O(``n * n_units``) in blocks of bounded memory, plus one O(n) evaluation. With the defaults
    the radius reaches one ring position after roughly ``ln(0.8 n) / 0.0003`` samples, so a
    run usually stops by ``"converged"`` well before the 100 epochs.

    **Callback events (D30).** ``"start"`` has no tour (the ring has not been decoded yet), with
    the ``extra`` keys ``radius``, ``learning_rate`` and ``n_units``; every epoch emits one
    ``"iteration"`` whose ``tour`` is the epoch's decoded ring and ``best_tour`` the best epoch's,
    with ``radius`` and ``learning_rate`` (after the epoch's decay) and ``n_samples``.

    **Supports:** symmetric and asymmetric cost matrices (the matrix only prices the decoded
    tours), coordinates required; stochastic; iterative. **Multi-trip:** not budget-aware (D6):
    under ``max_time_work`` the map ignores the budget during its search and ``fit`` warns; the
    result is still decoded into trips and priced under the multi-trip objective (``history_``
    records that objective).

    **Ceiling.** Like every 2.0 solver, SOM evaluates on a dense ``(n, n)`` cost matrix, so the
    practical ceiling is about 20 000 nodes (3.2 GB of float64). The four bundled instances
    above it (``vm22775``, ``sw24978``, ``bm33708``, ``ch71009``) can be read and subsampled
    (``load_tsp(name, n_nodes=5000)``) but not solved whole; coordinate-only fitting with
    Euclidean costs computed on the fly (``fit(None, coords=xy)``) is a 2.1 item (D18).
    :class:`~skroute.ensemble.MultiStart` with threads gains little on this numpy-bound solver
    (R9); prefer ``prefer="processes"`` there.

    References
    ----------
    .. [1] T. Kohonen, "Self-organized formation of topologically correct feature maps",
       Biological Cybernetics 43 (1982) 59-69.
    .. [2] B. Angéniol, G. de la Croix Vaubois and J.-Y. Le Texier, "Self-organizing feature
       maps and the travelling salesman problem", Neural Networks 1 (1988) 289-293.

    Examples
    --------
    >>> import numpy as np
    >>> from skroute.metaheuristics import SOM
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")  # Western Sahara, optimum 27603
    >>> som = SOM(random_state=0).fit(wi.distance_matrix(), coords=wi.coords, labels=wi.labels)
    >>> som.cost_ / wi.optimal_tour_length < 1.15  # the fast-tier tolerance of SPEC §6
    True
    >>> int(som.route_[0]) == int(som.route_[-1]) == int(som.depot_) == 1
    True
    >>> som.n_iter_ == len(som.history_) and som.stop_reason_ in {"converged", "max_iter"}
    True
    >>> bool(np.all(np.diff(som.history_) <= 0)) and som.n_samples_ <= som.n_iter
    True

    Without coordinates the solver refuses to fit:

    >>> SOM().fit(wi.distance_matrix())
    Traceback (most recent call last):
        ...
    ValueError: SOM needs node coordinates: fit(X, coords=...)
    """

    _parameter_constraints: dict = {
        "n_units": [Interval(Integral, 1, None, closed="left"), None],
        "learning_rate": [Interval(Real, 0.0, 1.0, closed="right")],
        "lr_decay": [Interval(Real, 0.0, 1.0, closed="right")],
        "radius": [Interval(Real, 0.0, None, closed="neither"), None],
        "radius_decay": [Interval(Real, 0.0, 1.0, closed="right")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    n_samples_: int

    def __init__(
        self,
        n_units: int | None = None,
        learning_rate: float = 0.8,
        lr_decay: float = 0.99997,
        radius: float | None = None,
        radius_decay: float = 0.9997,
        n_iter: int = 100_000,
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
    ) -> None:
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.radius = radius
        self.radius_decay = radius_decay
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def _get_tags(self) -> RouterTags:
        return RouterTags(
            kind="metaheuristic",
            exact=False,
            stochastic=True,
            iterative=True,
            budget_aware=False,
            requires_symmetric=False,
            requires_coords=True,
            max_nodes=None,
        )

    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        if rng is None:  # unreachable through fit: the tag says stochastic, so fit passes a Generator
            raise RuntimeError("SOM._solve needs the Generator handed by BaseRouter.fit (bug in the solver)")
        if problem.coords is None:  # unreachable through fit (tag requires_coords), kept for direct callers
            raise ValueError("SOM needs node coordinates: fit(X, coords=...)")
        n = problem.n
        xy = normalize_coords(problem.coords)  # aspect-preserving, longer side spans [0, 1]
        m = 8 * n if self.n_units is None else int(self.n_units)
        radius = m / 10.0 if self.radius is None else float(self.radius)
        lr = float(self.learning_rate)
        lr_decay, radius_decay = float(self.lr_decay), float(self.radius_decay)
        n_iter = int(self.n_iter)
        epoch_len = max(1, n_iter // 100)
        n_epochs = -(-n_iter // epoch_len)  # ceil: the last epoch may be shorter
        every = max(1, n_epochs // 10)

        # The ring starts at random inside the bounding box of the normalised cities (D10: every draw
        # comes from rng, pre-drawn in batches: the initial weights here, one index vector per epoch).
        weights = rng.random((m, 2)) * xy.max(axis=0)
        positions = np.arange(m, dtype=np.float64)
        # D30: no tour exists before the first epoch decodes the ring
        self._emit("start", 0, None, np.nan, radius=radius, learning_rate=lr, n_units=m)

        history: list[float] = []
        best_cost = np.inf
        best_tour: np.ndarray | None = None
        n_samples = 0
        reason = "max_iter"
        for k in range(n_epochs):
            count = min(epoch_len, n_iter - n_samples)
            idx = rng.integers(0, n, size=count)  # the cities of this epoch, pre-drawn
            for i in idx.tolist():
                city = xy[i]
                diff = weights - city
                d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                winner = int(np.argmin(d2))
                delta = np.abs(positions - winner)
                ring = np.minimum(delta, m - delta)  # wrapped ring distance
                sigma = radius if radius > _SIGMA_MIN else _SIGMA_MIN  # radius itself keeps decaying
                g = np.exp(-(ring * ring) / (2.0 * sigma * sigma))
                weights -= (lr * g)[:, None] * diff  # w += lr * g * (city - w)
                lr *= lr_decay
                radius *= radius_decay
            n_samples += count
            tour = _ring_to_tour(_winners(xy, weights), problem.depot)
            cost = float(problem.evaluate(tour))
            if best_tour is None or cost < best_cost - 1e-9 * max(1.0, abs(best_cost)):
                best_cost, best_tour = cost, tour
            history.append(best_cost)
            if self._callback is not None:
                # D30: the epoch's decoded ring is the current tour, the best epoch's tour the best-so-far
                self._emit(
                    "iteration",
                    k + 1,
                    tour,
                    cost,
                    best_tour,
                    best_cost,
                    radius=radius,
                    learning_rate=lr,
                    n_samples=n_samples,
                )
            if self._stop_requested:
                reason = "callback"
                break
            if radius < _RADIUS_FLOOR or lr < _LR_FLOOR:
                reason = "converged"
            if self.verbose >= 2 or (self.verbose and (k % every == 0 or reason == "converged")):
                log.info(
                    "SOM epoch %d/%d (%d samples): current %.6f, best %.6f, learning_rate %.4g, radius %.4g",
                    k + 1,
                    n_epochs,
                    n_samples,
                    cost,
                    best_cost,
                    lr,
                    radius,
                )
            if reason == "converged":
                break
        if self.verbose:
            log.info(
                "SOM stopped by %s after %d epochs (%d samples): best %.6f",
                reason,
                len(history),
                n_samples,
                best_cost,
            )
        assert best_tour is not None  # n_epochs >= 1, so at least one epoch ran
        self.history_ = np.asarray(history, dtype=np.float64)
        self.n_iter_ = len(history)
        self.n_samples_ = n_samples
        self.stop_reason_ = reason
        return best_tour
