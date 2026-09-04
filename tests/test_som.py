"""Acceptance tests of SPEC §4.4 for ``SOM`` (WP7): coordinates required, valid tours on the tiny tier,
fast-tier gaps on wi29/dj38, best-so-far ``history_`` and the best epoch's tour (not the last one),
bit-identical reproducibility, seed sensitivity, the multi-trip ``UserWarning`` with the result still
split into trips, the epoch accounting, the smallest legal sizes, the update rule replayed by hand
(wrapped ring distance, Gaussian of width ``radius``, both rates decayed per sample), degenerate
coordinates (coincident, collinear, duplicated points) and extreme ``radius``/``radius_decay`` values.
Slow-tier gaps live in ``tests/benchmarks/test_waterloo.py`` (WP8)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
import reference
from conftest import _euclid, fit_kwargs
from hypothesis import given, settings
from hypothesis import strategies as st

import skroute.metaheuristics._som as som_module
from skroute import RoutingProblem, all_solvers
from skroute.metaheuristics import SOM
from skroute.metaheuristics._som import _ring_to_tour, _winners
from skroute.preprocessing import normalize_coords

# --------------------------------------------------------------------------- registration and contract


def test_registered_and_tagged():
    assert SOM in all_solvers()
    tags = SOM()._get_tags()
    assert tags.kind == "metaheuristic"
    assert tags.stochastic and tags.iterative
    assert not tags.exact and not tags.budget_aware and not tags.requires_symmetric
    assert tags.requires_coords and tags.max_nodes is None
    names = SOM._get_param_names()
    assert "time_limit" not in names and "patience" not in names  # {"converged", "max_iter"} only


def test_repr_prints_changed_parameters_only():
    assert repr(SOM()) == "SOM()"
    assert repr(SOM(n_units=10, random_state=0)) == "SOM(n_units=10, random_state=0)"
    assert eval(repr(SOM(radius=2.5))) == SOM(radius=2.5)  # check 1 of §6: eval(repr(est)) == est


def test_raises_without_coords(small_euclidean):
    with pytest.raises(ValueError, match="needs node coordinates"):
        SOM(random_state=0).fit(small_euclidean["C"])


@pytest.mark.parametrize(
    ("params", "name"),
    [
        ({"n_iter": 0}, "n_iter"),
        ({"n_units": 0}, "n_units"),
        ({"learning_rate": 1.5}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"lr_decay": 0.0}, "lr_decay"),
        ({"radius": 0.0}, "radius"),
        ({"radius_decay": 1.5}, "radius_decay"),
        ({"verbose": -1}, "verbose"),
    ],
)
def test_parameter_validation_at_fit(small_euclidean, params, name):
    est = SOM(random_state=0, **params)
    with pytest.raises(ValueError, match=name):
        est.fit(small_euclidean["C"], coords=small_euclidean["coords"])


# --------------------------------------------------------------------------- tiny and fast tiers


def _assert_valid(est, C):
    n = C.shape[0]
    assert int(est.route_[0]) == int(est.route_[-1]) == int(est.depot_)
    assert sorted(est.tour_.tolist()) == list(range(n))
    assert est.cost_ == pytest.approx(
        reference.route_cost_from_labels(C, est.route_, est.labels_, est.depot_), rel=1e-9
    )


def test_valid_tour_on_tiny(tiny_instance):
    kw = fit_kwargs(SOM, tiny_instance)  # pytest.skip on the asymmetric fixtures (no coordinates)
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    est = SOM(random_state=0).fit(C, **kw)
    _assert_valid(est, C)
    assert opt <= est.cost_ + 1e-9
    assert est.history_[-1] == pytest.approx(est.cost_)


def test_fast_tier_gap(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    est = SOM(random_state=0).fit(C, coords=fast_instance["coords"], labels=fast_instance["labels"])
    assert opt <= est.cost_ + 1e-9
    assert int(est.route_[0]) == int(est.route_[-1]) == int(fast_instance["labels"][0])
    gap = est.cost_ / opt - 1
    assert gap <= 0.15, f"{fast_instance['name']}: gap {gap:.4f} > 0.15"


def test_asymmetric_matrix_is_priced_directionally():
    C, xy = _euclid(7, seed=71, asymmetric=True)
    est = SOM(random_state=0).fit(C, coords=xy)
    _assert_valid(est, C)  # route_cost_from_labels reads C[i, j] directionally


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["sym", "asym"])
def test_smallest_sizes(n, asymmetric):
    C, xy = _euclid(n, seed=n, asymmetric=asymmetric)
    est = SOM(random_state=0).fit(C, coords=xy)
    _assert_valid(est, C)
    assert est.n_iter_ == len(est.history_) >= 1
    assert est.stop_reason_ in {"converged", "max_iter"}


# --------------------------------------------------------------------------- iterative contract (D9, R8)


def test_history_is_best_so_far_and_ends_at_cost(small_euclidean):
    est = SOM(random_state=0).fit(small_euclidean["C"], coords=small_euclidean["coords"])
    assert est.history_.dtype == np.float64 and est.history_.ndim == 1
    assert est.n_iter_ == len(est.history_) >= 1
    assert np.all(np.diff(est.history_) <= 0.0)
    assert est.history_[-1] == pytest.approx(est.cost_, rel=1e-12)
    assert est.stop_reason_ in {"converged", "max_iter"}
    assert 1 <= est.n_samples_ <= est.n_iter


def test_default_run_converges_before_n_iter(small_euclidean):
    # radius = 0.8 n decays below one ring position after ~ln(0.8 n) / 0.0003 samples: 8 epochs at n = 12
    est = SOM(random_state=0).fit(small_euclidean["C"], coords=small_euclidean["coords"])
    assert est.stop_reason_ == "converged"
    assert est.n_samples_ < est.n_iter
    assert est.n_samples_ == est.n_iter_ * 1000  # epochs of n_iter // 100 samples


@pytest.mark.parametrize(
    ("n_iter", "epochs", "samples"),
    [(250, 125, 250), (150, 150, 150), (1000, 100, 1000), (1, 1, 1), (199, 199, 199), (201, 101, 201)],
)
def test_epoch_accounting_without_decay(small_euclidean, n_iter, epochs, samples):
    # lr_decay = radius_decay = 1.0: the run can only stop by max_iter, after exactly n_iter samples
    est = SOM(n_iter=n_iter, lr_decay=1.0, radius_decay=1.0, random_state=0)
    est.fit(small_euclidean["C"], coords=small_euclidean["coords"])
    assert est.stop_reason_ == "max_iter"
    assert est.n_iter_ == len(est.history_) == epochs
    assert est.n_samples_ == samples


@pytest.mark.parametrize("params", [{"radius": 0.5}, {"learning_rate": 5e-4}])
def test_converged_after_the_first_epoch(small_euclidean, params):
    # the convergence test runs at the end of an epoch: a radius already below 1 (or a learning rate
    # already below 1e-3) stops the run after exactly one epoch of n_iter // 100 samples
    est = SOM(random_state=0, **params).fit(small_euclidean["C"], coords=small_euclidean["coords"])
    assert est.stop_reason_ == "converged"
    assert est.n_iter_ == 1 and len(est.history_) == 1
    assert est.n_samples_ == 1000


def _spy_decoded_tours(monkeypatch) -> list[np.ndarray]:
    """Record the tour decoded at the end of every epoch (``_ring_to_tour`` is called once per epoch)."""
    decoded: list[np.ndarray] = []
    orig = som_module._ring_to_tour

    def spy(winners, depot):
        tour = orig(winners, depot)
        decoded.append(tour.copy())
        return tour

    monkeypatch.setattr(som_module, "_ring_to_tour", spy)
    return decoded


def test_returned_tour_is_the_best_epoch_not_the_last(small_euclidean, monkeypatch):
    # n_iter=2000 (epochs of 20 samples) stops by max_iter while the ring is still hot: the trace of
    # per-epoch costs is NOT monotone and the last epoch (321.10) is worse than the best one (316.95,
    # epoch 5), so an implementation returning the last ring, or recording the current cost, fails here
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    decoded = _spy_decoded_tours(monkeypatch)
    est = SOM(n_iter=2000, random_state=0).fit(C, coords=xy)
    problem = est.problem_
    costs = np.array([float(problem.evaluate(t)) for t in decoded])
    assert len(decoded) == est.n_iter_ == 100 and est.stop_reason_ == "max_iter"
    assert np.any(np.diff(costs) > 0)  # the natural trace goes up and down (R8)
    best = int(np.argmin(costs))
    assert 0 < best < len(costs) - 1 and costs[-1] > costs[best] + 1e-9
    returned = problem.to_index_tour(est.tour_)
    assert np.array_equal(returned, decoded[best])
    assert not np.array_equal(returned, decoded[-1])
    assert est.cost_ == pytest.approx(costs[best], rel=1e-12)
    assert np.array_equal(est.history_, np.minimum.accumulate(costs))  # best-so-far, never the current


def test_returned_tour_is_the_best_epoch_when_the_run_converges(small_euclidean):
    # history_ is monotone and ends at cost_: the tour of the best epoch is what the base class priced
    est = SOM(random_state=3).fit(small_euclidean["C"], coords=small_euclidean["coords"])
    problem = est.problem_
    assert problem.evaluate(problem.to_index_tour(est.tour_)) == pytest.approx(est.history_.min())


# --------------------------------------------------------------------------- randomness (D10, check 11)


def test_same_seed_is_bit_identical(small_euclidean):
    a, b = (SOM(random_state=7).fit(small_euclidean["C"], coords=small_euclidean["coords"]) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_
    assert np.array_equal(a.history_, b.history_) and a.n_samples_ == b.n_samples_


def test_seeds_zero_and_one_differ(medium_euclidean):
    C, xy = medium_euclidean["C"], medium_euclidean["coords"]
    a = SOM(random_state=0).fit(C, coords=xy)
    b = SOM(random_state=1).fit(C, coords=xy)
    assert not np.array_equal(a.tour_, b.tour_) or not np.array_equal(a.history_, b.history_)


def test_generator_is_advanced_and_matches_the_int_seed(small_euclidean):
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    a = SOM(random_state=rng).fit(C, coords=xy)
    assert rng.bit_generator.state != before
    b = SOM(random_state=5).fit(C, coords=xy)
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.history_, b.history_)


# --------------------------------------------------------------------------- multi-trip (D6)


def test_multi_trip_warns_and_still_splits(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]  # kw carries labels= and the LABEL depot
    est = SOM(random_state=0)
    with pytest.warns(UserWarning, match="ignores max_time_work"):
        est.fit(d.cost, time_matrix=d.time, coords=d.coords, **kw)
    budget = kw["max_time_work"]
    assert np.all(est.trip_times_ <= budget + 1e-9)
    assert est.n_trips_ == len(est.trips_) == len(est.trip_costs_) == len(est.trip_times_)
    assert all(int(t[0]) == int(t[-1]) == int(est.depot_) for t in est.trips_)
    tour = est.problem_.to_index_tour(est.tour_)
    expected = reference.problem_cost(d.cost, d.time, tour, budget, 10.0 * 2, "greedy")
    assert est.cost_ == pytest.approx(expected, rel=1e-9)
    assert alicante["optimum"]["greedy"] <= est.cost_ + 1e-9
    assert est.history_[-1] == pytest.approx(est.cost_)  # history_ records the multi-trip objective
    # the same tour under the optimal decoder is never worse (same tour, both decoders; check 8)
    p_opt = RoutingProblem(d.cost, time_matrix=d.time, split="optimal", **kw)
    assert p_opt.evaluate(tour) <= est.cost_ + 1e-9


def test_plain_fit_has_no_trip_times(alicante):
    d = alicante["bunch"]
    est = SOM(random_state=0).fit(d.cost, coords=d.coords, labels=d.labels, depot=d.depot)
    assert est.n_trips_ == 1 and not hasattr(est, "trip_times_")
    assert int(est.route_[0]) == int(est.route_[-1]) == int(d.depot)


# --------------------------------------------------------------------------- decoding


def test_single_unit_ring_decodes_to_index_order(small_euclidean):
    # every city wins neuron 0: ties by city index, rotated to the depot
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    est = SOM(n_units=1, random_state=0).fit(C, coords=xy, depot=3)
    assert est.tour_.tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]
    assert est.stop_reason_ == "converged"  # radius = n_units / 10 = 0.1 < 1 after the first epoch


@settings(derandomize=True, deadline=None, max_examples=200)
@given(st.data())
def test_ring_to_tour_orders_by_winner_then_index(data):
    n = data.draw(st.integers(3, 12))
    m = data.draw(st.integers(1, 40))
    winners = np.asarray(data.draw(st.lists(st.integers(0, m - 1), min_size=n, max_size=n)))
    depot = data.draw(st.integers(0, n - 1))
    tour = _ring_to_tour(winners, depot)
    assert tour.dtype == np.int64 and tour[0] == depot
    assert sorted(tour.tolist()) == list(range(n))
    keys = [(int(winners[c]), int(c)) for c in tour]
    descents = sum(keys[i] > keys[(i + 1) % n] for i in range(n))
    assert descents <= 1  # a rotation of the (winner, index)-sorted order


def test_winners_match_a_full_argmin_in_blocks(monkeypatch):
    rng = np.random.default_rng(0)
    xy, weights = rng.random((17, 2)), rng.random((23, 2))
    d2 = ((xy[:, None, :] - weights[None, :, :]) ** 2).sum(-1)
    expected = np.argmin(d2, axis=1)
    assert np.array_equal(_winners(xy, weights), expected)
    monkeypatch.setattr(som_module, "_DECODE_BLOCK", 1)  # one city per block
    assert np.array_equal(_winners(xy, weights), expected)


# --------------------------------------------------------------------------- update rule (§4.4), by hand


def _euclid_of(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.ascontiguousarray(np.sqrt((diff**2).sum(axis=-1)))


def _spy_epoch_weights(monkeypatch) -> list[np.ndarray]:
    """Record the ring at the end of every epoch (``_winners`` is called once per epoch to decode it)."""
    seen: list[np.ndarray] = []
    orig = som_module._winners

    def spy(xy, weights):
        seen.append(weights.copy())
        return orig(xy, weights)

    monkeypatch.setattr(som_module, "_winners", spy)
    return seen


def _replay(xy, weights, cities, *, lr, lr_decay, radius, radius_decay):
    """The §4.4 update written from the SPEC: one ring per presented city, rates decayed after each."""
    w, m, rings = np.array(weights, dtype=np.float64), len(weights), []
    for i in cities:
        x = xy[i]
        winner = int(np.argmin(((w - x) ** 2).sum(axis=1)))
        delta = np.abs(np.arange(m) - winner)
        ring = np.minimum(delta, m - delta)  # wrapped ring distance
        g = np.exp(-(ring**2) / (2.0 * radius**2))
        w = w + lr * g[:, None] * (x - w)
        lr, radius = lr * lr_decay, radius * radius_decay
        rings.append(w.copy())
    return rings


class _ScriptedRng:
    """Stands in for the Generator handed by ``fit``: a chosen initial ring and a scripted city sequence."""

    def __init__(self, unit_square_weights, cities):
        self.weights = np.asarray(unit_square_weights, dtype=np.float64)  # what rng.random((m, 2)) returns
        self.cities = list(cities)

    def random(self, size):
        assert tuple(size) == self.weights.shape
        return self.weights.copy()

    def integers(self, low, high, size):
        assert (low, high) == (0, 5) and size == 1  # one epoch of one sample at a time (n_iter < 200)
        return np.asarray([self.cities.pop(0)], dtype=np.int64)


@pytest.mark.parametrize(
    ("n_iter", "decays"),
    [(1, {}), (2, {"lr_decay": 0.5, "radius_decay": 0.5})],
    ids=["one-sample", "two-samples-decayed"],
)
def test_samples_replayed_by_hand(small_euclidean, monkeypatch, n_iter, decays):
    # n_iter < 200 -> epochs of ONE sample, decoded after each. D10 draw order: the initial ring
    # (rng.random((m, 2)) * xy.max(0)), then one index vector per epoch (rng.integers(0, n, size=1)).
    C, coords = small_euclidean["C"], small_euclidean["coords"]
    n, m = C.shape[0], 40
    xy = normalize_coords(coords)
    rng = np.random.default_rng(11)
    w0 = rng.random((m, 2)) * xy.max(axis=0)
    cities = [int(rng.integers(0, n, size=1)[0]) for _ in range(n_iter)]
    expected = _replay(
        xy,
        w0,
        cities,
        lr=0.8,
        lr_decay=decays.get("lr_decay", 0.99997),
        radius=m / 10,  # radius=None -> n_units / 10
        radius_decay=decays.get("radius_decay", 0.9997),
    )
    seen = _spy_epoch_weights(monkeypatch)
    est = SOM(n_units=m, n_iter=n_iter, random_state=11, **decays).fit(C, coords=coords)
    assert len(seen) == n_iter and all(np.array_equal(a, b) for a, b in zip(seen, expected, strict=True))
    tours = [_ring_to_tour(_winners(xy, w), 0) for w in expected]
    costs = [float(est.problem_.evaluate(t)) for t in tours]
    assert np.array_equal(est.tour_, tours[int(np.argmin(costs))])
    assert np.array_equal(est.history_, np.minimum.accumulate(costs))
    assert est.n_samples_ == est.n_iter_ == n_iter and est.stop_reason_ == "max_iter"
    if n_iter == 2:  # the second sample really used the decayed rates: replaying it undecayed disagrees
        undecayed = _replay(xy, w0, cities, lr=0.8, lr_decay=1.0, radius=m / 10, radius_decay=1.0)
        assert not np.array_equal(seen[1], undecayed[1])


def test_neighbourhood_wraps_around_the_ring(monkeypatch):
    # Unit 0 sits next to city 0 and wins; with radius=2 on 8 units the wrapped distances are
    # [0, 1, 2, 3, 4, 3, 2, 1], so units 1 and 7 (2 and 6, 3 and 5) must move by the same fraction
    # lr * exp(-d^2 / 8) of their gap to the city, and unit 4 (opposite) by the smallest one.
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    xy = normalize_coords(coords)  # the unit square itself: xy.max(0) == (1, 1)
    problem = RoutingProblem(_euclid_of(coords), coords=coords)
    m, lr, radius = 8, 0.5, 2.0
    w0 = np.full((m, 2), 0.9)
    w0[0] = 0.1
    est = SOM(n_units=m, learning_rate=lr, lr_decay=1.0, radius=radius, radius_decay=1.0, n_iter=1)
    seen = _spy_epoch_weights(monkeypatch)
    tour = est._solve(problem, _ScriptedRng(w0, cities=[0]))
    assert len(seen) == 1 and sorted(tour.tolist()) == list(range(5))
    frac = (seen[0] - w0) / (xy[0] - w0)  # lr * g_j, per unit and per coordinate
    d = np.array([0, 1, 2, 3, 4, 3, 2, 1])
    assert np.allclose(frac[:, 0], frac[:, 1]) and np.allclose(frac[:, 0], lr * np.exp(-(d**2) / 8.0))
    assert frac[1, 0] == frac[7, 0] and frac[2, 0] == frac[6, 0] and frac[3, 0] == frac[5, 0]
    assert frac[4, 0] == frac[:, 0].min() < frac[3, 0] < frac[1, 0] < frac[0, 0] == pytest.approx(lr)


# --------------------------------------------------------------------------- degenerate geometry, extremes


_LINE = np.array([0.0, 4.0, 1.0, 9.0, 4.0, 6.0, 9.0])  # collinear, with the points 4 and 9 duplicated


@pytest.mark.parametrize(
    ("coords", "depot"),
    [
        (np.full((6, 2), 3.0), 2),
        (np.c_[_LINE, np.zeros(7)], 0),
        (np.c_[np.zeros(7), _LINE], 3),
        (np.c_[_LINE, _LINE], 1),
    ],
    ids=["coincident", "horizontal", "vertical", "diagonal"],
)
def test_degenerate_coordinates(coords, depot):
    # coincident points: normalize_coords returns zeros, the ring collapses onto them and every city wins
    # neuron 0 -> index order rotated to the depot; collinear points with duplicates: the ring lies on
    # the line and the decoded tour is the line optimum 2 * length. No numpy warning anywhere.
    C = _euclid_of(coords)
    n = C.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = SOM(random_state=0).fit(C, coords=coords, depot=depot)
    _assert_valid(est, C)
    assert est.stop_reason_ == "converged"
    if np.ptp(coords) == 0.0:
        assert est.tour_.tolist() == [(depot + k) % n for k in range(n)] and est.cost_ == 0.0
    else:
        assert est.cost_ == pytest.approx(2.0 * np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))


def test_duplicated_points_in_the_plane():
    rng = np.random.default_rng(0)
    base = rng.random((8, 2)) * 100
    coords = np.vstack([base, base[:2]])  # two duplicated points
    C = _euclid_of(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = SOM(random_state=0).fit(C, coords=coords, depot=9)
    _assert_valid(est, C)


@pytest.mark.parametrize(
    "params",
    [{"radius_decay": 0.5}, {"radius": 1e-200}, {"n_iter": 1_000_000, "radius_decay": 0.96}],
    ids=["radius_decay=0.5", "radius=1e-200", "n_iter=1e6,radius_decay=0.96"],
)
def test_tiny_radius_never_underflows_the_gaussian(small_euclidean, monkeypatch, params):
    # Legal values drive the radius below ~1e-154 inside an epoch, where radius**2 underflows to 0.0:
    # the winner's Gaussian was 0/0 = NaN, the weights went NaN, every city won the first NaN neuron
    # and fit returned the index-order tour with nothing but numpy RuntimeWarnings. The width of the
    # Gaussian is floored (_SIGMA_MIN); the ring stays finite and the tour is a real one.
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    seen = _spy_epoch_weights(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = SOM(random_state=0, **params).fit(C, coords=xy)
    _assert_valid(est, C)
    assert seen and all(np.all(np.isfinite(w)) for w in seen)
    assert est.tour_.tolist() != list(range(C.shape[0]))
    assert est.stop_reason_ == "converged"  # radius < 1 at the end of the first epoch


# --------------------------------------------------------------------------- labels, depot, logging (D24)


def test_string_labels_and_depot_by_label(small_euclidean):
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    names = [f"c{i}" for i in range(C.shape[0])]
    est = SOM(random_state=0).fit(C, coords=xy, labels=names, depot="c4")
    assert est.depot_ == "c4" and est.route_[0] == "c4" == est.route_[-1]
    assert sorted(est.tour_.tolist()) == sorted(names)


def test_verbose_logs_and_never_prints(small_euclidean, caplog, capsys):
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    with caplog.at_level(logging.INFO, logger="skroute"):
        est1 = SOM(random_state=0, verbose=1).fit(C, coords=xy)
        n1 = len(caplog.records)
        caplog.clear()
        est2 = SOM(random_state=0, verbose=2).fit(C, coords=xy)
        n2 = len(caplog.records)
    assert n1 >= 1 and all(r.name.startswith("skroute") for r in caplog.records)
    assert n2 >= est2.n_iter_ + 1 > n1  # verbose=2 logs every epoch plus the stop line
    assert est1.cost_ == est2.cost_
    out, err = capsys.readouterr()
    assert out == "" and err == ""
    silent = SOM(random_state=0).fit(C, coords=xy)
    assert silent.cost_ == est1.cost_


# ------------------------------------------------------------------------ D31: the ring, in coordinate units
def test_ring_events_map_the_neurons_back_to_the_coordinates(small_euclidean, monkeypatch):
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    weights = _spy_epoch_weights(monkeypatch)
    events = []
    som = SOM(n_units=30, n_iter=500, random_state=0).fit(C, coords=xy, callback=events.append)
    iters = [e for e in events if e.stage == "iteration"]
    assert "ring" not in events[0].extra and "ring" not in events[-1].extra
    assert len(iters) == som.n_iter_ == len(weights)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = float((hi - lo).max())
    for e, w in zip(iters, weights, strict=True):
        ring = e.extra["ring"]
        assert isinstance(ring, np.ndarray) and ring.shape == (30, 2) and ring.dtype == np.float64
        np.testing.assert_allclose(ring, w * span + lo, rtol=0, atol=1e-9 * span)  # the scaling inverted
        # neurons start inside the normalised bounding box and only move towards cities: they stay inside
        assert (ring >= lo - 1e-9 * span).all() and (ring <= hi + 1e-9 * span).all()
    assert len({id(e.extra["ring"]) for e in iters}) == len(iters)  # a fresh array per event
    assert not np.array_equal(iters[0].extra["ring"], iters[-1].extra["ring"])  # and the ring did move


def test_ring_of_coincident_cities_sits_on_them():
    coords = np.full((5, 2), 3.0)
    events = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SOM(random_state=0, n_iter=50).fit(_euclid_of(coords), coords=coords, callback=events.append)
    rings = [e.extra["ring"] for e in events if e.stage == "iteration"]
    assert rings and all(
        np.array_equal(r, np.full((40, 2), 3.0)) for r in rings
    )  # span 0: back onto the point


def test_ring_is_not_built_without_a_callback(small_euclidean, monkeypatch):
    # the inverse scaling is two numbers; the per-epoch (m, 2) product happens only behind the callback guard
    C, xy = small_euclidean["C"], small_euclidean["coords"]
    a = SOM(n_units=20, n_iter=300, random_state=0).fit(C, coords=xy)
    b = SOM(n_units=20, n_iter=300, random_state=0).fit(C, coords=xy, callback=lambda e: None)
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
