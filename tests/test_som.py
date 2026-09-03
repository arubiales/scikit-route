"""Acceptance tests of SPEC §4.4 for ``SOM`` (WP7): coordinates required, valid tours on the tiny tier,
fast-tier gaps on wi29/dj38, best-so-far ``history_``, bit-identical reproducibility, seed sensitivity,
the multi-trip ``UserWarning`` with the result still split into trips, the epoch accounting and the
smallest legal sizes. Slow-tier gaps live in ``tests/benchmarks/test_waterloo.py`` (WP8)."""

from __future__ import annotations

import logging

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


def test_returned_tour_is_the_best_epoch(small_euclidean):
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
