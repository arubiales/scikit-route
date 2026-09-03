"""Acceptance tests of ``skroute.ensemble`` (SPEC §4.5, D17, D27): the ``check_router`` battery on
``MultiStart(SimulatedAnnealing(), n_restarts=4)``, ``n_jobs`` invariance, the refusal of deterministic
estimators, delegated tags, the parameter protocol with nested ``estimator__*`` keys, the tiny/fast
tolerances of the table (``0`` / ``<= SimulatedAnnealing``'s entry), and the two explicit-parameter
wrappers ``EnsembleGenetic``/``EnsembleSimulatedAnnealing`` (bit-identical to the ``MultiStart`` they
build, attributes copied, no runtime ``DeprecationWarning``). Slow-tier gaps live in
``tests/benchmarks/test_waterloo.py``."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
import tolerances
from conftest import _euclid, fit_kwargs, make

from skroute import RoutingProblem, all_solvers, clone
from skroute.base import BaseRouter, RouterTags
from skroute.construction import NRBS, ClarkeWright, NearestNeighbour
from skroute.ensemble import EnsembleGenetic, EnsembleSimulatedAnnealing, MultiStart
from skroute.exact import BruteForce
from skroute.metaheuristics import SOM, Genetic, SimulatedAnnealing, TabuSearch
from skroute.metrics import route_cost
from skroute.utils.estimator_checks import CheckSkipped, check_router

CHECKS = check_router.checks
CHECK_IDS = [name for name, _ in CHECKS]
CHECK_FNS = [fn for _, fn in CHECKS]
WRAPPERS = [EnsembleGenetic, EnsembleSimulatedAnnealing]
RESTARTS = {EnsembleGenetic: "n_genetics", EnsembleSimulatedAnnealing: "n_simulateds"}


def _ms(n_restarts=4, **kw):
    return MultiStart(SimulatedAnnealing(), n_restarts=n_restarts, **kw)


# --------------------------------------------------------------------------- the battery (SPEC §4.5, D27)
@pytest.mark.parametrize("check", CHECK_FNS, ids=CHECK_IDS)
def test_check_router_on_multistart(check, capsys):
    try:
        check(_ms())
    except CheckSkipped as e:
        pytest.skip(str(e))
    out, err = capsys.readouterr()
    assert out == "" and err == "", "nothing may be printed while a solver runs (D24)"


def test_roster_membership():
    roster = all_solvers()
    assert MultiStart not in roster  # needs an estimator (D27)
    assert EnsembleGenetic in roster and EnsembleSimulatedAnnealing in roster
    assert [s.__name__ for s in roster] == sorted(s.__name__ for s in roster)


# --------------------------------------------------------------------------- tags
def test_tags_delegate_to_the_estimator():
    tags = _ms()._get_tags()
    assert tags.kind == "ensemble" and tags.stochastic and tags.iterative and tags.budget_aware
    assert not tags.exact and not tags.requires_symmetric and not tags.requires_coords
    som = MultiStart(SOM())._get_tags()
    assert som.kind == "ensemble" and som.stochastic and som.requires_coords and not som.budget_aware
    # a deterministic estimator is still advertised as stochastic (the wrapper consumes random_state)
    cw = MultiStart(ClarkeWright())._get_tags()
    assert cw.stochastic and cw.requires_symmetric and cw.kind == "ensemble"
    bf = MultiStart(BruteForce())._get_tags()
    assert bf.exact and bf.max_nodes == 11


@pytest.mark.parametrize("estimator", [NearestNeighbour(), BruteForce(), ClarkeWright(), NRBS()])
def test_refuses_a_non_stochastic_estimator(estimator):
    C, _ = _euclid(8, seed=8)  # below every max_nodes cap, so the refusal is what fit reaches first
    with pytest.raises(
        ValueError, match=r"^MultiStart needs a stochastic estimator \(one with random_state\)$"
    ):
        MultiStart(estimator, n_restarts=2).fit(C)


def test_refuses_a_non_router_estimator(small_euclidean):
    with pytest.raises(ValueError, match="The 'estimator' parameter of MultiStart must be a BaseRouter"):
        MultiStart("SimulatedAnnealing").fit(small_euclidean["C"])


@pytest.mark.parametrize(
    ("params", "name"),
    [({"n_restarts": 0}, "n_restarts"), ({"n_jobs": 0}, "n_jobs"), ({"prefer": "gpu"}, "prefer")],
)
def test_invalid_parameters_raise_at_fit(params, name, small_euclidean):
    with pytest.raises(ValueError, match=f"The {name!r} parameter of MultiStart"):
        _ms(**params).fit(small_euclidean["C"])


# --------------------------------------------------------------------------- n_jobs invariance (D17)
def test_results_do_not_depend_on_n_jobs(medium_euclidean):
    C = medium_euclidean["C"]
    a = _ms(n_jobs=1, random_state=0).fit(C)
    b = _ms(n_jobs=2, random_state=0).fit(C)
    c = _ms(n_jobs=-1, random_state=0).fit(C)
    for other in (b, c):
        assert np.array_equal(a.tour_, other.tour_) and a.cost_ == other.cost_
        assert np.array_equal(a.costs_, other.costs_) and a.best_index_ == other.best_index_
        assert np.array_equal(a.history_, other.history_)


def test_results_do_not_depend_on_the_backend(medium_euclidean):
    C = medium_euclidean["C"]
    threads = _ms(n_jobs=2, prefer="threads", random_state=0).fit(C)
    processes = _ms(n_jobs=2, prefer="processes", random_state=0).fit(C)
    assert np.array_equal(threads.tour_, processes.tour_) and np.array_equal(threads.costs_, processes.costs_)
    assert threads.best_index_ == processes.best_index_


# --------------------------------------------------------------------------- fitted attributes
def test_fitted_attributes(small_euclidean):
    C = small_euclidean["C"]
    ms = _ms(random_state=0).fit(C)
    assert isinstance(ms.estimators_, list) and len(ms.estimators_) == 4
    assert all(isinstance(e, SimulatedAnnealing) and e is not ms.estimator for e in ms.estimators_)
    assert ms.costs_.dtype == np.float64 and ms.costs_.shape == (4,)
    assert np.array_equal(ms.costs_, [e.cost_ for e in ms.estimators_])
    assert ms.best_estimator_ is ms.estimators_[ms.best_index_]
    assert ms.cost_ == ms.costs_.min() == ms.best_estimator_.cost_
    assert np.array_equal(ms.tour_, ms.best_estimator_.tour_)
    assert np.array_equal(ms.history_, ms.best_estimator_.history_)
    assert ms.n_iter_ == ms.best_estimator_.n_iter_ and ms.stop_reason_ == ms.best_estimator_.stop_reason_
    # every restart was fitted on the SAME RoutingProblem (nothing is copied per worker)
    assert all(e.problem_ is ms.problem_ for e in ms.estimators_)
    assert ms.cost_ == pytest.approx(route_cost(C, ms.route_), rel=1e-9)
    # the estimator handed in is never fitted
    assert not hasattr(ms.estimator, "cost_")


def test_ties_go_to_the_lowest_index(tiny_instance):
    ms = _ms(random_state=0).fit(tiny_instance["C"])
    tied = np.flatnonzero(ms.costs_ <= ms.costs_.min() + 1e-9 * max(1.0, ms.costs_.min()))
    assert ms.best_index_ == int(tied[0])
    assert ms.cost_ == pytest.approx(tiny_instance["optimum"], rel=1e-9)  # table: MultiStart(SA, 4) -> 0
    assert len(tied) == 4, "every restart reaches the tiny optimum, so the tie rule is exercised"


def test_restarts_receive_independent_generators(small_euclidean):
    ms = _ms(random_state=0).fit(small_euclidean["C"])
    states = [e.random_state.bit_generator.state["state"]["state"] for e in ms.estimators_]
    assert all(isinstance(e.random_state, np.random.Generator) for e in ms.estimators_)
    assert len(set(states)) == 4, "SeedSequence.spawn must give every restart its own stream"


# --------------------------------------------------------------------------- tolerances (SPEC §6 table)
def test_fast_tier_within_the_wrapped_solvers_tolerance(fast_instance):
    C, opt = fast_instance["C"], fast_instance["optimum"]
    ms = _ms(random_state=0).fit(C, labels=fast_instance["labels"])
    assert opt <= ms.cost_ + 1e-9
    assert ms.cost_ / opt - 1 <= tolerances.FAST["SimulatedAnnealing"]
    assert int(ms.route_[0]) == int(ms.route_[-1]) == int(fast_instance["labels"][0])


def test_multi_trip_alicante_reaches_the_optimum(alicante, recwarn):
    d, kw = alicante["bunch"], alicante["kwargs"]
    ms = _ms(random_state=0).fit(d.cost, time_matrix=d.time, **kw)
    assert not [w for w in recwarn if "ignores max_time_work" in str(w.message)]  # budget-aware, delegated
    assert np.all(ms.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert ms.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert all(e.problem_ is ms.problem_ and e.problem_.multi_trip for e in ms.estimators_)


def test_budget_unaware_estimator_warns_once_at_the_outer_level(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ms = MultiStart(SOM(n_iter=2000), n_restarts=2, random_state=0).fit(
            d.cost, time_matrix=d.time, coords=d.coords, **kw
        )
    outer = [w for w in caught if "MultiStart ignores max_time_work" in str(w.message)]
    assert len(outer) == 1 and np.all(ms.trip_times_ <= kw["max_time_work"] + 1e-9)


# --------------------------------------------------------------------------- parameter protocol
def test_nested_parameter_protocol():
    ms = _ms(random_state=0)
    deep = ms.get_params(deep=True)
    assert deep["estimator"] is ms.estimator and deep["n_restarts"] == 4
    assert {k for k in deep if k.startswith("estimator__")} == {
        f"estimator__{k}" for k in SimulatedAnnealing._get_param_names()
    }
    assert "estimator__alpha" not in ms.get_params(deep=False)
    ms.set_params(estimator__alpha=0.9, n_restarts=3)
    assert ms.estimator.alpha == 0.9 and ms.n_restarts == 3
    with pytest.raises(ValueError, match="Invalid parameter 'beta'"):
        ms.set_params(estimator__beta=1)
    copy = clone(ms)
    assert copy == ms and copy is not ms and copy.estimator is not ms.estimator
    assert copy.estimator == ms.estimator and isinstance(copy.estimator, SimulatedAnnealing)
    assert repr(ms) == "MultiStart(estimator=SimulatedAnnealing(alpha=0.9), n_restarts=3, random_state=0)"
    assert repr(MultiStart(Genetic())) == "MultiStart(estimator=Genetic())"
    namespace = {"MultiStart": MultiStart, "SimulatedAnnealing": SimulatedAnnealing}
    assert eval(repr(ms), namespace) == ms


def test_init_stores_only_its_parameters():
    ms = _ms()
    assert (
        set(vars(ms))
        == set(ms.get_params(deep=False))
        == {
            "estimator",
            "n_restarts",
            "n_jobs",
            "prefer",
            "random_state",
            "verbose",
        }
    )
    assert ms.n_restarts == 4 and ms.n_jobs is None and ms.prefer == "threads" and ms.verbose == 0
    assert MultiStart(SimulatedAnnealing()).n_restarts == 10


# --------------------------------------------------------------------------- reproducibility (check 11, D10)
def test_same_seed_is_bit_identical_and_seeds_differ(small_euclidean):
    C = small_euclidean["C"]
    a, b = (_ms(random_state=7).fit(C) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and np.array_equal(a.costs_, b.costs_)
    assert np.array_equal(a.history_, b.history_) and a.best_index_ == b.best_index_
    c = _ms(random_state=8).fit(C)
    assert not np.array_equal(a.history_, c.history_) or a.n_iter_ != c.n_iter_
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    g = _ms(random_state=rng).fit(C)
    assert rng.bit_generator.state != before
    assert np.array_equal(g.tour_, a.tour_) and np.array_equal(g.costs_, a.costs_)


def test_verbose_logs_to_the_skroute_logger(small_euclidean, caplog):
    with caplog.at_level(logging.INFO, logger="skroute"):
        _ms(random_state=0, verbose=1).fit(small_euclidean["C"])
    records = [r.getMessage() for r in caplog.records if r.name == "skroute"]
    assert records and all(m.startswith("MultiStart") for m in records)
    assert any("best restart" in m for m in records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="skroute"):
        _ms(random_state=0, verbose=2).fit(small_euclidean["C"])
    assert sum("restart" in r.getMessage() and "/4" in r.getMessage() for r in caplog.records) == 4


def test_accepts_a_routing_problem_and_other_estimators(small_euclidean):
    problem = RoutingProblem(small_euclidean["C"])
    ms = MultiStart(TabuSearch(n_iter=50), n_restarts=2, random_state=0).fit(problem)
    assert ms.problem_ is problem and ms.stop_reason_ in {"max_iter", "patience"}
    assert all(isinstance(e, TabuSearch) and e.n_iter == 50 for e in ms.estimators_)


class _StochasticNonIterative(BaseRouter):
    """A stochastic solver without history_: MultiStart must not invent the iterative attributes."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def _get_tags(self):
        return RouterTags(kind="construction", stochastic=True)

    def _solve(self, problem, rng):
        return np.concatenate(
            ([problem.depot], rng.permutation(np.delete(np.arange(problem.n), problem.depot)))
        )


def test_non_iterative_estimator_gets_no_iterative_attributes(small_euclidean):
    ms = MultiStart(_StochasticNonIterative(), n_restarts=3, random_state=0).fit(small_euclidean["C"])
    assert not ms._get_tags().iterative
    assert not hasattr(ms, "history_") and not hasattr(ms, "n_iter_") and not hasattr(ms, "stop_reason_")
    assert ms.cost_ == ms.costs_.min() and len(ms.estimators_) == 3


# --------------------------------------------------------------------------- the two legacy wrappers
@pytest.mark.parametrize("Wrapper", WRAPPERS, ids=lambda w: w.__name__)
def test_wrapper_tags_and_signature(Wrapper):
    tags = Wrapper()._get_tags()
    assert tags == RouterTags(kind="ensemble", stochastic=True, iterative=True, budget_aware=True)
    names = Wrapper._get_param_names()
    assert {"n_jobs", "random_state", "verbose", "init", "patience", "time_limit"} <= set(names)
    with pytest.raises(TypeError):  # the inner knobs are keyword-only
        Wrapper(10, None, 0, 0, 50)


def test_wrapper_defaults_are_the_2_0_defaults():
    eg, es = EnsembleGenetic(), EnsembleSimulatedAnnealing()
    assert eg.n_genetics == 10 and es.n_simulateds == 10  # 1.0: n_simulateds=20
    assert eg._inner() == Genetic() and es._inner() == SimulatedAnnealing()
    assert repr(eg) == "EnsembleGenetic()" and repr(es) == "EnsembleSimulatedAnnealing()"
    assert eg.get_params(deep=False)["pop_size"] == 100 and eg.get_params()["n_generations"] == 500
    assert es.get_params()["alpha"] == 0.995 and es.get_params()["patience"] is None


@pytest.mark.parametrize(
    ("Wrapper", "Inner", "count"),
    [(EnsembleGenetic, Genetic, "n_genetics"), (EnsembleSimulatedAnnealing, SimulatedAnnealing, "count")],
    ids=["genetic", "annealing"],
)
def test_wrapper_is_bit_identical_to_the_multistart_it_builds(Wrapper, Inner, count, small_euclidean):
    C = small_euclidean["C"]
    count = "n_simulateds" if count == "count" else count
    wrapped = Wrapper(random_state=0, **{count: 3}).fit(C)
    ms = MultiStart(Inner(), n_restarts=3, random_state=0).fit(C)
    assert np.array_equal(wrapped.tour_, ms.tour_) and wrapped.cost_ == ms.cost_
    assert np.array_equal(wrapped.costs_, ms.costs_) and wrapped.best_index_ == ms.best_index_
    assert np.array_equal(wrapped.history_, ms.history_) and wrapped.stop_reason_ == ms.stop_reason_
    assert len(wrapped.estimators_) == 3 and all(isinstance(e, Inner) for e in wrapped.estimators_)
    assert wrapped.best_estimator_ is wrapped.estimators_[wrapped.best_index_]
    assert wrapped.n_iter_ == wrapped.best_estimator_.n_iter_ == len(wrapped.history_)


def test_wrapper_inner_knobs_propagate(small_euclidean):
    C = small_euclidean["C"]
    es = EnsembleSimulatedAnnealing(n_simulateds=2, random_state=0, alpha=0.9, moves="two_opt").fit(C)
    assert all(e.alpha == 0.9 and e.moves == "two_opt" and e.verbose == 0 for e in es.estimators_)
    eg = EnsembleGenetic(n_genetics=2, random_state=0, pop_size=20, local_search=("two_opt",)).fit(C)
    assert all(e.pop_size == 20 and e.local_search == ("two_opt",) for e in eg.estimators_)
    assert eg.cost_ == pytest.approx(route_cost(C, eg.route_), rel=1e-9)


@pytest.mark.parametrize("Wrapper", WRAPPERS, ids=lambda w: w.__name__)
def test_wrapper_does_not_warn_at_runtime(Wrapper, small_euclidean):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Wrapper(random_state=0, **{RESTARTS[Wrapper]: 1}).fit(small_euclidean["C"])
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("Wrapper", WRAPPERS, ids=lambda w: w.__name__)
def test_wrapper_multi_trip_reaches_the_optimum(Wrapper, alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]
    est = make(Wrapper).fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(est.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert est.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert not fit_kwargs(Wrapper, {"asymmetric": False})  # no coords needed
