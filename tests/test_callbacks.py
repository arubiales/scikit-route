"""D30, the progress-callback protocol: ``RouteEvent``, ``BaseRouter._emit``, the event trace of every
solver, ``MultiStart`` forwarding and the no-overhead guard.

Check 14 of ``check_router`` (run over the roster in ``tests/test_common.py`` and over ``MultiStart`` in
``tests/test_ensemble.py``) enforces the generic contract; this file holds the unit tests of the plumbing
and the solver-specific facts (which ``extra`` keys, one event per level/kick/generation, MILP edges...).
"""

from __future__ import annotations

import dataclasses
import math
import re
from itertools import pairwise

import numpy as np
import pytest
from conftest import _euclid, fit_kwargs, make

import skroute
from skroute import (
    MILP,
    SOM,
    AntColony,
    EnsembleGenetic,
    EnsembleSimulatedAnnealing,
    Genetic,
    IteratedLocalSearch,
    LocalSearch,
    MultiStart,
    RouteEvent,
    RoutingProblem,
    SimulatedAnnealing,
    TabuSearch,
    TwoOpt,
)
from skroute.base import BaseRouter, RouterTags
from skroute.metrics import split_trips

C4 = np.array([[0, 5, 9, 10], [5, 0, 4, 8], [9, 4, 0, 3], [10, 8, 3, 0]], dtype=float)
NAMES = ["d", "a", "b", "c"]

#: The ``extra`` keys every "iteration" event of a solver carries (documented in each solver's Notes).
ITERATION_EXTRA = {
    "SimulatedAnnealing": {"temperature", "accepted", "n_moves"},
    "TabuSearch": {"tenure"},
    "Genetic": {"generation", "n_evaluations", "mean_cost", "n_duplicates"},
    "AntColony": {"n_ants", "iteration_best", "deposit"},
    "SOM": {"radius", "learning_rate", "n_samples"},
    "IteratedLocalSearch": {"kick", "accepted", "current_cost"},
    "TwoOpt": {"moves_applied", "gain"},
    "OrOpt": {"moves_applied", "gain"},
    "LocalSearch": {"moves_applied", "gain"},
    "MILP": {"edges", "n_components", "lower_bound", "objective", "n_cuts"},
}
#: Solvers that emit no iteration event of their own: construction and exact (MILP apart) and the wrappers.
NO_ITERATIONS = {
    "BruteForce",
    "HeldKarp",
    "NearestNeighbour",
    "Insertion",
    "ClarkeWright",
    "NRBS",
    "EnsembleGenetic",
    "EnsembleSimulatedAnnealing",
}


def _record(est, C, **kw):
    """Fit with a recording callback; returns ``(est, events)``."""
    events: list[RouteEvent] = []
    est.fit(C, callback=events.append, **kw)
    return est, events


def _own(events, est):
    name = type(est).__name__
    return [e for e in events if e.solver == name and "restart" not in e.extra]


def _priced(problem, tour):
    return float(problem.evaluate(problem.to_index_tour(tour)))


# --------------------------------------------------------------------------- a probe solver for _emit
class Probe(BaseRouter):
    """Emits whatever its ``script`` says inside ``_solve`` (a list of ``_emit`` argument tuples/dicts) and
    returns the identity tour; ``script`` is a hyper-parameter so clone/get_params keep working."""

    def __init__(self, script=(), iterative=False, verbose=0):
        self.script = script
        self.iterative = iterative
        self.verbose = verbose

    def _get_tags(self):
        return RouterTags(kind="construction", iterative=self.iterative)

    def _solve(self, problem, rng):
        for args, kw in self.script:
            self._emit(*args, **kw)
        if self.iterative:
            self.history_, self.n_iter_, self.stop_reason_ = (
                [problem.evaluate(np.arange(problem.n))],
                1,
                "max_iter",
            )
        return np.roll(np.arange(problem.n, dtype=np.int64), -problem.depot)


# --------------------------------------------------------------------------- RouteEvent
def test_route_event_is_exported_and_frozen():
    assert skroute.RouteEvent is RouteEvent and "RouteEvent" in skroute.__all__
    problem = RoutingProblem(C4, labels=NAMES)
    tour = np.array(NAMES, dtype=object)
    e = RouteEvent("Probe", "iteration", 3, 22.0, 22.0, tour, tour, problem, {"k": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.cost = 1.0  # type: ignore[misc]
    twin = RouteEvent("Probe", "iteration", 3, 22.0, 22.0, tour, tour, problem, {"k": 1})
    assert (
        e != twin and e == e and len({e, twin}) == 2
    )  # identity semantics: arrays are not comparable/hashable
    assert repr(e) == "RouteEvent(solver='Probe', stage='iteration', iteration=3, best_cost=22)"
    assert re.fullmatch(r"RouteEvent\(solver='\w+', stage='\w+', iteration=\d+, best_cost=.+\)", repr(e))
    assert (
        e.extra == {"k": 1}
        and RouteEvent("P", "start", 0, math.nan, math.nan, None, None, problem).extra == {}
    )


def test_route_and_trips_decode_best_tour_with_the_problem_split_rule():
    C, _ = _euclid(8, seed=8)
    T = C / 10.0
    budget = 1.5 * float((T[0] + T[:, 0]).max())
    labels = [f"n{i}" for i in range(8)]
    est, events = _record(
        TwoOpt(), C, time_matrix=T, max_time_work=budget, extra_cost=3.0, labels=labels, depot="n2"
    )
    end = events[-1]
    assert end.stage == "end" and est.n_trips_ >= 2, "the fixture must need several trips to mean anything"
    assert end.route.tolist() == est.route_.tolist()
    assert [t.tolist() for t in end.trips] == [t.tolist() for t in est.trips_]
    assert [t.tolist() for t in split_trips(end.route, "n2")] == [t.tolist() for t in end.trips]
    assert end.route[0] == end.route[-1] == "n2" and all(t[0] == t[-1] == "n2" for t in end.trips)
    # a plain TSP decodes to one trip; the optimal split rule is honoured too
    p_opt = RoutingProblem(
        C, time_matrix=T, max_time_work=budget, extra_cost=3.0, labels=labels, split="optimal"
    )
    tour_lab = end.best_tour
    e_opt = RouteEvent("X", "end", 1, 0.0, 0.0, tour_lab, tour_lab, p_opt)
    starts = p_opt.trip_starts(p_opt.to_index_tour(tour_lab))
    assert len(e_opt.trips) == len(starts) - 1
    p_plain = RoutingProblem(C, labels=labels, depot="n2")
    plain = RouteEvent("X", "end", 1, 0.0, 0.0, tour_lab, tour_lab, p_plain)
    assert len(plain.trips) == 1 and plain.route.tolist() == [*tour_lab.tolist(), "n2"]


def test_route_and_trips_without_a_best_tour():
    problem = RoutingProblem(C4)
    e = RouteEvent("X", "start", 0, math.nan, math.nan, None, None, problem)
    assert e.route is None and e.trips == [] and math.isnan(e.cost) and math.isnan(e.best_cost)
    assert repr(e).endswith("best_cost=nan)")


# --------------------------------------------------------------------------- BaseRouter._emit
def test_emit_is_a_no_op_without_callback_even_on_garbage():
    class Poison:
        def __array__(self, *a, **k):
            raise AssertionError("_emit touched its arguments without a callback")

        def __iter__(self):
            raise AssertionError("_emit touched its arguments without a callback")

    est = Probe(script=[(("iteration", 1, Poison(), None), {})]).fit(C4)  # cost=None would price the tour
    assert est.cost_ == 22.0 and "_callback" not in vars(est) and "_callback_state" not in vars(est)
    assert est._callback is None and est._stop_requested is False
    # a bare _solve (no fit at all) works too: the class-level defaults stand in
    assert Probe(script=[(("start", 0, Poison(), 1.0), {})])._solve(RoutingProblem(C4), None).tolist() == [
        0,
        1,
        2,
        3,
    ]


def test_emit_converts_index_tours_to_labels_and_fills_the_defaults():
    problem = RoutingProblem(C4, labels=NAMES)
    tour = np.array([0, 2, 1, 3])
    best = np.array([0, 1, 2, 3])
    script = [
        (("start", 0, tour, 25.0), {}),  # best defaults to the current tour and cost
        (("iteration", 1, tour, None), {"note": "priced"}),  # cost=None -> problem.evaluate
        (("iteration", 2, tour, 25.0, best), {}),  # best_cost=None with a best tour -> priced
        (("iteration", 3, None, None, best, 22.0), {}),  # no current tour: cost nan
    ]
    seen = []
    Probe(script=script).fit(problem, callback=seen.append)
    start, e1, e2, e3, end = seen
    assert (
        start.stage == "start" and start.tour.tolist() == ["d", "b", "a", "c"] and start.tour.dtype == object
    )
    assert start.best_tour.tolist() == ["d", "b", "a", "c"] and start.cost == start.best_cost == 25.0
    assert e1.cost == e1.best_cost == pytest.approx(problem.evaluate(tour)) == 31.0  # d-b-a-c: 9+4+8+10
    assert e1.extra == {"note": "priced"}
    assert e2.cost == 25.0 and e2.best_tour.tolist() == NAMES and e2.best_cost == 22.0
    assert e3.tour is None and math.isnan(e3.cost) and e3.best_tour.tolist() == NAMES and e3.best_cost == 22.0
    assert end.stage == "end" and end.iteration == 3 and end.tour.tolist() == NAMES and end.best_cost == 22.0
    assert all(
        isinstance(e.cost, float) and isinstance(e.best_cost, float) and isinstance(e.iteration, int)
        for e in seen
    )
    assert all(e.solver == "Probe" and e.problem is problem for e in seen)


def test_emitted_tours_are_copies_of_the_solver_buffers():
    buffer = np.array([0, 1, 2, 3])
    seen = []
    Probe(script=[(("start", 0, buffer, 22.0), {})]).fit(C4, callback=seen.append)
    buffer[1], buffer[2] = 2, 1
    assert seen[0].tour.tolist() == [0, 1, 2, 3] and seen[0].best_tour.tolist() == [0, 1, 2, 3]


def test_start_is_synthesised_when_a_solver_emits_iterations_only():
    seen = []
    Probe(script=[(("iteration", 1, [0, 1, 2, 3], 22.0), {})]).fit(C4, callback=seen.append)
    assert [e.stage for e in seen] == ["start", "iteration", "end"]
    assert (
        seen[0].tour is None
        and seen[0].best_tour is None
        and math.isnan(seen[0].cost)
        and seen[0].iteration == 0
    )
    seen = []
    Probe().fit(
        C4, callback=seen.append
    )  # a solver that never calls _emit: start and end from the base class
    assert [(e.stage, e.iteration) for e in seen] == [("start", 0), ("end", 0)] and seen[1].best_cost == 22.0


@pytest.mark.parametrize(
    ("answer", "stops"),
    [(True, True), (np.True_, True), (False, False), (None, False), (1, False), ("yes", False), ([1], False)],
    ids=["True", "np.True_", "False", "None", "1", "str", "list"],
)
def test_only_a_true_bool_requests_a_stop(answer, stops):
    seen = {}

    class Peek(Probe):
        def _solve(self, problem, rng):
            tour = super()._solve(problem, rng)
            seen["stop"] = self._stop_requested  # observed inside the fit: the flag does not survive it
            return tour

    est = Peek(script=[(("iteration", 1, [0, 1, 2, 3], 22.0), {})]).fit(C4, callback=lambda e: answer)
    assert seen["stop"] is stops
    assert "_stop_requested" not in vars(est) and est._stop_requested is False


def test_emit_rejects_an_unknown_stage_an_end_or_a_second_start_and_fit_rejects_a_non_callable():
    with pytest.raises(ValueError, match="stage must be one of"):
        Probe(script=[(("middle", 1, [0, 1, 2, 3], 22.0), {})]).fit(C4, callback=lambda e: None)
    # "end" is the base class's, and "start" happens once: both mistakes are reported at the offending call
    with pytest.raises(ValueError, match="solvers never emit 'end': fit does"):
        Probe(script=[(("end", 1, [0, 1, 2, 3], 22.0), {})]).fit(C4, callback=lambda e: None)
    twice = [(("start", 0, None, math.nan), {}), (("start", 0, None, math.nan), {})]
    with pytest.raises(ValueError, match="'start' is emitted once per fit"):
        Probe(script=twice).fit(C4, callback=lambda e: None)
    # ...but only while a callback is set: without one _emit stays a no-op whatever it is handed
    assert Probe(script=twice).fit(C4).cost_ == 22.0
    assert Probe(script=[(("end", 1, [0, 1, 2, 3], 22.0), {})]).fit(C4).cost_ == 22.0
    for bad in (1, "plot", object(), [print]):
        with pytest.raises(TypeError, match="callback must be a callable"):
            Probe().fit(C4, callback=bad)
    problem = RoutingProblem(C4)
    seen = []
    assert (
        Probe().fit(problem, callback=seen.append).problem_ is problem
    )  # a problem plus callback is allowed
    assert len(seen) == 2 and seen[0].problem is problem


def test_callback_is_removed_even_when_solve_raises():
    class Boom(Probe):
        def _solve(self, problem, rng):
            self._emit("start", 0, None, math.nan)
            raise RuntimeError("kernel exploded")

    est = Boom()
    seen = []
    with pytest.raises(RuntimeError, match="kernel exploded"):
        est.fit(C4, callback=seen.append)
    assert (
        [e.stage for e in seen] == ["start"]
        and "_callback" not in vars(est)
        and "_callback_state" not in vars(est)
    )


class _Boom(Exception):
    pass


@pytest.mark.parametrize("stage", ["start", "iteration", "end"])
def test_a_callback_that_raises_leaves_the_estimator_unfitted_whatever_the_stage(stage):
    """An exception at the "end" event (or at the synthetic "start" of a construction solver) used to leave
    a fully fitted estimator behind a fit that raised; the post-condition is now the same at every stage."""

    def raise_at(e):
        if e.stage == stage:
            raise _Boom(stage)

    solvers = [SimulatedAnnealing(random_state=0), TwoOpt()]
    if stage != "iteration":
        solvers.append(Probe())  # emits nothing itself: synthetic start and end from the base class
    for est in solvers:
        with pytest.raises(_Boom, match=stage):
            est.fit(C4, callback=raise_at)
        fitted = [k for k in vars(est) if k.endswith("_") and not k.startswith("_")]
        assert fitted == [] and not hasattr(est, "cost_"), (type(est).__name__, stage, fitted)
        assert "_callback" not in vars(est) and "_callback_state" not in vars(est)
        assert "_stop_requested" not in vars(est)
        assert est.fit(C4).cost_ == 22.0  # a refit on the same instance is a plain fit


# --------------------------------------------------------------------------- every solver's trace
#: Solvers whose search starts from the ``init`` tour (nearest neighbour by default): "start" carries it.
START_WITH_INIT = {
    "SimulatedAnnealing",
    "TabuSearch",
    "Genetic",
    "IteratedLocalSearch",
    "TwoOpt",
    "OrOpt",
    "LocalSearch",
}
#: The ``extra`` keys the "start" event of a solver carries (documented in each solver's Notes); a solver
#: absent here (construction, exact) gets the base class's synthetic start, whose ``extra`` is empty.
START_EXTRA = {
    "SimulatedAnnealing": {"temperature"},
    "TabuSearch": set(),
    "Genetic": {"generation", "n_evaluations"},
    "AntColony": {"n_ants"},
    "SOM": {"radius", "learning_rate", "n_units"},
    "IteratedLocalSearch": set(),
    "TwoOpt": {"moves"},
    "OrOpt": {"moves"},
    "LocalSearch": {"moves"},
    "MILP": set(),
    "EnsembleGenetic": {"n_restarts"},
    "EnsembleSimulatedAnnealing": {"n_restarts"},
}


def _instance9():
    C, xy = _euclid(9, seed=9)  # n = 9: below BruteForce's cap, above the ILS double-bridge threshold
    return {"C": C, "coords": xy, "n": 9, "asymmetric": False}


def test_iteration_extra_keys_and_every_event_prices_its_tours(Solver):
    inst = _instance9()
    est, events = _record(make(Solver), inst["C"], **fit_kwargs(Solver, inst))
    problem = est.problem_
    for e in events:
        for tour, cost in ((e.tour, e.cost), (e.best_tour, e.best_cost)):
            if tour is not None:  # cost is the objective of tour AS THE SOLVER KNOWS IT: it must be right
                assert cost == pytest.approx(_priced(problem, tour), rel=1e-9), (e, tour.tolist())
    own = _own(events, est)
    iters = [e for e in own if e.stage == "iteration"]
    name = Solver.__name__
    if name in NO_ITERATIONS:
        assert iters == [] and [e.stage for e in own] == ["start", "end"]
    else:
        assert iters and all(set(e.extra) == ITERATION_EXTRA[name] for e in iters), name
    if name.startswith("Ensemble"):  # the wrapper forwards its restarts' events under the inner class's name
        assert {e.solver for e in events if "restart" in e.extra} == {name.removeprefix("Ensemble")}


def test_start_event_carries_the_init_tour_when_the_search_starts_from_one(Solver):
    inst = _instance9()
    est, events = _record(make(Solver), inst["C"], **fit_kwargs(Solver, inst))
    start = _own(events, est)[0]
    assert start.stage == "start" and start.iteration == 0
    keys = START_EXTRA.get(Solver.__name__, set())
    assert set(start.extra) == keys, (Solver.__name__, start.extra)
    assert all(f"``{k}" in Solver.__doc__ or f'"{k}"' in Solver.__doc__ for k in keys), (
        "document the start keys"
    )
    if Solver.__name__ in START_WITH_INIT:
        nn = est.problem_.to_label_tour(np.asarray(reference_nn(est.problem_)))
        assert start.tour.tolist() == nn.tolist() and start.best_tour is not None
        assert start.cost == pytest.approx(_priced(est.problem_, nn)) and start.best_cost <= start.cost + 1e-9
    else:
        assert start.tour is None and math.isnan(start.cost)


def reference_nn(problem):
    from skroute.utils import initial_tour

    return initial_tour(problem, "nearest_neighbour", None)


def test_a_stop_requested_at_start_is_honoured_after_the_first_iteration(Solver):
    est = make(Solver)
    if not est._get_tags().iterative:
        pytest.skip("only iterative solvers stop")
    inst = _instance9()
    est.fit(inst["C"], callback=lambda e: e.stage == "start", **fit_kwargs(Solver, inst))
    assert est.stop_reason_ == "callback" and est.n_iter_ == 1 == len(est.history_)
    assert est.history_[-1] == pytest.approx(est.cost_)


def test_fit_leaves_no_callback_trace_even_after_a_stop(Solver):
    inst = _instance9()
    est = make(Solver).fit(inst["C"], **fit_kwargs(Solver, inst))
    assert "_callback" not in vars(est) and "_callback_state" not in vars(est)
    assert "_stop_requested" not in vars(est)
    assert est._callback is None and est._callback_state is None and est._stop_requested is False
    stopped = make(Solver).fit(inst["C"], callback=lambda e: True, **fit_kwargs(Solver, inst))
    assert "_stop_requested" not in vars(stopped) and stopped._stop_requested is False
    if stopped._get_tags().iterative:
        assert stopped.stop_reason_ == "callback"
        # a bare _solve on the stopped instance starts clean: it does not inherit the stop request
        rng = np.random.default_rng(0) if stopped._get_tags().stochastic else None
        stopped._solve(stopped.problem_, rng)
        assert stopped.stop_reason_ != "callback" and stopped.n_iter_ >= 1


def test_sa_emits_one_event_per_temperature_level(small_euclidean):
    C = small_euclidean["C"]
    sa, events = _record(SimulatedAnnealing(random_state=0), C)
    iters = [e for e in events if e.stage == "iteration"]
    assert len(iters) == sa.n_iter_ == len(sa.history_) and [e.iteration for e in iters] == list(
        range(1, sa.n_iter_ + 1)
    )
    assert events[0].extra == {"temperature": sa.t0_}
    temps = [e.extra["temperature"] for e in iters]
    assert temps[0] == sa.t0_ and all(b == pytest.approx(a * sa.alpha) for a, b in pairwise(temps))
    assert all(0 <= e.extra["accepted"] <= e.extra["n_moves"] == 10 * 12 for e in iters)
    assert [e.best_cost for e in iters] == pytest.approx(sa.history_.tolist())
    short = SimulatedAnnealing(random_state=0).fit(C, callback=lambda e: e.iteration >= 7)
    assert short.n_iter_ == 7 and short.stop_reason_ == "callback"
    assert short.history_.tolist() == sa.history_[:7].tolist()  # same seed: the stopped run is a prefix


def test_ils_emits_one_event_per_kick_with_its_positions(small_euclidean):
    ils, events = _record(IteratedLocalSearch(perturbation_strength=2, random_state=0), small_euclidean["C"])
    iters = [e for e in events if e.stage == "iteration"]
    assert len(iters) == ils.n_iter_
    for e in iters:
        assert (
            len(e.extra["kick"]) == 2
        )  # perturbation_strength double bridges, cut positions 1 <= p1 < p2 < p3 <= n-1
        for kick in e.extra["kick"]:
            assert isinstance(kick, tuple) and len(kick) == 3 and all(type(p) is int for p in kick)
            assert 1 <= kick[0] < kick[1] < kick[2] <= 11
        assert type(e.extra["accepted"]) is bool
        assert e.extra["current_cost"] >= e.best_cost - 1e-9 and e.cost >= e.best_cost - 1e-9
        if e.extra["accepted"]:  # acceptance="better": the accepted candidate is the new current tour
            assert e.extra["current_cost"] == pytest.approx(e.cost)
    C6, _ = _euclid(6, seed=6)  # below 8 nodes the kick is a segment reversal (i, j)
    _, events6 = _record(IteratedLocalSearch(random_state=0), C6)
    kicks = [k for e in events6 if e.stage == "iteration" for k in e.extra["kick"]]
    assert kicks and all(len(k) == 2 and 1 <= k[0] < k[1] <= 5 for k in kicks)


def test_genetic_emits_one_event_per_generation(small_euclidean):
    ga, events = _record(
        Genetic(pop_size=20, n_generations=30, patience=None, random_state=0), small_euclidean["C"]
    )
    iters = [e for e in events if e.stage == "iteration"]
    assert len(iters) == ga.n_iter_ == 30
    assert [e.extra["generation"] for e in iters] == [e.iteration for e in iters] == list(range(1, 31))
    n_children = 20 - 2
    assert [e.extra["n_evaluations"] for e in iters] == [20 + g * n_children for g in range(1, 31)]
    assert events[0].extra == {"generation": 0, "n_evaluations": 20} and events[0].tour is not None
    assert all(
        e.extra["mean_cost"] >= e.cost - 1e-9 for e in iters
    )  # the generation's best is below its mean
    assert iters[-1].extra["n_duplicates"] == ga.n_duplicates_


def test_milp_iteration_events_carry_label_edges():
    C, _ = _euclid(9, seed=9)
    labels = list("abcdefghi")
    est, events = _record(MILP(), C, labels=labels)
    iters = [e for e in events if e.stage == "iteration"]
    assert len(iters) == est.n_solves_ and [e.iteration for e in iters] == list(range(1, est.n_solves_ + 1))
    for e in iters:
        edges = e.extra["edges"]
        assert isinstance(edges, list) and len(edges) == 9  # degree 2 on 9 nodes: 9 edges in the support
        assert all(isinstance(p, tuple) and len(p) == 2 and set(p) <= set(labels) for p in edges)
        assert type(e.extra["n_components"]) is int and e.extra["n_components"] >= 1
        assert e.extra["lower_bound"] <= est.cost_ + 1e-9 and type(e.extra["n_cuts"]) is int
        if e.extra["n_components"] > 1:
            assert e.tour is None and math.isnan(e.cost)
    last = iters[-1]
    assert last.extra["n_components"] == 1 and last.tour is not None and last.extra["n_cuts"] == est.n_cuts_
    assert last.cost == pytest.approx(est.cost_) and last.extra["objective"] == pytest.approx(est.cost_)
    Ca, _ = _euclid(7, seed=7, asymmetric=True)  # arcs on an asymmetric matrix: n of them
    _, events_a = _record(MILP(), Ca)
    assert all(len(e.extra["edges"]) == 7 for e in events_a if e.stage == "iteration")
    stopped = MILP().fit(C, callback=lambda e: True)  # a stop request ends the cut loop like a time-out
    assert stopped.n_solves_ == 1 and sorted(stopped.tour_.tolist()) == list(range(9))


def test_descents_report_the_moves_they_applied(small_euclidean):
    C = small_euclidean["C"]
    ls, events = _record(LocalSearch(), C)
    iters = [e for e in events if e.stage == "iteration"]
    assert events[0].extra == {"moves": ["two_opt", "or_opt"]}
    gains = [e.extra["gain"] for e in iters]
    assert all(g <= 0.0 for g in gains) and events[0].cost + sum(gains) == pytest.approx(ls.cost_)
    assert iters[-1].extra == {"moves_applied": [], "gain": 0.0}  # the converged sweep changed nothing
    assert all(set(e.extra["moves_applied"]) <= {"two_opt", "or_opt"} for e in iters)
    assert all((e.extra["gain"] < 0.0) == bool(e.extra["moves_applied"]) for e in iters)
    assert any(e.extra["moves_applied"] for e in iters), "nearest neighbour at n = 12 must be improvable"
    _, events2 = _record(TwoOpt(), C)
    assert all(set(e.extra["moves_applied"]) <= {"two_opt"} for e in events2 if e.stage == "iteration")


def test_som_reports_the_decaying_neighbourhood(small_euclidean):
    som, events = _record(SOM(random_state=0), small_euclidean["C"], coords=small_euclidean["coords"])
    iters = [e for e in events if e.stage == "iteration"]
    assert events[0].tour is None and events[0].extra == {
        "radius": 96 / 10,
        "learning_rate": 0.8,
        "n_units": 96,
    }
    radii = [e.extra["radius"] for e in iters]
    lrs = [e.extra["learning_rate"] for e in iters]
    samples = [e.extra["n_samples"] for e in iters]
    assert all(b < a for a, b in pairwise(radii))
    assert all(b < a for a, b in pairwise(lrs))
    assert all(b > a for a, b in pairwise(samples)) and samples[-1] == som.n_samples_


def test_aco_reports_the_iteration_best_and_the_deposit(small_euclidean):
    _, events = _record(AntColony(random_state=0), small_euclidean["C"])
    iters = [e for e in events if e.stage == "iteration"]
    assert events[0].tour is None and events[0].extra == {"n_ants": 12}
    assert all(e.extra["n_ants"] == 12 and e.extra["iteration_best"] == e.cost for e in iters)
    assert all(e.extra["deposit"] == ("global" if e.iteration % 5 == 0 else "iteration") for e in iters)


def test_tabu_reports_the_tenure(small_euclidean):
    C = small_euclidean["C"]
    tabu, events = _record(TabuSearch(random_state=0), C)
    tenures = [e.extra["tenure"] for e in events if e.stage == "iteration"]
    assert len(tenures) == tabu.n_iter_ and all(4 <= t <= 8 for t in tenures)  # ceil(sqrt(12)) = 4: [4, 8]
    _, fixed = _record(TabuSearch(tenure=3, random_state=0), C)
    assert all(e.extra["tenure"] == 3 for e in fixed if e.stage == "iteration")


# --------------------------------------------------------------------------- MultiStart forwarding
@pytest.mark.parametrize("n_jobs", [None, 1], ids=["n_jobs=None", "n_jobs=1"])
def test_multistart_forwards_sequentially_with_the_restart_index(small_euclidean, n_jobs):
    C = small_euclidean["C"]
    ms, events = _record(MultiStart(SimulatedAnnealing(), n_restarts=3, n_jobs=n_jobs, random_state=0), C)
    own = _own(events, ms)
    assert [e.stage for e in own] == ["start", "end"] and own[0].extra == {"n_restarts": 3}
    assert events[0] is own[0] and events[-1] is own[-1]  # outer start, the restarts in order, outer end
    inner = [e for e in events if "restart" in e.extra]
    assert {e.solver for e in inner} == {"SimulatedAnnealing"}
    assert [e.extra["restart"] for e in inner] == sorted(e.extra["restart"] for e in inner)
    for k in range(3):
        seq = [e for e in inner if e.extra["restart"] == k]
        assert seq[0].stage == "start" and seq[-1].stage == "end"
        assert len(seq) == ms.estimators_[k].n_iter_ + 2 and seq[-1].best_cost == ms.estimators_[k].cost_
        assert set(seq[1].extra) == {"temperature", "accepted", "n_moves", "restart"}
        assert all(e.problem is ms.problem_ for e in seq)


def test_multistart_does_not_forward_in_parallel_but_gives_the_same_result(small_euclidean):
    C = small_euclidean["C"]
    ms2, events = _record(MultiStart(SimulatedAnnealing(), n_restarts=3, n_jobs=2, random_state=0), C)
    assert [(e.solver, e.stage) for e in events] == [("MultiStart", "start"), ("MultiStart", "end")]
    ms1 = MultiStart(SimulatedAnnealing(), n_restarts=3, n_jobs=1, random_state=0).fit(
        C, callback=lambda e: None
    )
    assert (
        np.array_equal(ms1.tour_, ms2.tour_) and ms1.cost_ == ms2.cost_ and ms1.best_index_ == ms2.best_index_
    )
    assert np.array_equal(ms1.costs_, ms2.costs_)
    harmless = MultiStart(SimulatedAnnealing(), n_restarts=3, n_jobs=2, random_state=0).fit(
        C, callback=lambda e: True
    )
    assert harmless.stop_reason_ == "converged" and len(harmless.estimators_) == 3  # nothing to stop


def test_multistart_stop_request_ends_the_running_restart_and_launches_no_more(small_euclidean):
    def stop_in_the_second_restart(e):
        return e.extra.get("restart") == 1 and e.stage == "iteration" and e.iteration >= 3

    ms = MultiStart(SimulatedAnnealing(), n_restarts=5, random_state=0).fit(
        small_euclidean["C"], callback=stop_in_the_second_restart
    )
    assert len(ms.estimators_) == len(ms.costs_) == 2 and ms.stop_reason_ == "callback"
    assert ms.estimators_[0].stop_reason_ == "converged"
    assert ms.estimators_[1].stop_reason_ == "callback" and ms.estimators_[1].n_iter_ == 3
    assert ms.cost_ == min(ms.costs_) and ms.best_estimator_ is ms.estimators_[ms.best_index_]
    assert ms.n_iter_ == ms.best_estimator_.n_iter_ and np.array_equal(
        ms.history_, ms.best_estimator_.history_
    )


def test_multistart_forwards_sequentially_even_inside_a_parallel_config(small_euclidean):
    """``n_jobs=None`` defers to an enclosing ``joblib.parallel_config`` — except with a callback, which must
    never be invoked from a worker thread: the restarts then run one after another in the calling thread."""
    import threading

    from joblib import parallel_config

    threads: set[int] = set()
    events: list[RouteEvent] = []

    def record(e):
        threads.add(threading.get_ident())
        events.append(e)

    with parallel_config(n_jobs=2):
        ms = MultiStart(SimulatedAnnealing(), n_restarts=3, random_state=0).fit(
            small_euclidean["C"], callback=record
        )
    assert threads == {threading.get_ident()} and len(ms.estimators_) == 3
    assert {e.extra["restart"] for e in events if "restart" in e.extra} == {0, 1, 2}
    plain = MultiStart(SimulatedAnnealing(), n_restarts=3, random_state=0).fit(small_euclidean["C"])
    assert np.array_equal(ms.tour_, plain.tour_) and ms.best_index_ == plain.best_index_


def test_end_event_reports_n_iter_for_the_wrappers_too(small_euclidean):
    """The wrappers emit no iteration events of their own, so ``state.last_iteration`` stays 0; their end
    event must still report ``n_iter_`` (the winning restart's), like every other iterative solver."""
    C = small_euclidean["C"]
    for wrapper in (
        MultiStart(SimulatedAnnealing(), n_restarts=2, random_state=0),
        MultiStart(SimulatedAnnealing(), n_restarts=2, n_jobs=2, random_state=0),
        EnsembleSimulatedAnnealing(n_simulateds=2, random_state=0),
        EnsembleGenetic(n_genetics=2, pop_size=10, n_generations=5, patience=None, random_state=0),
    ):
        est, events = _record(wrapper, C)
        end = _own(events, est)[-1]
        assert end.stage == "end" and end.iteration == est.n_iter_ == est.best_estimator_.n_iter_ >= 1
        assert end.iteration >= max(e.iteration for e in events)


@pytest.mark.parametrize(
    ("Wrapper", "kw"),
    [(EnsembleSimulatedAnnealing, {"n_simulateds": 3}), (EnsembleGenetic, {"n_genetics": 3})],
    ids=["EnsembleSimulatedAnnealing", "EnsembleGenetic"],
)
def test_ensemble_wrappers_honour_a_stop_at_their_own_start(small_euclidean, Wrapper, kw):
    """A True answered to the WRAPPER's own start (and to nothing else) used to be lost: the wrapper set its
    flag, handed the raw callback to a fresh MultiStart and ran every restart. It now runs exactly one."""
    name = Wrapper.__name__
    seen: list[RouteEvent] = []

    def stop_at_own_start(e):
        seen.append(e)
        return e.solver == name and e.stage == "start"

    est = Wrapper(random_state=0, **kw).fit(small_euclidean["C"], callback=stop_at_own_start)
    assert len(est.estimators_) == len(est.costs_) == 1 and est.stop_reason_ == "callback"
    assert est.estimators_[0].stop_reason_ != "callback"  # the one restart ran in full
    assert [(e.solver, e.stage) for e in seen if "restart" not in e.extra] == [
        (name, "start"),
        ("MultiStart", "start"),
        ("MultiStart", "end"),
        (name, "end"),
    ]
    assert seen[-1].best_cost == est.cost_ and seen[-1].iteration == est.n_iter_
    # the plain MultiStart under the same rule, for comparison: one restart, 'callback'
    ms = MultiStart(est.estimators_[0].__class__(), n_restarts=3, random_state=0).fit(
        small_euclidean["C"], callback=lambda e: e.solver == "MultiStart" and e.stage == "start"
    )
    assert len(ms.estimators_) == 1 and ms.stop_reason_ == "callback"


def test_ensemble_wrappers_forward_through_multistart(small_euclidean):
    C = small_euclidean["C"]
    es, events = _record(EnsembleSimulatedAnnealing(n_simulateds=2, random_state=0), C)
    outer = [(e.solver, e.stage) for e in events if "restart" not in e.extra]
    assert outer == [
        ("EnsembleSimulatedAnnealing", "start"),
        ("MultiStart", "start"),
        ("MultiStart", "end"),
        ("EnsembleSimulatedAnnealing", "end"),
    ]
    inner = [e for e in events if "restart" in e.extra]
    assert {e.solver for e in inner} == {"SimulatedAnnealing"} and {e.extra["restart"] for e in inner} == {
        0,
        1,
    }
    assert events[-1].best_cost == es.cost_ and events[-1].best_tour.tolist() == es.tour_.tolist()
    stopped = EnsembleSimulatedAnnealing(n_simulateds=3, random_state=0).fit(
        C, callback=lambda e: e.stage == "iteration"
    )
    assert stopped.stop_reason_ == "callback" and stopped.n_iter_ == 1 and len(stopped.estimators_) == 1
