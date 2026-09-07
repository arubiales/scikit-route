"""Benchmark every scikit-route solver on the technician instance under one wall-clock protocol.

The instance is the worked case of `examples/technician_madrid.py`: the 182 Burger King
restaurants of the Comunidad de Madrid plus the office in Leganés, OSRM driving minutes
(asymmetric, captured 2026-09-05), thirty minutes of service per visit, 480-minute days,
``extra_cost=480`` so that a plan with fewer days always beats a plan with less driving, and
``split="optimal"``. One trip of the multi-trip objective is one working day, so the result of a
configuration is the pair ``(days, driving minutes)`` and the single number it is ranked by is
``objective = driving + 480 * (days - 1)``.

Protocol
--------
Every configuration of a round receives the **same wall-clock budget** and runs in its own
process, so a solver that spends its budget is compared with a solver that spends the same
budget, and nothing steals cores from a neighbour. Ensembles (`MultiStart`) run with
``prefer="processes"``: with threads the multi-trip kernels serialise on the interpreter and the
same eight restarts took 138 s instead of 36 s on the machine of the recorded campaign.
Deterministic configurations (the constructions, the descents) ignore the budget and finish in
milliseconds; their row records the budget of the round anyway, so the table stays readable.

Rounds
------
roster
    28 configurations, 120 s each: the whole solver catalogue with sensible settings. One pass
    over everything scikit-route ships, plus the exact `MILP`, which raises on a multi-trip
    instance by design -- that refusal is recorded as a row.
chained
    9 configurations, 240 s each, warm-started from the best tours found so far (``--seeds``):
    four solvers from each of three seeds, two long ensembles, and three runs of the hunt for a
    14-day plan with ``extra_cost=5000`` (days made overwhelming). Seeded configurations produce
    one row per seed.
long
    13 configurations, one per process, 30-60 min each and no iteration cap: the same families
    given real time, from scratch and from the best plan.

Records
-------
One JSON object per configuration is appended to ``--results`` (JSON Lines) with the keys of the
committed record: ``name``, ``family``, ``params``, ``round``, ``phase``, ``budget_s``,
``wall_s``, ``days``, ``driving_min``, ``objective``, ``stops_per_day``, ``day_minutes``,
``n_iter``, ``stop_reason``, ``fit_time_s`` and ``tour_sha256`` (the first 16 hex characters of
the sha256 of ``json.dumps(tour)``). The tour itself is written only with ``--keep-tours``, so a
rerun stays the size of the committed record; ``--keep-tours`` is what produces a ``--seeds``
file for the chained round. A configuration that raises is recorded too, with ``error`` and the
tail of its traceback instead of a result.

Reproducing the recorded rounds
-------------------------------
The wall-clock cost of each command is given: do not start the last one by accident.

    python benchmark.py --list                                    # the configuration keys
    python benchmark.py --round roster   --budget 120  --results out.jsonl   # ~50 min
    python benchmark.py --round chained  --budget 240  --results out.jsonl \
        --seeds results/best_plan.json                                       # ~35 min
    for k in ils_oropt5 ils_default ils_oropt5_best ils_default_best tabu20 \
             genetic_memetic genetic antcolony sa_slow; do                   # 1 h, in parallel
      python benchmark.py --round long --config $k --budget 3600 \
          --results out.jsonl --seeds seeds.jsonl &
    done; wait
    python benchmark.py --round long --config ms_ils_oropt5 --budget 3600 \
        --results out.jsonl --seeds seeds.jsonl                              # 1 h each, 8 processes
    python benchmark.py --round long --config ms_tabu20 --budget 1800 --results out.jsonl

Notes
-----
The searches are stochastic and time-limited: a rerun lands within a few minutes of driving of
the recorded numbers, it does not reproduce them exactly. ``random_state`` is fixed everywhere,
which pins the sequence of random choices but not the number of iterations that fit in a budget.

``stops_per_day`` here is the number of restaurants visited on each day. The rows of the
committed record (``results/results.jsonl``) carry one more per day for every solver row,
because the campaign recorded ``len(trip) - 1`` and ``trips_`` are closed ``[office, ...,
office]`` arrays -- that counts the legs driven, not the stops. ``days``, ``driving_min`` and
``objective``, the numbers the study reports, are unaffected; see README.md.

The instance loader and the path constants below are the same short section in the three scripts
of this directory, so each one runs on its own with nothing to install.

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import skroute
from skroute import (
    MILP,
    NRBS,
    SOM,
    AntColony,
    ClarkeWright,
    Genetic,
    Insertion,
    IteratedLocalSearch,
    LocalSearch,
    MultiStart,
    NearestNeighbour,
    OrOpt,
    RoutingProblem,
    SimulatedAnnealing,
    TabuSearch,
    TwoOpt,
)
from skroute.base import BaseRouter

# --------------------------------------------------------------------------- the instance
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = REPO / "examples" / "data"
RESULTS_DIR = HERE / "results"
STEM = "madrid_burger_king"
SERVICE_MIN = 30.0  # minutes of maintenance per restaurant
DAY_MIN = 480.0  # the eight-hour working day
EXTRA_DAY_MIN = 480.0  # charge per extra day: fewest days first, then least driving
N_RESTAURANTS = 182


class Instance(NamedTuple):
    """The committed case: labels (the office first), ``(lat, lon)`` coordinates and minutes."""

    labels: list[str]
    coords: np.ndarray
    time: np.ndarray


def load_instance(data_dir: Path) -> Instance:
    """Read the three committed CSVs of ``data_dir`` (see ``examples/data/README.md``)."""
    with (data_dir / f"{STEM}.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels = [row["label"] for row in rows]
    coords = np.array([[float(row["lat"]), float(row["lon"])] for row in rows], dtype=float)
    time = np.loadtxt(
        data_dir / f"{STEM}_times_min.csv", delimiter=",", skiprows=1, usecols=range(1, len(labels) + 1)
    )
    if time.shape != (len(labels), len(labels)):
        raise ValueError(f"{STEM}_times_min.csv is {time.shape}, expected {(len(labels),) * 2}")
    return Instance(labels=labels, coords=coords, time=time)


def make_problem(instance: Instance, extra_cost: float = EXTRA_DAY_MIN) -> RoutingProblem:
    """The multi-trip problem of the study: one trip is one working day."""
    return RoutingProblem(
        instance.time,
        labels=instance.labels,
        coords=instance.coords,
        time_matrix=instance.time,
        service_time=SERVICE_MIN,
        max_time_work=DAY_MIN,
        extra_cost=extra_cost,
        split="optimal",
    )


# --------------------------------------------------------------------------- the catalogue
ROUNDS: dict[str, tuple[str, Any, float]] = {
    # --round: (the label written to the record, the campaign's phase number, the default budget)
    "roster": ("roster (2 min)", 1, 120.0),
    "chained": ("chained (4 min)", 2, 240.0),
    "long": ("long run (30-60 min)", "long", 3600.0),
}
SEEDED = "{seed}"  # marker in a name: one run per seed, the seed's name substituted


class Config(NamedTuple):
    """One benchmarked configuration: its key, its round and how it is named in the record."""

    key: str
    round: str
    family: str
    name: str

    @property
    def seeded(self) -> bool:
        """Whether the configuration is warm-started, i.e. runs once per ``--seeds`` entry."""
        return SEEDED in self.name


CATALOGUE: tuple[Config, ...] = tuple(
    Config(key, rnd, family, name)
    for rnd, key, family, name in (
        # ---------------------------------------------------------------- roster: the catalogue
        ("roster", "nearest_neighbour", "construction", "NearestNeighbour"),
        ("roster", "insertion_farthest", "construction", "Insertion(farthest)"),
        ("roster", "insertion_nearest", "construction", "Insertion(nearest)"),
        ("roster", "insertion_cheapest", "construction", "Insertion(cheapest)"),
        ("roster", "clarke_wright", "construction", "ClarkeWright (symmetrised T)"),
        ("roster", "nrbs", "construction", "NRBS"),
        ("roster", "two_opt", "local_search", "TwoOpt"),
        ("roster", "or_opt", "local_search", "OrOpt"),
        ("roster", "local_search", "local_search", "LocalSearch(2opt+oropt)"),
        ("roster", "local_search_insertion", "local_search", "LocalSearch from Insertion"),
        ("roster", "ils_default", "ils", "ILS default (2opt+oropt, k=10)"),
        ("roster", "ils_oropt5", "ils", "ILS or_opt k=5"),
        ("roster", "ils_oropt5_metropolis", "ils", "ILS or_opt k=5 metropolis"),
        ("roster", "ils_oropt5_strength2", "ils", "ILS or_opt k=5 strength=2"),
        ("roster", "ils_oropt5_insertion", "ils", "ILS or_opt k=5 from Insertion"),
        ("roster", "sa", "metaheuristic", "SimulatedAnnealing"),
        ("roster", "sa_random", "metaheuristic", "SimulatedAnnealing init=random"),
        ("roster", "tabu", "metaheuristic", "TabuSearch"),
        ("roster", "tabu20", "metaheuristic", "TabuSearch k=20"),
        ("roster", "genetic", "metaheuristic", "Genetic"),
        ("roster", "genetic_memetic", "metaheuristic", "Genetic memetic (or_opt)"),
        ("roster", "antcolony", "metaheuristic", "AntColony"),
        ("roster", "som", "metaheuristic", "SOM"),
        ("roster", "som_polish", "metaheuristic", "SOM + LocalSearch polish"),
        ("roster", "milp", "exact", "MILP plain TSP (driving only)"),
        ("roster", "ms_ils_oropt5_x4", "ensemble", "MultiStart(ILS or_opt k=5) x4"),
        ("roster", "ms_sa_x4", "ensemble", "MultiStart(SA) x4"),
        ("roster", "ms_tabu_x4", "ensemble", "MultiStart(Tabu) x4"),
        # ---------------------------------------------------------------- chained: warm starts
        ("chained", "chain_ils_oropt5", "ils-chained", f"ILS or_opt k=5 from [{SEEDED}]"),
        ("chained", "chain_ils_default", "ils-chained", f"ILS 2opt+oropt k=10 from [{SEEDED}]"),
        ("chained", "chain_sa", "sa-chained", f"SA from [{SEEDED}]"),
        ("chained", "chain_tabu20", "tabu-chained", f"Tabu k=20 from [{SEEDED}]"),
        ("chained", "ms_ils_oropt5_x8", "ensemble", "MultiStart(ILS or_opt k=5) x8 long"),
        (
            "chained",
            "ms_ils_oropt5_metropolis_x8",
            "ensemble",
            "MultiStart(ILS or_opt k=5 metropolis) x8 long",
        ),
        (
            "chained",
            "hunt14_ms_nn",
            "hunt-14",
            "14-day hunt: MultiStart(ILS or_opt k=5, extra_cost=5000, init=nearest_neighbour) x8",
        ),
        (
            "chained",
            "hunt14_ms_random",
            "hunt-14",
            "14-day hunt: MultiStart(ILS or_opt k=5, extra_cost=5000, init=random) x8",
        ),
        (
            "chained",
            "hunt14_ils_best",
            "hunt-14",
            "14-day hunt: ILS 2opt+oropt from the best plan, extra_cost=5000",
        ),
        # ---------------------------------------------------------------- long: hours of budget
        ("long", "ils_oropt5", "ils", "ILS or_opt k=5 (1 h)"),
        ("long", "ils_default", "ils", "ILS 2opt+oropt k=10 (1 h)"),
        ("long", "ils_oropt5_best", "ils-chained", "ILS or_opt k=5 from the best plan (1 h)"),
        ("long", "ils_default_best", "ils-chained", "ILS 2opt+oropt k=10 from the best plan (1 h)"),
        ("long", "tabu20", "metaheuristic", "TabuSearch k=20 (1 h)"),
        ("long", "genetic_memetic", "metaheuristic", "Genetic memetic or_opt (1 h)"),
        ("long", "genetic", "metaheuristic", "Genetic (1 h)"),
        ("long", "antcolony", "metaheuristic", "AntColony (1 h)"),
        (
            "long",
            "sa_slow",
            "metaheuristic",
            "SimulatedAnnealing slow schedule alpha=0.99995 (1 h)",
        ),
        ("long", "ms_ils_oropt5", "ensemble", "MultiStart(ILS or_opt k=5) x8, 1 h per restart"),
        ("long", "ms_ils_default", "ensemble", "MultiStart(ILS 2opt+oropt k=10) x8, 1 h per restart"),
        ("long", "ms_tabu20", "ensemble", "MultiStart(TabuSearch k=20) x8, 30 min per restart"),
        ("long", "ms_sa_slow", "ensemble", "MultiStart(SA slow schedule) x8, 30 min per restart"),
    )
)
BY_ROUND: dict[str, dict[str, Config]] = {
    name: {cfg.key: cfg for cfg in CATALOGUE if cfg.round == name} for name in ROUNDS
}
# The names of the long round quote the budget of the recorded campaign; the budget of a run is
# in ``budget_s``. Keeping the label fixed is what makes a row comparable with the record.
NO_ITER_CAP = 10**8  # the long round runs on the clock alone


class Recipe(NamedTuple):
    """What ``build`` returns for one run: the estimator, its ``params`` row and its fit."""

    params: dict[str, Any]
    estimator: BaseRouter
    seed_name: str | None = None
    fit_matrix: np.ndarray | None = None  # None: the asymmetric matrix of the instance
    fit_problem: RoutingProblem | None = None  # the problem the result is priced with
    symmetric_fit: bool = False


@dataclass
class Context:
    """Everything a configuration needs to build itself: the data, the budget and the seeds."""

    instance: Instance
    problem: RoutingProblem
    budget: float
    jobs: int | None = None
    seeds: Sequence[tuple[str, np.ndarray]] = ()
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def kw(self) -> dict[str, Any]:
        """The ``fit`` keywords of the study, straight from the problem of the instance."""
        return {
            "labels": self.instance.labels,
            "coords": self.instance.coords,
            "time_matrix": self.instance.time,
            "service_time": SERVICE_MIN,
            "max_time_work": DAY_MIN,
            "extra_cost": EXTRA_DAY_MIN,
            "split": "optimal",
        }

    def restarts(self, n: int) -> int:
        """Workers for an ensemble of ``n`` restarts: one each, unless ``--jobs`` says otherwise."""
        return n if self.jobs is None else self.jobs

    def warm_tour(self, key: str, estimator: BaseRouter) -> np.ndarray:
        """The tour of a deterministic helper fit (`Insertion`, `SOM`), fitted once and cached."""
        if key not in self._cache:
            self._cache[key] = estimator.fit(self.instance.time, **self.kw).tour_
        return self._cache[key]

    def best_seed(self) -> tuple[str, np.ndarray]:
        """The best of ``--seeds``; the seeds arrive already sorted by objective."""
        if not self.seeds:
            raise SystemExit("this configuration needs a warm start: pass --seeds")
        return self.seeds[0]


def build(cfg: Config, ctx: Context) -> list[Recipe]:
    """The estimator(s) of one configuration, with the parameters recorded alongside the result.

    A seeded configuration returns one recipe per entry of ``--seeds``. The parameters of the
    recorded campaign are reproduced literally, including the iteration caps: the roster and the
    chained round leave the default ``n_iter`` of each solver in place (an iterated local search
    therefore stops at ``max_iter`` before its budget runs out on this instance), while the long
    round lifts the caps so that only the clock stops the search.
    """
    key, rnd, budget = cfg.key, cfg.round, ctx.budget
    if rnd == "roster":
        return _build_roster(key, ctx, budget)
    if rnd == "chained":
        return _build_chained(key, ctx, budget)
    return _build_long(key, ctx, budget)


def _build_roster(key: str, ctx: Context, budget: float) -> list[Recipe]:
    """The 28 configurations of the two-minute roster."""
    if key == "nearest_neighbour":
        return [Recipe({}, NearestNeighbour())]
    if key.startswith("insertion_"):
        strategy = key.removeprefix("insertion_")
        return [Recipe({"strategy": strategy}, Insertion(strategy=strategy))]
    if key == "clarke_wright":
        # ClarkeWright needs a symmetric matrix: fitted on (T + T.T) / 2, priced on T.
        symmetric = (ctx.instance.time + ctx.instance.time.T) / 2.0
        return [
            Recipe(
                {"matrix": "(T+T^T)/2"},
                ClarkeWright(),
                fit_matrix=symmetric,
                fit_problem=ctx.problem,
                symmetric_fit=True,
            )
        ]
    if key == "nrbs":
        return [Recipe({}, NRBS())]
    if key == "two_opt":
        return [Recipe({"init": "nearest_neighbour"}, TwoOpt(max_passes=500))]
    if key == "or_opt":
        return [Recipe({"init": "nearest_neighbour"}, OrOpt(max_passes=500))]
    if key == "local_search":
        return [Recipe({"init": "nearest_neighbour"}, LocalSearch(max_passes=500))]
    if key == "local_search_insertion":
        tour = ctx.warm_tour("insertion", Insertion())
        return [Recipe({"init": "Insertion(farthest)"}, LocalSearch(init=tour, max_passes=500))]
    if key == "ils_default":
        return [
            Recipe(
                {"local_search": "2opt+oropt", "n_candidates": 10},
                IteratedLocalSearch(patience=None, time_limit=budget, random_state=0),
            )
        ]
    if key.startswith("ils_oropt5"):
        common = {"local_search": ("or_opt",), "n_candidates": 5, "patience": None}
        if key == "ils_oropt5":
            return [
                Recipe(
                    {"local_search": "or_opt", "n_candidates": 5},
                    IteratedLocalSearch(**common, time_limit=budget, random_state=0),
                )
            ]
        if key == "ils_oropt5_metropolis":
            return [
                Recipe(
                    {"acceptance": "metropolis"},
                    IteratedLocalSearch(**common, acceptance="metropolis", time_limit=budget, random_state=0),
                )
            ]
        if key == "ils_oropt5_strength2":
            return [
                Recipe(
                    {"perturbation_strength": 2},
                    IteratedLocalSearch(**common, perturbation_strength=2, time_limit=budget, random_state=0),
                )
            ]
        tour = ctx.warm_tour("insertion", Insertion())
        return [
            Recipe(
                {"init": "Insertion(farthest)"},
                IteratedLocalSearch(**common, init=tour, time_limit=budget, random_state=0),
            )
        ]
    if key == "sa":
        return [
            Recipe(
                {"moves": "2opt+oropt+swap", "init": "nn"},
                SimulatedAnnealing(time_limit=budget, random_state=0),
            )
        ]
    if key == "sa_random":
        return [
            Recipe(
                {"init": "random"},
                SimulatedAnnealing(time_limit=budget, random_state=0, init="random"),
            )
        ]
    if key in ("tabu", "tabu20"):
        k = 10 if key == "tabu" else 20
        return [
            Recipe(
                {"n_candidates": k},
                TabuSearch(n_iter=10**6, n_candidates=k, patience=None, time_limit=budget, random_state=0),
            )
        ]
    if key == "genetic":
        return [
            Recipe(
                {"memetic": False},
                Genetic(n_generations=10**6, patience=None, time_limit=budget, random_state=0),
            )
        ]
    if key == "genetic_memetic":
        return [
            Recipe(
                {"memetic": "or_opt"},
                Genetic(
                    n_generations=10**6,
                    patience=None,
                    local_search=("or_opt",),
                    time_limit=budget,
                    random_state=0,
                ),
            )
        ]
    if key == "antcolony":
        return [Recipe({}, AntColony(n_iter=10**6, patience=None, time_limit=budget, random_state=0))]
    if key == "som":
        return [Recipe({"coords": "lat/lon"}, SOM(random_state=0))]
    if key == "som_polish":
        tour = ctx.warm_tour("som", SOM(random_state=0))
        return [Recipe({"init": "SOM tour"}, LocalSearch(init=tour, max_passes=500))]
    if key == "milp":
        # Recorded for the refusal: MILP certifies the plain tour only and raises under a budget.
        return [
            Recipe(
                {"note": "plain ATSP, days decoded afterwards"},
                MILP(time_limit=budget, max_nodes=300),
            )
        ]
    if key == "ms_ils_oropt5_x4":
        per = max(10.0, budget / 4)
        return [
            Recipe(
                {"n_restarts": 4, "per_restart_s": per},
                MultiStart(
                    IteratedLocalSearch(
                        local_search=("or_opt",), n_candidates=5, patience=None, time_limit=per
                    ),
                    n_restarts=4,
                    n_jobs=ctx.restarts(4),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    if key == "ms_sa_x4":
        per = max(10.0, budget / 4)
        return [
            Recipe(
                {"n_restarts": 4, "per_restart_s": per},
                MultiStart(
                    SimulatedAnnealing(time_limit=per),
                    n_restarts=4,
                    n_jobs=ctx.restarts(4),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    if key == "ms_tabu_x4":
        per = max(10.0, budget / 4)
        return [
            Recipe(
                {"n_restarts": 4, "per_restart_s": per},
                MultiStart(
                    TabuSearch(n_iter=10**6, patience=None, time_limit=per),
                    n_restarts=4,
                    n_jobs=ctx.restarts(4),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    raise SystemExit(f"unknown roster configuration {key!r}")


def _build_chained(key: str, ctx: Context, budget: float) -> list[Recipe]:
    """The nine warm-started configurations, and the hunt for a fourteenth day."""
    if key.startswith("chain_"):
        return [_chain(key, name, seed, budget) for name, seed in ctx.seeds]
    if key == "ms_ils_oropt5_x8":
        per = max(10.0, budget / 2)
        return [
            Recipe(
                {"n_restarts": 8, "per_restart_s": per},
                MultiStart(
                    IteratedLocalSearch(
                        local_search=("or_opt",), n_candidates=5, patience=None, time_limit=per
                    ),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=7,
                ),
            )
        ]
    if key == "ms_ils_oropt5_metropolis_x8":
        per = max(10.0, budget / 2)
        return [
            Recipe(
                {"n_restarts": 8, "acceptance": "metropolis"},
                MultiStart(
                    IteratedLocalSearch(
                        local_search=("or_opt",),
                        n_candidates=5,
                        acceptance="metropolis",
                        patience=None,
                        time_limit=per,
                    ),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=7,
                ),
            )
        ]
    if key in ("hunt14_ms_nn", "hunt14_ms_random"):
        # Days made overwhelming (extra_cost = 5000) to see whether 14 days exist at all.
        init = "nearest_neighbour" if key == "hunt14_ms_nn" else "random"
        return [
            Recipe(
                {"extra_cost": 5000, "init": init, "per_restart_s": budget},
                MultiStart(
                    IteratedLocalSearch(
                        local_search=("or_opt",),
                        n_candidates=5,
                        patience=None,
                        time_limit=budget,
                        init=init,
                    ),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=11,
                ),
                fit_problem=make_problem(ctx.instance, extra_cost=5000.0),
            )
        ]
    if key == "hunt14_ils_best":
        name, seed = ctx.best_seed()
        return [
            Recipe(
                {"extra_cost": 5000, "init": name},
                IteratedLocalSearch(patience=None, time_limit=budget * 2, random_state=3, init=seed),
                seed_name=name,
                fit_problem=make_problem(ctx.instance, extra_cost=5000.0),
            )
        ]
    raise SystemExit(f"unknown chained configuration {key!r}")


def _chain(key: str, seed_name: str, seed: np.ndarray, budget: float) -> Recipe:
    """One warm-started solver of the chained round, from one seed tour."""
    params = {"init": seed_name}
    if key == "chain_ils_oropt5":
        est: BaseRouter = IteratedLocalSearch(
            local_search=("or_opt",),
            n_candidates=5,
            patience=None,
            time_limit=budget,
            random_state=1,
            init=seed,
        )
    elif key == "chain_ils_default":
        est = IteratedLocalSearch(patience=None, time_limit=budget, random_state=1, init=seed)
    elif key == "chain_sa":
        est = SimulatedAnnealing(time_limit=budget, random_state=1, init=seed)
    elif key == "chain_tabu20":
        est = TabuSearch(
            n_iter=10**6, n_candidates=20, patience=None, time_limit=budget, random_state=1, init=seed
        )
    else:
        raise SystemExit(f"unknown chained solver {key!r}")
    return Recipe(params, est, seed_name=seed_name)


def _build_long(key: str, ctx: Context, budget: float) -> list[Recipe]:
    """The 13 long configurations: no iteration cap, one process each, hours of budget."""
    if key == "ils_oropt5":
        return [
            Recipe(
                {},
                IteratedLocalSearch(
                    n_iter=NO_ITER_CAP,
                    local_search=("or_opt",),
                    n_candidates=5,
                    patience=None,
                    time_limit=budget,
                    random_state=0,
                ),
            )
        ]
    if key == "ils_default":
        return [
            Recipe(
                {},
                IteratedLocalSearch(n_iter=NO_ITER_CAP, patience=None, time_limit=budget, random_state=0),
            )
        ]
    if key in ("ils_oropt5_best", "ils_default_best"):
        name, seed = ctx.best_seed()
        moves = {"local_search": ("or_opt",), "n_candidates": 5} if key == "ils_oropt5_best" else {}
        return [
            Recipe(
                {"init": name},
                IteratedLocalSearch(
                    n_iter=NO_ITER_CAP,
                    patience=None,
                    time_limit=budget,
                    random_state=5,
                    init=seed,
                    **moves,
                ),
                seed_name=name,
            )
        ]
    if key == "tabu20":
        return [
            Recipe(
                {},
                TabuSearch(
                    n_iter=NO_ITER_CAP,
                    n_candidates=20,
                    patience=None,
                    time_limit=budget,
                    random_state=0,
                ),
            )
        ]
    if key in ("genetic", "genetic_memetic"):
        memetic = {"local_search": ("or_opt",)} if key == "genetic_memetic" else {}
        return [
            Recipe(
                {},
                Genetic(
                    n_generations=NO_ITER_CAP,
                    patience=None,
                    time_limit=budget,
                    random_state=0,
                    **memetic,
                ),
            )
        ]
    if key == "antcolony":
        return [Recipe({}, AntColony(n_iter=NO_ITER_CAP, patience=None, time_limit=budget, random_state=0))]
    if key == "sa_slow":
        return [
            Recipe(
                {"alpha": 0.99995, "init": "random"},
                SimulatedAnnealing(
                    alpha=0.99995, patience=None, time_limit=budget, random_state=0, init="random"
                ),
            )
        ]
    if key in ("ms_ils_oropt5", "ms_ils_default"):
        moves = {"local_search": ("or_opt",), "n_candidates": 5} if key == "ms_ils_oropt5" else {}
        return [
            Recipe(
                {"n_restarts": 8, "per_restart_s": budget},
                MultiStart(
                    IteratedLocalSearch(n_iter=NO_ITER_CAP, patience=None, time_limit=budget, **moves),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    if key == "ms_tabu20":
        return [
            Recipe(
                {"n_restarts": 8, "per_restart_s": budget},
                MultiStart(
                    TabuSearch(n_iter=NO_ITER_CAP, n_candidates=20, patience=None, time_limit=budget),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    if key == "ms_sa_slow":
        return [
            Recipe(
                {"n_restarts": 8, "per_restart_s": budget},
                MultiStart(
                    SimulatedAnnealing(alpha=0.99995, patience=None, time_limit=budget, init="random"),
                    n_restarts=8,
                    n_jobs=ctx.restarts(8),
                    prefer="processes",
                    random_state=0,
                ),
            )
        ]
    raise SystemExit(f"unknown long configuration {key!r}")


# --------------------------------------------------------------------------- pricing and records
def price(problem: RoutingProblem, tour: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Days, driving and the shape of the plan, recomputed from a giant tour by the problem itself.

    The tour is cut into days by the problem's own optimal split, so the numbers come from the
    matrix and not from what a solver reported. ``objective`` is always priced at the study's
    ``extra_cost`` of 480 minutes per extra day, even for the ``extra_cost=5000`` hunt, so every
    row of the record is comparable.
    """
    index_tour = problem.to_index_tour(tour)
    starts = problem.trip_starts(index_tour)
    costs = problem.trip_costs(index_tour, starts)
    times = problem.trip_times(index_tour, starts)
    days = len(starts) - 1
    driving = round(float(costs.sum()), 1)
    return {
        "days": days,
        "driving_min": driving,
        # priced from the rounded minutes, as the campaign did: the raw sum sits on a rounding
        # tie often enough (NRBS: 1845.3500000000001) that the two spellings differ by 0.1
        "objective": round(driving + EXTRA_DAY_MIN * (days - 1), 1),
        "stops_per_day": [int(b - a) for a, b in pairwise(starts.tolist())],
        "day_minutes": [round(float(x)) for x in times],
    }


def tour_hash(tour: Sequence[str]) -> str:
    """The record's fingerprint of a plan: 16 hex characters of the sha256 of its JSON."""
    return hashlib.sha256(json.dumps(list(tour)).encode("utf-8")).hexdigest()[:16]


def record_row(
    cfg: Config,
    recipe: Recipe,
    est: BaseRouter,
    problem: RoutingProblem,
    wall: float,
    budget: float,
) -> dict[str, Any]:
    """One result row: the identity of the configuration, the plan it found and how it stopped."""
    tour = [str(x) for x in est.tour_.tolist()]
    label, phase, _ = ROUNDS[cfg.round]
    row: dict[str, Any] = {
        "name": cfg.name.format(seed=recipe.seed_name) if cfg.seeded else cfg.name,
        "family": cfg.family,
        "params": recipe.params,
        "round": label,
        "phase": phase,
        "key": cfg.key,
        "budget_s": budget,
        "wall_s": round(wall, 1),
        **price(problem, tour),
        "n_iter": int(getattr(est, "n_iter_", 0) or 0),
        "stop_reason": getattr(est, "stop_reason_", None),
        "fit_time_s": round(float(getattr(est, "fit_time_", wall)), 1),
        "tour_sha256": tour_hash(tour),
    }
    if recipe.symmetric_fit:
        row["cost_sym"] = round(float(est.cost_), 1)
    for attribute in ("lower_bound_", "gap_", "t0_", "tenure_"):  # whatever the solver exposes
        if hasattr(est, attribute):
            value = getattr(est, attribute)
            row[attribute.rstrip("_")] = float(value) if np.isscalar(value) else str(value)
    row["tour"] = tour
    return row


def error_row(cfg: Config, recipe: Recipe, exc: BaseException, wall: float) -> dict[str, Any]:
    """A configuration that raised: recorded, because a refusal by design is a result too."""
    label, phase, _ = ROUNDS[cfg.round]
    return {
        "name": cfg.name.format(seed=recipe.seed_name) if cfg.seeded else cfg.name,
        "family": cfg.family,
        "params": recipe.params,
        "round": label,
        "phase": phase,
        "key": cfg.key,
        "wall_s": round(wall, 1),
        "error": f"{type(exc).__name__}: {exc}",
        "trace": traceback.format_exc()[-800:],
    }


def run_one(cfg: Config, recipe: Recipe, ctx: Context, budget: float) -> dict[str, Any]:
    """Fit one configuration, price it and return its row -- an exception becomes an error row."""
    matrix = ctx.instance.time if recipe.fit_matrix is None else recipe.fit_matrix
    kw = dict(ctx.kw)
    problem = recipe.fit_problem or ctx.problem
    if recipe.symmetric_fit:
        kw["time_matrix"] = matrix  # the day budget must be measured on the matrix that was fitted
    elif recipe.fit_problem is not None:
        kw["extra_cost"] = recipe.fit_problem.fixed_cost
    started = time.perf_counter()
    try:
        est = recipe.estimator.fit(matrix, **kw)
    except Exception as exc:  # one refusal or crash must not kill a 50-minute round: record it
        return error_row(cfg, recipe, exc, time.perf_counter() - started)
    return record_row(cfg, recipe, est, problem, time.perf_counter() - started, budget)


def append(path: Path, row: dict[str, Any], keep_tours: bool) -> None:
    """Append one row to the JSON Lines record, dropping the tour unless it was asked for."""
    if not keep_tours:
        row = {k: v for k, v in row.items() if k != "tour"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(row: dict[str, Any]) -> str:
    """The one-line console summary of a finished configuration."""
    if "error" in row:
        return f"{row['name'][:60]:60s} ERROR {row['error'][:100]}"
    return (
        f"{row['name'][:60]:60s} days {row['days']:>3} driving {row['driving_min']:>7.1f} min"
        f"  wall {row['wall_s']:7.1f}s  iter {row['n_iter']:>8}  {row['stop_reason'] or ''}"
    )


# --------------------------------------------------------------------------- seeds
def load_seeds(path: Path, keep: int = 3) -> list[tuple[str, np.ndarray]]:
    """Warm starts for the chained and the long round: the best ``keep`` tours of a recorded file.

    Accepts a ``results.jsonl`` written with ``--keep-tours`` (the best rows by objective) or a
    ``best_plan.json`` (its single plan). The committed ``results/results.jsonl`` carries only
    the hash of each tour, so it cannot seed a rerun -- ``results/best_plan.json`` can, and is
    the default.
    """
    rows: list[dict[str, Any]]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        rows = [r for r in rows if r.get("tour") and "objective" in r]
        if not rows:
            raise SystemExit(f"{path} has no row with a tour: rerun the round with --keep-tours")
        rows.sort(key=lambda r: (r["objective"], r["wall_s"]))
    else:
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not plan.get("tour"):
            raise SystemExit(f"{path} carries no tour")
        rows = [{"name": plan.get("found_by", path.stem), "tour": plan["tour"]}]
    return [(str(r["name"]), np.array(r["tour"], dtype=object)) for r in rows[:keep]]


# --------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    """The command line: one round, one budget, one JSON Lines file to append to."""
    parser = argparse.ArgumentParser(
        description="Benchmark the scikit-route solvers on the technician instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The long round takes one configuration at a time: one process per configuration.",
    )
    parser.add_argument("--round", choices=sorted(ROUNDS), help="which round to run")
    parser.add_argument("--budget", type=float, help="wall-clock seconds per configuration")
    parser.add_argument("--config", metavar="KEY", help="one configuration of the round (see --list)")
    parser.add_argument("--list", action="store_true", help="print the configuration keys and exit")
    parser.add_argument(
        "--jobs", type=int, metavar="N", help="workers of an ensemble (default: one per restart)"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"the CSVs (default {DEFAULT_DATA})")
    parser.add_argument("--out", type=Path, default=Path("study_out"), help="directory of the results file")
    parser.add_argument(
        "--results", type=Path, help="the JSON Lines record to append to (default <out>/results.jsonl)"
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=RESULTS_DIR / "best_plan.json",
        help="warm starts for the chained and the long round (a --keep-tours .jsonl, or a best_plan.json)",
    )
    parser.add_argument("--keep-tours", action="store_true", help="also write the tour of every row")
    parser.add_argument("--verbose", action="store_true", help="let the solvers log their progress")
    return parser


def selected(round_name: str, key: str | None) -> list[Config]:
    """The configurations to run: one, or every one of the round."""
    configs = BY_ROUND[round_name]
    if key is None:
        return list(configs.values())
    if key not in configs:
        known = ", ".join(configs)
        raise SystemExit(f"unknown configuration {key!r} for --round {round_name}; known: {known}")
    return [configs[key]]


def print_catalogue() -> None:
    """``--list``: every configuration, round by round, as ``round  key  name``."""
    width = max(len(cfg.key) for cfg in CATALOGUE)
    for name, (label, _, budget) in ROUNDS.items():
        print(f"# {name}: {label}, default budget {budget:.0f} s per configuration")
        for cfg in BY_ROUND[name].values():
            print(f"{name:8s} {cfg.key:{width}s}  {cfg.name}")


def needs_seeds(configs: Sequence[Config]) -> bool:
    """Whether any selected configuration is warm-started from a recorded tour."""
    return any(
        cfg.seeded or cfg.key in ("hunt14_ils_best", "ils_oropt5_best", "ils_default_best") for cfg in configs
    )


def iter_runs(configs: Sequence[Config], ctx: Context) -> Iterator[tuple[Config, Recipe]]:
    """Every (configuration, recipe) pair to fit: a seeded configuration yields one per seed."""
    for cfg in configs:
        for recipe in build(cfg, ctx):
            yield cfg, recipe


def main(argv: Sequence[str] | None = None) -> int:
    """Run one round (or one configuration of it) and append a row per configuration."""
    args = build_parser().parse_args(argv)
    if args.list:
        print_catalogue()
        return 0
    if args.round is None:
        raise SystemExit("--round is required (or --list); see --help")
    if args.round == "long" and args.config is None:
        keys = " ".join(BY_ROUND["long"])
        raise SystemExit(
            "--round long runs one configuration per process, and each one takes 30-60 minutes.\n"
            f"Pick one with --config KEY, or launch the wave yourself:\n"
            f"  for k in {keys}; do python {Path(__file__).name} --round long --config $k &\n  done; wait"
        )
    skroute.set_log_level("INFO" if args.verbose else "WARNING")
    label, _, default_budget = ROUNDS[args.round]
    budget = default_budget if args.budget is None else args.budget
    results = args.results or args.out / "results.jsonl"

    instance = load_instance(args.data)
    configs = selected(args.round, args.config)
    ctx = Context(instance=instance, problem=make_problem(instance), budget=budget, jobs=args.jobs)
    if needs_seeds(configs):
        ctx.seeds = load_seeds(args.seeds)
        print(f"warm starts from {args.seeds}: {[name for name, _ in ctx.seeds]}", flush=True)
    print(
        f"{label}: {len(instance.labels)} nodes, {budget:.0f} s per configuration, "
        f"{len(configs)} configuration(s) -> {results}",
        flush=True,
    )
    for cfg, recipe in iter_runs(configs, ctx):
        row = run_one(cfg, recipe, ctx, budget)
        append(results, row, args.keep_tours)
        print(report(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
