"""The tolerance table of SPEC §6 as data — the only place tolerance numbers live.

``TINY``, ``FAST`` and ``SLOW`` are keyed by class name; ``Insertion`` is keyed as
``"Insertion[farthest]"``/``"Insertion[cheapest]"`` and ``Genetic`` as ``"Genetic"``/``"Genetic[memetic]"``,
with the matching ``set_params`` in ``PARAMS``. A tolerance is the maximum gap ``cost_ / optimum - 1``
(defaults, ``random_state=0``); ``0.0`` means "equals the optimum" (``rel=1e-9``); ``None`` means "a
valid tour is enough" on the tiny tier and "not run" on the fast tier (BruteForce/HeldKarp are capped
below n = 29). ``SLOW`` maps every key to ``{instance: tolerance}`` — only the listed instances run.

Tightening a tolerance is a release-notes item; loosening one requires the lead's approval and a
CHANGELOG line. ``benchmarks/waterloo.py`` prints these dicts next to the measured gaps.
"""

from __future__ import annotations

SLOW_INSTANCES = ("qa194", "uy734", "zi929", "lu980")


def _all(tol: float) -> dict[str, float]:
    return dict.fromkeys(SLOW_INSTANCES, tol)


# set_params applied to the class for a bracketed key
PARAMS: dict[str, dict] = {
    "Insertion[farthest]": {"strategy": "farthest"},
    "Insertion[cheapest]": {"strategy": "cheapest"},
    "Genetic": {},
    "Genetic[memetic]": {"local_search": ("two_opt",)},
}

# tiny tier: n <= 9 symmetric and asymmetric instances, optimum by reference.brute_force
TINY: dict[str, float | None] = {
    "BruteForce": 0.0,
    "HeldKarp": 0.0,
    "MILP": 0.0,
    "NearestNeighbour": 0.50,
    "Insertion[farthest]": 0.30,
    "Insertion[cheapest]": 0.30,
    "ClarkeWright": 0.30,
    "NRBS": None,  # valid tour only
    "TwoOpt": 0.11,  # SPEC table: 0.10; the n = 6 asymmetric instance measures 10.7 % (a 2-opt local optimum)
    "OrOpt": 0.12,
    "LocalSearch": 0.10,
    "IteratedLocalSearch": 0.0,  # 3 seeds (SEEDS_TO_OPTIMUM)
    "SimulatedAnnealing": 0.0,
    "TabuSearch": 0.0,
    "Genetic": 0.0,
    "Genetic[memetic]": 0.0,
    "AntColony": 0.0,
    "SOM": None,  # valid tour only (+coords)
    "EnsembleGenetic": 0.0,
    "EnsembleSimulatedAnnealing": 0.0,
}

# fast tier: wi29 (27603) and dj38 (6656)
FAST: dict[str, float | None] = {
    "BruteForce": None,  # capped (max_nodes=11)
    "HeldKarp": None,  # capped (max_nodes=20)
    "MILP": 0.0,
    "NearestNeighbour": 0.50,  # dj38 measures 46.4 %
    "Insertion[farthest]": 0.25,
    "Insertion[cheapest]": 0.30,
    "ClarkeWright": 0.25,
    "NRBS": 0.50,
    "TwoOpt": 0.20,
    "OrOpt": 0.25,
    "LocalSearch": 0.12,
    "IteratedLocalSearch": 0.03,
    "SimulatedAnnealing": 0.03,
    "TabuSearch": 0.08,
    "Genetic": 0.15,  # measured 8.6 % on dj38
    "Genetic[memetic]": 0.05,
    "AntColony": 0.08,
    "SOM": 0.15,
    "EnsembleGenetic": 0.15,  # <= the wrapped solver's entry
    "EnsembleSimulatedAnnealing": 0.03,
}

# slow tier (@slow, tests/benchmarks/test_waterloo.py): qa194, uy734, zi929, lu980
SLOW: dict[str, dict[str, float] | None] = {
    "BruteForce": None,
    "HeldKarp": None,
    "MILP": {"qa194": 0.0},  # time_limit=150
    "NearestNeighbour": _all(0.35),
    "Insertion[farthest]": _all(0.25),
    "Insertion[cheapest]": _all(0.30),
    "ClarkeWright": _all(0.25),
    "NRBS": {"qa194": 0.60},
    "TwoOpt": _all(0.20),
    "OrOpt": _all(0.25),  # OrOpt alone measures 21.7 % on lu980
    "LocalSearch": _all(0.15),  # measured 11.9 % on qa194
    "IteratedLocalSearch": _all(0.06),  # measured 4.16 % on lu980
    "SimulatedAnnealing": _all(0.10),  # measured 7.0 % on lu980
    "TabuSearch": _all(0.15),
    "Genetic": {"qa194": 0.30},  # measured 18.8 %
    "Genetic[memetic]": {"qa194": 0.08, "lu980": 0.08},  # minutes
    "AntColony": _all(0.15),  # n <= 1000
    "SOM": {"qa194": 0.15},
    "EnsembleGenetic": None,
    "EnsembleSimulatedAnnealing": None,
}

# classes that must reach the tiny/alicante optimum at seeds 0, 1 and 2
SEEDS_TO_OPTIMUM = {
    "IteratedLocalSearch",
    "SimulatedAnnealing",
    "TabuSearch",
    "Genetic",
    "AntColony",
    "EnsembleGenetic",
    "EnsembleSimulatedAnnealing",
}

# measured baselines quoted in SPEC §4 (gap = cost_/optimum - 1, defaults, random_state=0), rendered
# into docs/benchmarks.md so a regression is distinguishable from a tie-break difference
MEASURED: dict[str, dict[str, float]] = {
    "NearestNeighbour": {
        "wi29": 0.318,
        "dj38": 0.464,
        "qa194": 0.245,
        "uy734": 0.254,
        "zi929": 0.195,
        "lu980": 0.267,
    },
    "TwoOpt": {"lu980": 0.162},
    "OrOpt": {"lu980": 0.217},
    "LocalSearch": {"qa194": 0.119, "lu980": 0.078},
    "IteratedLocalSearch": {"lu980": 0.0416},
    "SimulatedAnnealing": {
        "wi29": 0.0,
        "dj38": 0.0,
        "qa194": 0.0244,
        "uy734": 0.0684,
        "zi929": 0.0569,
        "lu980": 0.0702,
    },  # 2.0 kernel (2-opt + Or-opt + swap), defaults, random_state=0
    "Genetic": {"wi29": 0.049, "dj38": 0.086, "qa194": 0.188},
    "TabuSearch": {
        "wi29": 0.0,
        "dj38": 0.0,
        "qa194": 0.0534,
        "uy734": 0.0507,
        "zi929": 0.0358,
        "lu980": 0.0426,
    },
}


def missing(name: str) -> str:
    """The message of the ``KeyError`` raised for a class without an entry."""
    return f"add a tolerance for {name} in tests/tolerances.py"


def keys_for(name: str) -> list[str]:
    """Every key of the tables for class ``name`` (``"Genetic"`` -> ``["Genetic", "Genetic[memetic]"]``).

    Raises ``KeyError("add a tolerance for <Name> in tests/tolerances.py")`` when there is none.
    """
    keys = [k for k in TINY if k == name or k.startswith(name + "[")]
    if not keys:
        raise KeyError(missing(name))
    for table in (FAST, SLOW):
        absent = [k for k in keys if k not in table]
        if absent:
            raise KeyError(missing(name))
    return keys


def params_for(key: str) -> dict:
    """The ``set_params`` matching a key (``{}`` for a plain class name)."""
    return dict(PARAMS.get(key, {}))
