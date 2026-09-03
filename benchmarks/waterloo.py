"""Gap table of every solver on the bundled Waterloo instances -> ``docs/benchmarks.md`` (SPEC §7, WP8).

Fits every class of :func:`skroute.all_solvers` (one row per tolerance key: ``Insertion[farthest]``,
``Genetic[memetic]``...) plus ``MultiStart(SimulatedAnnealing(), n_restarts=4)`` with its defaults and
``random_state=--seed`` on the chosen instances, records the gap to the published optimum and the wall
time, and writes a markdown table with a provenance header (date, commit, CPU, OS, Python/numpy versions,
the seed) and, next to every measured gap, the tolerance of ``tests/tolerances.py`` (``FAST`` for wi29 and
dj38, ``SLOW`` for the larger instances) and the baseline quoted in SPEC §4 (``tolerances.MEASURED``).

The only parameter that differs from the defaults is ``MILP(time_limit=150)`` on the slow instances, as
in the tolerance table (the default 60 s does not always prove qa194); the header says so. Solvers capped
by ``max_nodes`` (BruteForce, HeldKarp, MILP above 300) show ``capped``.

Usage::

    python benchmarks/waterloo.py                      # wi29 dj38 qa194 lu980 -> docs/benchmarks.md
    python benchmarks/waterloo.py --instances wi29 dj38 --solvers SimulatedAnnealing TabuSearch --out -
    python benchmarks/waterloo.py --seed 1 --out /tmp/bench.md

Works from a source checkout without installing the package: when ``skroute`` is not importable the
checkout root is put on ``sys.path`` (D29), like ``benchmarks/kernels.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    import skroute
except ModuleNotFoundError:  # a source checkout without an install (D29): use the tree next to us
    sys.path.insert(0, str(ROOT))
    import skroute

if str(ROOT / "tests") not in sys.path:  # tests/ is not a package (D16): import tolerances as a module
    sys.path.insert(0, str(ROOT / "tests"))
import tolerances  # noqa: E402

from skroute.datasets import load_tsp  # noqa: E402
from skroute.ensemble import MultiStart  # noqa: E402
from skroute.metaheuristics import SimulatedAnnealing  # noqa: E402

OPTIMA = {"wi29": 27603, "dj38": 6656, "qa194": 9352, "uy734": 79114, "zi929": 95345, "lu980": 11340}
DEFAULT_INSTANCES = ("wi29", "dj38", "qa194", "lu980")
FAST_INSTANCES = ("wi29", "dj38")
MULTISTART = "MultiStart[SimulatedAnnealing, 4]"
SLOW_OVERRIDES: dict[str, dict[str, Any]] = {"MILP": {"time_limit": 150.0}}


# --------------------------------------------------------------------------- one measurement
def tolerance_for(key: str, instance: str) -> float | None:
    """The tolerance of ``tests/tolerances.py`` for ``key`` on ``instance`` (``None`` = not tested there)."""
    if key == MULTISTART:
        key = "SimulatedAnnealing"
    if instance in FAST_INSTANCES:
        return tolerances.FAST.get(key)
    slow = tolerances.SLOW.get(key)
    return None if slow is None else slow.get(instance)


def baseline_for(key: str, instance: str) -> float | None:
    return tolerances.MEASURED.get(key, {}).get(instance)


def build(key: str, seed: int, instance: str) -> Any:
    """An unfitted estimator for a tolerance key, ``random_state=seed`` when accepted, defaults otherwise."""
    if key == MULTISTART:
        return MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=seed)
    name = key.split("[")[0]
    cls = getattr(skroute, name)
    params = dict(tolerances.params_for(key))
    if "random_state" in cls._get_param_names():
        params["random_state"] = seed
    if instance not in FAST_INSTANCES:
        params.update(SLOW_OVERRIDES.get(name, {}))
    return cls(**params)


def measure(key: str, bunch: Any, C: np.ndarray, seed: int) -> dict[str, Any]:
    """Fit ``key`` on one instance; ``{"cost", "gap", "time"}`` or ``{"skipped": reason}``."""
    est = build(key, seed, bunch.name)
    tags = est._get_tags()
    n = C.shape[0]
    if tags.max_nodes is not None and n > tags.max_nodes:
        return {"n": n, "skipped": "capped"}
    kw: dict[str, Any] = {"labels": bunch.labels}
    if tags.requires_coords:
        kw["coords"] = bunch.coords
    t0 = time.perf_counter()
    est.fit(C, **kw)
    elapsed = time.perf_counter() - t0
    optimum = OPTIMA[bunch.name]
    return {"n": n, "cost": float(est.cost_), "gap": est.cost_ / optimum - 1.0, "time": elapsed}


def keys(solvers: Sequence[str] | None) -> list[str]:
    """Tolerance keys of the selected solvers (every ``all_solvers()`` class plus MultiStart by default)."""
    names = [cls.__name__ for cls in skroute.all_solvers()]
    out: list[str] = []
    for name in names:
        if solvers is None or name in solvers:
            out.extend(tolerances.keys_for(name))
    if solvers is None or "MultiStart" in solvers:
        out.append(MULTISTART)
    unknown = [] if solvers is None else [s for s in solvers if s not in names and s != "MultiStart"]
    if unknown:
        raise SystemExit(f"unknown solver(s) {unknown}; choose among {[*names, 'MultiStart']}")
    return out


# --------------------------------------------------------------------------- provenance
def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_name() -> str:
    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def provenance(seed: int, instances: Sequence[str]) -> list[str]:
    import scipy

    commit = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    return [
        f"- **Date:** {dt.date.today().isoformat()}",
        f"- **Commit:** `{commit}` (branch `{branch}`), scikit-route {skroute.__version__}",
        f"- **CPU:** {cpu_name()}",
        f"- **OS:** {platform.platform()}",
        f"- **Python:** {platform.python_version()} · numpy {np.__version__} · scipy {scipy.__version__}",
        f"- **Seed:** `random_state={seed}` for every stochastic solver; one fit per cell, "
        "wall time in seconds",
        "- **Parameters:** the defaults of every class, except `MILP(time_limit=150)` on the slow instances "
        "(as in the tolerance table); bracketed keys apply the `set_params` of `tests/tolerances.py` "
        '(`Insertion[farthest]` = `strategy="farthest"`, `Genetic[memetic]` = `local_search=("two_opt",)`)',
        f"- **Instances:** {', '.join(f'{i} (optimum {OPTIMA[i]})' for i in instances)} — "
        "TSPLIB `EUC_2D` with `nint` rounding (D15)",
        "- **Tolerance:** the maximum gap asserted by the test-suite (`tests/tolerances.py`: `FAST` on "
        "wi29/dj38, `SLOW` elsewhere; `—` = not asserted on that instance). **Baseline:** the gap "
        "measured while the specification was written (`tolerances.MEASURED`, SPEC §4), so a regression "
        "is distinguishable from a tie-break difference.",
    ]


# --------------------------------------------------------------------------- the table
def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.2f} %"


def table_rows(results: dict[str, dict[str, dict[str, Any]]], instances: Sequence[str]) -> list[str]:
    rows = []
    for key in results:
        for inst in instances:
            r = results[key].get(inst)
            if r is None:
                continue
            n = r["n"]
            tol, base = tolerance_for(key, inst), baseline_for(key, inst)
            if "skipped" in r:
                cells = [
                    f"`{key}`",
                    inst,
                    str(n),
                    str(OPTIMA[inst]),
                    r["skipped"],
                    "—",
                    fmt_pct(tol),
                    fmt_pct(base),
                    "—",
                ]
            else:
                cells = [
                    f"`{key}`",
                    inst,
                    str(n),
                    str(OPTIMA[inst]),
                    f"{r['cost']:.0f}" if float(r["cost"]).is_integer() else f"{r['cost']:.2f}",
                    fmt_pct(r["gap"]),
                    fmt_pct(tol),
                    fmt_pct(base),
                    f"{r['time']:.2f}",
                ]
            rows.append("| " + " | ".join(cells) + " |")
    return rows


HEADER = (
    "| Solver | Instance | n | Optimum | Cost | Gap | Tolerance | Baseline | Time (s) |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---:|"
)


def render(results: dict[str, dict[str, dict[str, Any]]], instances: Sequence[str], seed: int) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Gap of every solver to the published optimum of the bundled Waterloo instances, produced by",
        "`python benchmarks/waterloo.py` on the release candidate and committed here (SPEC §7). The",
        "test-suite asserts the **Tolerance** column on every run (`tests/test_common.py` for the fast",
        "tier, `tests/benchmarks/test_waterloo.py` for the slow one); this page records the gaps actually",
        "measured, with their provenance. Gaps are `cost_ / optimum - 1` with `random_state=0`; a",
        "different machine may move a stochastic solver's gap by a tie-break (libm `exp` differs by",
        "ulps and flips Metropolis decisions), never below the optimum.",
        "",
        *provenance(seed, instances),
        "",
        HEADER,
        *table_rows(results, instances),
        "",
        "Regenerate with `python benchmarks/waterloo.py` (a few minutes on a laptop: the `MILP` proof",
        "of qa194 dominates). Tightening a tolerance is a release-notes item; loosening one requires",
        "the lead's approval and a CHANGELOG line.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def run(
    instances: Sequence[str], solver_keys: Iterable[str], seed: int, log=print
) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in solver_keys}
    for inst in instances:
        bunch = load_tsp(inst)
        C = bunch.distance_matrix()
        log(f"=== {inst}: n = {C.shape[0]}, optimum {OPTIMA[inst]}")
        for key in results:
            r = measure(key, bunch, C, seed)
            results[key][inst] = r
            if "skipped" in r:
                log(f"  {key:38s} {r['skipped']}")
            else:
                tol = tolerance_for(key, inst)
                flag = "" if tol is None or r["gap"] <= tol + 1e-12 else "   <-- ABOVE TOLERANCE"
                gap, secs = 100 * r["gap"], r["time"]
                log(f"  {key:38s} gap {gap:7.2f} %  tol {fmt_pct(tol):>8s}  {secs:8.2f} s{flag}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--instances", nargs="+", default=list(DEFAULT_INSTANCES), choices=sorted(OPTIMA), metavar="NAME"
    )
    parser.add_argument(
        "--solvers", nargs="+", default=None, metavar="CLASS", help="class names (default: all + MultiStart)"
    )
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "benchmarks.md"), help="markdown file, or - for stdout"
    )
    parser.add_argument("--seed", type=int, default=0, help="random_state of every stochastic solver")
    args = parser.parse_args(argv)
    instances = [i for i in OPTIMA if i in args.instances]  # canonical order
    results = run(instances, keys(args.solvers), args.seed, log=lambda s: print(s, file=sys.stderr))
    text = render(results, instances, args.seed)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
