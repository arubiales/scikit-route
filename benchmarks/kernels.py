"""Timings of the compiled core on the bundled Waterloo instances (SPEC §8, WP1 definition of done).

Targets, measured on the design-phase prototype (Apple M-series, ``-O3``):

* ``tour_cost`` at n = 10 639 (fi10639): <= 30 microseconds.
* 2-opt + Or-opt descent from the nearest-neighbour tour with 10-NN candidate lists on
  fi10639: <= 150 ms.

The script also reports the gap of every construction / descent result to the published
optimum, the cost of one multi-trip evaluation under both decoders and the throughput of the
full-evaluation generic descent on the instances small enough for it.

The ``.tsp`` files are read directly here (``NODE_COORD_SECTION``, ``EUC_2D`` with TSPLIB's
``nint(x) = floor(x + 0.5)`` rounding, D15) so that the benchmark does not depend on
``skroute.datasets``.

Usage::

    python benchmarks/kernels.py                       # wi29 dj38 qa194 lu980 fi10639
    python benchmarks/kernels.py --instances wi29 qa194 --repeat 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

try:
    from skroute._core import _routing as core
except ModuleNotFoundError:  # a source checkout without an install (D29): use the tree next to us
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skroute._core import _routing as core

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "skroute" / "datasets" / "_data" / "tsplib"
OPTIMA = {
    "wi29": 27603,
    "dj38": 6656,
    "qa194": 9352,
    "uy734": 79114,
    "zi929": 95345,
    "lu980": 11340,
    "fi10639": 520527,
}
DEFAULT_INSTANCES = ("wi29", "dj38", "qa194", "lu980", "fi10639")
GREEDY = int(core.SplitRule.SPLIT_GREEDY)
OPTIMAL = int(core.SplitRule.SPLIT_OPTIMAL)
TWO_OPT, OR_OPT, SWAP = 1, 2, 4


# --------------------------------------------------------------------------- data
def read_tsp(path: Path) -> np.ndarray:
    """Coordinates of a TSPLIB ``EUC_2D`` file (``NODE_COORD_SECTION``), in file order."""
    coords = []
    in_section = False
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if not line or line == "EOF":
                continue
            if in_section:
                _, x, y = line.split()
                coords.append((float(x), float(y)))
    return np.asarray(coords, dtype=np.float64)


def nint(x: np.ndarray) -> np.ndarray:
    """TSPLIB rounding, ``floor(x + 0.5)`` (D15) - never ``np.rint`` (half-to-even)."""
    return np.floor(x + 0.5)


def euc2d_matrix(xy: np.ndarray, block: int = 2048) -> np.ndarray:
    """``EUC_2D`` distance matrix, built blockwise to bound the peak memory."""
    n = len(xy)
    D = np.empty((n, n), dtype=np.float64)
    for s in range(0, n, block):
        diff = xy[s : s + block, None, :] - xy[None, :, :]
        D[s : s + block] = nint(np.sqrt((diff**2).sum(-1)))
    return D


def neighbour_lists(D: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` int64 candidate lists: the k nearest nodes of each row, self excluded, ascending."""
    n = D.shape[0]
    idx = np.argpartition(D, k + 1, axis=1)[:, : k + 1]  # the k + 1 smallest per row, self among them
    rows = np.arange(n)[:, None]
    # Overwrite self with the last of the k + 1 entries and keep the first k: whether self sat in the
    # first k slots or in the last one, the k survivors are k distinct foreign nodes (no copy of D).
    idx = np.where(idx == rows, idx[:, -1:], idx)[:, :k]
    order = np.argsort(D[rows, idx], axis=1, kind="stable")
    return np.ascontiguousarray(np.take_along_axis(idx, order, axis=1), dtype=np.int64)


# --------------------------------------------------------------------------- timing
def best_of(fn: Callable[..., Any], *args: Any, repeat: int = 5) -> tuple[float, Any]:
    """Minimum wall time of ``repeat`` calls and the last result."""
    best = math.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def gap(cost: float, name: str) -> str:
    opt = OPTIMA.get(name)
    return f"gap {100.0 * (cost / opt - 1.0):6.2f} %" if opt else "gap    n/a"


def descend(D: np.ndarray, tour: np.ndarray, cand: np.ndarray, rounds: int = 50) -> tuple[float, int]:
    """Alternate 2-opt and Or-opt (L <= 3, reversals allowed) with persistent buffers until neither
    improves; returns (total gain, rounds used). This is the LocalSearch protocol of §4.3."""
    n = len(tour)
    pos = np.empty(n, dtype=np.int64)
    core.rebuild_pos(tour, pos)
    dlb = np.zeros(n, dtype=np.uint8)
    total = 0.0
    used = 0
    for r in range(1, rounds + 1):
        used = r
        dlb[:] = 0
        g1 = core.two_opt_descent(D, tour, pos, cand, dlb, True, 1000)
        dlb[:] = 0
        g2 = core.or_opt_descent(D, tour, pos, cand, dlb, 3, True, 1000)
        total += g1 + g2
        if g1 + g2 == 0.0:
            break
    return total, used


def bench_instance(name: str, repeat: int, rng: np.random.Generator) -> None:
    xy = read_tsp(DATA / f"{name}.tsp")
    n = len(xy)
    t0 = time.perf_counter()
    D = euc2d_matrix(xy)
    t_matrix = time.perf_counter() - t0
    print(
        f"\n=== {name}: n = {n}, matrix {D.nbytes / 1e6:.0f} MB in {t_matrix * 1e3:.0f} ms, "
        f"optimum {OPTIMA.get(name, '?')}"
    )
    tour = np.concatenate(([0], rng.permutation(np.arange(1, n)))).astype(np.int64)

    # ---- evaluation
    t_np, c_np = best_of(lambda: float(D[tour[:-1], tour[1:]].sum() + D[tour[-1], tour[0]]), repeat=repeat)
    t_cy, c_cy = best_of(core.tour_cost_py, D, tour, repeat=10 * repeat)
    assert abs(c_np - c_cy) < 1e-6 * max(1.0, abs(c_np))
    print(
        f"tour_cost          cython {t_cy * 1e6:9.2f} us | numpy fancy-index {t_np * 1e6:9.1f} us"
        + ("   <-- target <= 30 us" if name == "fi10639" else "")
    )
    T = D / 60.0  # pretend minutes -> hours
    budget = 3.0 * float((T[0, 1:] + T[1:, 0]).max())
    t_g, c_g = best_of(core.problem_cost_py, D, T, tour, budget, 100.0, GREEDY, repeat=10 * repeat)
    t_o, c_o = best_of(core.problem_cost_py, D, T, tour, budget, 100.0, OPTIMAL, repeat=repeat)
    out = np.empty(n + 1, dtype=np.int64)
    k_g = core.trip_starts(T, tour, budget, GREEDY, D, 100.0, out)
    k_o = core.trip_starts(T, tour, budget, OPTIMAL, D, 100.0, out)
    print(
        f"multi-trip cost    greedy {t_g * 1e6:9.2f} us ({k_g} trips) | optimal {t_o * 1e6:9.2f} us "
        f"({k_o} trips, {100.0 * (1.0 - c_o / c_g):.2f} % cheaper)"
    )

    # ---- construction + candidate lists
    nn = np.empty(n, dtype=np.int64)
    t_nn, _ = best_of(core.nearest_neighbour_tour, D, 0, nn, repeat=max(1, repeat // 2))
    c_nn = core.tour_cost_py(D, nn)
    K = 10
    t_cand, cand = best_of(neighbour_lists, D, K, repeat=1)
    print(f"nearest neighbour  {t_nn * 1e3:9.1f} ms  {gap(c_nn, name)} | {K}-NN lists {t_cand * 1e3:9.1f} ms")

    # ---- 2-opt + Or-opt with candidate lists and don't-look bits, from the NN tour
    t_ls, (total_gain, rounds) = best_of(lambda: descend(D, nn.copy(), cand), repeat=1)
    ls = nn.copy()
    total_gain, rounds = descend(D, ls, cand)
    c_ls = core.tour_cost_py(D, ls)
    assert sorted(ls.tolist()) == list(range(n)) and ls[0] == 0
    assert abs(c_nn + total_gain - c_ls) < 1e-6 * max(1.0, c_ls), (c_nn, total_gain, c_ls)
    print(
        f"2-opt + Or-opt     {t_ls * 1e3:9.1f} ms  {gap(c_ls, name)} ({rounds} rounds, {K}-NN, DLB)"
        + ("   <-- target <= 150 ms" if name == "fi10639" else "")
    )

    # ---- the full-evaluation generic path (multi-trip / ATSP), only where it is meant to run
    if n <= 1000:
        pos = np.empty(n, dtype=np.int64)
        gen = nn.copy()
        core.rebuild_pos(gen, pos)
        scratch = np.empty(n, dtype=np.int64)
        dp = np.empty(n, dtype=np.float64)
        pred = np.empty(n, dtype=np.int64)
        t0 = time.perf_counter()
        g = core.local_search_generic(
            D, T, gen, pos, cand, budget, 100.0, GREEDY, TWO_OPT | OR_OPT | SWAP, 3, 1000, scratch, dp, pred
        )
        t_gen = time.perf_counter() - t0
        c0 = core.problem_cost_py(D, T, nn, budget, 100.0, GREEDY)
        c1 = core.problem_cost_py(D, T, gen, budget, 100.0, GREEDY)
        assert abs(c0 + g - c1) < 1e-6 * max(1.0, c1)
        print(f"generic descent    {t_gen * 1e3:9.1f} ms  multi-trip greedy, from NN: {c0:.0f} -> {c1:.0f}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=list(DEFAULT_INSTANCES),
        help="Waterloo instance names (files under skroute/datasets/_data/tsplib)",
    )
    parser.add_argument("--repeat", type=int, default=5, help="repetitions per timing (best of)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random starting tour")
    args = parser.parse_args(None if argv is None else list(argv))
    rng = np.random.default_rng(args.seed)
    print(f"numpy {np.__version__}, core {core.__file__}")
    for name in args.instances:
        bench_instance(name, args.repeat, rng)


if __name__ == "__main__":
    main()
