"""Watch a scikit-route solver work on a bundled instance.

    python examples/live_demo.py                                   # IteratedLocalSearch on dj38
    python examples/live_demo.py --instance wi29 --solver SimulatedAnnealing --every 10
    python examples/live_demo.py --instance barcelona --solver TabuSearch --backend plotly --map
    python examples/live_demo.py --solver SimulatedAnnealing --init random --every 30 --gif run.gif   # record

The script loads the instance, builds its cost matrix and fits the solver with a
``skroute.viz.LivePlot`` callback (or a ``Recorder`` when ``--gif`` is given). Needs
``pip install "scikit-route[viz]"`` (``[viz-map]`` for ``--backend plotly`` / ``--map``).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import skroute
from skroute.datasets import load_barcelona, load_tsp
from skroute.viz import LivePlot, Recorder
from skroute.viz._live import backend_is_interactive  # the probe LivePlot itself uses for plt.pause

log = logging.getLogger("skroute")


def load(name: str) -> tuple:
    """``(cost matrix, labels, xy coordinates, (lat, lon) coordinates or None)`` of a bundled instance."""
    if name == "barcelona":
        b = load_barcelona()
        return b.cost, b.labels, b.coords[:, ::-1], b.coords  # x = longitude, y = latitude
    b = load_tsp(name)
    return b.distance_matrix(), b.labels, b.coords, None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a scikit-route solver work.")
    parser.add_argument("--instance", choices=["wi29", "dj38", "barcelona"], default="dj38")
    parser.add_argument(
        "--solver", default="IteratedLocalSearch", help="class name of a solver (see skroute.all_solvers())"
    )
    parser.add_argument("--every", type=int, default=1, help="redraw every n-th iteration")
    parser.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib")
    parser.add_argument("--map", action="store_true", help="OpenStreetMap tiles (Plotly; barcelona only)")
    parser.add_argument(
        "--gif", metavar="PATH", help="record the run and save it as a GIF instead of watching"
    )
    parser.add_argument("--seed", type=int, default=0, help="random_state of a stochastic solver")
    parser.add_argument(
        "--init",
        choices=["nearest_neighbour", "random"],
        default=None,
        help="starting tour of solvers with init=",
    )
    args = parser.parse_args(argv)

    if args.map and args.instance != "barcelona":
        parser.error("--map needs --instance barcelona (the only bundled instance with latitude/longitude)")
    if args.map and args.backend != "plotly":
        args.backend = "plotly"

    skroute.set_log_level("INFO")
    cost, labels, xy, latlon = load(args.instance)
    solvers = {cls.__name__: cls for cls in skroute.all_solvers()}  # the classes that take no arguments
    if args.solver not in solvers:
        parser.error(f"unknown solver {args.solver!r}; choose one of {sorted(solvers)}")
    est = solvers[args.solver]()
    if "random_state" in est.get_params():
        est.set_params(random_state=args.seed)
    if args.init is not None and "init" in est.get_params():
        est.set_params(init=args.init)

    if args.gif:
        rec = Recorder(every=args.every)
        est.fit(cost, labels=labels, coords=xy, callback=rec)
        rec.animate(xy, interval=80, figsize=(6, 6)).save(args.gif, writer="pillow", dpi=80)
        log.info(
            "%s on %s: cost %.2f, %d frames written to %s",
            args.solver,
            args.instance,
            est.cost_,
            rec.n_frames,
            args.gif,
        )
        return 0

    live = LivePlot(latlon if args.map else xy, backend=args.backend, map=args.map, every=args.every)
    est.fit(cost, labels=labels, coords=xy, callback=live)
    log.info(
        "%s on %s: cost %.2f in %.2f s (%s), %d events, %d redraws",
        args.solver,
        args.instance,
        est.cost_,
        est.fit_time_,
        getattr(est, "stop_reason_", "done"),
        live.n_events,
        live.n_redraws,
    )
    if args.backend == "matplotlib" and backend_is_interactive():  # headless (Agg): nothing to keep open
        import matplotlib.pyplot as plt

        plt.show()  # LivePlot never blocks; keep the final picture on screen until it is closed
    return 0


if __name__ == "__main__":
    sys.exit(main())
