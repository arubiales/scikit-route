"""Watch a scikit-route solver work on a bundled instance — live, recorded, or replayed at time-lapse speed.

    python examples/live_demo.py                                   # IteratedLocalSearch on dj38, live
    python examples/live_demo.py --instance wi29 --solver SimulatedAnnealing --every 10 --show current
    python examples/live_demo.py --instance barcelona --solver TabuSearch --backend plotly --map
    python examples/live_demo.py --solver SimulatedAnnealing --init random --every 30 --record run.gif
    python examples/live_demo.py --solver Insertion --record insertion.gif --fps 8    # one frame per node
    python examples/live_demo.py --solver SOM --set n_iter=20000 --record som.gif --fps 12
    python examples/live_demo.py --solver SimulatedAnnealing --init random --every 30 --speed 10  # replay

The script loads the instance, builds its cost matrix and fits the solver with a
``skroute.viz.LivePlot`` callback — or a ``Recorder`` when ``--record`` or ``--speed`` is given:
``--record PATH`` saves the run as a GIF (pillow) or an MP4 (ffmpeg), at a fixed ``--fps`` or, with
``--speed``, as a time-lapse of the recorded clock; ``--speed`` alone records silently and then
replays the run on screen ``--speed`` times faster than it happened. Needs
``pip install "scikit-route[viz]"`` (``[viz-map]`` for ``--backend plotly`` / ``--map``).
"""

from __future__ import annotations

import argparse
import ast
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a scikit-route solver work, record it or replay it.")
    parser.add_argument("--instance", choices=["wi29", "dj38", "barcelona"], default="dj38")
    parser.add_argument(
        "--solver", default="IteratedLocalSearch", help="class name of a solver (see skroute.all_solvers())"
    )
    parser.add_argument("--every", type=int, default=1, help="redraw (or keep) every n-th iteration")
    parser.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib")
    parser.add_argument("--map", action="store_true", help="OpenStreetMap tiles (Plotly; barcelona only)")
    parser.add_argument(
        "--show",
        choices=["both", "best", "current"],
        default="both",
        help="which tours to draw during the run",
    )
    parser.add_argument("--trail", type=int, default=0, help="fading copies of the last K current tours")
    parser.add_argument(
        "--record",
        "--gif",
        dest="record",
        metavar="PATH",
        help="record the run and save it (.gif through pillow, .mp4 and the like through ffmpeg)",
    )
    parser.add_argument("--fps", type=int, default=20, help="frame rate of the file written by --record")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="time-lapse factor: with --record the file follows the recorded clock divided by SPEED; "
        "alone, the run is recorded silently and replayed on screen SPEED times faster",
    )
    parser.add_argument("--seed", type=int, default=0, help="random_state of a stochastic solver")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help="a solver parameter as a Python literal (repeatable): --set n_iter=20000 --set strategy=nearest",
    )
    parser.add_argument(
        "--init",
        choices=["nearest_neighbour", "random"],
        default=None,
        help="starting tour of solvers with init=",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.map and args.instance != "barcelona":
        parser.error("--map needs --instance barcelona (the only bundled instance with latitude/longitude)")
    if args.map and args.backend != "plotly":
        args.backend = "plotly"
    if args.speed is not None and args.speed <= 0:
        parser.error("--speed must be > 0")

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
    for item in args.set:
        name, sep, text = item.partition("=")
        if not sep or name not in est.get_params():
            parser.error(f"--set expects PARAM=VALUE with a parameter of {args.solver}; got {item!r}")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text  # a bare word: a string parameter such as strategy=cheapest
        est.set_params(**{name: value})

    if args.record or args.speed is not None:
        rec = Recorder(every=args.every)
        est.fit(cost, labels=labels, coords=xy, callback=rec)
        log.info(
            "%s on %s: cost %.2f in %.2f s, %d events kept (%d frames)",
            args.solver,
            args.instance,
            est.cost_,
            est.fit_time_,
            len(rec),
            rec.n_frames,
        )
        if args.record:
            out = rec.save(args.record, xy, fps=args.fps, show=args.show, speed=args.speed, trail=args.trail)
            log.info("written %s (%.0f kB)", out, out.stat().st_size / 1024)
            return 0
        live = rec.replay(
            latlon if args.map else xy,
            speed=args.speed,
            backend=args.backend,
            show=args.show,
            trail=args.trail,
            map=args.map,
        )
        log.info("replayed %d events at %gx", live.n_events, args.speed)
    else:
        live = LivePlot(
            latlon if args.map else xy,
            backend=args.backend,
            map=args.map,
            every=args.every,
            show=args.show,
            trail=args.trail,
        )
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
