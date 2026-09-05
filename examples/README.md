# Examples

## `live_demo.py` — watch a solver work, record it, replay it

Runs a solver on a bundled instance with a `skroute.viz.LivePlot` callback, so you see
the current tour (thin, light line) and the best tour so far (thick line) being redrawn
while `fit` runs, with the iteration, the costs and the solver's most informative scalar
fact (temperature, tenure, generation... — one; lists such as the kick of an iterated
local search never) in the title. Solvers that report the structure they are building
(D31) show it too: the growing partial tour of a construction heuristic as orange edges,
the pheromone trails of the ant colony fading with their strength, the SOM's elastic ring
in teal. Needs the `viz` extra (`pip install "scikit-route[viz]"`; `[viz-map]` adds Plotly
for `--backend plotly` and `--map`).

```bash
python examples/live_demo.py                                                     # IteratedLocalSearch on Djibouti (dj38)
python examples/live_demo.py --instance wi29 --solver SimulatedAnnealing --every 10 --show current --trail 5
python examples/live_demo.py --instance barcelona --solver TabuSearch --every 5
python examples/live_demo.py --instance barcelona --solver SimulatedAnnealing --every 30 --backend plotly --map
python examples/live_demo.py --solver SimulatedAnnealing --init random --every 30 --record live_demo.gif --fps 12
python examples/live_demo.py --solver Insertion --record construction_demo.gif --fps 8
python examples/live_demo.py --solver SimulatedAnnealing --init random --every 30 --speed 10
```

| Option | Meaning |
|---|---|
| `--instance wi29 \| dj38 \| barcelona` | Western Sahara (29), Djibouti (38) or the Barcelona road-cost table (19 places, `(lat, lon)` coordinates) |
| `--solver NAME` | any class of `skroute.all_solvers()`; stochastic ones get `random_state=--seed` (default 0) |
| `--every N` | redraw (or keep, when recording) every N-th outer iteration — the knob that keeps a fast solver fast |
| `--backend matplotlib \| plotly` | a matplotlib window, or a Plotly figure shown when the run ends |
| `--map` | OpenStreetMap tiles (Plotly `Scattermap`; Barcelona only, the other instances have planar coordinates; the solver is fitted with the `(lat, lon)` coordinates, so a SOM ring lands on the tiles; not with `--record`, which draws plain axes) |
| `--show both \| best \| current` | which tours to draw: the attempt and the best, only the best, or only the attempt |
| `--trail K` | keep the last K current tours on screen, fading with age |
| `--record PATH` | record the run with `Recorder` and save it: `.gif` through pillow, `.mp4` (and the rest) through ffmpeg |
| `--fps N` | frame rate of the recorded file (default 20) |
| `--speed S` | with `--record`, a time-lapse: frames follow the recorded clock divided by S (never shorter than 10 ms per frame — pointless for a construction whose steps are microseconds apart); alone, record silently and replay the run on screen S times faster |
| `--init nearest_neighbour \| random` | starting tour of the solvers that have `init=` (a random start makes the search visible) |

The matplotlib window appears through `plt.pause` while the solver runs; the script
calls `plt.show()` at the end to keep the final picture open. Headless machines
(`MPLBACKEND=Agg`) run the whole thing silently — the picture is rendered once at the end
and `plt.show()` is skipped — which is how the GIFs of the documentation are produced:

```bash
python examples/live_demo.py --instance dj38 --solver SimulatedAnnealing --init random --every 30 \
    --record docs/images/live_demo.gif --fps 12
python examples/live_demo.py --instance dj38 --solver Insertion --record docs/images/construction_demo.gif --fps 8
python examples/live_demo.py --instance dj38 --solver SOM --set n_iter=20000 --record docs/images/som_demo.gif --fps 12
```

The first one untangles a random tour of about 27 700 — four times the optimum 6656 —
down to the optimum on the machine that recorded it (another platform may land within a
percent); the annealer's attempts are the thin grey line, the best tour so far the thick
one. The second shows farthest insertion growing the tour node by node (a construction
runs in a millisecond, so the file is written at a fixed frame rate, one frame per inserted
city, rather than as a time-lapse of the recorded clock); the third the SOM ring closing on
the cities epoch by epoch (`n_iter=20000` gives 58 epochs on `dj38`).

## `LivePlot` in a notebook

```python
from skroute import SimulatedAnnealing
from skroute.datasets import load_tsp
from skroute.viz import LivePlot

dj = load_tsp("dj38")
live = LivePlot(dj.coords, every=10)
sa = SimulatedAnnealing(random_state=0).fit(dj.distance_matrix(), labels=dj.labels, callback=live)
```

- With `%matplotlib inline` every redraw replaces the cell output (`IPython.display.display(fig, clear=True)`);
  raise `every=` if the notebook feels sluggish.
- With `%matplotlib widget` (ipympl) the canvas is updated in place.
- `LivePlot(dj.coords, backend="plotly")` uses a Plotly `FigureWidget` (needs `anywidget`), updated in
  place; `map=True` draws on OpenStreetMap tiles for `(lat, lon)` coordinates such as `load_barcelona().coords`.

To stop a long run from another cell, fit in a thread and call `live.stop()`; the solver
finishes its current iteration and returns with `stop_reason_ == "callback"`. This is for
`%matplotlib inline` and `backend="plotly"`, whose redraws are IPython display calls; a
desktop window (macosx, Tk, Qt) must be drawn from the main thread, so there `fit` stays
on it and a key handler on `live.fig` calls `live.stop()` (see the user guide).

```python
import threading

live = LivePlot(dj.coords, every=10)
sa = SimulatedAnnealing(random_state=0)
worker = threading.Thread(
    target=sa.fit, args=(dj.distance_matrix(),), kwargs={"labels": dj.labels, "callback": live}
)
worker.start()
# ... later, in another cell:
live.stop()
worker.join()
sa.stop_reason_  # 'callback'
```

## `Recorder`: replay at time-lapse speed

`Recorder` is the non-interactive counterpart: it keeps every event with a wall-clock
stamp, and afterwards `rec.replay(coords, speed=10)` drives a `LivePlot` through the run
ten times faster than it happened, `rec.animate(coords, speed=10)` builds the matching
matplotlib animation (`plt.show()`, or `IPython.display.HTML(anim.to_jshtml())`),
`rec.save("run.gif", coords, speed=10)` writes it (`.mp4` with ffmpeg), and
`rec.to_plotly(coords, map=...)` gives a figure with Play/Pause, a speed menu (0.5x to 8x)
and a slider. The user guide page *Live visualisation* covers all of this in detail.

## `technician_madrid.py` — a real plan: every Burger King of Madrid, one technician, eight-hour days

The worked case of the 2.1 release, in the owner's words: *"imaginemos que tenemos que hacer
los mantenimientos de los Burger King de Madrid y nuestra oficina está en Leganés en Calle
Ramón y Cajal 18. Un mantenimiento medio tarda 30 min. Quiero ver todas las rutas como serían
para cubrir todos los Burger King por un técnico de mantenimiento de sistemas de alarma día
tras día hasta que los haga todos, lo más óptimo posible. Su jornada es de 8 horas."* The
182 restaurants come from OpenStreetMap (`fetch_pois`), the office from Nominatim (`geocode`),
the driving times from OSRM (`travel_time_matrix`); the plan is a multi-trip fit where a trip
is a working day, with `service_time=30`, `max_time_work=480`, `extra_cost=480` (a day costs
more than any driving) and `split="optimal"`. The data are committed under `examples/data/`
(see its README for the provenance and the attribution), so the script runs offline. On the
machine that wrote the documentation the default two-minute run gives **15 days and 25.8 hours
of driving** (the constructions need 16 days; 12 is the service-only lower bound); the user
guide page *A real case: the technician's plan* narrates the results.

```bash
python examples/technician_madrid.py                                   # the committed data, two minutes
python examples/technician_madrid.py --quick                           # a few seconds (what the test runs)
python examples/technician_madrid.py --solver sa --time-limit 60 --live
python examples/technician_madrid.py --google-key AIza...              # + a Google Maps page of the plan
python examples/technician_madrid.py --refresh                         # fetch the live services again
python examples/technician_madrid.py --refresh --provider google --google-key AIza...   # with traffic
```

| Option | Meaning |
|---|---|
| `--data DIR` | directory of the three CSVs (default `examples/data`) |
| `--refresh` | fetch the restaurants (Overpass), the office (Nominatim) and the matrix (`--provider osrm \| google`) again, drop near-duplicates within 60 m and rewrite the CSVs |
| `--provider osrm \| google` | the routing service of `--refresh`: the public OSRM demo server (free, no traffic model) or the Google Distance Matrix API with `departure_time="now"` (needs the key) |
| `--limit N` | with `--refresh`: keep only the first N restaurants — a check of the three services in seconds |
| `--google-key KEY` | a Google Maps API key (or `GOOGLE_MAPS_API_KEY`): the Google matrix with `--refresh --provider google`, and always the Maps JavaScript page `technician_madrid_google.html` |
| `--service MIN`, `--hours H`, `--start HH:MM` | minutes per visit (30), hours per day (8), departure time (08:00) |
| `--solver multistart \| ils \| sa \| tabu \| genetic` | the search: `MultiStart(IteratedLocalSearch(local_search=("or_opt",), n_candidates=5), n_restarts=8, n_jobs=-1, prefer="processes")` by default — relocations are the move that repacks a day |
| `--time-limit SECONDS` | wall-clock budget of the whole run (120): the search takes 85 % of it, a polish of the winner with the full move set (2-opt and Or-opt) the rest; everything is priced with `split="optimal"` |
| `--seed N` | `random_state` (0) |
| `--quick` | a few iterations instead of the time budget — deterministic, a few seconds |
| `--extra-day-cost MIN` | the charge per extra day in minutes of driving (default: the day's budget, so fewer days always win) |
| `--out DIR` | output directory (`./technician_madrid_out`) |
| `--live`, `--record GIF` | watch the search with `LivePlot`, or record it with `Recorder` and save a GIF (the restarts then run sequentially) |
| `--verbose` | also log the solvers' own progress (restarts, iterations) through the `skroute` logger |

The output directory receives `technician_madrid_timetable.csv` (every stop with its
arrival and departure), `technician_madrid_days.csv` (the totals per day: stops, minutes and
kilometres of driving, service, the time back at the office),
`technician_madrid.kml` (for Google My Maps and Google Earth),
`technician_madrid_google_urls.txt` (one Google Maps Directions link per leg of at most
nine stops), `technician_madrid_map.html` (the Plotly map on OpenStreetMap tiles, one
colour per day), `technician_madrid_google.html` when a key is given, and the two PNGs of
the documentation; the console shows the per-day table, the totals, the lower bound and
the two construction baselines through the `skroute` logger.
