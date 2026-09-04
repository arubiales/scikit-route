# Examples

## `live_demo.py` — watch a solver work

Runs a solver on a bundled instance with a `skroute.viz.LivePlot` callback, so you see
the current tour (thin, light line) and the best tour so far (thick line) being redrawn
while `fit` runs, with the iteration, the costs and the solver's own facts (temperature,
tenure, kick...) in the title. Needs the `viz` extra (`pip install "scikit-route[viz]"`;
`[viz-map]` adds Plotly for `--backend plotly` and `--map`).

```bash
python examples/live_demo.py                                                     # IteratedLocalSearch on Djibouti (dj38)
python examples/live_demo.py --instance wi29 --solver SimulatedAnnealing --every 10
python examples/live_demo.py --instance barcelona --solver TabuSearch --every 5
python examples/live_demo.py --instance barcelona --solver SimulatedAnnealing --every 30 --backend plotly --map
python examples/live_demo.py --instance dj38 --solver SimulatedAnnealing --init random --every 30 --gif live_demo.gif
```

| Option | Meaning |
|---|---|
| `--instance wi29 \| dj38 \| barcelona` | Western Sahara (29), Djibouti (38) or the Barcelona road-cost table (19 places, `(lat, lon)` coordinates) |
| `--solver NAME` | any class of `skroute.all_solvers()`; stochastic ones get `random_state=--seed` (default 0) |
| `--every N` | redraw every N-th outer iteration — the knob that keeps a fast solver fast |
| `--backend matplotlib \| plotly` | a matplotlib window, or a Plotly figure shown when the run ends |
| `--map` | OpenStreetMap tiles (Plotly `Scattermap`; Barcelona only, the other instances have planar coordinates) |
| `--gif PATH` | record the run with `Recorder` and save a GIF (Pillow) instead of watching |
| `--init nearest_neighbour \| random` | starting tour of the solvers that have `init=` (a random start makes the search visible) |

The matplotlib window appears through `plt.pause` while the solver runs; the script
calls `plt.show()` at the end to keep the final picture open. Headless machines
(`MPLBACKEND=Agg`) run the whole thing silently, which is how `docs/images/live_demo.gif`
is produced: `--instance dj38 --solver SimulatedAnnealing --init random --every 30 --gif live_demo.gif`
(a random tour of 27 722 becomes the optimum 6656).

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
finishes its current iteration and returns with `stop_reason_ == "callback"`:

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

`Recorder` is the non-interactive counterpart: it keeps every event, and
`rec.animate(coords)` (matplotlib `FuncAnimation`, `anim.save("run.gif", writer="pillow")`)
or `rec.to_plotly(coords, map=...)` (frames and a slider) replay the run afterwards.
The user guide page *Live visualisation* covers all of this in detail.
