# Which solver wins, and how far from optimal? The technician's plan, measured

`examples/technician_madrid.py` produces **one** plan for the maintenance technician's case: 182
Burger King restaurants of the Comunidad de Madrid from an office in Leganés, thirty minutes per
visit, eight-hour days. It picks a search and a budget and hands you a timetable. This directory
answers the two questions the example cannot: **which of the solvers scikit-route ships is actually
best on this instance**, and **how far the plan it gives you is from the best possible**.

The short answer:

| | |
|---|---|
| Best plan found | **15 days, 1532.8 min of driving** (25.5 h, 1206.7 km) |
| Stops per day | 13, 13, 12, 11, 12, 13, 13, 12, 13, 12, 10, 11, 13, 11, 13 — days of 450 to 479 min |
| Found by | `IteratedLocalSearch` (2-opt + Or-opt, `n_candidates=10`) for one hour from the best plan so far, confirmed unimprovable by the exact per-day polish |
| Proven lower bound on the driving | **1078.3 min** (the exact ATSP optimum over the 183 nodes, proved in 559 s) — the plan is 42 % above it |
| Proven lower bound on the days | **14** — so 15 is *not* proved optimal, but 23 of 59 configurations reach 15 and **none** reached 14, including a hunt where an extra day was charged ten times its own budget |
| Best solver in two minutes | `MultiStart(IteratedLocalSearch(local_search=("or_opt",), n_candidates=5))` — the only configuration of the roster that reached 15 days at all |
| Best single solver in two minutes | `IteratedLocalSearch(local_search=("or_opt",), n_candidates=5)`: 16 days, 1554.7 min |
| Worth knowing | relocations (`or_opt`) are the move that repacks a day; every configuration whose move set starts with 2-opt needs more time to find the fifteenth day |

Everything above is reproducible from this directory, and `report.py` prints it from the
committed record without running anything.

## The instance

The committed data of `examples/data/` (see its README for the provenance): 183 nodes — the
office first, then the 182 restaurants — with the OSRM **driving minutes** matrix, asymmetric,
captured **2026-09-05**, and the matching kilometres matrix. The model is the one the example
uses, and one trip of the multi-trip objective is one working day:

| | |
|---|---|
| `service_time` | 30.0 minutes per restaurant (5460 minutes in total) |
| `max_time_work` | 480.0 minutes — the eight-hour day, driving *and* service |
| `extra_cost` | 480.0 — one day's budget, so a plan with fewer days always beats a plan with less driving |
| `split` | `"optimal"` — the giant tour is cut into days optimally at every move |
| Objective reported | the pair `(days, driving minutes)`; ranked by `objective = driving + 480 * (days - 1)` |

The roster runs every solver of `skroute.all_solvers()` that can face 183 nodes: the two exact
ones are out (`BruteForce` caps at 11 nodes, `HeldKarp` at 20 — they reappear inside
`exact_polish.py`, where a day is 14 nodes), and so are the two legacy `Ensemble*` shims, whose
work `MultiStart` does. That leaves fourteen solvers, several of them in more than one setting,
plus three `MultiStart` ensembles.

The matrix is asymmetric (one-way streets, motorway ramps), which rules `ClarkeWright` out — it
requires a symmetric matrix, so it is fitted on `(T + T.T) / 2` and re-priced on `T` — and makes
`MILP` refuse the multi-trip instance outright: it certifies the plain tour and nothing else. Both
refusals are recorded rather than hidden.

## The protocol

* **Equal wall-clock budgets.** Every configuration of a round gets the same `--budget` seconds.
  A solver that spends 120 s is compared with a solver that spends 120 s, never with one that ran
  to convergence. Deterministic configurations (constructions, descents) finish in milliseconds
  and their row still records the round's budget.
* **One process per configuration.** Nothing steals cores from a neighbour, and a crash takes one
  row down rather than the round.
* **`prefer="processes"` for every ensemble.** With threads the multi-trip kernels serialise on
  the interpreter: the same eight restarts took **138 s with threads against 36 s with
  processes** on the machine of this campaign. If you rerun a `MultiStart` configuration with
  threads, you are measuring the GIL, not the solver.
* **Fixed `random_state`.** It pins the sequence of random choices, not how many iterations fit
  in a budget, so a rerun is *comparable*, not identical (see "Rerunning" below).
* **Nothing believes a solver's own cost.** Every row's `days`, `driving_min` and `objective` are
  recomputed from the matrix by `RoutingProblem`, cutting the tour with the optimal split.

## The rounds, and what each one cost

The recorded campaign ran on a **10-core Apple Silicon laptop, 2026-09-05 to 2026-09-07**. Read
the wall-clock column before starting anything: the long round is a night's work.

| Round | Configurations | Budget each | Wall clock | What it is |
|---|---|---|---|---|
| `roster` | 28 | 120 s | ~23 min | one pass over the catalogue with sensible settings (the deterministic half finishes instantly; the budget caps the round at 56 min) |
| `chained` | 9 (17 rows) | 240 s | ~45 min | warm starts from the best tours so far, plus the hunt for a 14-day plan |
| `long` | 13 | 30-60 min | 11.5 h of CPU, ~4 h wall clock (nine single-process runs in parallel, then the four eight-process ensembles one after another) | the same families given real time, no iteration cap |
| `exact polish` | 2 | 1500-1800 s | ~3 min | Held-Karp per day and exact exchanges between days |
| `lower_bound` | 2 | 1500 s | ~19 min | the MILP bound, once on the matrix and once on its metric closure |

```bash
cd examples/technician_madrid_study

python benchmark.py --list                                              # the configuration keys

# Round 1 -- the roster, ~23 min. --keep-tours because the next round warm-starts from it.
python benchmark.py --round roster --budget 120 --results out/all.jsonl --keep-tours

# The exact polish of whatever plan you are delivering, ~2 min.
python exact_polish.py --plan results/best_plan.json --max-seconds 1800 \
    --results out/all.jsonl --keep-tours

# Round 2 -- warm starts from the three best tours so far, ~45 min.
python benchmark.py --round chained --budget 240 --results out/all.jsonl \
    --seeds out/all.jsonl --keep-tours

# Round 3 -- one configuration per process. Nine in parallel is one hour; do NOT run the
# thirteen sequentially unless you have twelve hours.
for k in ils_oropt5 ils_default ils_oropt5_best ils_default_best tabu20 \
         genetic_memetic genetic antcolony sa_slow; do
  python benchmark.py --round long --config $k --budget 3600 \
      --results out/long.jsonl --seeds out/all.jsonl --keep-tours &
done; wait
# then the four ensembles, which take eight processes each, one after another (~3 h)
python benchmark.py --round long --config ms_ils_oropt5 --budget 3600 --results out/long.jsonl --seeds out/all.jsonl
python benchmark.py --round long --config ms_ils_default --budget 3600 --results out/long.jsonl --seeds out/all.jsonl
python benchmark.py --round long --config ms_tabu20     --budget 1800 --results out/long.jsonl
python benchmark.py --round long --config ms_sa_slow    --budget 1800 --results out/long.jsonl

# The bounds, ~10 min each. The recorded files are the truth; this only re-proves them.
python lower_bound.py --time-limit 1500 --out out/lower_bounds.json
python lower_bound.py --time-limit 1500 --metric-closure --out out/lower_bounds_metric_closure.json

# The report, instant, from whatever record you point it at.
python report.py --results out/all.jsonl --bounds out/lower_bounds.json
```

Every script takes `--data DIR` (the CSVs, default `examples/data`) and writes only where it is
told; nothing overwrites `results/`.

## What the numbers said

**Two minutes is enough to get close, and not enough to get there.** Of the 28 roster
configurations only `MultiStart(ILS or_opt k=5) x4` reached 15 days (1574.7 min); the best single
solver, `ILS or_opt k=5`, stopped at 16 days and 1554.7 min — *less driving over more days*, which
the objective rightly ranks worse. Then `TabuSearch k=20` (1600.2), `SOM + LocalSearch` (1601.2),
`Genetic` memetic (1607.0), `MultiStart(SA)` (1611.0), down to `NearestNeighbour` (1843.6) and
`NRBS` (1845.3). A plain `LocalSearch` from an `Insertion` tour reaches 1619.2 in **0.4 seconds**,
which is the cheapest good answer in the whole study.

**An hour buys one day, not much driving.** `ILS or_opt k=5` goes from 16 days / 1554.7 to 15 days
/ 1546.3; from the best plan it reaches 1532.9 in 25 366 iterations. The default move set (2-opt
first) is slower to repack a day: from scratch it still ends at 16 days after an hour, but warm
started it produced the best plan of the study, 1532.8 in 9973 iterations. `SimulatedAnnealing`
with a slow schedule (`alpha=0.99995`) **converged** after 31 minutes and 184 203 iterations at 16
days / 1554.2 — a genuinely finished search that still misses the fifteenth day. `Genetic` managed
5.7 M generations and 16 days; memetic `Genetic` managed 90 generations in the same hour and 15
days / 1555.3, which says everything about where the work belongs. `AntColony` got through 612
iterations: each one is expensive at n = 183.

**The exact hybrid found nothing left to find.** `exact_polish.py` re-solves every day exactly
with Held-Karp (at most 13 stops plus the office) and then tries every relocation and swap of
stops between two days, re-solving both days exactly. On the 15-day plans it changed **nothing**:
the days the iterated local search produced were already exactly optimal, and no day can be
emptied by greedy relocation. On an earlier 15-day plan it found 0.9 minutes by exchanging stops —
its note in the record reads `1541.5 -> 1541.5 (exact days) -> 1540.6 min`. That is why the plan is called unimprovable *by these moves* — not optimal.

**Fourteen days is not excluded, and was never found.** The day bound is 14, and 23
configurations reached 15 — including a dedicated hunt with `extra_cost=5000` (ten times the
day's own budget, so any plan with fewer days wins by a landslide) from two different starts.
Nothing reached 14. The honest statement is the one `report.py` prints: 15 days is the best of
everything that was tried, one day above a bound that may or may not be attainable.

## The files

| File | What it is |
|---|---|
| `benchmark.py` | the harness: `--round {roster,chained,long}`, `--budget`, `--config`, `--seeds`, `--results`, `--jobs`, `--keep-tours`, `--list`. Every configuration of the record, with the parameters it was recorded with |
| `lower_bound.py` | the MILP bound (`--time-limit`, `--metric-closure`), the triangle-inequality statistics and the two day bounds |
| `exact_polish.py` | the Held-Karp hybrid: `--plan` (a `best_plan.json` or a timetable CSV of the example), `--max-seconds` |
| `report.py` | reads the record and prints the ranking, the two-minutes-against-an-hour comparison and the gap against the bound; `--format {markdown,text}`, `--sort {objective,name,family}` |
| `results/results.jsonl` | 60 rows: 59 plans and one recorded refusal (`MILP` on a multi-trip instance). One JSON object per configuration, with `name`, `family`, `params`, `round`, `days`, `driving_min`, `objective`, `stops_per_day`, `day_minutes`, `n_iter`, `stop_reason`, `wall_s`, `budget_s`, `fit_time_s` and `tour_sha256` — the first 16 hex characters of the sha256 of the tour's JSON. The tours themselves are dropped (they would be 60 × 183 labels); `--keep-tours` writes them |
| `results/best_plan.json` | the plan the study reports, with its tour, its days as labels, and the minutes and kilometres of each day. `tests/test_study.py` rebuilds it from the committed matrices and checks all of it |
| `results/lower_bounds.json` | the ATSP bound on the OSRM matrix: proven optimal, 1078.32 min, 559 s, plus the tour, the triangle statistics and the tail of the cut loop's bound trace |
| `results/lower_bounds_metric_closure.json` | the same on the metric closure of the matrix: 1078.32 min again, 0 violating triples — which is what removes the triangle-inequality caveat |

`params` is what makes the record checkable: it carries the settings each row was run with, so a
rerun can be compared configuration by configuration.

## Rerunning: what you should expect

The searches are **stochastic and time-limited**. `random_state` is fixed, but how many iterations
fit into 120 s depends on the machine, the numpy build and what else is running, and the
acceptance of a move depends on the iteration it happens at. A rerun therefore lands **within a
few minutes of driving** of the recorded numbers and usually on the same day count — it does not
reproduce them exactly, and a table that matched to the decimal would mean something was cached.

What *is* exactly reproducible: the deterministic configurations. `NearestNeighbour`, `Insertion`
(all three strategies), `NRBS`, `TwoOpt`, `OrOpt`, `LocalSearch`, `SOM`, `ClarkeWright` and the
exact polish give the same tour byte for byte — their `tour_sha256` is a regression test. (The one
wobble: `NRBS` records 1845.3 minutes where the harness now computes 1845.4 on the identical tour.
The raw sum is 1845.3500000000001, which sits on a rounding tie.)

The MILP bound is deterministic too: HiGHS runs single-threaded and deterministically, so the
1078.32 comes back, in about the same nine and a half minutes.

## What the record gets wrong

`results/` is committed **verbatim** as the campaign wrote it, because its tours and their hashes
are the study's evidence. Two *derived* fields in it are wrong, both from the same off-by-one, and
both are pinned by `tests/test_study.py` so that nobody rediscovers them by surprise:

1. **`stops_per_day` counts legs, not stops, on every solver row.** The campaign recorded
   `len(trip) - 1`, and `trips_` are closed `[office, ..., office]` arrays, so each day's figure
   is one too high and the row sums to 182 + `days` instead of 182. The two `hybrid` rows (the
   exact polish) carry the true counts, as does `best_plan.json`, and `benchmark.py` now records
   the true count. `days`, `driving_min` and `objective` are unaffected.
2. **The `ClarkeWright (symmetrised T)` row records 17 days and 1180.4 minutes of driving.** Both
   are wrong. `RoutingProblem.trip_starts` returns `n_trips + 1` positions and the campaign read
   that length as the day count, then derived the driving by subtracting `480 * (days - 1)` from a
   correct objective. The plan is **16 days and 1660.4 minutes** — the same day count as the other
   constructions, mid-table on driving — and its objective, 8860.4, is right. So the tempting
   reading, "fewer minutes over two more days", is an artefact: re-run it and check.

   ```bash
   python benchmark.py --round roster --config clarke_wright --budget 120 --results /tmp/cw.jsonl
   # ClarkeWright (symmetrised T)   days 16  driving 1660.4 min   (tour_sha256 deddfe827e718a4c,
   # the same tour as the record)
   ```

The same off-by-one was in the plan reader of the prototypes, where it produced an empty leading
and an empty trailing day; `exact_polish.days_from_tour` cuts on `starts[1:-1]`, asserts every day
non-empty and asserts that the days partition the 182 restaurants exactly once.

Two smaller differences between the record and what the harness writes today, neither of them a
correction: `params.passes` and `n_iter` of an exact-polish row now count the *accepted* passes
(the campaign counted scans, so its 1 and 2 mean 0 and 1 accepted), and every row now carries the
`key` of its configuration, which is what `tests/test_study.py` uses to keep `--list` and the
record in sync.

---

Data © OpenStreetMap contributors (ODbL); routing by OSRM (router.project-osrm.org). Show it
wherever these numbers or the pictures made from them appear.
