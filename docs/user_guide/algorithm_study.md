# Which solver wins: an algorithm study

The [previous page](real_world.md) built one plan for one real instance. This page answers
the question that comes right after it: every name in `skroute.all_solvers()` is fitted the
same way, so which of the eighteen do you run on your own data, what does more wall clock buy,
and is the answer any good in absolute terms? The case is the technician's: 59 configurations
were run on it under equal budgets, the plain tour was proved optimal to get a rigorous bound,
and every day of the winning plan was re-solved exactly. What follows is what those runs say —
including where a solver looks bad and why.

The short version: [`IteratedLocalSearch`][skroute.IteratedLocalSearch] with Or-opt
relocations, wrapped in [`MultiStart`][skroute.MultiStart], is the configuration to run — in
an hour it plans **15 days and 1 533.5 minutes of driving**, and the best plan of the whole
campaign, 1 532.8 minutes, is one more hour of the same search continued from it. That plan is
at most one day away from the optimum and 42 % above a driving bound that no day-partitioned
plan can reach.

## The instance and the protocol

The instance is the one of the previous page, unchanged: every configuration was fitted with
the same arguments, on the same matrix, and its result was priced with the same objective, so
the only difference between two rows of the tables below is the search. The recorded rows —
one JSON object per configuration, with its parameters, wall clock, iterations and stop
reason — are committed under `examples/technician_madrid_study/results/`.

- **Nodes:** 183 — the 182 Burger King restaurants of the Comunidad de Madrid plus the office
  in Leganés, from `examples/data/madrid_burger_king*.csv` (OpenStreetMap and OSRM, captured
  2026-09-05). The travel times are **asymmetric** driving minutes.
- **Model:** `service_time=30`, `max_time_work=480`, `extra_cost=480`, `split="optimal"`,
  `depot="office"` — the fit of the previous page, one trip per working day.
- **Objective:** reported as **(days, driving minutes)**. The single number the solvers
  minimise is `driving + 480 * (days - 1)`, so one day saved beats any amount of driving: the
  best plan scores `1532.8 + 14 * 480 = 8252.8`.
- **Budget:** equal wall clock per configuration, given as `time_limit=` and never as an
  iteration count — 120 s in the roster, 3 600 s in the long campaign. A roster
  `MultiStart(...)` spends the same 120 s of CPU (four restarts of 30 s, in parallel); the
  long `MultiStart`s do **not** — eight processes for the whole hour, eight times the CPU of
  a single run. Read those four rows as "what the wrapper does with a core each", not as a
  fair duel.
- **One process per configuration**, and `prefer="processes"` inside every `MultiStart`.
- **Hardware and date:** a 10-core Apple Silicon laptop, September 2026. The long campaign
  ran nine single-process configurations in parallel and then the four `MultiStart`s one
  after another: four hours of wall clock, 11.5 hours of process time.
- **The searches are stochastic and stopped by the clock.** `random_state=0` everywhere, but
  a time limit makes a run irreproducible bit for bit (the same seed sees a different number
  of iterations). Expect a few minutes of driving either way on your machine, and, for the
  configurations that sit on the 15/16-day boundary, the odd extra day.

The plan to beat is the best of the campaign: **15 days, 1 532.8 minutes of driving** (25.5 h,
1 206.7 km), 10 to 13 stops a day (`[13, 13, 12, 11, 12, 13, 13, 12, 13, 12, 10, 11, 13, 11,
13]`) and day totals between 450 and 479 of the 480 available minutes. It was found by an hour
of `IteratedLocalSearch` with the default move set continuing from the best plan then known,
and the exact polish described below could not improve it. It is 16 minutes of driving better
than the two-minute plan narrated on the previous page (1 549), which is the honest size of
the prize for an hour of computing.

## How good can it get

Two bounds make the answer measurable, and neither depends on any heuristic.

**The driving bound.** A plan over $k$ days is a closed walk that starts at the office, visits
every restaurant once and passes through the office $k$ times. Delete the intermediate office
visits — replace `... a -> office -> b ...` by `a -> b` — and what is left is a closed tour
over the 183 nodes. Under the triangle inequality that shortcut can never lengthen the walk,
so **the driving of any plan, on any number of days, is at least the optimal closed tour over
the 183 nodes**. That tour is small enough to be *proved*: [`MILP`][skroute.MILP] (the
Dantzig–Fulkerson–Johnson formulation with lazy subtour cuts on HiGHS, arc variables because
the matrix is asymmetric) returns **1 078.3 minutes, proven optimal** — `gap_ == 0.0` — in
559 s.

The caveat is that road times are not exactly metric: OSRM's matrix violates the triangle
inequality in **738 of 6 128 487 triples (0.012 %)**, mostly where a motorway ramp makes the
direct arc slower than a detour. That is small, but "small" is not an argument, so the same
bound was recomputed on the **metric closure** of the matrix (all-pairs shortest paths, zero
violations by construction, every entry at most the original): the proven optimum there is
**1 078.3 as well**, in 570 s. The bound holds on a matrix where the shortcutting argument is
exact, and the caveat is gone.

**The day bounds.** Every day holds at most 480 minutes of service plus driving, the service
is fixed at $182 \times 30 = 5460$ minutes, and the driving is at least 1 078.3, so

$$\text{days} \ \ge\ \left\lceil \frac{5460 + 1078.3}{480} \right\rceil = \lceil 13.62 \rceil = 14$$

A second bound needs no driving bound at all: sixteen visits are 480 minutes of service on
their own and leave nothing for the road, while fifteen leave 30 minutes — more than the
cheapest conceivable set of legs for a fifteen-stop day (2.6 out, 2.2 back and 14 hops of
0.6 add up to 14). So a day holds at most 15 stops and `days >= ceil(182 / 15) = 13`. It is
weaker here, and it is the bound to reach for when no tour bound is available. Both improve on
the 12 days of the service-only count quoted on the previous page.

**What the searches reach.** Of the 59 configurations that produced a plan, **23 reach 15
days and none reaches 14** — including three configurations run specifically to hunt for a
14-day plan, with `extra_cost=5000` instead of 480: at that price a search accepts a day
fewer even if it costs 5 000 extra minutes of driving, and it still came back with 15 days.

So, honestly stated:

- **Days: the optimum is 14 or 15, and the plan uses 15.** At most one day from optimal. The
  14 is not excluded by anything; it is simply never found.
- **Driving: 1 532.8 against a bound of 1 078.3 is 42 % above it** — and almost none of that
  42 % is slack. The 455 minutes between the two are the 14 extra returns to the office that a
  15-day plan must drive and a single tour does not: 32.5 minutes each, against the 45 that an
  average pair of office legs costs (the mean leg to a restaurant is 22.5 minutes), because the
  optimal split cuts the tour where returning is cheapest. A plan with fewer days would recover
  some of those returns; a better routing inside 15 days will not.

```python
>>> import math
>>> import numpy as np
>>> T = np.loadtxt("examples/data/madrid_burger_king_times_min.csv", delimiter=",",
...                skiprows=1, usecols=range(1, 184))
>>> bound = 1078.32          # results/lower_bounds.json: proven optimal ATSP tour, 559 s
>>> service = 182 * 30       # 182 visits of half an hour
>>> service, math.ceil((service + bound) / 480)
(5460, 14)
>>> violations = sum(int(np.sum(T[:, [k]] + T[[k], :] < T - 1e-9)) for k in range(len(T)))
>>> violations, round(100 * violations / len(T) ** 3, 3)      # of 183 ** 3 triples
(738, 0.012)
>>> round(1532.8 - bound, 1), round((1532.8 - bound) / 14, 1), round(100 * (1532.8 / bound - 1), 1)
(454.5, 32.5, 42.1)

```

## The ranking at two minutes

Twenty-eight configurations, 120 s each, ranked by the objective (`driving + 480 * (days -
1)`), which is why row 1 has *more* driving than row 2 and row 18 has the least driving of the
table and sits near the bottom. Every row is `fit` on the same matrix with the same arguments;
`Wall` is the measured wall clock — a construction heuristic that answers in a millisecond
returns the rest of its budget — and `Iterations` is `n_iter_`, whose unit differs per family
(a kick and descent for an iterated local search, a temperature level for annealing, a
generation for the genetic algorithm, a colony pass for the ant colony).

| # | Configuration | Family | Days | Driving | Objective | Wall | Iterations |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `MultiStart(ILS or_opt k=5) x4` | ensemble | 15 | 1 574.7 | 8 294.7 | 30.9 s | 94 |
| 2 | `ILS or_opt k=5` | ils | 16 | 1 554.7 | 8 754.7 | 120.1 s | 862 |
| 3 | `ILS or_opt k=5 from Insertion` | ils | 16 | 1 565.1 | 8 765.1 | 120.2 s | 807 |
| 4 | `ILS or_opt k=5 strength=2` | ils | 16 | 1 570.3 | 8 770.3 | 120.0 s | 623 |
| 5 | `ILS default (2opt+oropt, k=10)` | ils | 16 | 1 572.5 | 8 772.5 | 120.2 s | 271 |
| 6 | `ILS or_opt k=5 metropolis` | ils | 16 | 1 587.1 | 8 787.1 | 120.4 s | 698 |
| 7 | `TabuSearch k=20` | metaheuristic | 16 | 1 600.2 | 8 800.2 | 120.0 s | 440 |
| 8 | `SOM + LocalSearch polish` | metaheuristic | 16 | 1 601.2 | 8 801.2 | 0.6 s | 4 |
| 9 | `Genetic memetic (or_opt)` | metaheuristic | 16 | 1 607.0 | 8 807.0 | 121.0 s | 2 |
| 10 | `MultiStart(SA) x4` | ensemble | 16 | 1 611.0 | 8 811.0 | 30.1 s | 1462 |
| 11 | `MultiStart(Tabu) x4` | ensemble | 16 | 1 612.8 | 8 812.8 | 30.1 s | 138 |
| 12 | `TabuSearch` | metaheuristic | 16 | 1 612.8 | 8 812.8 | 120.1 s | 1073 |
| 13 | `SimulatedAnnealing init=random` | metaheuristic | 16 | 1 615.0 | 8 815.0 | 27.5 s | 1838 |
| 14 | `LocalSearch from Insertion` | local search | 16 | 1 619.2 | 8 819.2 | 0.4 s | 5 |
| 15 | `LocalSearch(2opt+oropt)` | local search | 16 | 1 620.1 | 8 820.1 | 0.3 s | 4 |
| 16 | `OrOpt` | local search | 16 | 1 640.3 | 8 840.3 | 0.4 s | 6 |
| 17 | `Genetic` | metaheuristic | 16 | 1 656.4 | 8 856.4 | 120.0 s | 134373 |
| 18 | `ClarkeWright (symmetrised T)` | construction | 17 | 1 180.4 | 8 860.4 | 0.0 s | 0 |
| 19 | `SimulatedAnnealing` | metaheuristic | 16 | 1 667.4 | 8 867.4 | 23.8 s | 1838 |
| 20 | `Insertion(cheapest)` | construction | 16 | 1 709.6 | 8 909.6 | 0.0 s | 0 |
| 21 | `Insertion(nearest)` | construction | 16 | 1 720.4 | 8 920.4 | 0.0 s | 0 |
| 22 | `Insertion(farthest)` | construction | 16 | 1 720.7 | 8 920.7 | 0.0 s | 0 |
| 23 | `TwoOpt` | local search | 16 | 1 743.7 | 8 943.7 | 0.1 s | 4 |
| 24 | `AntColony` | metaheuristic | 16 | 1 745.0 | 8 945.0 | 127.5 s | 18 |
| 25 | `SOM` | metaheuristic | 16 | 1 790.3 | 8 990.3 | 0.8 s | 17 |
| 26 | `NearestNeighbour` | construction | 16 | 1 843.6 | 9 043.6 | 0.0 s | 0 |
| 27 | `NRBS` | construction | 16 | 1 845.3 | 9 045.3 | 0.0 s | 0 |

The tables of this page are generated from the recorded rows by `python
examples/technician_madrid_study/report.py`.

The twenty-eighth configuration is [`MILP`][skroute.MILP], which does not appear because it
refuses the instance on purpose: `ValueError: MILP optimises the plain tour and cannot certify
a multi-trip optimum; use BruteForce (n <= 11) or a heuristic solver`. That refusal is the
right behaviour — a "proven optimum" that ignored the day budget would be a lie — and it is
also why the bound above is a *plain tour* bound.

Three readings of the table. Leaving aside Clarke-Wright's 17-day plan, the ranking spans 291
minutes of driving, from 1 554.7 to 1 845.3: the choice of solver is worth 19 % of the driving,
and one day. The step to 15 days is worth 480 points of objective and only one configuration in
the roster reached it, so at short budgets the interesting question is not "how little driving"
but "who lands on the lower step". And the two cheapest searches in the table (`LocalSearch`
from a construction, 0.3–0.4 s, 1 619–1 620) sit 4 % above the best two-minute driving: a third
of a percent of the budget gets you within 4 %, and the whole rest of it buys those last few
percent plus the day.

## Two minutes against one hour

The same families, at 120 s and at 3 600 s. `Iterations` shows what the extra wall clock
actually bought each of them, which varies by five orders of magnitude.

| Configuration | 2 min | 1 h | Iterations (2 min, 1 h) |
|---|---|---|---|
| `ILS or_opt k=5` | 16 / 1 554.7 | **15 / 1 546.3** | 862, 25 070 |
| `ILS 2opt+oropt k=10` (default) | 16 / 1 572.5 | 16 / 1 557.1 | 271, 9 312 |
| `MultiStart(ILS or_opt k=5)` | 15 / 1 574.7 (x4, 30 s each) | **15 / 1 533.5** (x8, 1 h each) | 94, 13 165 |
| `MultiStart(ILS 2opt+oropt k=10)` | — | **15 / 1 541.4** (x8, 1 h each) | —, 6 181 |
| `TabuSearch k=20` | 16 / 1 600.2 | **15 / 1 567.0** | 440, 17 542 |
| `MultiStart(TabuSearch k=20)` | 16 / 1 612.8 (x4, 30 s) | 16 / 1 592.0 (x8, 30 min) | 138, 8 950 |
| `SimulatedAnnealing` | 16 / 1 667.4 (alpha=0.995) | 16 / 1 554.2 (alpha=0.99995) | 1 838, 184 203 |
| `MultiStart(SimulatedAnnealing)` | 16 / 1 611.0 (x4, 30 s) | **15 / 1 555.2** (x8 slow, 30 min) | 1 462, 159 014 |
| `Genetic memetic (or_opt)` | 16 / 1 607.0 | **15 / 1 555.3** | 2, 90 |
| `Genetic` | 16 / 1 656.4 | 16 / 1 617.8 | 134 373, 5 757 448 |
| `AntColony` | 16 / 1 745.0 | 16 / 1 624.0 | 18, 612 |
| `ILS` continued from the best plan | — | **15 / 1 532.8** (from 1 537.1) | —, 9 973 |

Thirty times the budget moves four of the ten families that appear in both columns from 16 days
to 15, and buys between 8 and 121 minutes of driving. It does not change the ranking much: what
was ahead at two minutes is ahead at an hour, with one exception worth noting — the annealer,
next to last at two minutes and, with a slow enough schedule, better driving at an hour than
seven of the eleven other rows (while still needing 16 days).

## What each solver is good for

### Iterated local search, with Or-opt relocations

[`IteratedLocalSearch`][skroute.IteratedLocalSearch] is the default and it wins here, in every
budget and from every start. The setting that matters is the move set: `local_search=("or_opt",),
n_candidates=5` beat the default `("two_opt", "or_opt")` with `n_candidates=10` at two minutes
(1 554.7 against 1 572.5) and at an hour from scratch (1 546.3 and 15 days against 1 557.1 and
16), because it does three times the iterations — 862 against 271 in two minutes — and because
**relocation is the move that repacks a day**. Under `split="optimal"` the plan is decoded from
a giant tour, so moving a segment of one to three stops to another part of the tour is what
moves a restaurant from one day to another; a 2-opt reversal shortens the driving *inside* a day
but rarely changes what fits in it. The order flips once the plan is already good: continuing
from the best plan, the full move set gave 1 532.8 and Or-opt alone 1 532.9 — a tie, with 9 973
iterations against 25 366, so at that point the expensive move is worth its price. Practical
rule: search with Or-opt, polish with both.

### MultiStart

[`MultiStart`][skroute.MultiStart] earns its place on a staircase objective. At two minutes,
four restarts of 30 s reached 15 days while the same solver given the whole 120 s in one run
reached 16 — the same CPU, four samples instead of one deep descent. That is the shape of the
problem: crossing from 16 days to 15 needs several coordinated relocations that each add
driving, so it is a matter of landing in the right basin, and four independent draws beat one
long run at finding it. At an hour per restart, eight restarts of the Or-opt configuration gave
1 533.5, within 0.7 minutes of the best chained run and 13 minutes better than the single
one-hour run; the wrapper also lifted the slow annealer from 16 days to 15 (1 555.2). Use
`prefer="processes"` (see below) and `n_jobs=-1`; the restarts are independent, so the result is
the same for any `n_jobs`.

### Tabu search

[`TabuSearch`][skroute.TabuSearch] is the mid-table solver of this instance: 15 days at an hour
(1 567.0), 34 minutes behind the best. Widening the candidate lists helps — `n_candidates=20`
gave 1 600.2 against 1 612.8 for the default 10 at two minutes, with 440 iterations against
1 073, so fewer and better-informed moves beat more and blinder ones. Wrapped in `MultiStart`
it does *not* improve (1 592.0 with eight restarts of 30 minutes, still 16 days): a
best-admissible scan with a randomised tenure is already close to deterministic, so restarts
mostly repeat each other. Its niche is elsewhere — a structured instance where annealing
stalls; here the ILS relocations are simply better aimed.

### Simulated annealing and its schedule

[`SimulatedAnnealing`][skroute.SimulatedAnnealing] with the defaults *converges* in 24 s
(1 838 temperature levels, `stop_reason_ == "converged"`) and then cannot use the remaining 96
seconds: at 1 667.4 it beats only the construction heuristics, a bare 2-opt descent, the raw SOM
ring and the ant colony, and giving it more wall clock changes nothing. The schedule is the knob,
not the budget. With `alpha=0.99995` it ran 184 203 levels and converged after 31 minutes at
**1 554.2**, better driving than every other one-hour run except the three iterated local
searches — and still 16 days, which is the pattern of this instance: annealing is very good at
the driving inside a fixed number of days and unlucky at crossing the day boundary. Eight
restarts of that slow schedule did cross it (15 days, 1 555.2). Note that "converged after 31
minutes" is genuine convergence, not a time limit: if you want more, lower `alpha` further or add
restarts, do not add minutes.

### Genetic, plain and memetic

The plain [`Genetic`][skroute.Genetic] is the cheapest iteration of the library and the weakest
result at this size: 134 373 generations in two minutes for 1 656.4, and **5.76 million**
generations in an hour for 1 617.8 — 85 minutes behind the best, still 16 days. Recombining
permutations of 183 nodes without a descent simply does not find the structure. The memetic form
(`local_search=("or_opt",)`, an Or-opt descent on every child) is a different algorithm: 40
seconds per generation, so **2 generations** in two minutes (1 607.0) and **90** in an hour —
and those 90 generations reach 15 days at 1 555.3, fourth among the single-run configurations of
the long campaign, behind three iterated local searches and nothing else. If you
run a population-based search on an instance this size, run the memetic one and budget in tens
of minutes; its by-product is a pool of good distinct plans in `estimators_`, which the
[warm starts and ensembles](warm_starts_and_ensembles.md) page uses.

### Ant colony at n = 183

[`AntColony`][skroute.AntColony] is the solver most penalised by the size. Each iteration builds
50 ant tours and polishes every one of them with a 2-opt descent priced under the optimal split,
which at 183 nodes costs seconds: **18 iterations** in two minutes (1 745.0, behind all three
insertion variants) and **612** in an hour (1 624.0). A MAX-MIN colony needs thousands of
iterations before the pheromone means anything, and this instance never gives it that many. It is
not a bad solver here, it is an unfinished one — on the plain Euclidean benchmarks, where an
iteration is milliseconds, it lands within 2 % of the optimum.

### SOM as a one-second warm start

[`SOM`][skroute.SOM] is the only solver that never reads the cost matrix during its search (it
pulls a ring of neurons towards the coordinates), and it is not budget-aware, so its tour is
split into days afterwards. On its own that gives 1 790.3 in 0.8 s. Follow it with one
[`LocalSearch`][skroute.LocalSearch] descent and the pair gives **1 601.2 in 0.6 s** — a
whisker behind two minutes of `TabuSearch` with `n_candidates=20` (1 600.2) and ahead of the
default tabu search (1 612.8) and of the memetic genetic algorithm (1 607.0), for a
two-hundredth of the budget. That is its role on a real instance: a geometric first tour good
enough to hand to a budget-aware search.

### The construction heuristics as starting points

The five construction rows answer in milliseconds and every one of them needs 16 days:
[`Insertion`][skroute.Insertion] at 1 709.6 (`cheapest`), 1 720.4 (`nearest`) and 1 720.7
(`farthest`), [`NearestNeighbour`][skroute.NearestNeighbour] at 1 843.6 and
[`NRBS`][skroute.NRBS] at 1 845.3. Two details are worth taking away. First, `cheapest` beats
`farthest` here, the reverse of the Euclidean benchmarks — a reminder that insertion strategies
are a property of the metric, and on asymmetric road times you should try all three. Second, a
better start is not a better search: `IteratedLocalSearch` started from `Insertion(farthest)`
finished at 1 565.1 while the same search from the default nearest-neighbour start finished at
1 554.7. What the constructions are really for is what goes on top of them: one
[`LocalSearch`][skroute.LocalSearch] descent from `Insertion(farthest)` gives 1 619.2 in 0.4 s,
while [`TwoOpt`][skroute.TwoOpt] (1 743.7) or [`OrOpt`][skroute.OrOpt] (1 640.3) alone show
which half of that descent does the work — the relocations, again.

### Clarke-Wright, which optimises something else

[`ClarkeWright`][skroute.ClarkeWright] is the one budget-aware construction — its savings merges
refuse a trip that would not fit the day — and the one solver that **requires a symmetric
matrix**, which this instance is not (up to ten minutes between the two directions of a pair). Run
on `(T + T.T) / 2` and re-priced on the real matrix it returns **17 days and 1 180.4 minutes**:
the least driving of the whole campaign, only 9.5 % above the proven tour bound, and the
tenth-worst objective of the 27, because two extra days cost 960. It is a good answer to a different
question — "keep every trip short" — and `extra_cost=480` is the parameter that says a day is
expensive. Take it as the warning that a solver aware of the *budget* is not automatically aware
of the *charge per trip*.

### The exact solvers: one for the bound, one for the days

[`MILP`][skroute.MILP] refuses the multi-trip instance by design and proves the plain tour
instead: 183 asymmetric nodes in 559 s, which is well beyond the "about 60 asymmetric nodes
within a minute" rule of thumb of [choosing a solver](choosing_a_solver.md) — the rule is about a
minute, not about feasibility, and here nine minutes bought a proof. That proof is the only
absolute statement on this page. [`HeldKarp`][skroute.HeldKarp] is capped at 20 nodes and is
therefore useless on the instance and exactly right *inside a day*: 13 stops plus the office is
14 nodes, three milliseconds of dynamic programming, an optimal day.
[`BruteForce`][skroute.BruteForce] is the only solver that certifies a multi-trip optimum, at 11
nodes — the office plus ten stops, which is the size of day 11 of this plan and of no other.

## The exact hybrid: a useful negative result

The campaign closed with something the library does not offer as a solver, composed from two
things it does offer: **re-solve every day exactly with [`HeldKarp`][skroute.HeldKarp]** (at most
13 stops plus the office, so at most 14 nodes), then **try every relocation and every swap of
stops between two days**, re-solving both affected days exactly and accepting an exchange only
when it improves the driving and both days still fit in 480 minutes.

Applied to the 15-day plans it changed **nothing on the routes**: every day of the ILS plans was
already optimal for its own set of restaurants, and the pass over the 1 532.8 plan converged in
56 s with no improvement at all. On an earlier 15-day plan it found the routes optimal too and
gained under a minute from the exchanges (1 541.5 to 1 540.6, 115 s, two passes). No day could be
emptied by relocating its stops elsewhere, either.

That is a negative result and it is the most useful one of the study, because it splits the
remaining doubt in two and closes one half. The per-day sequencing is **proved optimal**; the
driving of the plan is entirely determined by *which restaurants share a day*. So an
`IteratedLocalSearch` polish is not leaving minutes on the table inside the days, and anything
better than 1 532.8 — or a 14-day plan — has to come from a different assignment of restaurants
to days, not from a better route through the same ones.

```python
>>> import json
>>> from pathlib import Path
>>> results = Path("examples/technician_madrid_study/results")
>>> lines = (results / "results.jsonl").read_text(encoding="utf-8").splitlines()  # doctest: +SKIP
>>> rows = [json.loads(line) for line in lines]  # doctest: +SKIP
>>> ranked = sorted((r for r in rows if "objective" in r), key=lambda r: r["objective"])  # doctest: +SKIP
>>> len(rows), len(ranked)                       # one row raised: MILP  # doctest: +SKIP
(60, 59)
>>> best = ranked[0]  # doctest: +SKIP
>>> best["days"], best["driving_min"], best["family"], best["wall_s"]  # doctest: +SKIP
(15, 1532.8, 'hybrid', 55.9)
>>> [sum(1 for r in ranked if r["days"] == d) for d in (14, 15, 16, 17)]  # doctest: +SKIP
[0, 23, 35, 1]

```

## Reproducing it

The harness lives in `examples/technician_madrid_study/` (its README lists the flags); it reads
the committed CSVs, so nothing here touches the network. The wall clock is what the campaign
cost on ten cores:

```bash
python examples/technician_madrid_study/roster.py        # 28 configurations, 120 s each: ~23 min of process time
python examples/technician_madrid_study/long_runs.py     # 13 configurations, 1 h each: ~4 h wall clock
python examples/technician_madrid_study/bounds.py        # the MILP tour bound + its metric closure: ~10 min each
python examples/technician_madrid_study/exact_polish.py  # Held-Karp days + inter-day exchanges: ~1 min
python examples/technician_madrid_study/report.py        # rebuild the tables of this page from results/
```

!!! warning "Processes, not threads"
    Every `MultiStart` of the campaign runs with `prefer="processes"`. The multi-trip kernels hold
    the interpreter lock while they price a move, so with the default threads the restarts
    serialise: the same eight restarts took **138 s against 36 s**. Threads are the right default
    for the plain symmetric TSP, where the hot loops release the lock; on a multi-trip instance
    they cost you the cores.

Two more notes for a rerun. The long campaign ran nine single-process configurations at once on
ten cores, which is deliberate: eight `MultiStart` processes plus nine searches would oversubscribe
the machine and every wall-clock budget would measure contention instead of the algorithm. And the
numbers will not repeat exactly — the budgets are wall clock, so a faster machine does more
iterations with the same seed. The days should repeat; the driving will land within a few minutes.

## What we would do next

**Settle 14 against 15 exactly.** Every heuristic in the campaign says 15 and the counting bound
says 14 is not impossible, and no amount of extra search will close that gap — the exact hybrid
showed the doubt lives entirely in the assignment of restaurants to days. The way to close it is
to model the assignment itself: ask whether a **14-day** partition exists at all, as a
set-partitioning problem over candidate day-routes (cover the 182 restaurants with exactly 14
routes, each priced by an exact day solve, each within 480 minutes) or as a compact integer
programme with a day index per restaurant and a per-day time constraint. A feasibility answer
either produces a 14-day plan or proves 15 optimal. That model is outside what
[`MILP`][skroute.MILP] does today — it optimises the plain tour and says so — so it is a candidate
for a future release rather than a recipe you can run now.

**Do not expect much more from the driving.** Within 15 days the margin is essentially gone: the
best four configurations of the long campaign lie between 1 532.8 and 1 541.4, an hour of
`IteratedLocalSearch` continuing from a 1 537.1 plan bought 4.3 minutes over 9 973 iterations, and
the exact per-day re-solve found zero. If the technician needs a materially shorter plan, the lever
is not the solver: it is the model — a second technician (`people=2`), a longer day, per-restaurant
service times, or starting the day from home instead of the office.
