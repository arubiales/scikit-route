# Choosing a solver

scikit-route ships sixteen solvers behind one interface: every one of them takes the
same `fit(X, ...)` arguments (see [the problem model](problem_model.md)) and fills the
same fitted attributes (`tour_`, `route_`, `trips_`, `cost_`, ...). What differs is
*what each solver promises*: a certificate of optimality, a tour in microseconds, a
search that sees the multi-trip budget, or a result you can reproduce bit for bit. This
page is the decision guide. It starts with the capability table, walks through the five
families with one runnable example per solver, and closes with a table of "if you want X,
use Y".

If you only want an answer: [`IteratedLocalSearch`][skroute.IteratedLocalSearch] is the
recommended default. It reaches the published optimum on the small bundled instances,
stays within a few percent on the 1 000-node ones, accepts asymmetric matrices and the
multi-trip objective, and its result is reproducible from `random_state` alone.

## The capability table

The table below is generated at build time from the solvers' own
[`RouterTags`][skroute.RouterTags], so it cannot drift from the code. The same table sits
in the README.

{% include-markdown "user_guide/_capability_table.md" %}

How to read the columns:

| Column | Meaning |
|---|---|
| **Kind** | the family: `exact` certifies the optimum, `construction` builds a tour once, `local search` improves a tour by moves, `metaheuristic` escapes local optima, `ensemble` restarts another solver |
| **Exact** | `is_optimal_` is set and the tour is provably optimal for the objective the solver accepts |
| **Stochastic** | the solver has a `random_state` parameter; it can be wrapped in [`MultiStart`][skroute.MultiStart] |
| **Multi-trip aware** | the *search itself* prices every move with the multi-trip objective. A solver that is not budget-aware still returns feasible trips (the base class decodes and prices its tour) but warns that its search ignored `max_time_work`; an exact solver that is not budget-aware refuses the budget outright |
| **Asymmetric** | accepts a cost matrix with `C[i, j] != C[j, i]` (an asymmetric TSP). Only `ClarkeWright` refuses one |
| **Needs coordinates** | `fit(X, coords=...)` is mandatory (`SOM`) |
| **Max nodes** | a hard cap on `n`, raised only by passing a larger `max_nodes` |

[`MultiStart`][skroute.MultiStart] is not in the table because it needs an estimator:
its capabilities are those of the solver it wraps.

The examples below share two instances from [`skroute.datasets`][skroute.datasets.load_tsp]:
Western Sahara (`wi29`, 29 cities, optimum 27603) and Djibouti (`dj38`, 38 cities,
optimum 6656). Their `distance_matrix()` uses the TSPLIB `EUC_2D` rounding, so costs are
integers and the published optima apply.

```python
>>> import numpy as np
>>> from skroute.datasets import load_tsp
>>> wi, dj = load_tsp("wi29"), load_tsp("dj38")
>>> C_wi, C_dj = wi.distance_matrix(), dj.distance_matrix()
>>> wi.optimal_tour_length, dj.optimal_tour_length, int(wi.depot)
(27603, 6656, 1)

```

!!! warning "Loader matrices carry no labels"
    `wi.distance_matrix()` and the `cost`/`time` matrices of the cost loaders are plain
    `float64` arrays. Pass `labels=wi.labels` to `fit`, otherwise `tour_` and `route_`
    come back as positions `0..n-1` instead of the file's ids, and `depot=b.depot` is
    rejected as "not a label of X". Every example on this page does so.

## Exact solvers

The three exact solvers set `is_optimal_` and are the yardstick of the library: every
heuristic is tested against `BruteForce` on small instances, and `MILP` proves the
published optima of the small Waterloo instances. Details: [Exact solvers](../api/exact.md).

### BruteForce

[`BruteForce`][skroute.BruteForce] enumerates the $(n-1)!$ giant tours in lexicographic
order and prices each one with the problem's own decoder. It is therefore the only solver
that certifies the **multi-trip** optimum: under `split="greedy"` it is exact over
greedy-decoded giant tours, under `split="optimal"` it is exact for the
distance-constrained multi-trip problem. Its one parameter is `max_nodes` (default 11):
$10!$ is 3.6 million tours, a few hundredths of a second; every extra node multiplies the
work by $n$.

The 4-node example of the README, hand-checkable (the single trip would need 5 h > 4 h, so
the tour breaks after node 3; trips cost 18 and 20, plus one extra-trip charge of 3):

```python
>>> from skroute import BruteForce
>>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
>>> hours = {1: {1: 0, 2: 1, 3: 2, 4: 2}, 2: {1: 1, 2: 0, 3: 1, 4: 2},
...          3: {1: 2, 2: 1, 3: 0, 4: 1}, 4: {1: 2, 2: 2, 3: 1, 4: 0}}
>>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> bf.route_.tolist(), bf.cost_, bf.n_trips_, bf.is_optimal_
([1, 2, 3, 1, 4, 1], 41.0, 2, True)
>>> bf.trip_times_.tolist()
[4.0, 4.0]
>>> BruteForce().fit(C_wi)
Traceback (most recent call last):
    ...
ValueError: BruteForce handles at most 11 nodes, got 29; raise max_nodes only if you accept the time/memory cost

```

Pick it when $n \le 11$ and you need a certificate, in particular for the multi-trip
objective, where nothing else certifies anything.

### HeldKarp

[`HeldKarp`][skroute.HeldKarp] is the Bellman–Held–Karp dynamic programme over subsets:
$O(2^{n-1} n^2)$ time and $O(2^{n-1} n)$ memory (about 90 MB at the default
`max_nodes=20`). It reads every arc directionally, so asymmetric matrices are solved
exactly, but it certifies the **plain tour only**: under a budget it raises instead of
returning a misleading "optimal" result.

```python
>>> from skroute import HeldKarp, MILP
>>> sub = load_tsp("wi29", n_nodes=12)          # a fixed 12-city subsample (random_state=2019 by default)
>>> hk = HeldKarp().fit(sub.distance_matrix(), labels=sub.labels)
>>> milp = MILP().fit(sub.distance_matrix(), labels=sub.labels)
>>> hk.cost_ == milp.cost_, hk.is_optimal_, hk.n_nodes_, sub.optimal_tour_length is None
(True, True, 12, True)
>>> HeldKarp().fit(cost, time_matrix=hours, max_time_work=4.0)
Traceback (most recent call last):
    ...
ValueError: HeldKarp optimises the plain tour and cannot certify a multi-trip optimum; use BruteForce (n <= 11) or a heuristic solver

```

In practice `MILP` solves the same sizes instantly, so `HeldKarp` is mostly a reference
implementation; it is the right choice when you want an exact answer with no external
solver in the loop.

### MILP

[`MILP`][skroute.MILP] is the Dantzig–Fulkerson–Johnson integer programme solved by
HiGHS through `scipy.optimize.milp`: binary edge variables (symmetric) or arc variables
(asymmetric) with degree constraints, and subtour-elimination cuts added *lazily* — each
solve is split into connected components and one cut per component is appended until
the solution is a single Hamiltonian cycle. That final solution is a **proven optimum**.

Key parameters: `time_limit` (default 60 s; `None` runs until proven), `max_nodes`
(default 300; realistic sizes within a minute are about 200 symmetric or 60 asymmetric
nodes) and `mip_rel_gap` (0.0 by default: every relaxation is solved to proven
optimality). Beyond `is_optimal_` it exposes `lower_bound_`, `gap_`, `n_solves_` and
`n_cuts_`. When the budget runs out the fit still returns a valid tour, with
`is_optimal_ == False` and the gap to the best bound seen.

```python
>>> milp = MILP().fit(C_wi, labels=wi.labels)
>>> int(milp.cost_) == wi.optimal_tour_length == 27603, milp.is_optimal_, milp.gap_
(True, True, 0.0)
>>> milp.lower_bound_ == milp.cost_ and milp.n_solves_ >= 1 and milp.n_cuts_ >= 0
True
>>> hurried = MILP(time_limit=1e-6).fit(C_wi, labels=wi.labels)   # far too little time: a tour, no certificate
>>> hurried.is_optimal_, hurried.cost_ >= milp.cost_, 0.0 < hurried.gap_ <= 1.0
(False, True, True)

```

Pick it whenever you need a certificate above 11 nodes (plain tour, symmetric or
asymmetric). `docs/benchmarks.md` records `qa194` (194 cities) proved in about 40 s.

## Construction heuristics

Construction heuristics build a tour from scratch, deterministically, in
$O(n^2)$ to $O(n^2 \log n)$: no `random_state`, no `history_`. Their results range from
the optimum itself (farthest insertion on dj38) to 46 % above it (nearest neighbour on
the same instance; see the measured table below), so they are baselines and
**warm starts** (`init=est.tour_`) rather than answers. Details:
[Construction heuristics](../api/construction.md).

### NearestNeighbour

[`NearestNeighbour`][skroute.NearestNeighbour] walks from the depot to the closest
unvisited node until every node is visited. It has no parameters; it is the default
`init` of every iterative solver. Expect 25–45 % above the optimum on Euclidean instances
(`docs/benchmarks.md`: 31.8 % on wi29, 46.4 % on dj38).

### Insertion

[`Insertion`][skroute.Insertion] grows the tour one node at a time, inserting each at its
cheapest position. `strategy` picks the next node: `"farthest"` (default; the node
farthest from the partial tour — fixes the global shape first and is the best of the
three on Euclidean data), `"cheapest"` (smallest insertion cost) or `"nearest"`. Every
insertion cost is read in driving direction, so it is exact arithmetic on asymmetric
matrices too. There are no `FarthestInsertion`/`CheapestInsertion` classes.

```python
>>> from skroute import Insertion, NearestNeighbour
>>> far = Insertion().fit(C_dj, labels=dj.labels)                    # strategy="farthest"
>>> cheap = Insertion(strategy="cheapest").fit(C_dj, labels=dj.labels)
>>> far.cost_ <= 1.25 * dj.optimal_tour_length, far.cost_ <= cheap.cost_
(True, True)
>>> nn = NearestNeighbour().fit(C_dj, labels=dj.labels)
>>> far.cost_ <= nn.cost_ <= 1.5 * dj.optimal_tour_length
True

```

### ClarkeWright

[`ClarkeWright`][skroute.ClarkeWright] is the parallel savings heuristic: every customer
starts as its own out-and-back trip, the savings
$s_{ij} = C_{di} + C_{jd} - \lambda\, C_{ij}$ are sorted in descending order and two
trips are merged at their endpoints whenever the merged trip is still feasible. The
parameter `shape` is $\lambda$ (1.0 is the classical rule). Two things make it special
among construction heuristics: it is **budget-aware** — under `max_time_work` a merge is
refused when the merged closed trip would not fit, so the trips it builds are the ones it
optimised — and it requires a **symmetric** cost matrix.

```python
>>> from skroute import ClarkeWright
>>> cw = ClarkeWright().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> cw.route_.tolist(), cw.cost_, cw.n_trips_        # the BruteForce optimum of the 4-node example
([1, 2, 3, 1, 4, 1], 41.0, 2)
>>> A = np.array([[0, 1, 9, 9], [9, 0, 1, 9], [9, 9, 0, 1], [1, 9, 9, 0]], dtype=float)
>>> ClarkeWright().fit(A)
Traceback (most recent call last):
    ...
ValueError: ClarkeWright requires a symmetric cost matrix
>>> HeldKarp().fit(A).tour_.tolist(), HeldKarp().fit(A).cost_   # the reverse direction would cost 36
([0, 1, 2, 3], 4.0)

```

Pick it as the construction heuristic for a multi-trip instance (the other three build a
plain tour and let the decoder split it).

### NRBS

[`NRBS`][skroute.NRBS] (Node Ranking Based on Stats) is the construction heuristic of
scikit-route 1.0 (2020), ported faithfully: every node gets a priority from the mean and
standard deviation of its row of `C`, and two passes over the priority order link each
node to its highest-scoring candidate without closing a premature cycle. Its five
exponents (`mean_priority`, `std_priority`, `mean_connection`, `std_connection`,
`distance_weight`) default to `1.0` and must be `>= 0`. It is kept for continuity with
1.0 results (Barcelona with every exponent at 0.5 reproduces the pinned 1.0 tour); for a
new project prefer `Insertion` or `ClarkeWright`.

### The budget warning

Under `max_time_work`, `NearestNeighbour`, `Insertion` and `NRBS` build a plain tour and
hand it to the decoder. The trips you get back fit the budget and are priced correctly,
but the search did not see the objective, and `fit` says so with a `UserWarning`:

```python
>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     nn = NearestNeighbour().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> print([str(w.message) for w in caught if "max_time_work" in str(w.message)][0])
NearestNeighbour ignores max_time_work during its search; the result is still split into trips and priced under the multi-trip objective
>>> nn.route_.tolist(), nn.cost_, bool(np.all(nn.trip_times_ <= 4.0))
([1, 2, 3, 1, 4, 1], 41.0, True)

```

## Local search

The local searches start from `init` (the nearest-neighbour tour by default, or the
`tour_`/`route_` of another solver) and apply moves while they shorten the tour. All four
are budget-aware and accept asymmetric matrices. On a symmetric plain TSP they use
$O(1)$ move deltas, candidate lists and don't-look bits and scale to tens of thousands of
nodes; asymmetric matrices and the multi-trip objective take the full-evaluation path,
$O(n)$ per candidate move. Details: [Local search](../api/local_search.md).

### TwoOpt, OrOpt and LocalSearch

[`TwoOpt`][skroute.TwoOpt] reverses a segment of the tour when that removes a crossing;
[`OrOpt`][skroute.OrOpt] relocates a segment of one to `max_segment` (default 3)
consecutive nodes; [`LocalSearch`][skroute.LocalSearch] alternates the descents listed in
`moves` (default `("two_opt", "or_opt")`) until neither improves. The three are
**deterministic** (no `random_state`; `init="random"` is refused) and share three knobs:

- `n_candidates` (default 10): each node only tries moves towards its $k$ nearest
  neighbours. `None` scans the full neighbourhood — a better local optimum at small $n$
  for a higher cost per pass.
- `max_passes` (default 50): the maximum number of outer iterations.
- `first_improvement` (2-opt only): apply the first improving move of a node, or the best.

**Iteration accounting.** One outer iteration is one pass of each listed descent over the
active nodes; `history_[k]` is the cost *after* iteration $k$ (non-increasing, never
above the cost of `init`), `n_iter_ == len(history_)` and `stop_reason_` is
`"converged"` (a pass that started with every node active changed nothing) or
`"max_iter"`. These are the only two values the descents emit.

```python
>>> from skroute import TwoOpt, OrOpt, LocalSearch
>>> two, oo, ls = (est.fit(C_wi, labels=wi.labels) for est in (TwoOpt(), OrOpt(), LocalSearch()))
>>> ls.cost_ <= 1.12 * wi.optimal_tour_length and ls.cost_ <= min(two.cost_, oo.cost_)   # the latter: on this instance
True
>>> two.stop_reason_, two.n_iter_ == len(two.history_), bool(np.all(np.diff(two.history_) <= 0))
('converged', True, True)
>>> float(two.history_[0]) <= NearestNeighbour().fit(C_wi, labels=wi.labels).cost_   # never worse than init
True
>>> TwoOpt(n_candidates=None).fit(C_wi, labels=wi.labels).cost_ <= two.cost_   # full neighbourhood
True

```

Pick `LocalSearch` when you need a decent tour in milliseconds on a large instance, or as
the polish of a tour that came from elsewhere. `TwoOpt` and `OrOpt` alone are mainly
useful to study the moves; Or-opt by itself leaves crossings that 2-opt removes.

### IteratedLocalSearch — the recommended default

[`IteratedLocalSearch`][skroute.IteratedLocalSearch] descends to a local optimum, then
repeats: *kick* the incumbent, descend again, accept the new local optimum when it is
better, and remember the best tour seen. The kick is a double bridge
(`A B C D -> A C B D`), which no 2-opt or Or-opt move can undo, so every iteration
explores a genuinely new basin.

| Parameter | Default | Role |
|---|---|---|
| `n_iter` | 1000 | maximum kick + descent iterations (`"max_iter"`) |
| `patience` | 100 | stop after this many iterations without improving the best tour (`"patience"`) |
| `perturbation_strength` | 1 | kicks per iteration |
| `acceptance` | `"better"` | `"metropolis"` also accepts a worse local optimum with probability $e^{-\Delta/T}$ at fixed `temperature` (default 0.5 % of the initial cost) |
| `local_search` | `("two_opt", "or_opt")` | the descents run after every kick; `None` leaves a random walk of kicks |
| `n_candidates`, `init`, `time_limit`, `random_state`, `verbose` | | as everywhere |

```python
>>> from skroute import IteratedLocalSearch
>>> ils = IteratedLocalSearch(random_state=0).fit(C_wi, labels=wi.labels)
>>> ils.cost_ <= 1.03 * wi.optimal_tour_length        # the fast-tier tolerance of the test-suite
True
>>> ils.stop_reason_ in {"patience", "max_iter"}, ils.n_iter_ == len(ils.history_)
(True, True)
>>> bool(np.all(np.diff(ils.history_) <= 0)) and float(ils.history_[-1]) == ils.cost_   # best-so-far
True

```

Pick it first. `docs/benchmarks.md` measures it at the optimum on wi29 and dj38, 0.27 %
above on qa194 and 1.19 % on lu980 (980 cities, 0.25 s), the best quality-per-second of
the library. On asymmetric or multi-trip problems the confirming sweep after every kick
costs $O(n^2 k)$, so keep to a few hundred nodes at the default `n_iter` or set
`time_limit`.

## Metaheuristics

The metaheuristics trade time for the ability to escape local optima. All of them are
stochastic (seed them with `random_state`), iterative (`history_` is the best-so-far cost
per outer iteration) and — except `SOM` — budget-aware.

### SimulatedAnnealing

[`SimulatedAnnealing`][skroute.SimulatedAnnealing] walks the space of giant tours with
random proposals drawn from `moves` (`("two_opt", "or_opt", "swap")` by default): a
downhill move is always accepted, an uphill one with probability $e^{-\Delta/T}$. One
outer iteration is one temperature level of `n_moves` proposals (default $10n$), after
which $T \leftarrow \alpha T$ (`alpha=0.995`). `t0="auto"` prices 1 000 random
proposals on the initial tour and sets $T_0$ so that the median uphill move is accepted
with probability one half; `t_min="auto"` is $10^{-4} T_0$, which makes the number of
levels $\lceil \ln 10^4 / -\ln \alpha \rceil = 1838$ whatever the instance. The best
tour is kept in its own buffer, so `history_[-1]` equals `cost_` exactly.

`patience` is `None` by default on purpose: with a calibrated $T_0$ the hot phase is
non-improving by design (the current cost first drops below the nearest-neighbour start
after several hundred levels), so a small patience applied from level 0 would return the
start tour. When you set it, the count starts only once the current cost has first
fallen below the initial cost.

```python
>>> from skroute import SimulatedAnnealing
>>> sa = SimulatedAnnealing(random_state=0).fit(C_wi, labels=wi.labels)
>>> sa.cost_ <= 1.03 * wi.optimal_tour_length, sa.t0_ > 0
(True, True)
>>> sa.stop_reason_, sa.n_iter_                    # T fell below t_min after ln(1e4) / -ln(0.995) levels
('converged', 1838)
>>> bool(np.all(np.diff(sa.history_) <= 0)) and float(sa.history_[-1]) == sa.cost_
True

```

Pick it for multi-trip instances where you want to trade time for quality: its random
proposals are priced with the full objective, and `MultiStart(SimulatedAnnealing())`
uses every core (`docs/benchmarks.md`: at the optimum on wi29/dj38, 2.4 % on qa194,
7.0 % on lu980).

### TabuSearch

[`TabuSearch`][skroute.TabuSearch] applies, every iteration, the *best admissible* move
of a candidate neighbourhood (2-opt reversals and Or-opt relocations that join a node to
one of its `n_candidates` nearest neighbours) — even when it worsens the tour — and
forbids re-adding the edges that move removed for `tenure` iterations. A tabu move is
still allowed when it beats the best tour so far (aspiration). `tenure="auto"` redraws
the tenure every iteration uniformly from $[\lceil\sqrt{n}\rceil, 2\lceil\sqrt{n}\rceil]$
(Taillard's robust tabu search); an `int` fixes it. `n_iter` (1000) and `patience` (200)
bound the run. The tabu attributes live in an `int32` `(n, n)` matrix, which puts the
practical ceiling at about 5 000 nodes (100 MB).

```python
>>> from skroute import TabuSearch
>>> ts = TabuSearch(random_state=0).fit(C_dj, labels=dj.labels)
>>> ts.cost_ <= 1.08 * dj.optimal_tour_length, ts.stop_reason_ in {"max_iter", "patience", "time_limit"}
(True, True)
>>> ts.n_iter_ == len(ts.history_) and float(ts.history_[-1]) == ts.cost_
True

```

Pick it when simulated annealing stalls on a structured instance: its systematic
best-move scan is deterministic given the tenure draws and behaves well on asymmetric
matrices (where a reversal makes every arc of the reversed span tabu).

### Genetic

[`Genetic`][skroute.Genetic] evolves a population of `pop_size` giant tours for up to
`n_generations`: tournament selection (`tournament_size`), a real permutation crossover
(`crossover="ox"` order crossover or `"pmx"` partially mapped, with probability
`p_crossover`), a mutation (`"inversion"`, `"swap"` or `"insertion"`, probability
`p_mutation`), `n_elite` parents copied unchanged, and exact-duplicate children mutated
once more. The fitness is the problem objective itself, so a budget steers the
population directly. The plain GA is the weakest metaheuristic of the library on large
instances (`docs/benchmarks.md`: 9 % above the optimum on dj38, 16 % on qa194); the
**memetic** configuration `local_search=("two_opt",)`, which polishes every child with a
2-opt descent, is competitive with `IteratedLocalSearch` (0.32 % on qa194, 0.67 % on
lu980) at a much higher cost per generation.

```python
>>> from skroute import Genetic
>>> ga = Genetic(random_state=0).fit(C_dj, labels=dj.labels)
>>> memetic = Genetic(local_search=("two_opt",), random_state=0).fit(C_dj, labels=dj.labels)
>>> ga.cost_ <= 1.15 * dj.optimal_tour_length, memetic.cost_ <= 1.05 * dj.optimal_tour_length
(True, True)
>>> ga.stop_reason_ in {"max_iter", "patience", "time_limit"}, memetic.n_iter_ == len(memetic.history_)
(True, True)

```

Pick the memetic form when you want a population-based search (for instance to harvest
several good tours from `estimators_` of a `MultiStart`); pick the plain form only for
teaching or for 1.0 continuity.

### AntColony

[`AntColony`][skroute.AntColony] is a MAX–MIN Ant System: every iteration `n_ants` ants
(default $\min(n, 50)$) build tours from the depot by a roulette wheel over the unvisited
nodes of their `n_candidates` (20) nearest-neighbour list, weighting node $j$ by
$\tau_{ij}^{\alpha} (1/C_{ij})^{\beta}$; each tour is polished by `local_search`
(`("two_opt",)` by default) and priced with the problem objective; the pheromone
evaporates (`rho=0.02`) and the iteration-best ant — the best-so-far ant every fifth
iteration — deposits $1/\text{cost}$ on its arcs, with the trail clipped to
$[\tau_{\max}/2n, \tau_{\max}]$ so the colony never stagnates. Three `(n, n)` matrices
besides the cost matrix bound the practical size to a few thousand nodes; the trail is
exposed as `pheromone_`.

```python
>>> from skroute import AntColony
>>> aco = AntColony(random_state=0).fit(C_wi, labels=wi.labels)
>>> aco.cost_ <= 1.08 * wi.optimal_tour_length, aco.pheromone_.shape == (29, 29)
(True, True)
>>> aco.stop_reason_ in {"max_iter", "patience", "time_limit"} and aco.n_iter_ == len(aco.history_)
True

```

Pick it when you want diversity: `docs/benchmarks.md` measures it at 1.7 % on qa194 and
5.7 % on lu980 with the defaults, and the pheromone matrix is a readable summary of which
arcs the colony believes in.

### SOM

[`SOM`][skroute.SOM] is the neural approach: a one-dimensional Kohonen ring of `n_units`
neurons (default $8n$) is pulled towards the cities one random city at a time, with a
Gaussian neighbourhood of standard deviation `radius` (default `n_units / 10`) that
shrinks by `radius_decay` while `learning_rate` shrinks by `lr_decay`; reading the ring
in order gives the tour. It is the one solver that **needs coordinates**
(`fit(X, coords=...)`), it never reads the cost matrix during its search (the matrix
prices the tour decoded after every epoch, which is what `history_` records), and it is
**not budget-aware**: under `max_time_work` it warns and its tour is split afterwards.
`n_iter` counts samples, grouped in epochs of `n_iter // 100`; the run stops by
`"converged"` when the radius falls below one ring position or the learning rate below
$10^{-3}$.

```python
>>> from skroute import SOM
>>> som = SOM(random_state=0).fit(C_wi, coords=wi.coords, labels=wi.labels)
>>> som.cost_ <= 1.15 * wi.optimal_tour_length, som.stop_reason_ in {"converged", "max_iter"}
(True, True)
>>> som.n_iter_ == len(som.history_) and som.n_samples_ <= som.n_iter
True
>>> SOM().fit(C_wi)
Traceback (most recent call last):
    ...
ValueError: SOM needs node coordinates: fit(X, coords=...)

```

Pick it when you have planar coordinates and want a geometric, visualisable search; for
quality per second the local searches win.

## Ensembles

[`MultiStart`][skroute.MultiStart] runs any stochastic solver from `n_restarts`
independent seeds — in parallel with `n_jobs`, threads by default — and keeps the best
tour; the result is identical for any `n_jobs`. [`EnsembleGenetic`][skroute.EnsembleGenetic]
and [`EnsembleSimulatedAnnealing`][skroute.EnsembleSimulatedAnnealing] are the 1.0
ensembles kept as thin wrappers over it. The
[warm starts and ensembles](warm_starts_and_ensembles.md) page covers them in depth.

```python
>>> from skroute import MultiStart
>>> ms = MultiStart(SimulatedAnnealing(), n_restarts=4, random_state=0).fit(C_dj, labels=dj.labels)
>>> ms.cost_ <= 1.03 * dj.optimal_tour_length, len(ms.estimators_), ms.cost_ == float(ms.costs_.min())
(True, 4, True)

```

## Size ceilings

Every solver of 2.0 works on a dense `float64` `(n, n)` matrix, so the ceiling shared by
all of them is memory: about **20 000 nodes** (3.2 GB). The four bundled instances above
that size can be read and subsampled (`load_tsp(name, n_nodes=5000)`) but not solved
whole. Within that ceiling:

| Solver | Practical limit | Why |
|---|---|---|
| `BruteForce` | 11 nodes | $(n-1)!$ tours |
| `HeldKarp` | 20 nodes | $2^{n-1}$ subsets, ~90 MB of tables |
| `MILP` | ~200 symmetric / ~60 asymmetric nodes within a minute; hard cap 300 | integer programme with lazy cuts |
| `TabuSearch` | ~5 000 nodes | `int32` tabu matrix (100 MB) |
| `AntColony` | a few thousand nodes | three extra `(n, n)` matrices |
| `SimulatedAnnealing`, `IteratedLocalSearch`, `TabuSearch`, `Genetic` under a budget or on an asymmetric matrix | a few hundred to ~2 000 nodes | no $O(1)$ delta exists: every move is re-priced in $O(n)$ |
| `LocalSearch`, `TwoOpt`, `OrOpt`, `IteratedLocalSearch(time_limit=...)` on a symmetric plain TSP | the memory ceiling | $O(1)$ deltas, candidate lists, don't-look bits |

`fit_time_` records the seconds spent in the search, so you can measure your own
instance before scaling up.

## Common pitfalls

Four mistakes that every solver catches for you, with the messages you will see. The
first one is the reason `time_matrix` is keyword-only: 1.0 took the two matrices
positionally, so a migrated call that kept the old order would swap them silently and
price hours as euros.

```python
>>> from skroute import SimulatedAnnealing
>>> from skroute.datasets import load_barcelona
>>> bcn = load_barcelona()                     # cost in EUR, time in hours, depot 10000007
>>> SimulatedAnnealing().fit(bcn.cost, bcn.time, max_time_work=8.0)          # positional time matrix
Traceback (most recent call last):
    ...
TypeError: ...fit() takes 2 positional arguments but 3 ...
>>> SimulatedAnnealing().fit(bcn.cost, labels=bcn.labels, max_time_work=8.0)  # a budget needs durations
Traceback (most recent call last):
    ...
ValueError: max_time_work given but no time_matrix; pass time_matrix=X to use the cost matrix as durations
>>> SimulatedAnnealing().fit(bcn.cost, depot=bcn.depot)                       # loader matrix without labels=
Traceback (most recent call last):
    ...
ValueError: depot 10000007 is not a label of X
>>> BruteForce().fit(cost, time_matrix=hours, max_time_work=3.0)              # nodes 3 and 4 need 4 h round trips
Traceback (most recent call last):
    ...
skroute.exceptions.InfeasibleProblemError: nodes [3, 4] cannot be served in one trip: depot round trip exceeds max_time_work=3.0
>>> sa = SimulatedAnnealing(random_state=0).fit(bcn.cost, time_matrix=bcn.time, labels=bcn.labels,
...                                             depot=bcn.depot, max_time_work=8.0, extra_cost=12.83, people=2)
>>> int(sa.depot_), bool(np.all(sa.trip_times_ <= 8.0)), sa.n_trips_ == len(sa.trips_)
(10000007, True, True)

```

[`InfeasibleProblemError`][skroute.exceptions.InfeasibleProblemError] is a `ValueError`
raised before any search runs: a node whose round trip from the depot exceeds the budget
can never be served, whatever the solver. The multi-trip objective itself is the subject
of [multi-trip routing](multi_trip.md).

## Measured quality

The gaps below are copied from [`docs/benchmarks.md`](../benchmarks.md) (gap =
`cost_ / optimum - 1`, defaults, `random_state=0`, one fit per cell on the machine
recorded in that page's header); the tolerances the test-suite asserts on every run live in
`tests/tolerances.py` and are looser on purpose, so that a tie-break on another machine is
not a failure. A different machine may move a stochastic solver's gap by a tie-break,
never below the optimum.

| Solver | wi29 (29) | dj38 (38) | qa194 (194) | lu980 (980) | Time on lu980 |
|---|---:|---:|---:|---:|---:|
| `MILP` | 0.00 % | 0.00 % | 0.00 % (39.6 s) | capped | — |
| `IteratedLocalSearch` | 0.00 % | 0.00 % | 0.27 % | 1.19 % | 0.25 s |
| `Genetic(local_search=("two_opt",))` | 0.00 % | 0.00 % | 0.32 % | 0.67 % | 0.81 s |
| `AntColony` | 0.00 % | 0.00 % | 1.71 % | 5.66 % | 1.14 s |
| `SimulatedAnnealing` | 0.00 % | 0.00 % | 2.44 % | 7.02 % | 1.02 s |
| `TabuSearch` | 0.00 % | 0.00 % | 5.34 % | 4.26 % | 1.36 s |
| `LocalSearch` | 0.00 % | 2.28 % | 6.28 % | 4.74 % | 0.02 s |
| `SOM` | 0.00 % | 0.06 % | 6.44 % | 10.22 % | 2.77 s |
| `TwoOpt` | 3.29 % | 0.06 % | 7.74 % | 6.15 % | 0.01 s |
| `Insertion` (farthest) | 1.94 % | 0.00 % | 6.63 % | 10.18 % | 0.00 s |
| `ClarkeWright` | 4.89 % | 0.12 % | 12.25 % | 8.68 % | 0.07 s |
| `Genetic` (plain) | 0.53 % | 9.04 % | 16.49 % | 26.72 % | 0.08 s |
| `OrOpt` | 4.87 % | 19.77 % | 15.67 % | 17.60 % | 0.01 s |
| `NRBS` | 18.31 % | 7.80 % | 14.18 % | 19.99 % | 0.08 s |
| `NearestNeighbour` | 31.83 % | 46.41 % | 24.47 % | 26.72 % | 0.00 s |

## If you want X, use Y

| You want... | Use | Notes |
|---|---|---|
| a certificate of optimality | [`MILP`][skroute.MILP] up to ~200 nodes; [`BruteForce`][skroute.BruteForce] up to 11; [`HeldKarp`][skroute.HeldKarp] up to 20 | only `BruteForce` certifies the multi-trip objective |
| the best quality in seconds | [`IteratedLocalSearch`][skroute.IteratedLocalSearch] | then `Genetic(local_search=("two_opt",))` or `MultiStart(SimulatedAnnealing())` if you can spend more time |
| a multi-trip budget inside the search | any budget-aware solver: `IteratedLocalSearch`, `SimulatedAnnealing`, `TabuSearch`, `Genetic`, `AntColony`; `ClarkeWright` as construction | add `split="optimal"` for the minimum-cost partition of the tour; see [multi-trip routing](multi_trip.md) |
| asymmetric costs | everything except `ClarkeWright`; exact: `HeldKarp`, `MILP` | metaheuristics take the $O(n)$-per-move path: a few hundred to ~2 000 nodes |
| a very large $n$ | [`LocalSearch`][skroute.LocalSearch] or `IteratedLocalSearch(time_limit=...)` on a symmetric plain TSP | dense-matrix ceiling ~20 000 nodes |
| a baseline or a warm start in microseconds | [`NearestNeighbour`][skroute.NearestNeighbour], [`Insertion`][skroute.Insertion], [`ClarkeWright`][skroute.ClarkeWright] | pass `init=est.tour_` to the next solver |
| bit-exact reproducibility | any stochastic solver with an integer `random_state`, without `time_limit` | same seed, same machine, same tour; see [reproducibility](warm_starts_and_ensembles.md#random_state-and-reproducibility) |
| all your cores | [`MultiStart`][skroute.MultiStart] with `n_jobs=-1` | threads by default; `prefer="processes"` for `Genetic` and `SOM` |
| a geometric picture of the search | [`SOM`][skroute.SOM] | needs `coords=` |
| to watch progress | `verbose=1` plus `skroute.set_log_level("INFO")` | nothing is ever printed; records go to the `skroute` logger |
