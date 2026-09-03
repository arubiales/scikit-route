# scikit-route 2.0 — Definitive specification

Status: FROZEN for implementation. Changes go through a pull request against this document, approved by the lead. Written 2026-09-03 after reading the 1.0.0a2 tree at `/Users/albertorubialesborrego/scikit-route` (branch `modernization/v2`), the four proposals and the three scorecards. Facts below marked *(verified)* were checked against the repository today.

---

## 1. Executive summary and decisions

scikit-route 2.0 is a scikit-learn-flavoured library of route optimisation solvers with one problem model: a closed tour from a depot over a dense cost matrix, optionally decoded into several trips under a per-trip working-time budget with a fixed charge per additional trip. Every solver is an estimator (`__init__` stores knobs, `fit(X, ...)` returns `self`, results in `*_` attributes). One shared Cython 3 core evaluates costs and move deltas over typed memoryviews; each solver may ship its own `.pyx` next to its `.py` and `cimport`s the core's inline primitives, so eight people can write hot loops without touching a shared file. The base class builds one immutable `RoutingProblem`, validates the tour every `_solve` returns and **recomputes `cost_` from it**, which makes the 2020 "route does not match cost" bug impossible by construction. Plain TSP is `est.fit(C)`; the multi-trip objective is `est.fit(C, time_matrix=T, max_time_work=8, extra_cost=12.83, people=2)` and is available to every budget-aware solver through the same kernels. 2.0 ships 15 solver classes (16 with `AntColony`; 14 if `HeldKarp` is also deferred) plus `MultiStart` and the two compatibility wrappers `EnsembleGenetic`/`EnsembleSimulatedAnnealing`; pandas and googlemaps are optional. Tests are deterministic through `random_state`, exact on n ≤ 9, and benchmarked against the Waterloo optima with measured tolerances. Version is 2.0.0 with a migration guide.

### Decisions (binding)

- **D1 Objective = giant tour + split rule.** A solution is a permutation of all nodes with the depot first (the *giant tour*). Its cost is travel cost of the decoded trips plus `people * extra_cost * (n_trips - 1)`. Two decoders exist in the core, chosen by `fit(..., split=)`: `"greedy"` (default: leg `a→b` joins the open trip iff `t + T[a,b] + T[b,depot] <= max_time_work`, else close at `a`, reopen `depot→b`) and `"optimal"` (Prins 2004: the minimum-cost partition into consecutive feasible trips, a DAG shortest path, O(n·L)). Greedy is the default because it is O(n), it is what the measured prototype implements, and it is the honest successor of the 1.0 rule; optimal is one flag away and is never worse for a given tour. Every trip, including the return leg, fits the budget under both rules.
- **D2 The base class recomputes `cost_`** from the returned tour with the problem's own decoder and raises `RuntimeError` if `_solve` returns anything but a permutation with the depot at position 0. Solvers never report a cost.
- **D3 `max_time_work` without `time_matrix` raises** `ValueError("max_time_work given but no time_matrix; pass time_matrix=X to use the cost matrix as durations")`. Silent defaulting hides unit mistakes (euros read as hours).
- **D4 `people` multiplies only `extra_cost`.** 1.0 multiplied travel cost too (contradicting its docstring); a global factor never changes the argmin, so only reported costs change. Documented in the migration guide.
- **D5 A node whose round trip exceeds the budget raises `InfeasibleProblemError`** at fit, naming the labels.
- **D6 Exact solvers that cannot certify the multi-trip objective (HeldKarp, MILP) raise** when a budget is given; non-exact budget-unaware solvers (NearestNeighbour, Insertion, NRBS, SOM) warn and their result is still decoded and priced under the objective. An "exact" result that is not optimal for the objective is a trap; a heuristic one is merely a heuristic.
- **D7 Data to `fit`, knobs to `__init__`.** Signature: `fit(X, *, time_matrix=None, depot=None, coords=None, labels=None, max_time_work=None, extra_cost=0.0, people=1, split="greedy")`. **`time_matrix` is keyword-only** (also in `RoutingProblem`): 1.0 took `(route_example, time_matrix, cost_matrix)`, so a migrating user who keeps the positional order would swap the two square matrices without any error and price hours as euros; the keyword makes the swap impossible. `X` may also be a ready `RoutingProblem`. No `y`. Warm starts are `init=` in `__init__` (KMeans precedent).
- **D8 Outputs are label-space ndarrays**: `tour_` (open, depot first), `route_` (as driven: depot first and last, re-inserted between trips), `trips_` (list of closed per-trip arrays). Index space never leaks. `init=` accepts `tour_` or `route_` of another solver.
- **D9 `history_` is best-so-far per outer iteration** (monotone non-increasing), `n_iter_ == len(history_)`, `stop_reason_ in {"converged","max_iter","patience","time_limit"}`; these three exist only on iterative solvers. Each solver documents the **subset** of stop reasons it can emit (a solver without `patience`/`time_limit` parameters never emits those values; the fitted-attribute table of §3.4 lists the subsets). `trip_times_` exists only when a time matrix was given.
- **D10 Randomness is pre-drawn** in Python from `numpy.random.default_rng(random_state)` and passed to `nogil` kernels as arrays. Bit-exact reproducibility, no numpy C-API at build time, thread-parallel `MultiStart`. No `bitgen_t` in 2.0.
- **D11 Kernel topology:** one core extension `skroute/_core/_routing` whose `cdef inline` primitives are **defined, with their bodies, in `_routing.pxd`** (the only way Cython shares inline code across extension modules — a body-less `cdef inline` declaration does not compile in the cimporting module, verified with Cython 3.3); the non-inline `cdef`/`cpdef` functions are declared in the `.pxd` and defined in `_routing.pyx`. Per-solver `.pyx` files live next to their `.py`; `setup.py` globs `skroute/**/*.pyx`. No `cimport numpy` anywhere, so wheels are numpy-ABI independent.
- **D12 No pure-Python fallback ships.** Wheels cover CPython 3.11–3.14 × Linux/macOS/Windows; the pure-Python oracles live in `tests/reference.py` and are written by a non-kernel owner.
- **D13 Depot is stored as an index; the tour is rotated so `tour[0] == depot`.** No matrix permutation/copy (a second 900 MB at n = 10 639).
- **D14 ATSP is allowed everywhere** except `ClarkeWright` (tag `requires_symmetric`). Reversal deltas are O(1) only when symmetric; otherwise kernels take the full-evaluation path (same path as multi-trip). MILP handles ATSP with arc variables.
- **D15 TSPLIB rounding is `nint(x) = floor(x + 0.5)`**, never `np.rint` (half-to-even proved 9351 on qa194 against the published 9352).
- **D16 Layout:** flat package, `tests/` at the repo root and **not a package** (no `tests/__init__.py`, no `tests/benchmarks/__init__.py`): with a package, pytest's default import mode would put the repo root on `sys.path` and `import skroute` would resolve to the uncompiled source tree instead of the installed wheel in the cibuildwheel and sdist jobs (verified). `pyproject.toml` sets `[tool.pytest.ini_options] testpaths = ["tests"]`, `pythonpath = ["tests"]`, `addopts = "-m 'not slow' --strict-markers"`, `doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS NUMBER"`; every test imports the oracle as `import reference`. cibuildwheel: `test-command = "pytest {project}/tests -x -q"` (no `-m` clause: `addopts` already deselects `slow`, and single quotes are not quoting characters for `cmd.exe`, which is what cibuildwheel uses on Windows), `test-environment = { SKROUTE_EXPECT_WHEEL = "1" }`; `tests/test_base.py::test_runs_against_installed_copy` asserts `"site-packages" in skroute.__file__` whenever that variable is set (the sdist job sets it too). Never `python -m pytest` in the wheel/sdist jobs (it adds the cwd to `sys.path`).
- **D17 `MultiStart` uses joblib threads by default** (`prefer="threads"`), seeds from `SeedSequence.spawn`, results independent of `n_jobs`. `EnsembleGenetic`/`EnsembleSimulatedAnnealing` are explicit-parameter subclasses.
- **D18 Roster:** exact `BruteForce, HeldKarp (P1), MILP`; construction `NearestNeighbour, Insertion, ClarkeWright, NRBS`; local search `TwoOpt, OrOpt, LocalSearch, IteratedLocalSearch`; metaheuristics `SimulatedAnnealing, TabuSearch, Genetic, AntColony (P1), SOM`; ensemble `MultiStart, EnsembleGenetic, EnsembleSimulatedAnnealing`. That is 15 solvers (16 with AntColony; 14 if HeldKarp is also deferred) + `MultiStart` + two compatibility wrappers. There are **no** `CheapestInsertion`/`FarthestInsertion` aliases: `Insertion(strategy=...)` is the only spelling (aliases add nothing over the parameter, break `check_router` item 1 and inflate the roster). Deferred to 2.1: Christofides (needs a real blossom matching; never a greedy fake), Or-3opt/Lin–Kernighan, VNS, multi-trip MILP, float32 matrices, on-the-fly distances and coordinate-only fitting (SOM without a matrix).
- **D19 Benchmark tolerances are the measured ones** (Proposal 4's baseline), tightened via release notes; no exact-float golden pins across platforms — reproducibility is asserted as "same seed, same machine, bit-identical".
- **D20 pandas is optional** (`scikit-route[pandas]`); DataFrames are recognised by duck typing; loaders return numpy `Bunch`es and take `as_frame=True`; CSVs are parsed with the stdlib `csv` module *(verified: addresses contain commas)*.
- **D21 Python ≥ 3.11**; CI 3.11–3.14 plus 3.15 pre-release allowed-failure; runtime `numpy>=1.26`, `scipy>=1.11`, `joblib>=1.3`.
- **D22 Version 2.0.0.** `fit` no longer returns a tuple, problem data moves from `__init__` to `fit`, four hard dependencies disappear, objective semantics change.
- **D23 Declarative parameter validation** (`_parameter_constraints` with `Interval`/`Options`), deep `get_params`/`set_params` with `__` nesting, print-changed-only `repr`, a `RouterTags` protocol, and a public `check_router()` battery.
- **D24 No printing anywhere.** `verbose` routes to `logging.getLogger("skroute")`. Because Python's last-resort handler shows only WARNING and above, `verbose=1` would otherwise look like "does nothing": `skroute/__init__.py` attaches a `logging.NullHandler()` to the `skroute` logger and exposes `skroute.set_log_level(level)` (sets the level and attaches a `StreamHandler` to stderr if the logger has only the `NullHandler`); every `verbose` docstring ends with the sentence *"Records go to the `skroute` logger at INFO; enable them with `logging.basicConfig(level=logging.INFO)` or `skroute.set_log_level("INFO")`."*; `getting_started.md` shows it once.
- **D25 Hyper-parameter glossary** (§4.0) is mandatory vocabulary in every solver and docstring.
- **D26 Priorities and dates.** P0 = must ship in 2.0.0. P1 = ships only if its owner finishes it (code, tests, docs page) by the **feature freeze on 2026-10-16**; otherwise the owner hands the lead a written deferral PR by that date and the item moves to 2.1 with its name not exported (procedure in §4.4). P1 items: `HeldKarp` (WP2; MILP is instant at n ≤ 20 and BruteForce covers n ≤ 11), `AntColony` (WP6), the `GEO` and `ATT` edge-weight types of `read_tsplib` (WP7; `EUC_2D`, `CEIL_2D`, `EXPLICIT` and `read_tsplib_tour` are P0 — all 27 bundled files are `EUC_2D`), `GoogleDistanceMatrix` (WP7). `split="optimal"` inside `local_search_generic` stays P0 (D1 promises the optimal split to every budget-aware solver). Release candidate 2026-10-23 (benchmarks page regenerated, nightly green on three OSes), release 2026-10-30.
- **D27 `all_solvers()` and the two halves of the test battery.** `skroute.all_solvers() -> list[type[BaseRouter]]` returns, sorted by `__name__`, every concrete solver class exported from `skroute` that can be instantiated with no arguments: the 15 (16) solvers of D18 plus `EnsembleGenetic` and `EnsembleSimulatedAnnealing`; `MultiStart` is never returned (it needs an estimator) and is covered by `tests/test_ensemble.py`. `check_router(estimator)` (an **unfitted instance**) runs only the structural checks 1–11 and 13 of §6 on instances it builds itself; the tolerance checks live in `tests/test_common.py` and are driven by `tests/tolerances.py`, the single source of every tolerance number.
- **D28 `RouterTags.kind` and an honest default.** `RouterTags` gains `kind in {"exact", "construction", "local_search", "metaheuristic", "ensemble"}` (used by the capability table and by the tolerance tests) and `budget_aware` defaults to **`False`**: a solver whose author forgets `_get_tags()` must be advertised as budget-unaware (it warns), never as budget-aware.
- **D29 Execution model (lead, 2026-09-03).** The work packages are executed by parallel agents in separate git worktrees (`~/scikit-route-wt/<wp>`, branch `wp/<wp>`) merged by the lead, in two waves: wave A = spine (base, problem, utils incl. `estimator_checks.py`, `tests/conftest.py`, `tests/tolerances.py`, `tests/test_common.py`), core (WP1), datasets/preprocessing (WP7 data), community files (WP8 docs part); wave B = every solver package (WP2–WP6, WP7 SOM, WP8 ensemble/shims/user guide) against the merged, compiled core. Consequences: there is no `SKROUTE_REFERENCE` switch and no `_core/_reference.py`; during development the package is never installed — tests run as `python -m pytest tests` from the worktree root (the interpreter puts the cwd on `sys.path`; the wheel and sdist CI jobs use the `pytest` entry point exactly as D16 requires); the calendar of D26 becomes 'ship in this delivery, or defer with the written procedure'.

---

## 2. Repository layout and dependencies

Every file that will exist at the 2.0.0 tag (plus the development-only `docs/gen_pages.py` and `docs/check_api_coverage.py`, which ship in the repository but not in the wheel). Owner codes in brackets refer to §8 (L = lead). Files marked *keep* already exist.

```
scikit-route/
  pyproject.toml                 [L] PEP 621 metadata, deps, extras, tool config (ruff, mypy, pytest, cibuildwheel);
                                     [tool.setuptools.package-data] skroute = ["py.typed", "**/*.pxd", "**/*.pyi",
                                     "datasets/_data/tsplib/*.tsp", "datasets/_data/costs/*.csv", "datasets/_descr/*.md"]
                                     (MANIFEST.in only governs sdists; wheels need this list)
  setup.py                       [L] only cythonize(glob("skroute/**/*.pyx")) with the directives of §3.5
  MANIFEST.in                    [L] recursive-include skroute *.pyx *.pxd *.pyi *.tsp *.csv *.md; include skroute/py.typed; drops *.pkl
  README.md                      [L] badges, 30-second example, install, links (outline §7)
  CHANGELOG.md                   [L] Keep a Changelog; 2.0.0 entry skeleton in §7
  CITATION.cff                   [L] cff-version 1.2.0, title scikit-route, authors Alberto Rubiales, version 2.0.0, MIT, repository-code, date-released
  LICENSE                        keep (MIT)
  CONTRIBUTING.md                [WP8]
  CODE_OF_CONDUCT.md             [WP8] Contributor Covenant 2.1
  SECURITY.md                    [WP8]
  mkdocs.yml                     [L] nav in §7
  .pre-commit-config.yaml        [L] ruff (lint+format), cython-lint, end-of-file, trailing-whitespace
  .gitignore                     [L] keep + *.so *.c build/ site/ .venv*/
  .github/CODEOWNERS             [L] one owner per path of §8
  .github/FUNDING.yml            keep
  .github/workflows/ci.yml       [L]
  .github/workflows/wheels.yml   [L]
  .github/workflows/docs.yml     [L]
  .github/workflows/nightly.yml  [L]
  .github/ISSUE_TEMPLATE/bug_report.yml, feature_request.yml   [WP8]
  .github/PULL_REQUEST_TEMPLATE.md                             [WP8]
  benchmarks/kernels.py          [WP1] micro-benchmarks of the core (the prototype bench.py, ported)
  benchmarks/waterloo.py         [WP8] gap table of every solver on the bundled instances (feeds docs/benchmarks.md)
  skroute/
    __init__.py                  [L] __version__ re-export, PEP 562 lazy exports + `if TYPE_CHECKING:` eager imports of every
                                     public name (mypy and mkdocstrings do not see __getattr__), __all__, all_solvers() (D27),
                                     set_log_level(), NullHandler on the "skroute" logger (D24)
    _version.py                  [L] __version__ = "2.0.0" (single source, read statically by setuptools)
    py.typed                     [L]
    base.py                      [L] RouterTags, BaseRouter, clone, is_router
    problem.py                   [L] RoutingProblem
    exceptions.py                [L] NotFittedError, InfeasibleProblemError
    metrics.py                   [L] route_cost, split_trips (public recomputation helpers)
    utils/__init__.py            [L] re-exports Bunch, check_random_state, check_is_fitted, initial_tour
    utils/validation.py          [L] coerce_matrix, coerce_labels, check_random_state, check_is_fitted
    utils/_param_validation.py   [L] Interval, Options, validate_parameter_constraints
    utils/_bunch.py              [L] Bunch (dict with attribute access, 25 lines)
    utils/_init_tour.py          [L] initial_tour(problem, init, rng)
    utils/estimator_checks.py    [WP8] check_router(estimator) and its structural sub-checks (the one file under utils/ the lead does not own)
    _core/__init__.py            [L] imports _routing or raises a one-line-fix ImportError (final, D29)
    _core/_routing.pxd           [WP1] the frozen contract of §3.5 — inline primitives DEFINED here with their bodies
    _core/_routing.pyx           [WP1] the non-inline cdef/cpdef functions
    _core/_routing.pyi           [WP1] typing stubs of every cpdef and `class SplitRule(IntEnum)`
    exact/__init__.py            [WP2] __all__ = ["BruteForce", "HeldKarp", "MILP"]  (HeldKarp only if shipped, D26)
    exact/_brute_force.py, _brute.pyx                            [WP2]
    exact/_held_karp.py, _held_karp.pyx                          [WP2] P1
    exact/_milp.py                                               [WP2]
    construction/__init__.py     [WP3] __all__ = ["NearestNeighbour","Insertion","ClarkeWright","NRBS"]
    construction/_nearest_neighbour.py, _insertion.py, _insertion.pyx, _clarke_wright.py, _nrbs.py   [WP3]
    local_search/__init__.py     [WP4] __all__ = ["TwoOpt","OrOpt","LocalSearch","IteratedLocalSearch"]
    local_search/_two_opt.py, _or_opt.py, _local_search.py, _iterated.py                          [WP4]
    metaheuristics/__init__.py   [L] append-only, one export per line
    metaheuristics/_simulated_annealing.py, _sa.pyx, _tabu_search.py, _tabu.pyx                   [WP5]
    metaheuristics/_genetic.py, _ga.pyx, _ant_colony.py, _aco.pyx                                 [WP6]
    metaheuristics/_som.py                                                                        [WP7]
    ensemble/__init__.py, _multistart.py, _legacy.py             [WP8]
    heuristics/__init__.py, heuristics/brute/__init__.py, heuristics/NRBS/__init__.py            [WP8] deprecated shims
    metaheuristics/genetics/__init__.py, simulated_annealing/__init__.py, tabu_search/__init__.py, som/__init__.py  [WP8] deprecated shims
    datasets/__init__.py         [WP7]
    datasets/_loaders.py         [WP7] load_tsp, list_tsp, 27 country wrappers, 5 cost loaders
    datasets/_tsplib.py          [WP7] read_tsplib, read_tsplib_tour
    datasets/_descr/*.md         [WP7] one DESCR per dataset family
    datasets/_data/tsplib/*.tsp  [WP7] the 27 files, moved from _latitude_longitude/
    datasets/_data/costs/*.csv   [WP7] the 5 CSVs, moved from _money_cost/; .pkl deleted
    preprocessing/__init__.py    [WP7]
    preprocessing/_distances.py  [WP7] distance_matrix, euclidean_matrix, haversine_matrix, tsplib_nint
    preprocessing/_convert.py    [WP7] pairs_to_matrix, to_dict_of_dicts, from_dict_of_dicts, normalize_coords
    preprocessing/google.py      [WP7] GoogleDistanceMatrix (lazy googlemaps import) P1; CostScraper deprecated wrapper (§5.2)
  tests/                         NOT a package: no __init__.py anywhere below (D16)
    conftest.py                  [WP8] fixtures of §6
    reference.py                 [WP8] pure-Python oracles (imported as `import reference`)
    tolerances.py                [WP8] TINY/FAST/SLOW dicts keyed by class name + SEEDS_TO_OPTIMUM (the §6 table as data)
    test_base.py                 [L]  RoutingProblem coercion (incl. neighbours() with 4 coincident points and a non-zero
                                      diagonal, coerce_labels dtypes, keyword-only time_matrix), BaseRouter protocol,
                                      exceptions, installed-copy guard (SKROUTE_EXPECT_WHEEL)
    test_common.py               [WP8] check_router over all_solvers() + the tolerance tests driven by tolerances.py
    test_core.py                 [WP1] hypothesis tests of every kernel vs reference.py
    test_exact.py                [WP2]
    test_construction.py         [WP3]
    test_local_search.py         [WP4]
    test_simulated_annealing.py, test_tabu_search.py   [WP5]
    test_genetic.py, test_ant_colony.py                [WP6]
    test_som.py, test_datasets.py, test_preprocessing.py   [WP7]
    test_ensemble.py, test_legacy_shims.py             [WP8]
    benchmarks/test_waterloo.py  [WP8] ALL @slow gap tests of every solver (no other file has slow-tier tests)
    data/explicit_matrix.tsp, tiny.tour, ulysses16.tsp, ulysses16.opt.tour, att48.tsp, att48.opt.tour   [WP7] reader fixtures
    data/nrbs_barcelona_1_0.json [WP3] the pinned 1.0 NRBS result (§4.2)
  docs/
    gen_pages.py                  [WP8] run by mkdocs-gen-files: writes user_guide/_capability_table.md from all_solvers() tags;
                                        `python docs/gen_pages.py --readme` refreshes the README table between markers
    check_api_coverage.py         [WP8] fails if a name in any skroute __all__ has no `:::` directive under docs/api
    index.md, getting_started.md, installation.md          [WP8]
    user_guide/problem_model.md   (objective, split rules, ATSP, the ~20 000-node dense-matrix ceiling)   [WP8]
    user_guide/choosing_a_solver.md (includes the generated capability table) [WP8]
    user_guide/multi_trip.md, user_guide/warm_starts_and_ensembles.md      [WP8]
    api/base.md                   [L]  (directives listed in §7)
    api/utils.md                  [L]  ::: skroute.utils, ::: skroute.utils.estimator_checks.check_router
    api/exact.md [WP2]  api/construction.md [WP3]  api/local_search.md [WP4]
    api/simulated_annealing.md, api/tabu_search.md [WP5]  api/genetic.md, api/ant_colony.md [WP6]
    api/som.md, api/datasets.md, api/preprocessing.md [WP7]  api/ensemble.md [WP8]
    benchmarks.md                 [WP8] COMMITTED table produced by benchmarks/waterloo.py on the release candidate (header: date,
                                        commit, CPU, OS, Python/numpy versions, random_state=0, tolerances, measured baselines)
    migration.md                  [L] 1.0.0a2 → 2.0 (outline §7)
    contributing.md (includes ../CONTRIBUTING.md), changelog.md (includes ../CHANGELOG.md), about.md   [WP8]
    images/logo.png               keep
```

Deleted: `skroute/_utils/`, `skroute/_validators/`, `skroute/cluster/`, every in-package `tests/` and `pytest.ini`, `requirements.txt`, `skroute/datasets/DESCR.txt`, `columns_*.txt`, all `.pkl`, `.c`/`.html` artefacts, the Travis badge.

### Dependencies (minimum versions)

| Group | Packages |
|---|---|
| runtime (`dependencies`) | `numpy>=1.26`, `scipy>=1.11`, `joblib>=1.3` |
| extra `pandas` | `pandas>=2.2` |
| extra `google` | `googlemaps>=4.10` |
| extra `test` | `pytest>=8.0`, `pytest-cov>=5.0`, `hypothesis>=6.100`, `pandas>=2.2` |
| extra `dev` | `test` + `ruff>=0.6`, `mypy>=1.10`, `cython-lint>=0.16`, `pre-commit>=3.7`, `build>=1.2`, `cibuildwheel>=3.0` (needs `test-environment`), `Cython>=3.1` |
| extra `docs` | `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.25`, `mkdocs-gen-files>=0.5`, `mkdocs-include-markdown-plugin>=6.0` (`mkdocs-literate-nav` dropped: nothing uses it) |
| build (`[build-system].requires`) | `setuptools>=77`, `Cython>=3.1` — **no numpy** (memoryviews only) |

`requires-python = ">=3.11"`. Classifiers 3.11, 3.12, 3.13, 3.14. Dev venv: `.venv` is Python 3.13.3 / numpy 2.5.2 / scipy 1.18.1 / Cython 3.3.0 / pandas 3.0.5; `.venv314` is 3.14.7 *(verified)*. No compiled `.so` is present in the tree *(verified)*; build with `pip install -e .`.

---

## 3. Public API contract

### 3.1 Conventions

- **Index space** (internal): nodes are `0..n-1` in the order of the input matrix rows; `problem.depot` is the depot's index; a *tour* is an `int64` array of length `n`, a permutation with `tour[0] == problem.depot`. Kernels read `depot = tour[0]`.
- **Label space** (public): `problem.labels[i]` is the user's label of node `i` (`arange(n)` for a plain ndarray, `X.index` for a DataFrame, key order for a dict-of-dicts, or `labels=`). Everything a user sees (`tour_`, `route_`, `trips_`, `depot_`, `init=`) is labels.
- **Closed tour**: the objective always includes the return to the depot.
- **Trips**: `trip_starts` is an `int64` array of length `n_trips + 1` with `starts[0] == 1`, `starts[-1] == n`; trip `k` is `tour[starts[k]:starts[k+1]]`. Plain TSP has `starts == [1, n]`.
- **Asymmetric matrices** (`not problem.symmetric`) are ATSP instances: every evaluation reads `C[i, j]` directionally; only `ClarkeWright` refuses them.
- **Diagonal** entries are never read by any kernel; they must still be finite.
- **Numbers**: costs `float64`, indices `int64_t` (from `libc.stdint`, never `long`), `Py_ssize_t` for positions.

### 3.2 `skroute/exceptions.py`

```python
class NotFittedError(ValueError, AttributeError):
    """Raised by check_is_fitted when a fitted attribute is accessed before fit."""

class InfeasibleProblemError(ValueError):
    """Raised when a node cannot be served in a single trip within max_time_work."""
```

### 3.3 `skroute/problem.py` — `RoutingProblem` (complete contract)

```python
from __future__ import annotations
import numpy as np
from ._core import _routing as core
from .exceptions import InfeasibleProblemError
from .utils.validation import coerce_matrix, coerce_labels

# Cython 3 exposes a cpdef enum to Python as the IntEnum class `_routing.SplitRule`; its members are
# NOT module-level names (`core.SPLIT_GREEDY` raises AttributeError — verified with Cython 3.3).
_SPLIT = {"greedy": int(core.SplitRule.SPLIT_GREEDY), "optimal": int(core.SplitRule.SPLIT_OPTIMAL)}


class RoutingProblem:
    """One instance in index space. Immutable after construction; shareable across solvers and threads.

    Parameters
    ----------
    X : (n, n) array-like, DataFrame or dict-of-dicts
        Cost matrix. Rows are origins, columns destinations. Must be square, n >= 3, all finite.
    time_matrix : same kinds as X, optional, keyword-only (D7)
        Durations, same shape and labels as X, all finite and >= 0. Required iff max_time_work is given.
    depot : label, optional
        Label of the depot (a position for plain arrays without labels=). Default: first node.
    coords : (n, 2) array-like, optional
        Coordinates in row order of X. Needed by SOM; carried, never validated beyond shape.
    labels : sequence of n hashables, optional
        Labels for a plain ndarray X. Must equal the labels X already carries, if any.
    max_time_work : float > 0, optional
        Per-trip budget in the units of time_matrix. None = plain TSP.
    extra_cost : float >= 0, default 0.0
        Fixed charge per trip beyond the first.
    people : int >= 1, default 1
        Multiplies extra_cost only.
    split : {"greedy", "optimal"}, default "greedy"
        Decoder of the giant tour into trips (see D1).
    """

    def __init__(self, X, *, time_matrix=None, depot=None, coords=None, labels=None,
                 max_time_work=None, extra_cost=0.0, people=1, split="greedy"):
        C, lab = coerce_matrix(X, "X")                      # float64 C-contiguous, labels or None
        n = C.shape[0]
        if n < 3:
            raise ValueError(f"X must have at least 3 nodes, got {n}")
        if labels is not None:
            given = coerce_labels(labels, n)                # 1-D int64 or object array, n unique entries (contract below)
            if lab is not None and not np.array_equal(lab, given):
                raise ValueError("labels= disagrees with the labels carried by X")
            lab = given
        self.labels = np.arange(n, dtype=np.int64) if lab is None else lab
        self._index = {label: i for i, label in enumerate(self.labels.tolist())}
        if len(self._index) != n:
            raise ValueError("labels must be unique")
        self.cost = C
        self.n = n
        if depot is None:
            self.depot = 0
        else:
            try:
                self.depot = self._index[depot]
            except (KeyError, TypeError):
                raise ValueError(f"depot {depot!r} is not a label of X") from None
        if split not in _SPLIT:
            raise ValueError(f"split must be 'greedy' or 'optimal', got {split!r}")
        self.split = split
        if max_time_work is None:
            if time_matrix is not None:
                raise ValueError("time_matrix given but no max_time_work; pass max_time_work=<hours per trip>")
            if extra_cost != 0.0 or people != 1 or split != "greedy":
                raise ValueError("extra_cost, people and split have no effect without max_time_work")   # D3: no silent knobs
            self.time = None
            self.max_time_work = np.inf
        else:
            if time_matrix is None:
                raise ValueError("max_time_work given but no time_matrix; "
                                 "pass time_matrix=X to use the cost matrix as durations")
            if not (np.isfinite(max_time_work) and float(max_time_work) > 0):   # inf would store T and set multi_trip for nothing
                raise ValueError(f"max_time_work must be a finite number > 0, got {max_time_work!r}")
            T, tlab = coerce_matrix(time_matrix, "time_matrix")
            if T.shape != C.shape:
                raise ValueError(f"time_matrix has shape {T.shape}, X has shape {C.shape}")
            if tlab is not None and not np.array_equal(tlab, self.labels):
                raise ValueError("time_matrix labels differ from the labels of X")
            if (T < 0).any():
                raise ValueError("time_matrix contains negative durations")
            self.time = T
            self.max_time_work = float(max_time_work)
            d = self.depot
            bad = (T[d, :] + T[:, d] > self.max_time_work)
            bad[d] = False
            if bad.any():
                raise InfeasibleProblemError(
                    f"nodes {self.labels[bad].tolist()} cannot be served in one trip: "
                    f"depot round trip exceeds max_time_work={self.max_time_work}")
        if not (float(extra_cost) >= 0) or not np.isfinite(extra_cost):
            raise ValueError(f"extra_cost must be a finite number >= 0, got {extra_cost!r}")
        if not isinstance(people, (int, np.integer)) or isinstance(people, bool) or people < 1:
            raise ValueError(f"people must be an integer >= 1, got {people!r}")
        self.extra_cost, self.people = float(extra_cost), int(people)
        self.coords = None
        if coords is not None:
            xy = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
            if xy.shape != (n, 2):
                raise ValueError(f"coords must have shape ({n}, 2), got {xy.shape}")
            self.coords = xy
        self.symmetric = bool(np.array_equal(C, C.T))
        self._neigh: dict[int, np.ndarray] = {}

    # ----- derived, read-only -----
    @property
    def multi_trip(self) -> bool: return self.time is not None
    @property
    def fixed_cost(self) -> float: return self.extra_cost * self.people
    @property
    def time_or_cost(self) -> np.ndarray:
        """What kernels receive as T: the time matrix, or the cost matrix when there is no budget (never read then)."""
        return self.cost if self.time is None else self.time
    @property
    def split_code(self) -> int: return _SPLIT[self.split]
    @property
    def depot_label(self): return self.labels[self.depot]

    # ----- label <-> index -----
    def index_of(self, label) -> int:
        try: return self._index[label]
        except (KeyError, TypeError): raise ValueError(f"{label!r} is not a label of X") from None

    def to_index_tour(self, seq) -> np.ndarray:
        """Labels -> int64 tour with the depot at position 0.

        Accepts an open tour (n labels), a closed route (depot repeated at the end)
        or a multi-trip route (depot repeated between trips): every occurrence of the
        depot is removed, then the depot is prepended. Raises ValueError unless the
        remaining labels are exactly the non-depot labels, each once."""
        idx = np.fromiter((self.index_of(x) for x in seq), dtype=np.int64, count=len(seq))
        body = idx[idx != self.depot]
        expected = np.delete(np.arange(self.n), self.depot)
        if not np.array_equal(np.sort(body), expected):
            raise ValueError("init tour must contain every label exactly once (the depot may repeat)")
        return np.concatenate(([self.depot], body)).astype(np.int64)

    def to_label_tour(self, tour) -> np.ndarray:
        return self.labels[np.asarray(tour, dtype=np.int64)]

    # ----- kernels -----
    # Typed memoryviews accept only C-contiguous int64 arrays (a list or an int32 array raises TypeError /
    # "Buffer dtype mismatch"); these public methods coerce first.
    @staticmethod
    def _as_index(a) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.int64)

    def evaluate(self, tour) -> float:
        """Objective of an index tour (D1). O(n) greedy / plain, O(n*L) optimal."""
        return core.problem_cost_py(self.cost, self.time_or_cost, self._as_index(tour),
                                    self.max_time_work, self.fixed_cost, self.split_code)

    def trip_starts(self, tour) -> np.ndarray:
        out = np.empty(self.n + 1, dtype=np.int64)
        k = core.trip_starts(self.time_or_cost, self._as_index(tour), self.max_time_work, self.split_code,
                             self.cost, self.fixed_cost, out)
        return out[: k + 1]

    def trip_costs(self, tour, starts) -> np.ndarray:
        out = np.empty(len(starts) - 1)
        core.trip_costs(self.cost, self._as_index(tour), self._as_index(starts), out); return out

    def trip_times(self, tour, starts) -> np.ndarray:
        out = np.empty(len(starts) - 1)
        core.trip_times(self.time, self._as_index(tour), self._as_index(starts), out); return out

    def neighbours(self, k: int = 10) -> np.ndarray:
        """k nearest neighbours of every node by C[i, :], as int64 (n, k), sorted ascending; cached.

        The diagonal is excluded regardless of its value (only finiteness is required of it) and
        ties are broken by node index (stable sort). Six bundled instances have coincident points
        (lu980: 346 duplicate rows, ho14473: 7 370), so zero off-diagonal distances are normal input.
        """
        k = min(int(k), self.n - 1)
        if k not in self._neigh:
            D = self.cost.copy()                       # one transient (n, n) copy; accepted
            np.fill_diagonal(D, np.inf)
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(D, idx, 1), axis=1, kind="stable")
            self._neigh[k] = np.ascontiguousarray(np.take_along_axis(idx, order, 1), dtype=np.int64)
        return self._neigh[k]

    def __repr__(self):
        kind = "multi-trip" if self.multi_trip else "TSP"
        sym = "symmetric" if self.symmetric else "asymmetric"
        return f"RoutingProblem(n={self.n}, {kind}, {sym}, depot={self.depot_label!r})"
```

`coerce_matrix(M, name) -> (ndarray, labels | None)` in `utils/validation.py` — the exact coercion contract:

```python
def coerce_matrix(M, name):
    if isinstance(M, dict):                                        # legacy dict-of-dicts
        labels = list(M)
        try:
            arr = np.array([[M[i][j] for j in labels] for i in labels], dtype=np.float64)
        except KeyError as e:
            raise ValueError(f"{name}: dict-of-dicts is not square, missing key {e}") from None
        lab = coerce_labels(labels, len(labels))
    elif hasattr(M, "to_numpy") and hasattr(M, "index") and hasattr(M, "columns"):   # DataFrame, duck-typed
        if list(M.index) != list(M.columns):
            raise ValueError(f"{name}: index and columns must hold the same labels in the same order")
        arr, lab = np.ascontiguousarray(M.to_numpy(dtype=np.float64)), coerce_labels(M.index, len(M.index))
    else:
        arr, lab = np.ascontiguousarray(np.asarray(M, dtype=np.float64)), None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, lab


def coerce_labels(seq, n):
    """Label dtype is ALWAYS int64 or object, whatever the input path (ndarray+labels=, DataFrame index,
    dict keys), so tour_/labels_/depot_ compare equal across paths and `np.array_equal` never mixes kinds.
    Integer-like labels (numpy or Python ints, never bool) -> int64; anything else (strings, mixed, tuples)
    -> object. `np.asarray(["a", "b"])` would give '<U1' and a DataFrame index 'object' — hence the rule."""
    items = list(seq)
    if len(items) != n or len(set(items)) != n:
        raise ValueError(f"labels must be {n} unique hashables")
    if all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in items):
        return np.array(items, dtype=np.int64)
    return np.array(items, dtype=object)
```

Non-numeric matrix input raises the numpy/`ValueError` naturally ("could not convert string to float"), which is acceptable. Unhashable labels raise `TypeError` from `set()`, also acceptable.

### 3.4 `skroute/base.py` — `RouterTags`, `BaseRouter`, `clone` (complete)

```python
from __future__ import annotations
import inspect, logging, warnings
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .problem import RoutingProblem
from .utils._param_validation import validate_parameter_constraints
from .utils.validation import check_random_state

log = logging.getLogger("skroute")

_FIT_KWARGS = ("depot", "coords", "labels", "max_time_work", "extra_cost", "people", "split")


@dataclass(frozen=True)
class RouterTags:
    kind: str = "metaheuristic"      # "exact" | "construction" | "local_search" | "metaheuristic" | "ensemble" (D28)
    exact: bool = False              # provably optimal for the objective it accepts; sets is_optimal_
    stochastic: bool = False         # consumes random_state; MultiStart-able
    iterative: bool = False          # sets history_, n_iter_, stop_reason_
    budget_aware: bool = False       # the search itself sees the multi-trip objective; solvers opt IN (D28)
    requires_symmetric: bool = False # raises on asymmetric X
    requires_coords: bool = False    # raises without coords=
    max_nodes: int | None = None     # hard cap on n (BruteForce, HeldKarp, MILP)

# budget_aware=True is declared explicitly by: BruteForce, ClarkeWright, TwoOpt, OrOpt, LocalSearch,
# IteratedLocalSearch, SimulatedAnnealing, TabuSearch, Genetic, AntColony, EnsembleGenetic,
# EnsembleSimulatedAnnealing; MultiStart delegates to its estimator. Everything else warns under a budget.


class BaseRouter:
    """Base class of every solver.

    Subclasses: store every __init__ argument verbatim as an attribute of the same name and
    nothing else; declare `_parameter_constraints`; override `_get_tags()`; implement
    `_solve(problem, rng) -> int64 array` (a permutation of range(n) with problem.depot first).
    Iterative solvers set self.history_, self.n_iter_, self.stop_reason_ inside _solve;
    exact solvers set self.is_optimal_. Everything else is set here.
    """

    _parameter_constraints: dict = {}

    # ---------- scikit-learn parameter protocol ----------
    @classmethod
    def _get_param_names(cls):
        sig = inspect.signature(cls.__init__)
        return sorted(p.name for p in sig.parameters.values()
                      if p.name != "self" and p.kind is not p.VAR_KEYWORD and p.kind is not p.VAR_POSITIONAL)

    def get_params(self, deep=True):
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{key}__{k}": v for k, v in value.get_params().items()})
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            return self
        valid = self._get_param_names()
        nested = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for estimator {self!r}. "
                                 f"Valid parameters are: {valid}.")
            if delim:
                nested.setdefault(key, {})[sub_key] = value
            else:
                setattr(self, key, value)
        for key, sub_params in nested.items():
            getattr(self, key).set_params(**sub_params)
        return self

    def __repr__(self):
        sig = inspect.signature(type(self).__init__).parameters
        parts = []
        for k, v in self.get_params(deep=False).items():
            default = sig[k].default
            same = (v is default) or (isinstance(v, type(default)) and not isinstance(v, np.ndarray) and v == default)
            if not same:
                parts.append(f"{k}={v!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __eq__(self, other):            # equality of type and parameters; used by tests and clone checks
        if type(self) is not type(other):
            return False
        a, b = self.get_params(deep=False), other.get_params(deep=False)
        return all(_param_equal(a[k], b[k]) for k in a)   # dict == would raise on an ndarray init=
    __hash__ = None

    # ---------- capability protocol ----------
    def _get_tags(self) -> RouterTags:
        return RouterTags()

    # ---------- template method ----------
    def _solve(self, problem: RoutingProblem, rng: np.random.Generator | None) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X, *, time_matrix=None, depot=None, coords=None, labels=None,
            max_time_work=None, extra_cost=0.0, people=1, split="greedy"):
        """Solve the instance and store the result in trailing-underscore attributes.

        Returns
        -------
        self
        """
        if isinstance(X, RoutingProblem):
            if time_matrix is not None or any(v is not None for v in (depot, coords, labels, max_time_work)) \
               or extra_cost != 0.0 or people != 1 or split != "greedy":
                raise ValueError("X is a RoutingProblem: pass it alone, without other fit arguments")
            problem = X
        else:
            problem = RoutingProblem(X, time_matrix=time_matrix, depot=depot, coords=coords, labels=labels,
                                     max_time_work=max_time_work, extra_cost=extra_cost,
                                     people=people, split=split)
        validate_parameter_constraints(self._parameter_constraints, self.get_params(deep=False),
                                       caller_name=type(self).__name__)
        tags = self._get_tags()
        name = type(self).__name__
        if tags.requires_symmetric and not problem.symmetric:
            raise ValueError(f"{name} requires a symmetric cost matrix")
        if tags.requires_coords and problem.coords is None:
            raise ValueError(f"{name} needs node coordinates: fit(X, coords=...)")
        if tags.max_nodes is not None and problem.n > tags.max_nodes:
            raise ValueError(f"{name} handles at most {tags.max_nodes} nodes, got {problem.n}; "
                             "raise max_nodes only if you accept the time/memory cost")
        if problem.multi_trip and not tags.budget_aware:
            if tags.exact:
                raise ValueError(f"{name} optimises the plain tour and cannot certify a multi-trip optimum; "
                                 "use BruteForce (n <= 11) or a heuristic solver")
            warnings.warn(f"{name} ignores max_time_work during its search; the result is still "
                          "split into trips and priced under the multi-trip objective", UserWarning, stacklevel=2)
        rng = check_random_state(getattr(self, "random_state", None)) if tags.stochastic else None
        self._reset_fitted()
        t0 = perf_counter()
        tour = self._solve(problem, rng)
        fit_time = perf_counter() - t0
        tour = np.ascontiguousarray(tour, dtype=np.int64)
        if tour.shape != (problem.n,) or tour[0] != problem.depot \
           or not np.array_equal(np.sort(tour), np.arange(problem.n)):
            raise RuntimeError(f"{name}._solve returned an invalid tour (bug in the solver): "
                               "expected a permutation of range(n) starting at the depot index")
        if tags.iterative:
            for attr in ("history_", "n_iter_", "stop_reason_"):
                if not hasattr(self, attr):
                    raise RuntimeError(f"{name}._solve must set {attr} (bug in the solver)")
            self.history_ = np.asarray(self.history_, dtype=np.float64)
        if tags.exact and not hasattr(self, "is_optimal_"):
            raise RuntimeError(f"{name}._solve must set is_optimal_ (bug in the solver)")
        self._set_results(problem, tour, fit_time)
        return self

    def _reset_fitted(self):
        for k in [k for k in vars(self) if k.endswith("_") and not k.startswith("_")]:
            delattr(self, k)

    def _set_results(self, problem, tour, fit_time):
        starts = problem.trip_starts(tour)
        lab = problem.labels
        d = lab[problem.depot : problem.depot + 1]              # 1-element array, keeps the label dtype
        self.problem_ = problem
        self.n_nodes_ = problem.n
        self.labels_ = lab.copy()
        self.depot_ = lab[problem.depot]
        self.tour_ = lab[tour]
        self.trips_ = [np.concatenate((d, lab[tour[a:b]], d)) for a, b in zip(starts[:-1], starts[1:])]
        self.route_ = np.concatenate([self.trips_[0]] + [t[1:] for t in self.trips_[1:]])
        self.n_trips_ = len(self.trips_)
        self.trip_costs_ = problem.trip_costs(tour, starts)
        if problem.multi_trip:
            self.trip_times_ = problem.trip_times(tour, starts)
        self.cost_ = float(problem.evaluate(tour))            # D2: recomputed, never reported
        self.fit_time_ = float(fit_time)


def _param_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def clone(estimator):
    """New unfitted estimator with the same parameters (deep copies nothing; parameters are values)."""
    params = estimator.get_params(deep=False)
    return type(estimator)(**{k: (clone(v) if hasattr(v, "get_params") else v) for k, v in params.items()})


def is_router(obj) -> bool:
    return isinstance(obj, BaseRouter)
```

**Top-level surface (`skroute/__init__.py`, lead)**: every solver class of D18 (AntColony/HeldKarp only if shipped), plus `RoutingProblem`, `BaseRouter`, `RouterTags`, `clone`, `is_router`, `all_solvers`, `check_router`, `set_log_level`, `__version__`, is lazily exported (PEP 562 `__getattr__`), listed in `skroute.__all__` and imported eagerly under `if TYPE_CHECKING:` so mypy and mkdocstrings see real types. `skroute.datasets`, `skroute.preprocessing`, `skroute.metrics`, `skroute.exceptions` and `skroute.utils` are imported as subpackages/modules. `all_solvers()` (D27) imports eagerly from the `__all__` lists of `exact`, `construction`, `local_search`, `metaheuristics` and `ensemble`, drops `MultiStart`, and returns the classes sorted by `__name__`. `set_log_level(level)`: `log.setLevel(level)`; if the logger's only handler is the `NullHandler` attached at import, add a `StreamHandler` (stderr) with the format `"%(name)s %(levelname)s %(message)s"`.

**Fitted attributes (exact types)** — set by `BaseRouter` for every solver:

| attribute | type / shape | meaning |
|---|---|---|
| `problem_` | `RoutingProblem` | the coerced instance (reusable: `Other().fit(est.problem_)`) |
| `n_nodes_` | `int` | n |
| `labels_` | 1-D array `(n,)`, label dtype | labels in matrix row order |
| `depot_` | scalar, label dtype | the depot's label |
| `tour_` | 1-D array `(n,)`, label dtype | open giant tour, depot first — the warm-start format |
| `route_` | 1-D array `(n + n_trips,)`, label dtype | as driven: depot, trip 1, depot, trip 2, …, depot |
| `trips_` | `list[np.ndarray]`, each closed `[depot, …, depot]` | one per trip; `len == 1` for plain TSP |
| `n_trips_` | `int` | `len(trips_)` |
| `trip_costs_` | `float64 (n_trips,)` | travel cost of each closed trip (fixed charge excluded) |
| `trip_times_` | `float64 (n_trips,)` | **only when a time matrix was given**; each `<= max_time_work + 1e-9` |
| `cost_` | `float` | `trip_costs_.sum() + fixed_cost * (n_trips_ - 1)` |
| `fit_time_` | `float` | seconds in `_solve` |
| `history_` | `float64 (n_iter_,)` | iterative only: best-so-far cost after each outer iteration |
| `n_iter_` | `int` | iterative only: outer iterations actually run |
| `stop_reason_` | `str` | iterative only: `"converged" | "max_iter" | "patience" | "time_limit"`. Allowed subsets: TwoOpt/OrOpt/LocalSearch/SOM `{"converged", "max_iter"}`; ILS/Tabu/Genetic/AntColony `{"max_iter", "patience", "time_limit"}`; SimulatedAnnealing `{"converged", "patience", "time_limit"}`; MultiStart and the Ensemble wrappers copy the best estimator's value |
| `is_optimal_` | `bool` | exact only |

`check_is_fitted(est)` (in `utils/validation.py`) raises `NotFittedError(f"This {name} instance is not fitted yet. Call 'fit' first.")` when `cost_` is missing.

`check_random_state(seed)`: `None` or `Integral` → `np.random.default_rng(seed)`; a `Generator` → returned as is (and therefore advanced by the fit); anything else → `TypeError("random_state must be None, an int or a numpy.random.Generator")`. Legacy `RandomState` is not accepted.

**Parameter validation** (`utils/_param_validation.py`, ≈120 lines, lead): `Interval(type, low, high, closed="both"|"left"|"right"|"neither")`, `Options(type, {values})`, `"array-like"`, `"random_state"`, `"boolean"`, `"verbose"`, `None`, and callables. Message format (sklearn's): `The 'alpha' parameter of SimulatedAnnealing must be a float in the range (0.0, 1.0). Got 1.5 instead.` Validation runs at fit time, never in `__init__`.

**Common helper** `utils/_init_tour.py::initial_tour(problem, init, rng) -> int64 tour`: `"nearest_neighbour"` → `core.nearest_neighbour_tour`; `"random"` → depot followed by `rng.permutation` of the rest (requires `rng`; raises `ValueError` if the solver is not stochastic); array-like → `problem.to_index_tour(init)`; any other string → `ValueError("init must be 'nearest_neighbour', 'random' or an array of labels")`.

**Three usage examples** — written in the exact shape every docstring, docs page and the README must use: a `>>>` session whose printed results are **platform-stable facts** (booleans, structural equalities, `.tolist()` of small arrays) for stochastic solvers, and exact numbers only for deterministic solvers and dataset constants (`optimal_tour_length`). numpy 2 prints scalars as `np.int64(0)` / `np.float64(1.0)`, so examples convert with `int()`, `float()`, `.item()` or `.tolist()` before printing. A line whose output is run-dependent carries `# doctest: +SKIP`. These three run under `pytest --doctest-modules` in `docs.yml` (§7, R7).

```python
>>> # 1. Plain TSP from numpy: Western Sahara, optimum 27603 -- labels are the file's 1-based ids
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")                       # TSPBunch: coords, labels, depot, optimal_tour_length, ...
>>> C = wi.distance_matrix()                    # metric="tsplib_euc_2d": nint rounding, optima match the literature
>>> ils = IteratedLocalSearch(random_state=0).fit(C, labels=wi.labels)   # route_ comparable with the published tour
>>> ils.cost_ / wi.optimal_tour_length < 1.03   # the fast-tier tolerance of §6; the run itself reaches 27603 on the author's machine
True
>>> int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == 1
True
>>> ils.n_iter_ == len(ils.history_) and ils.stop_reason_ in {"patience", "max_iter"}
True

>>> # 2. Multi-trip from Barcelona: 8-hour days, 12.83 EUR per extra day, two people.
>>> # Loader matrices are PLAIN ndarrays: pass labels=bcn.labels, otherwise depot=bcn.depot is not a label of X.
>>> import numpy as np
>>> from skroute import SimulatedAnnealing, TabuSearch, RoutingProblem
>>> from skroute.datasets import load_barcelona
>>> bcn = load_barcelona()                      # cost (EUR), time (h), labels (int64 ids), depot == 10000007
>>> sa = SimulatedAnnealing(random_state=0).fit(bcn.cost, time_matrix=bcn.time, labels=bcn.labels,
...                                             depot=bcn.depot, max_time_work=8.0, extra_cost=12.83, people=2)
>>> bool(np.all(sa.trip_times_ <= 8.0))          # every trip fits, return leg included
True
>>> int(sa.route_[0]) == int(sa.route_[-1]) == 10000007 and sa.n_trips_ == len(sa.trips_)
True
>>> problem = RoutingProblem(bcn.cost, time_matrix=bcn.time, labels=bcn.labels, depot=bcn.depot,
...                          max_time_work=8.0, extra_cost=12.83, people=2, split="optimal")
>>> costs = {type(est).__name__: est.fit(problem).cost_ for est in (sa, TabuSearch(random_state=0))}
>>> sorted(costs)                                 # one instance, several solvers, optimal split
['SimulatedAnnealing', 'TabuSearch']

>>> # 3. Legacy dict-of-dicts input; the depot is the first key, as 1.0's route_example[0]. Deterministic: exact output.
>>> from skroute import BruteForce
>>> cost = {1: {1: 0, 2: 5, 3: 9, 4: 10}, 2: {1: 5, 2: 0, 3: 4, 4: 8},
...         3: {1: 9, 2: 4, 3: 0, 4: 3}, 4: {1: 10, 2: 8, 3: 3, 4: 0}}
>>> hours = {1: {1: 0, 2: 1, 3: 2, 4: 2}, 2: {1: 1, 2: 0, 3: 1, 4: 2},
...          3: {1: 2, 2: 1, 3: 0, 4: 1}, 4: {1: 2, 2: 2, 3: 1, 4: 0}}
>>> bf = BruteForce().fit(cost, time_matrix=hours, max_time_work=4.0, extra_cost=3.0)
>>> bf.route_.tolist(), bf.cost_, bf.n_trips_
([1, 2, 3, 1, 4, 1], 41.0, 2)
```

Hand check of example 3 against the greedy rule (D1): the single trip 1-2-3-4-1 needs 1+1+1+2 = 5 h > 4, so it must break. Giant tour 1-2-3-4: leg 1→2 (t = 1), leg 2→3 (t = 2, return 2 → 4 h fits), leg 3→4 would need 2+1+2 = 5 h → close at 3, reopen at 4: trips [1,2,3,1] (cost 5+4+9 = 18) and [1,4,1] (cost 20); 18 + 20 + 3.0 = 41.0. The six giant tours evaluate to 41, 54, 41, 54, 41, 41; BruteForce returns the lexicographically first, 1-2-3-4 (§4.1). `split="optimal"` also gives 41.0 here ({2},{3,4} is infeasible: 2+1+2 = 5 h). This arithmetic must still be confirmed by the doctest (R7).

### 3.5 The core extension: `skroute/_core/_routing.pxd` (frozen contract)

Directives for every `.pyx` in the package (set in `setup.py`, not per file): `language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False, embedsignature=True`. Compiler flags: `-O3` unless MSVC (`/O2`). No `cimport numpy`. No OpenMP. `malloc`/`free` from `libc.stdlib`, never VLAs.

**How the contract is shared (verified with Cython 3.3.0 in the dev venv):**

- Every **`cdef inline`** function below is **defined with its full body in `_routing.pxd`** — that is how Cython inlines across extension modules. A body-less `cdef inline` declaration in a `.pxd` makes every cimporting module fail in the C compiler (`'inline' can only appear on functions`, because Cython emits a function-pointer import for the symbol); it does not "still link". The listing shows the signatures only for brevity; the reference bodies are the prototype's `skbench/bench_core.pyx`. `_routing.pyx` must not re-define them.
- The **non-inline `cdef`/`cpdef`** functions (`optimal_split_cost`, `problem_cost_py`, `trip_starts`, `trip_costs`, `trip_times`, `move_segment*`, `double_bridge`, `rebuild_pos`, the descents, `local_search_generic`, `nearest_neighbour_tour`) are declared body-less in the `.pxd` and defined in `_routing.pyx`; cimporting modules reach them through the module's C-API.
- **What is frozen** is the set of signatures and the documented semantics. WP1 owns the bodies and may change them without an R2 issue as long as the hypothesis suite stays green.
- `noexcept nogil` applies to every function below **except** `problem_cost_py` and `trip_starts`: those two hold the GIL, `malloc`/`free` their own `dp`/`pred` scratch for the optimal split (raising `MemoryError` on failure) and are the only two such functions in the file.
- The `cpdef enum SplitRule` is reached from Python as the IntEnum class **`_routing.SplitRule`** (`_routing.SplitRule.SPLIT_GREEDY == 0`); its members are not module attributes. `.pyx` files use the bare names via `cimport`; the `.pyi` declares `class SplitRule(IntEnum): SPLIT_GREEDY = 0; SPLIT_OPTIMAL = 1`; C-level code compares `int split` against the enum values.

```cython
# skroute/_core/_routing.pxd
from libc.stdint cimport int64_t, uint8_t

cpdef enum SplitRule:          # Python: _routing.SplitRule.SPLIT_GREEDY / .SPLIT_OPTIMAL (IntEnum)
    SPLIT_GREEDY = 0
    SPLIT_OPTIMAL = 1

# NOTE: every `cdef inline` function in this file carries its body here (elided in the SPEC listing).

# ------------------------------------------------------------------ evaluation
# C, T: (n, n) C-contiguous float64; tour: int64 permutation, tour[0] is the depot.
# max_time: +inf means plain TSP (T is then never read). fixed_cost = people * extra_cost.
# dp (float64[n]) and pred (int64[n]) are caller-owned scratch buffers used ONLY by the
# optimal split; pass zero-length views when split == SPLIT_GREEDY. Never share scratch across threads.

cdef inline double tour_cost(const double[:, ::1] C, const int64_t[::1] tour) noexcept nogil
cdef inline double greedy_split_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                     double max_time, double fixed_cost) noexcept nogil
cdef double optimal_split_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                               double max_time, double fixed_cost,
                               double[::1] dp, int64_t[::1] pred) noexcept nogil
cdef inline double problem_cost(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                                double max_time, double fixed_cost, int split,
                                double[::1] dp, int64_t[::1] pred) noexcept nogil
    # dispatch: max_time == inf -> tour_cost; split == SPLIT_GREEDY -> greedy; else optimal
cpdef double problem_cost_py(const double[:, ::1] C, const double[:, ::1] T, const int64_t[::1] tour,
                             double max_time, double fixed_cost, int split)
    # Python entry point; holds the GIL, malloc/frees its own dp/pred scratch (MemoryError on failure).
    # NOT noexcept nogil. Used by RoutingProblem.evaluate and tests.
cpdef Py_ssize_t trip_starts(const double[:, ::1] T, const int64_t[::1] tour, double max_time, int split,
                             const double[:, ::1] C, double fixed_cost, int64_t[::1] out)
    # writes out[0..k] (out[0] == 1, out[k] == n) and returns k = n_trips; out has length n + 1.
    # C and fixed_cost are needed only by the optimal split, for which it malloc/frees its own dp/pred
    # scratch while holding the GIL (NOT noexcept nogil, MemoryError on failure). Plain TSP -> k == 1.
cpdef void trip_costs(const double[:, ::1] C, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil       # closed-trip travel cost per trip
cpdef void trip_times(const double[:, ::1] T, const int64_t[::1] tour, const int64_t[::1] starts,
                      double[::1] out) noexcept nogil       # closed-trip duration per trip

# ------------------------------------------------------------------ move deltas
# Position domains (position 0 is the depot and never moves):
#   2-opt and swap: 1 <= i < j <= n-1.
#   Or-opt: 1 <= i, i + L - 1 <= n-1 (the segment never wraps), 0 <= j <= n-1 (j == 0 places the segment
#           right after the depot), j not in [i-1, i+L-1].
# Every [i..j] range in this file is INCLUSIVE, except double_bridge, whose segments are half-open.
# delta = cost(after) - cost(before) for the PLAIN closed tour. Successor of position n-1 is position 0.
cdef inline double two_opt_delta(const double[:, ::1] C, const int64_t[::1] tour,
                                 Py_ssize_t i, Py_ssize_t j) noexcept nogil
    # reverse tour[i..j], i < j. Exact iff the matrix is symmetric. O(1).
cdef inline double two_opt_delta_asym(const double[:, ::1] C, const int64_t[::1] tour,
                                      Py_ssize_t i, Py_ssize_t j) noexcept nogil
    # same move, exact for asymmetric matrices, O(j - i).
cdef inline double or_opt_delta(const double[:, ::1] C, const int64_t[::1] tour,
                                Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, bint reverse) noexcept nogil
    # move segment tour[i..i+L-1] (L in 1..3) so that it follows the node at position j
    # (j not in [i-1, i+L-1]); optionally reversed. reverse=False is exact for asymmetric
    # matrices; reverse=True is exact only when symmetric. O(1).
cdef inline double swap_delta(const double[:, ::1] C, const int64_t[::1] tour,
                              Py_ssize_t i, Py_ssize_t j) noexcept nogil
    # exchange the nodes at positions i and j; i < j required; adjacent (j == i+1) handled.
    # Exact for asymmetric matrices. O(1).

# ------------------------------------------------------------------ apply moves (in place)
# `_pos` variants keep pos[node] == position consistent; the plain variants do not touch pos.
cdef inline void reverse_segment(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j) noexcept nogil
    # reverses tour[i..j] INCLUSIVE, i < j — the move priced by two_opt_delta(C, tour, i, j)
cdef inline void reverse_segment_pos(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t j) noexcept nogil
    # same, inclusive
cdef inline void swap_positions(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t j) noexcept nogil
cdef inline void swap_positions_pos(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t j) noexcept nogil
cdef void move_segment(int64_t[::1] tour, Py_ssize_t i, Py_ssize_t L, Py_ssize_t j, bint reverse) noexcept nogil
    # the Or-opt move matching or_opt_delta; O(|i - j| + L) by rotating the affected span
cdef void move_segment_pos(int64_t[::1] tour, int64_t[::1] pos, Py_ssize_t i, Py_ssize_t L,
                           Py_ssize_t j, bint reverse) noexcept nogil
cpdef void double_bridge(const int64_t[::1] tour, Py_ssize_t p1, Py_ssize_t p2, Py_ssize_t p3,
                         int64_t[::1] out) noexcept nogil
    # A B C D -> A C B D with A = tour[0..p1), B = [p1..p2), C = [p2..p3), D = [p3..n); 1 <= p1 < p2 < p3 <= n-1.
    # Orientation-preserving: exact on ATSP. Writes to out (length n).
cpdef void rebuild_pos(const int64_t[::1] tour, int64_t[::1] pos) noexcept nogil

# ------------------------------------------------------------------ descents
# Return value = cost_after - cost_before, always <= 0 (callers do `cost += returned`); 0.0 means "local
# optimum for this move, nothing changed". A pass = one sweep over the nodes whose don't-look bit is active.
# cand: int64 (n, k) candidate lists from RoutingProblem.neighbours(k); dont_look: uint8[n] (0 = active).
# Both use Bentley's neighbour-list scan with the pruning `C[a, succ(a)] > C[a, c]` and reset the
# don't-look bits of the touched endpoints on improvement. Stop at a local optimum or after max_passes.
# The pos/cand/dont_look buffers are caller-owned and persist across calls (LocalSearch calls with max_passes=1).
cpdef double two_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                             const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                             bint first_improvement, int max_passes) noexcept nogil
cpdef double or_opt_descent(const double[:, ::1] C, int64_t[::1] tour, int64_t[::1] pos,
                            const int64_t[:, ::1] cand, uint8_t[::1] dont_look,
                            int max_segment, bint allow_reverse, int max_passes) noexcept nogil
cpdef double local_search_generic(const double[:, ::1] C, const double[:, ::1] T, int64_t[::1] tour,
                                  int64_t[::1] pos, const int64_t[:, ::1] cand, double max_time,
                                  double fixed_cost, int split, int moves, int max_segment, int max_passes,
                                  int64_t[::1] scratch_tour, double[::1] dp, int64_t[::1] pred) noexcept nogil
    # Full-re-evaluation FIRST-IMPROVEMENT descent over the candidate neighbourhoods for the
    # multi-trip objective and/or asymmetric matrices. moves is a bit mask: 1 = two_opt,
    # 2 = or_opt (no reversal, segment lengths 1..max_segment), 4 = swap. `first_improvement=False`
    # is honoured only on the symmetric plain path (TwoOpt/LocalSearch document this).
    # O(n) per candidate move; documented ceiling ~2000 nodes.

# ------------------------------------------------------------------ construction
cpdef void nearest_neighbour_tour(const double[:, ::1] C, int64_t depot, int64_t[::1] out) noexcept nogil
```

Semantics of the two decoders, stated once (kernels implement exactly this):

- **greedy**: `t = 0; trips = 1; cost = 0`; for `k in 0..n-2`: `a = tour[k], b = tour[k+1]`; if `t + T[a,b] + T[b,d] <= max_time` then `t += T[a,b]; cost += C[a,b]` else `cost += C[a,d] + C[d,b]; t = T[d,b]; trips += 1`; finally `cost += C[tour[n-1], d]`; return `cost + (trips-1) * fixed_cost`. A trip start is every position `k+1` where the else-branch fired, plus position 1. *(This is the prototype's `multitrip_cost`, verified.)*
- **optimal** (Prins): customers `c_1..c_m` = `tour[1..n-1]`; `dp[0] = 0`; for `j in 0..m-1`: open a trip at `c_{j+1}`, extend `i = j+1, j+2, …` while the **open path** `open = T[d,c_{j+1}] + Σ_{k<i} T[c_k,c_{k+1}] <= max_time` — this quantity is monotone in `i`, so breaking there is exact; a closed trip `open + T[c_i,d] <= max_time` is NOT monotone when `T` violates the triangle inequality (road matrices occasionally, the asymmetric test fixtures systematically), so the loop must not stop at the first infeasible closed trip. For each `i` whose closed trip is feasible: `cand = dp[j] + C[d,c_{j+1}] + Σ C + C[c_i,d] + (fixed_cost if j > 0 else 0)`; if `cand < dp[i]`: `dp[i] = cand; pred[i] = j`. Return `dp[m]`; trip starts by following `pred` from `m`. O(n·L), L = longest feasible open path. `reference.optimal_split` implements exactly this rule (its O(n²) DP tests every `(j, i)` pair, which is the same set once the open-path bound is applied). Because D5 guarantees every single-customer trip is feasible, `dp[m]` is always finite.

`skroute/_core/__init__.py`:

```python
try:
    from . import _routing
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-route's compiled core is missing. Install a wheel (pip install scikit-route) "
                      "or build from source with a C compiler: pip install -e .") from e
```

`skroute/metrics.py` (public recomputation, used by tests and users): `route_cost(X, route, *, depot=None, labels=None, time_matrix=None, max_time_work=None, extra_cost=0.0, people=1, split="greedy") -> float` builds a `RoutingProblem`, converts the label route with `to_index_tour` and returns `evaluate`. **`depot=None` means `route[0]`** (the first label of the route — so a `route_`/`tour_` produced with any `depot=` re-evaluates without repeating the depot); a given `depot` must equal `route[0]`, else `ValueError("depot must be the first label of route")`. `split_trips(route, depot=None) -> list[np.ndarray]` splits a label route at depot occurrences (`depot=None` → `route[0]`) and returns **closed** trips `[depot, …, depot]` in route order; an open tour (depot only at the front) is returned as one closed trip. `reference.route_cost_from_labels` mirrors the same default.

---

## 4. Per-solver specifications

### 4.0 Glossary (mandatory names)

`n_iter` (outer iterations), `max_passes` (outer iterations of a local search, see §4.3), `patience` (outer iterations without improvement of the best-so-far before stopping; `None` disables; SA counts them only once the current cost has first fallen below the initial cost, see §4.4), `time_limit` (seconds, checked once per outer iteration, `None` disables; documented as breaking bit-exact reproducibility), `init` (`"nearest_neighbour" | "random" | array-like of labels`), `n_candidates` (k of the candidate lists; `None` = all `n-1` nodes, i.e. the full neighbourhood), `random_state`, `verbose` (0 silent; 1 logs every `max(1, n_outer // 10)` outer iterations at INFO; 2 logs every iteration; every `verbose` docstring ends with the D24 sentence on enabling the `skroute` logger), `max_nodes`. **`local_search`** (one meaning everywhere): `None`, or a tuple of move names ⊆ `{"two_opt", "or_opt"}`; a single string is accepted and normalised to a 1-tuple; there is no `"both"`. Defaults: ILS `("two_opt", "or_opt")`, Genetic `None`, AntColony `("two_opt",)`. SA: `t0`, `t_min`, `alpha`, `n_moves`, `moves`. GA: `pop_size`, `n_generations`, `crossover`, `p_crossover`, `mutation`, `p_mutation`, `tournament_size`, `n_elite`, `local_search`. Tabu: `tenure`. ILS: `perturbation_strength`, `acceptance`, `temperature`, `local_search`. ACO: `n_ants`, `alpha`, `beta`, `rho`. SOM: `n_units`, `learning_rate`, `lr_decay`, `radius`, `radius_decay`. Clarke–Wright: `shape` (savings weight λ in `s_ij = C[d,i] + C[j,d] - shape·C[i,j]`). Ensemble: `estimator`, `n_restarts`, `n_jobs`, `prefer`. Move names: `"two_opt"`, `"or_opt"`, `"swap"`. Improvement test everywhere: `new < best - 1e-9 * max(1.0, abs(best))`.

Every iterative solver: `history_[k]` = best-so-far after outer iteration `k`; `n_iter_ = len(history_)`; `stop_reason_`. Every stochastic solver draws all randomness from the `rng` handed to `_solve`, in batches per outer iteration (D10). Generic path = `local_search_generic` / full evaluation when `problem.multi_trip or not problem.symmetric`.

Acceptance tests below are in addition to `check_router` (§6). "tiny" = the `tiny_instance` fixture (n = 5..9 symmetric and asymmetric Euclidean instances with coordinates, optimum by `reference.brute_force`); "alicante" = the multi-trip fixture (8 nodes, budget = 1.5 × max round trip, optimum by `reference.brute_force` under both split rules). Gaps are `cost_/optimum - 1`. Tiny/fast-tier/alicante acceptance tests live in the WP's own test file; **every slow-tier gap test lives in `tests/benchmarks/test_waterloo.py` (WP8)**, driven by `tests/tolerances.py` and parametrised over `all_solvers()` — WP files never contain slow-tier tests.

### 4.1 `skroute.exact`

**`BruteForce(max_nodes=11)`** — `exact/_brute_force.py` + `_brute.pyx`. Tags: `kind="exact", exact, budget_aware, max_nodes`. Enumerates the `(n-1)!` permutations of positions `1..n-1` in **lexicographic order by `next_permutation`** (Knuth 7.2.1.2 Algorithm L, O(1) amortised) inside a `nogil` loop — not Heap's algorithm, whose order is not lexicographic — evaluating each with `problem_cost` (so it is exact for the chosen split rule too) and keeping the first strictly better tour; when `problem.symmetric and not problem.multi_trip` it skips permutations with `tour[1] > tour[n-1]` to halve the work (reversal changes the split, so never under a budget; the kept orientation is the lexicographically smaller one). `is_optimal_ = True`. Ties: the lexicographically first optimal permutation, identical to `itertools.permutations` in `reference.brute_force`. Complexity O((n-1)!·n); 10! at n = 11 ≈ 3.6 M tours in seconds. Acceptance: equals `reference.brute_force` on every tiny instance (symmetric, asymmetric) and on alicante under `split="greedy"` and `"optimal"`.

**`HeldKarp(max_nodes=20)`** — `_held_karp.py` + `_held_karp.pyx`. **Priority P1 (D26)**: ships only if WP2 finishes it by the feature freeze; MILP already solves n ≤ 20 instantly, so a deferral costs users nothing. Tags: `kind="exact", exact, budget_aware=False, max_nodes`. Bitmask DP over the `n-1` non-depot nodes: `dp[S][j]` = cheapest path from the depot through set `S` ending at `j`, `int8`/`int64` parent table for reconstruction, `malloc`ed (2^(n-1)·(n-1) doubles ≈ 80 MB at n = 20; the cap is documented, `max_nodes` may be raised to 23). Direction-aware, so ATSP works. Raises under a budget (D6). `is_optimal_ = True`. Acceptance: equals BruteForce on every tiny instance; runs wi29? no (29 > 20) — runs `n = 16` random in < 0.1 s.

**`MILP(time_limit=60.0, max_nodes=300, mip_rel_gap=0.0)`** — `_milp.py`. Tags: `kind="exact", exact, budget_aware=False, max_nodes`. Dantzig–Fulkerson–Johnson with lazy subtour cuts on `scipy.optimize.milp` (HiGHS): symmetric → `n(n-1)/2` binary edge variables with degree-2 equalities; asymmetric → `n(n-1)` arc variables with in/out-degree-1 equalities. Loop: solve; round `x`; find connected components of the support (`scipy.sparse.csgraph.connected_components`, weak); if one component → optimal; else add one cut per component `Σ_{e ⊂ S} x_e <= |S| - 1` (edges inside S, or arcs with both ends in S) and re-solve with the remaining wall budget. DFJ over MTZ because MTZ's relaxation is weak (hours at n ≈ 200 where DFJ took 39 s on qa194). `lower_bound_` = the **largest valid bound seen**: the objective of every solve that returned `res.status == 0` (a DFJ relaxation solved to optimality is a valid bound) and `res.mip_dual_bound` of a time-limited solve when it is not `None`; `res.fun` of a time-limited solve is the PRIMAL value of the incumbent (an upper bound) and is never used as a bound, and a solve that stops without an incumbent returns `fun is None` and `mip_dual_bound is None` (verified with scipy 1.18), in which case the previous bound is kept. When the last solve returned an integral single-component solution, `lower_bound_ == cost_` and `is_optimal_ = True`. On time-out: `is_optimal_ = False`, the returned tour is the best single-component integral solution seen or, failing that, `core.nearest_neighbour_tour` polished by `core.two_opt_descent` (WP2 depends on core M2 for this path only; never another WP's solver); `gap_ = max(0.0, (cost_ - lower_bound_) / cost_)`, `n_cuts_`, `n_solves_`. Realistic sizes: ~200 symmetric nodes within a minute, ~60 asymmetric. Acceptance: equals BruteForce on tiny (symmetric and asymmetric); `cost_ == 27603` on wi29 and `6656` on dj38 (fast tier); `9352` on qa194 (`@slow`, `time_limit=150`).

### 4.2 `skroute.construction`

**`NearestNeighbour()`** — Tags: `kind="construction", budget_aware=False`. Greedy from the depot over `C`, ties by lowest index; O(n²) in the core kernel. Acceptance: valid; tiny gap ≤ 0.50; fast tier ≤ 0.50 (measured with exactly this rule: wi29 31.8 %, dj38 46.4 %; slow tier qa194 24.5 %, uy734 25.4 %, zi929 19.5 %, lu980 26.7 % — recorded in `benchmarks.md`).

**`Insertion(strategy="farthest")`**, `strategy in {"farthest", "cheapest", "nearest"}` — `_insertion.py` + `_insertion.pyx`. Start with the depot and the farthest (cheapest: nearest) node; repeatedly pick the unrouted node by the strategy (farthest: max over unrouted of min distance to the partial tour; cheapest: min insertion cost; nearest: min distance to the tour) and insert it at its cheapest position; incremental `min_dist[node]` array; O(n²); direction-aware so exact on ATSP. Tags: `kind="construction", budget_aware=False`. There are no `CheapestInsertion`/`FarthestInsertion` classes (D18). Acceptance: tiny ≤ 0.30; fast tier: farthest ≤ 0.25, cheapest ≤ 0.30 (the tolerance tests run `Insertion(strategy=s)` for each strategy).

**`ClarkeWright(shape=1.0)`** — Tags: `kind="construction", requires_symmetric, budget_aware=True`. Parallel savings: every non-depot node starts as its own trip with creation index = its node index; savings `s_ij = C[d,i] + C[j,d] - shape·C[i,j]` sorted descending (stable, then by `(i, j)`); merge two trips at their endpoints when neither is interior and, under a budget, the merged closed trip's duration `<= max_time_work`; a merged trip keeps the **smaller creation index** of its two parts; without a budget it degenerates to the greedy-edge heuristic and merges everything into one tour. The returned giant tour is the concatenation of the trips in increasing creation index, each oriented so that its first node is the endpoint with the smaller `C[d, ·]` (ties: the smaller node index); the base class re-decodes it (greedy may merge trips further and never breaks feasibility; the docs state that the reported trips may differ from the savings trips). O(n² log n). Acceptance: fast tier ≤ 0.25; alicante ≤ 0.25 gap under greedy split; refuses an asymmetric matrix.

**`NRBS(mean_priority=1.0, std_priority=1.0, mean_connection=1.0, std_connection=1.0, distance_weight=1.0)`** — Tags: `kind="construction", budget_aware=False`. The five exponents default to the value `1.0` (1.0.0a2 had no defaults at all and rejected ints); 1.0's misspelt `distance_weigth` is renamed `distance_weight`. Faithful port of the 2020 heuristic: per node, `μ_i` and `σ_i` over the full row of `C` **including the zero diagonal** (population std), priority `μ_i^a · σ_i^b` sorted descending; connection score of `j` for `i`: `μ_j^c · σ_j^e / max(C[i,j], 1e-12)^f` sorted descending (the depot participates like any node; pairs with `C[i,j] == 0` — coincident points, present in lu980 and rw1621 — therefore get the maximum score and are connected first, instead of an `inf`/`nan` whose sort order is platform-dependent). Two passes over the priority order: for each node with degree < 2, connect it to the highest-scoring candidate with degree < 2 that is not already its neighbour and does not close a cycle before all nodes are covered — cycle detection by union-find plus degree counters (replacing the `deepcopy` per candidate), which preserves the 2020 selection order and therefore its results; the hard-coded 19 becomes `n`. The resulting Hamiltonian path is closed and rotated to the depot; if the passes leave the graph disconnected (degenerate ties), the remaining endpoints are joined greedily by nearest endpoint. Complexity O(n² log n). Acceptance: valid on tiny; on Barcelona with all five parameters `= 0.5` the plain tour **sequence and cost** equal the 1.0 result pinned in `tests/data/nrbs_barcelona_1_0.json` (`abs=0.01` on the cost). Pinning procedure (once, WP3): the 1.0 tree is gone from `modernization/v2` (commit `e15a6af`), so `git worktree add /tmp/skroute10 533f320`, a Python ≤ 3.11 venv (the 1.0 `_cost` kernel is Cython 0.29 and returns a C `float`, i.e. float32), replace `skroute._utils._utils._cost` by `tests/reference.tour_cost` (float64), run `NRBS(0.5, 0.5, 0.5, 0.5, 0.5).fit(10000007, ids, cost_dict)` on Barcelona built with `pairs_to_matrix` + `to_dict_of_dicts`, store `{"cost": ..., "route": [...]}` in the JSON with the commit and the command in a `"provenance"` field. Proposals reported 446.59 — verify, never copy. Fast tier ≤ 0.50.

### 4.3 `skroute.local_search`

Common parameters: `init="nearest_neighbour"`, `n_candidates=10`, `max_passes=50`, `verbose=0` (no `time_limit`, no `patience`, no `random_state`: these three descents are deterministic). Tags: `kind="local_search"`, `budget_aware=True`, `iterative=True`. **Iteration accounting (binding for all three):** one outer iteration = one call of each listed descent kernel with `max_passes=1`; the `pos`, `cand` and `dont_look` buffers are allocated once and persist across calls; `history_[k]` = cost after outer iteration `k` (`= history_[k-1] + Σ returned gains`, gains being `cost_after - cost_before <= 0`); `n_iter_` = outer iterations run; `stop_reason_ = "converged"` when an iteration returns `0.0` for every listed move, `"max_iter"` after `max_passes` iterations — the only two values these solvers emit. Symmetric plain TSP → `two_opt_descent`/`or_opt_descent` with `pos` and don't-look bits; otherwise → `local_search_generic` with the corresponding move mask and `max_segment` (generic path is first-improvement only; `first_improvement=False` is honoured only on the symmetric plain path — documented). `n_candidates=None` scans the full neighbourhood (`neighbours(n-1)`), which gives a markedly better local optimum at small n (lu980: 9.4 % vs 16.2 % for 2-opt alone); the default stays 10 for speed.

**`TwoOpt(first_improvement=True, ...)`** — 2-opt only. Acceptance: never worse than its `init`; tiny ≤ 0.10; fast tier ≤ 0.20.
**`OrOpt(max_segment=3, ...)`** — segment relocation 1..`max_segment`, both orientations when symmetric. Acceptance: tiny ≤ 0.12; fast tier ≤ 0.25.
**`LocalSearch(moves=("two_opt", "or_opt"), first_improvement=True, max_segment=3, ...)`** — alternates the listed descents (iteration accounting above) until none improves. `moves` ⊆ `{"two_opt", "or_opt"}`; `"swap"` is reachable only through `local_search_generic`'s mask and is **not** accepted here (`Options`). Acceptance: tiny ≤ 0.10; fast tier ≤ 0.12; slow (qa194, lu980) ≤ 0.15 (measured 11.9 % / 7.8 % from NN with 10-NN lists; the margin absorbs scan-order differences).
**`IteratedLocalSearch(n_iter=1000, patience=100, perturbation_strength=1, acceptance="better", temperature=None, local_search=("two_opt", "or_opt"), n_candidates=10, init="nearest_neighbour", time_limit=None, random_state=None, verbose=0)`** — Tags: `kind="local_search", stochastic, iterative, budget_aware`. Descend from `init` (same iteration accounting as `LocalSearch`, run to convergence); then each outer iteration: copy the incumbent, apply `perturbation_strength` kicks, descend to convergence, accept if better (`"better"`) or with Metropolis probability at fixed `temperature` (`"metropolis"`, `temperature=None` → 0.5 % of the initial cost); track the best. **Kick:** for `n >= 8` a double bridge with `p1 < p2 < p3 = np.sort(rng.choice(np.arange(1, n), size=3, replace=False))` (a sample without replacement, never three sorted uniforms that may coincide), pre-drawn per iteration; for `n < 8` (double bridge is undefined at n = 3 and degenerate below 8) the kick is a random segment reversal `reverse_segment(tour, i, j)` with `i < j` drawn without replacement from `1..n-1` (at n = 3 the only kick is the swap (1, 2)). `local_search` per the glossary (tuple of moves). Multi-trip and ATSP use the generic descent. `stop_reason_ ∈ {"max_iter", "patience", "time_limit"}`. This is the recommended default solver. Acceptance: tiny and alicante equal the optimum at seeds 0, 1, 2; fast tier ≤ 0.03; slow ≤ 0.06 (measured 4.16 % on lu980, stopped by `patience` at iteration 340).

### 4.4 `skroute.metaheuristics`

**`SimulatedAnnealing(t0="auto", t_min="auto", alpha=0.995, n_moves=None, moves=("two_opt", "or_opt", "swap"), patience=None, init="nearest_neighbour", time_limit=None, random_state=None, verbose=0)`** — `_simulated_annealing.py` + `_sa.pyx`. Tags: `kind="metaheuristic", stochastic, iterative, budget_aware`. Outer iteration = one temperature level of `n_moves` (default `10·n`) proposals; per level Python pre-draws `u ~ U(0,1)`, `ri, rj ~ U{1..n-1}`, `mv ~ U{moves}` and calls `anneal_level(...)`. **Draw → move mapping (binding):** 2-opt: `i, j = min(ri, rj), max(ri, rj)`, invalid if `i == j`; Or-opt: `i = ri`, `L = 1 + (rj % 3)`, `j = rj`, invalid if `i + L - 1 > n-1` or `j in [i-1, i+L-1]`; swap: `i, j = min, max`, invalid if `i == j`. **An invalid draw is a rejected proposal**: it counts towards `n_moves`, consumes its `u` and changes nothing — so `n_moves` means "proposals per level" identically for SA and `EnsembleSimulatedAnnealing`. Symmetric plain path: O(1) deltas (`two_opt_delta`, `or_opt_delta`, `swap_delta`), apply on accept. Generic path: copy `tour → scratch`, apply on scratch, `problem_cost(scratch)`, accept by copying back. The kernel keeps a **separate best buffer** and copies into it on strict improvement (the 1.0 aliasing bug is impossible). `t0="auto"`: sample 1000 random moves on the initial tour and set `t0 = -median(Δ⁺)/ln(0.5)` so a median uphill move is accepted with probability 0.5 (`t0 = 1.0` if no uphill move); `t_min="auto"` → `1e-4·t0`. Geometric cooling `T *= alpha`. Stop when `T < t_min` (`"converged"`), `patience` levels without improvement (`"patience"`), or `time_limit`. **`patience=None` by default**: with `t0="auto"` the hot phase is non-improving by design — measured with the prototype kernel, the current cost first drops below the NN start at level 436 (wi29), 413 (dj38), 836 (qa194), 1011 (uy734), 1023 (lu980), so a `patience=50` default would return the NN tour (25–46 % gap). When a user sets `patience`, the count starts only once the current cost has first fallen below the initial cost (glossary); the docstring says so. `t0_` is stored. `stop_reason_ ∈ {"converged", "patience", "time_limit"}`. Acceptance: tiny and alicante optimum at seeds 0, 1, 2; fast tier ≤ 0.03; slow ≤ 0.10 (measured 0.0/0.0/3.5/5.4/7.0 % for wi29/dj38/qa194/uy734/lu980 with a 2-opt-only kernel; the three-move mix is untested, hence the margin); two fits with `random_state=0` bit-identical.

**`TabuSearch(n_iter=1000, tenure="auto", n_candidates=10, patience=200, init="nearest_neighbour", time_limit=None, random_state=None, verbose=0)`** — `_tabu_search.py` + `_tabu.pyx`. Tags: `kind="metaheuristic", stochastic, iterative, budget_aware`. Rewrite. Neighbourhood: 2-opt moves `(i, j)` where `tour[j]` is in the candidate list of `tour[i-1]` (symmetric plain path, O(1) deltas); generic path: for each such candidate pair, the 2-opt move and the no-reversal Or-opt relocations of the segments starting at `i` with `L = 1, 2, 3`, with full evaluation. Each iteration applies the best admissible move even if worsening. **Tabu attributes are the removed edges** (arcs when asymmetric): `tabu_until[a, b]` **int32** `(n, n)` matrix (tenure counters never exceed `n_iter`; int64 would be 905 MB at n = 10 639 — the documented practical ceiling is ~5 000 nodes, 100 MB); a move is tabu if any edge it *adds* is tabu, unless it beats the incumbent (aspiration). `tenure="auto"` → `rng.integers(ceil(sqrt(n)), 2*ceil(sqrt(n)) + 1, size=n_iter)`, i.e. uniform on `[⌈√n⌉, 2⌈√n⌉]` **both ends inclusive**, pre-drawn (robust tabu); an `int` is a fixed tenure (`Interval(Integral, 1, None, closed="left")`). `history_` = best-so-far per iteration; `stop_reason_ ∈ {"max_iter", "patience", "time_limit"}`. Acceptance: tiny and alicante optimum at seeds 0, 1, 2; fast tier ≤ 0.08; slow ≤ 0.15.

**`Genetic(pop_size=100, n_generations=500, crossover="ox", p_crossover=0.9, mutation="inversion", p_mutation=0.2, tournament_size=3, n_elite=2, local_search=None, patience=100, init="nearest_neighbour", time_limit=None, random_state=None, verbose=0)`** — `_genetic.py` + `_ga.pyx`. Tags: `kind="metaheuristic", stochastic, iterative, budget_aware`. Chromosome = `tour[1:]` (int64 `(pop_size, n-1)` array); initial population: one individual from `init`, the rest random permutations; fitness = `problem_cost` of every row in one kernel call (`evaluate_population`). Per generation: tournament selection (size `tournament_size`), **real crossover** — `"ox"` (a segment `[a, b]` of parent 1, the rest in parent-2 order) or `"pmx"` (segment copy plus mapping repair) — with probability `p_crossover`, mutation with probability `p_mutation` per child: `"inversion"` (reverse a random segment = a 2-opt move), `"swap"`, `"insertion"`; optional memetic polish `local_search` (glossary: `None` or a tuple ⊆ `{"two_opt", "or_opt"}`, a single string normalised to a 1-tuple) applied to every child with the §4.3 descents run to convergence (generic path under multi-trip/ATSP); generational replacement with the `n_elite` best parents kept and exact-duplicate children rejected (re-mutated once). The caller's data is never mutated. All randomness pre-drawn per generation (`pop_size` pairs of cut points, tournament indices, uniforms). Stop: `n_generations` (`"max_iter"`), `patience`, `time_limit`. Acceptance: tiny and alicante optimum at seeds 0, 1, 2; plain GA: fast tier ≤ 0.15, slow ≤ 0.30 on qa194 only (measured with exactly these defaults: 4.9 % wi29, 8.6 % dj38, 18.8 % qa194; at n ≈ 1000 a 50 000-evaluation GA essentially returns its elitist NN individual, so no plain slow test there); memetic `local_search=("two_opt",)`: fast tier ≤ 0.05, slow ≤ 0.08 on qa194 and lu980 (`@slow`, several minutes: pop × generations 2-opt descents); OX/PMX children are permutations (hypothesis).

**`AntColony(n_ants=None, alpha=1.0, beta=2.0, rho=0.02, n_iter=200, n_candidates=20, local_search=("two_opt",), patience=50, time_limit=None, random_state=None, verbose=0)`** — `_ant_colony.py` + `_aco.pyx`. Tags: `kind="metaheuristic", stochastic, iterative, budget_aware`. **Priority P1 (D26): ships in 2.0 only if WP6 finishes it with tests and docs by the feature freeze; otherwise it moves to 2.1 and its name is not exported.** Deferral procedure (applies to every P1 item, with its own files): WP6's deferral PR removes `docs/api/ant_colony.md` and its nav line, the CHANGELOG line, its rows in `tests/tolerances.py`, and deletes `tests/test_ant_colony.py`, `_ant_colony.py`, `_aco.pyx`; the lead removes the export from `metaheuristics/__init__.py` and `skroute/__init__.py` in the same PR (the one cross-ownership PR allowed besides R2); the migration guide never mentions a deferred item. `stop_reason_ ∈ {"max_iter", "patience", "time_limit"}`. MAX–MIN Ant System: `n_ants=None → min(n, 50)`; pheromone `τ` (n, n) initialised to `1/(ρ·L_NN)` with bounds `τ_max = 1/(ρ·L_best)`, `τ_min = τ_max/(2n)`; each ant starts at the depot and chooses the next node by roulette over `τ^α · (1/C)^β` restricted to the unvisited candidate list (all unvisited nodes when the list is exhausted), consuming one pre-drawn uniform per step; optional 2-opt polish per ant; the deposit uses the **problem cost** (so the budget steers the colony) of the iteration-best ant, alternating with the global best every 5 iterations; evaporation `τ *= (1-ρ)`. Acceptance: tiny optimum at seeds 0, 1, 2; fast tier ≤ 0.08; slow ≤ 0.15 (n ≤ 1000).

**`SOM(n_units=None, learning_rate=0.8, lr_decay=0.99997, radius=None, radius_decay=0.9997, n_iter=100_000, random_state=None, verbose=0)`** — `_som.py`, numpy only. Tags: `kind="metaheuristic", stochastic, iterative, budget_aware=False, requires_coords`. 1-D Kohonen ring over min-max-normalised coordinates (aspect-preserving); `n_units=None → 8n`, `radius=None → n_units/10`; each sample picks one random city (pre-drawn indices), finds the winner by vectorised argmin, updates the ring with a Gaussian of the ring distance (wrapped) and decays `learning_rate` and `radius`. **Outer iteration = an epoch of `max(1, n_iter // 100)` samples.** After each epoch the ring is decoded to a tour (cities ordered by winner index, ties by city index, rotated to the depot) and evaluated with `problem.evaluate`; `history_[k]` = best-so-far cost up to epoch `k` (monotone, D9/R8 — never the current cost), the returned tour is the best epoch's tour; `n_iter_` = epochs run; `n_samples_` = samples actually drawn; `stop_reason_ = "converged"` (radius < 1 or learning_rate < 1e-3, checked at epoch end) or `"max_iter"`. No `time_limit`/`patience` parameters. **Ceiling (stated in the docstring, `problem_model.md` and migration item 9):** 2.0 evaluates every solution on a dense cost matrix, so SOM too needs the `(n, n)` matrix — the practical ceiling is ~20 000 nodes (3.2 GB); the four bundled instances above it (vm22775, sw24978, bm33708, ch71009) can be read and subsampled (`load_tsp(name, n_nodes=5000)`) but not solved whole; coordinate-only fitting (`fit(None, coords=xy)` with Euclidean cost on the fly) is deferred to 2.1 with on-the-fly distances (D18). Acceptance: fast tier ≤ 0.15 on wi29/dj38 coordinates; slow ≤ 0.15 on qa194; raises `ValueError` without `coords`; `pytest.skip` on asymmetric fixtures (no meaningful coordinates).

### 4.5 `skroute.ensemble`

**`MultiStart(estimator, n_restarts=10, n_jobs=None, prefer="threads", random_state=None, verbose=0)`** — `_multistart.py`. Tags: delegate to the estimator's tags with `kind="ensemble"`, `stochastic=True`. Not in `all_solvers()` (needs an estimator, D27); `tests/test_ensemble.py` runs `check_router(MultiStart(SimulatedAnnealing(), n_restarts=4))` and the `n_jobs` invariance test. Refuses a non-stochastic estimator (`ValueError("MultiStart needs a stochastic estimator (one with random_state)")`). `_solve`: `seeds = np.random.SeedSequence(<int drawn from rng>).spawn(n_restarts)`; `clone(estimator).set_params(random_state=np.random.default_rng(seed_k))`; `joblib.Parallel(n_jobs, prefer=prefer)` runs `est.fit(problem)` on the **shared** `RoutingProblem`; keeps the best `cost_` (ties → lowest index); exposes `estimators_`, `costs_` (float64 `(n_restarts,)`), `best_index_`, `best_estimator_`, and copies `history_`/`n_iter_`/`stop_reason_` from the best estimator when present. Results are identical for any `n_jobs`. `get_params(deep=True)` exposes `estimator__*`. Threads by default because kernels are `nogil` and a 900 MB matrix must not be pickled; the docs say GA/SOM scale less on threads.

**`EnsembleGenetic(n_genetics=10, n_jobs=None, random_state=None, verbose=0, *, pop_size=100, n_generations=500, crossover="ox", p_crossover=0.9, mutation="inversion", p_mutation=0.2, tournament_size=3, n_elite=2, local_search=None, patience=100, init="nearest_neighbour", time_limit=None)`** and **`EnsembleSimulatedAnnealing(n_simulateds=10, n_jobs=None, random_state=None, verbose=0, *, t0="auto", t_min="auto", alpha=0.995, n_moves=None, moves=("two_opt","or_opt","swap"), patience=None, init="nearest_neighbour", time_limit=None)`** — `_legacy.py`. Subclasses of `BaseRouter`; the inner knobs take the 2.0 defaults of §4.4 (1.0 defaulted `n_simulateds=20`, now 10 — a changed default, listed in the migration table). Tags: `kind="ensemble", stochastic=True, iterative=True, budget_aware=True`. `_solve(problem, rng)`: `ms = MultiStart(Genetic(**inner), n_restarts=self.n_genetics, n_jobs=self.n_jobs, prefer="threads", random_state=rng, verbose=self.verbose).fit(problem)` (the outer `rng` Generator is passed so the whole run consumes exactly the outer `random_state`); copy `history_, n_iter_, stop_reason_, estimators_, costs_, best_index_, best_estimator_` from `ms` onto `self`; return `problem.to_index_tour(ms.tour_)`. Same for `EnsembleSimulatedAnnealing` with `SimulatedAnnealing(**inner)` and `n_simulateds`. Parameters are explicit so `get_params` works; they carry a `DeprecationWarning` in the docstring only (removal in 3.0), not at runtime.

### 4.6 Legacy name map (the eight names)

| 1.0.0a2 | 2.0 home | notes |
|---|---|---|
| `skroute.heuristics.brute.BruteForce` | `skroute.exact.BruteForce` | `skroute.heuristics` shim re-exports with `DeprecationWarning` |
| `skroute.heuristics.NRBS.NRBS` | `skroute.construction.NRBS` | the five exponents default to the value `1.0` (1.0.0a2 had no defaults); `distance_weigth` → `distance_weight`; `fit(start, ids, cost)` → `fit(cost, depot=start)` |
| `skroute.metaheuristics.genetics.Genetic` | `skroute.metaheuristics.Genetic` | `p_c→p_crossover`, `p_m→p_mutation`, `pop→pop_size`, `gen→n_generations`, `k→tournament_size`, `early_stopping→patience` |
| `…genetics.EnsembleGenetic` | `skroute.ensemble.EnsembleGenetic` | `n_genetics` kept |
| `…simulated_annealing.SimulatedAnnealing` | `skroute.metaheuristics.SimulatedAnnealing` | `temp→t0`, `delta→alpha` (1.0 silently rescaled it into 0.9–1), `tol→t_min`, `neighbours→n_moves` |
| `…simulated_annealing.EnsembleSimulatedAnnealing` | `skroute.ensemble.EnsembleSimulatedAnnealing` | `n_simulateds` kept as a name (default changes 20 → 10); the SA knobs take the 2.0 defaults of §4.4 |
| `…tabu_search.TabuSearch` | `skroute.metaheuristics.TabuSearch` | `searchs→n_iter`, `tabu_length/tabu_var→tenure`, `p_m` dropped |
| `…som.SOM` | `skroute.metaheuristics.SOM` | `units→n_units`, `lr→learning_rate`, `fit(nodes, epochs)` → `fit(X, coords=..., )` with `n_iter` in `__init__` |

Top-level exports: see §3.4 ("Top-level surface") — every solver of D18 plus `RoutingProblem`, `clone`, `is_router`, `all_solvers`, `check_router`, `set_log_level`, `__version__`.

**Shim contents (WP8, exact):** each legacy `__init__.py` re-exports exactly the 1.0 `__all__` of that package — `heuristics/brute`: `BruteForce`; `heuristics/NRBS`: `NRBS`; `metaheuristics/genetics`: `Genetic`, `EnsembleGenetic`; `metaheuristics/simulated_annealing`: `SimulatedAnnealing`, `EnsembleSimulatedAnnealing` (1.0's `__all__` misspelt it `SimmulatedAnnealing`; the shim exports the real name); `metaheuristics/tabu_search`: `TabuSearch`; `metaheuristics/som`: `SOM`; `skroute/heuristics/__init__.py` exposes the `brute` and `NRBS` subpackages. Each shim emits at import `warnings.warn("skroute.<old path> is deprecated since 2.0 and will be removed in 3.0; import <Name> from skroute.<new path>", DeprecationWarning, stacklevel=2)`. `tests/test_legacy_shims.py` asserts identity with the new classes (`shim.Genetic is skroute.metaheuristics.Genetic`) and the warning text.

**Every public name of 1.0.0a2** (this table is reproduced verbatim in `docs/migration.md`; nothing public from 1.0 may be missing from it):

| 1.0 name | status | 2.0 replacement | reason |
|---|---|---|---|
| the 8 classes above | kept (moved) | see the table above | — |
| `EnsembleGenetic`, `EnsembleSimulatedAnnealing` | kept (moved) | `skroute.ensemble.*`; or `MultiStart(Genetic(...))` | explicit-parameter wrappers over `MultiStart` |
| `preprocessing.dfcolumn_to_dict` | replaced | `pairs_to_matrix` | result depended on row order |
| `preprocessing.matrix_to_dict` | replaced | `to_dict_of_dicts` | referenced an undefined variable |
| `preprocessing.normalize(df, lat, lon)` | replaced | `normalize_coords(coords)` | numpy in, numpy out |
| `preprocessing.df_to_tuple` | removed | `Bunch.coords`/`labels`, or `df[[...]].to_numpy()` | one-line pandas idiom |
| `preprocessing.matrix_parse` | removed | `pairs_to_matrix` + `distance_matrix` | superseded |
| `preprocessing.DataLossWarning` | removed | — | nothing emits it any more |
| `preprocessing.CostScraper` | deprecated wrapper | `GoogleDistanceMatrix(api_key).fetch(coords, labels)` | §5.2; removed in 3.0 |
| `cluster.KMeansTruncate` | removed | `sklearn.cluster.KMeans(max_iter=1)` directly | scikit-learn is no longer a dependency |
| `cluster.{KMeans, DBSCAN, AffinityPropagation, GaussianMixture}` re-exports | removed | import from scikit-learn | same |
| 27 `load_<country>` + 5 cost loaders | kept | same names; return a `Bunch`/`TSPBunch` (§5.1) | return shape changed: `"DataFrame"` → `frame` (with `as_frame=True`), `"feature_names"` dropped, `DESCR` kept |
| `load_costs_qatar` | deprecated alias | `load_qatar_costs` | 1.0 loaded Valencia |
| `mode=` in TSP loaders | deprecated | `n_nodes=` | 1.0 returned 0 rows for wi29 "small" |
| `fit(route_example, time_matrix, cost_matrix)` | changed | `fit(cost, time_matrix=..., depot=..., init=...)` | D7; boxed warning in the guide |

**Defaults that changed** (results with default parameters differ from 1.0; also in `docs/migration.md`):

| class | 1.0 default | 2.0 default |
|---|---|---|
| `BruteForce`, `TabuSearch` | `max_time_work=8`, `extra_cost`, `people` in `__init__` | plain TSP unless `max_time_work=` is passed to `fit` |
| `Genetic` | `p_c=0.6, p_m=0.4, pop=400, gen=1600, k=3` | `p_crossover=0.9, p_mutation=0.2, pop_size=100, n_generations=500, tournament_size=3, patience=100` |
| `EnsembleGenetic` | `pop=400, gen=1000` | `pop_size=100, n_generations=500` |
| `SimulatedAnnealing` | `temp=12.0, neighbours=250, delta=0.78, tol=1.29` | `t0="auto", n_moves=10·n, alpha=0.995, t_min="auto", patience=None` |
| `EnsembleSimulatedAnnealing` | `n_simulateds=20` | `n_simulateds=10` |
| `TabuSearch` | `searchs=1250, p_m=0.6, tabu_length=45, tabu_var=10` | `n_iter=1000, tenure="auto"` (`p_m` dropped) |
| `SOM` | `radius_decay=0.9991, lr_decay=0.9991, fit(..., epochs=10_000)` | `radius_decay=0.9997, lr_decay=0.99997, n_iter=100_000` |
| `NRBS` | no defaults (`distance_weigth`) | all five `1.0` (`distance_weight`) |

---

## 5. Datasets and preprocessing API

### 5.1 `skroute.datasets`

`Bunch` (`utils/_bunch.py`): `dict` subclass with attribute access and a `__repr__` listing keys. `TSPBunch(Bunch)` (`datasets/_loaders.py`) adds the **real method** `distance_matrix(self, *, force=False)` (not a key: `keys()` lists only data fields) with the result cached under the private attribute `_distance_matrix`.

**Loader matrices carry no labels** (they are plain `float64` ndarrays): pass `labels=b.labels` to `fit`/`RoutingProblem` (or use `as_frame=True`, whose DataFrames carry the ids), otherwise `depot=b.depot` is rejected as "not a label of X" — every example and fixture in this document does so. The depot of every cost dataset is row 0.

**TSPLIB instances** — `load_tsp(name, *, n_nodes=None, random_state=2019, mode=None) -> TSPBunch` with fields `name` (e.g. `"wi29"`), `coords` (`float64 (n, 2)`, x/y as in the file), `labels` (`int64`, the file's 1-based ids), `depot` (first label), `edge_weight_type` (`"EUC_2D"` for all 27), `optimal_tour_length` (from the table in the brief; `None` when subsampled), `DESCR`, and the method `distance_matrix(*, force=False)` → `preprocessing.distance_matrix(coords, metric="tsplib_euc_2d")`, cached, refusing `n > 20_000` unless `force=True` (ch71009 would need 40 GB). The DESCR of the four instances above 20 000 nodes (vm22775, sw24978, bm33708, ch71009) states that 2.0 solves only dense matrices, that the whole instance cannot be solved in 2.0, and shows `load_tsp(name, n_nodes=5000)`. They stay bundled in 2.0 (R6 decided: readers and subsampling keep working offline; a lazy download is a 2.1 option behind the same `load_tsp`). `n_nodes` subsamples without replacement with `default_rng(random_state)`, keeping the first node; `mode="small"|"medium"|"big"` is accepted for 2.x with a `DeprecationWarning` and maps to `max(10, round(0.005 n))` / `round(0.2 n)` / all (1.0 returned 0 rows for wi29 "small"). `list_tsp() -> list[str]` returns the 27 names. The 27 country functions (`load_sahara`, `load_djibouti`, `load_qatar`, … exactly the 1.0 names) are one-line wrappers `load_tsp("wi29", **kwargs)`. Matrices are never built by the loader.

**Cost datasets** — `load_alicante_murcia(*, as_frame=False)`, `load_barcelona`, `load_madrid`, `load_valencia`, `load_qatar_costs` (the 1.0 `load_costs_qatar` loaded Valencia; it remains as a deprecated alias that now loads Qatar). Return `Bunch(cost, time, distance, coords, labels, depot, units, DESCR, frame)`: square `float64` matrices pivoted from the long tables with `pairs_to_matrix(..., symmetric=True)` *(verified: the files are upper triangles including the diagonal — Barcelona 190 rows for 19 nodes, Madrid 171/18, Valencia 105/14, Alicante 36/8; Qatar 18 336 rows for 192 nodes without the diagonal)*; `cost` in EUR (Spanish files) or kilometres (Qatar, which has no money column), `time` in hours (Qatar's `seconds/3600`), `distance` in metres, `coords` as `(lat, lon)`, `labels` int64 ids in first-appearance order, `depot` = first id (`10000002`, `10000007`, `10000016`, `10000022`, `1`), `units = {"cost": "EUR"|"km", "time": "h", "distance": "m"}`. `as_frame=True` requires pandas (`ImportError("pandas is required for as_frame=True: pip install scikit-route[pandas]")`) and returns `cost`/`time`/`distance` as labelled DataFrames plus `frame` (the long table). Parsing uses the stdlib `csv` module.

**TSPLIB reader** — `read_tsplib(path_or_file) -> Bunch(name, comment, type, dimension, edge_weight_type, edge_weight_format, coords | cost, display_coords)`: keywords tolerant of `KEY : value` and `KEY: value` *(verified: dj38 mixes both and lacks `EOF`)*, CRLF, missing `EOF`; `NODE_COORD_SECTION` for `EUC_2D`, `CEIL_2D`, `MAN_2D` (P0) and `ATT`, `GEO` (**P1**, D26; coordinates kept raw; conversion happens in `distance_matrix`); `EDGE_WEIGHT_SECTION` for `EXPLICIT` with `FULL_MATRIX`, `UPPER_ROW`, `LOWER_ROW`, `UPPER_DIAG_ROW`, `LOWER_DIAG_ROW` (returns `cost`; P0 — the hypothesis round-trip and `tests/data/explicit_matrix.tsp` need it); `read_tsplib_tour(path) -> int64 array` of 1-based ids (P0). Pure Python, no pandas, no regex separators. An unsupported `EDGE_WEIGHT_TYPE` raises `ValueError("EDGE_WEIGHT_TYPE GEO is not supported in this version")`.

### 5.2 `skroute.preprocessing`

- `distance_matrix(coords, metric="euclidean", *, block_size=2048) -> float64 (n, n)`; `metric in {"euclidean", "manhattan", "tsplib_euc_2d", "tsplib_ceil_2d", "tsplib_man_2d", "tsplib_att", "tsplib_geo", "haversine"}` (`manhattan`/`tsplib_man_2d` close issue #24, §10; `MAN_2D = nint(|dx| + |dy|)`). TSPLIB metrics follow the TSPLIB 95 definitions exactly: `nint(x) = floor(x + 0.5)` (D15); `CEIL_2D`: `ceil(sqrt(dx² + dy²))`; `ATT` (P1): `r = sqrt((dx² + dy²) / 10.0); t = nint(r); d = t + 1 if t < r else t`; `GEO` (P1), spelled out because it is NOT nint-rounded: `PI = 3.141592`; per coordinate `deg = int(x)` (truncation towards zero, as in the TSPLIB reference code — `nint` evaluates the ulysses16 optimal tour to 6917, not 6859); `m = x - deg; rad = PI * (deg + 5.0 * m / 3.0) / 180.0` (same for `y`); `RRR = 6378.388`; `q1 = cos(lon_i - lon_j); q2 = cos(lat_i - lat_j); q3 = cos(lat_i + lat_j)`; `d = int(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)` (truncation plus one). Regression tests (WP7, active when the P1 metric ships): the published optimal tours `tests/data/ulysses16.opt.tour` and `tests/data/att48.opt.tour` evaluate to **6859** (`GEO`) and **10628** (`ATT`) on `ulysses16.tsp`/`att48.tsp`. `haversine` on `(lat, lon)` degrees with radius `6371.0088` km. Computed blockwise to bound peak memory; warns above 20 000 nodes (1.8 GB at 15 000). `euclidean_matrix(coords)` and `haversine_matrix(latlon)` are one-line conveniences.
- `pairs_to_matrix(origin, destination, value, *, symmetric=True, labels=None, fill=None) -> (matrix, labels)`: sequences (or DataFrame columns) of equal length; labels in first-appearance order unless given; `symmetric=True` mirrors missing entries; missing entries without a mirror raise unless `fill` is given; the diagonal defaults to 0. Replaces `dfcolumn_to_dict`, whose result depended on row order.
- `to_dict_of_dicts(matrix, labels=None) -> dict` and `from_dict_of_dicts(d) -> (matrix, labels)` (the 1.0 `matrix_to_dict` referenced an undefined variable).
- `normalize_coords(coords) -> float64 (n, 2)`: aspect-preserving min-max into `[0, 1]²` (what SOM uses).
- `skroute.preprocessing.google.GoogleDistanceMatrix(api_key, mode="driving", *, batch_size=10)` (**P1**, D26), `.fetch(coords, labels=None) -> Bunch(distance, time, labels, units)` with metres and hours: `googlemaps` is imported lazily inside `__init__` (`ImportError("googlemaps is required: pip install scikit-route[google]")`); requests are batched `batch_size × batch_size` origins × destinations (the API caps at 100 elements per request; 1.0 issued one request per pair — 18 336 for Qatar); progress via `logging`; tests mock the client. **`CostScraper` is a thin deprecated wrapper, not a bare alias** (a bare alias would bind 1.0's `nodes` argument to `mode` and fail with an opaque googlemaps error after spending the user's quota): `class CostScraper: def __init__(self, api, nodes, mode="driving")` warns `DeprecationWarning("CostScraper is deprecated since 2.0 and will be removed in 3.0; use GoogleDistanceMatrix(api_key).fetch(coords, labels)")`, stores `labels = [n[0] for n in nodes]`, `coords = [n[1:] for n in nodes]`; `scrap()` calls `fetch` and returns its Bunch; `pandas()` returns the long table (requires pandas); `to_pickle()` raises `NotImplementedError("to_pickle was removed in 2.0; use pandas().to_pickle(...)")`. If `GoogleDistanceMatrix` is deferred, `CostScraper` goes with it and both appear as "removed, returns in 2.1" in the migration table.

---

## 6. Testing conventions

**Layout**: §2 and D16 — `tests/` is not a package; `pyproject.toml` carries `[tool.pytest.ini_options] testpaths = ["tests"]`, `pythonpath = ["tests"]`, `addopts = "-m 'not slow' --strict-markers"`, `doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS NUMBER"`, `markers = ["slow: ...", "benchmark: ..."]`. **Command**: `pytest tests` (slow deselected by `addopts`); `pytest tests -m slow` for benchmarks; `pytest tests --cov=skroute --cov-report=term-missing` in CI. **Markers**: `slow` (Waterloo n ≥ 194, MILP on qa194, anything > 10 s) and `benchmark` (alias on the same tests for tooling). **Coverage target**: 90 % of `.py` lines (kernels are covered through `test_core.py`; no Cython linetrace in CI). Nothing in the suite touches the network. **Every slow-tier gap test lives in `tests/benchmarks/test_waterloo.py`** (WP8), driven by `tests/tolerances.py` and parametrised over `all_solvers()`.

**`tests/conftest.py`**:

```python
import numpy as np, pytest
import reference                                    # tests/ is on sys.path via pythonpath = ["tests"] (D16)
from skroute import all_solvers, RoutingProblem
from skroute.datasets import load_tsp, load_alicante_murcia, load_barcelona
from skroute.preprocessing import distance_matrix

OPTIMA = {"wi29": 27603, "dj38": 6656, "qa194": 9352, "uy734": 79114, "zi929": 95345, "lu980": 11340}

def _euclid(n, seed, asymmetric=False):
    """(C, coords). Every instance keeps the coordinates that generated it (SOM needs them)."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    C = distance_matrix(xy)
    if asymmetric:
        C = C * rng.uniform(0.7, 1.3, C.shape); np.fill_diagonal(C, 0.0)
    return np.ascontiguousarray(C), xy

@pytest.fixture(scope="session", params=[(5, False), (7, False), (9, False), (6, True), (8, True)],
                ids=lambda p: f"n{p[0]}{'-asym' if p[1] else ''}")
def tiny_instance(request):
    n, asym = request.param
    C, xy = _euclid(n, seed=n, asymmetric=asym)
    return {"C": C, "coords": xy, "n": n, "asymmetric": asym, "optimum": reference.brute_force(C)[0]}

@pytest.fixture(scope="session")
def small_euclidean():                      # n = 12, for reproducibility and label round-trip tests
    C, xy = _euclid(12, seed=12)
    return {"C": C, "coords": xy, "n": 12, "asymmetric": False}

@pytest.fixture(scope="session")
def medium_euclidean():                     # n = 40, where seeds 0 and 1 give different tours (check 11)
    C, xy = _euclid(40, seed=40)
    return {"C": C, "coords": xy, "n": 40, "asymmetric": False}

@pytest.fixture(scope="session")
def alicante():                             # multi-trip fixture, 8 nodes; the depot is row 0 == label d.depot
    d = load_alicante_murcia()
    budget = 1.5 * float((d.time[0, :] + d.time[:, 0]).max())
    fit_kw = dict(labels=d.labels, depot=d.depot, max_time_work=budget, extra_cost=10.0, people=2)  # LABEL space
    ref_kw = dict(depot=0, max_time_work=budget, extra_cost=10.0, people=2)                         # INDEX space
    opt = {s: reference.brute_force(d.cost, d.time, split=s, **ref_kw)[0] for s in ("greedy", "optimal")}
    return {"bunch": d, "kwargs": fit_kw, "ref_kwargs": ref_kw, "optimum": opt}

@pytest.fixture(scope="session")
def barcelona():
    return load_barcelona()

@pytest.fixture(scope="session", params=["wi29", "dj38"])
def fast_instance(request):
    b = load_tsp(request.param)
    return {"name": b.name, "C": b.distance_matrix(), "coords": b.coords, "labels": b.labels,
            "asymmetric": False, "optimum": OPTIMA[b.name]}

@pytest.fixture(scope="session", params=["qa194", "uy734", "zi929", "lu980"])
def slow_instance(request):
    b = load_tsp(request.param)
    return {"name": b.name, "C": b.distance_matrix(), "coords": b.coords, "labels": b.labels,
            "asymmetric": False, "optimum": OPTIMA[b.name]}

def pytest_generate_tests(metafunc):
    if "Solver" in metafunc.fixturenames:
        metafunc.parametrize("Solver", all_solvers(), ids=lambda s: s.__name__)   # D27: no MultiStart here

def make(Solver, **overrides):
    """Instantiate with random_state=0 when accepted; used by every parametrised test."""
    params = {"random_state": 0} if "random_state" in Solver._get_param_names() else {}
    params.update(overrides)
    return Solver(**params)

def fit_kwargs(Solver, inst):
    """coords= for requires_coords solvers; those are skipped on asymmetric instances (no meaningful coordinates)."""
    if make(Solver)._get_tags().requires_coords:
        if inst.get("asymmetric"):
            pytest.skip(f"{Solver.__name__} needs coordinates; asymmetric instances have none")
        return {"coords": inst["coords"]}
    return {}
```

**`tests/reference.py`** (WP8, pure Python/numpy, ≤ 200 lines) — **everything works in index space**: `tour_cost(C, tour)`, `greedy_split(C, T, tour, max_time, fixed_cost) -> (cost, starts)`, `optimal_split(...)` (O(n²) DP over every `(j, i)` pair with the open-path bound of §3.5, so it agrees with the kernel on non-metric `T`), `problem_cost(...)`, `brute_force(C, T=None, *, depot=0, max_time_work=None, extra_cost=0.0, people=1, split="greedy") -> (cost, tour)` via `itertools.permutations` (`depot` is an **index**; the alicante fixture passes `ref_kw`, never the label), `two_opt_delta_by_recompute`, `or_opt_apply`, `ox`, `pmx`. The one label-space helper is `route_cost_from_labels(X, route, *, depot=None, labels=None, time_matrix=None, ...)`, which mirrors `skroute.metrics.route_cost` (`depot=None` → `route[0]`). Every core kernel is tested against these.

**`tests/tolerances.py`** (WP8) — the tolerance table below as data, the only place tolerance numbers live: `TINY`, `FAST`, `SLOW` dicts keyed by class name (`Insertion` keyed as `"Insertion[farthest]"`/`"Insertion[cheapest]"`, `Genetic` as `"Genetic"`/`"Genetic[memetic]"` with the matching `set_params`), `SEEDS_TO_OPTIMUM = {"IteratedLocalSearch", "SimulatedAnnealing", "TabuSearch", "Genetic", "AntColony"}`, `MEASURED` (the per-instance baselines quoted in §4, rendered into `benchmarks.md`). A class in `all_solvers()` without an entry fails with `KeyError("add a tolerance for <Name> in tests/tolerances.py")`; `benchmarks/waterloo.py` prints the same dicts next to the measured gaps.

**`check_router(estimator)`** (`skroute/utils/estimator_checks.py`, WP8, public): takes an **unfitted instance** (so `MultiStart(SimulatedAnnealing())` can be checked too), runs the structural checks 1–11 and 13 below in-process on instances it builds itself — n = 6 symmetric Euclidean with coordinates, n = 6 asymmetric, n = 3 and n = 4 symmetric and asymmetric, and the alicante bunch loaded from `skroute.datasets` — and raises `AssertionError` prefixed with the check number. `check_router.checks` is the list of `(name, fn)` pairs, each `fn(estimator)`. **`tests/test_common.py`** exposes every entry of `check_router.checks` as a parametrised pytest test over `all_solvers()` (via `make(Solver)`, so fixtures such as `capsys` are usable there) and adds the tolerance tests (12) from `tests/tolerances.py`; it is the merge gate of every solver PR:

1. `__init__` stores every parameter verbatim and sets no other attribute (`vars(est)` equals its `get_params(deep=False)`); `get_params/set_params/clone` round-trip; `eval(repr(est))` equals `est` (with the class name imported).
2. Not fitted → `check_is_fitted` raises `NotFittedError`; no trailing-underscore attribute before `fit`.
3. `fit` returns `self`; `route_[0] == route_[-1] == depot_`; `route_` minus depot occurrences is a permutation of `labels_` minus depot; `tour_[0] == depot_`; `trips_` closed; `n_trips_ == len(trips_) == count(depot in route_) - 1`.
4. `cost_ == pytest.approx(reference.route_cost_from_labels(...), rel=1e-9)` and `cost_ == pytest.approx(trip_costs_.sum() + fixed*(n_trips_-1), rel=1e-12)` (two float summation orders; never exact equality).
5. ndarray, DataFrame (`pytest.importorskip("pandas")`) and dict-of-dicts inputs give identical `tour_` up to labels; string labels survive; `depot=` by label works; `labels=` on an ndarray works.
6. Invalid inputs raise `ValueError` with the messages of §3.3 (non-square, NaN, unknown depot, `max_time_work` without `time_matrix`, `time_matrix` without `max_time_work`, RoutingProblem plus kwargs); infeasible node → `InfeasibleProblemError`.
7. Tags honoured: `requires_symmetric` → raises on asymmetric; `requires_coords` → raises without coords; exact and `budget_aware=False` → raises with a budget; non-exact `budget_aware=False` → `UserWarning`; `max_nodes` → raises above.
8. Multi-trip (alicante; skipped for solvers with `exact and not budget_aware`, which raise under a budget per D6 and check 7; non-exact budget-unaware solvers run it under the `UserWarning`): `trip_times_ <= budget + 1e-9`, `trip_times_` absent for plain TSP; **same tour, both decoders**: after a fit under `split="greedy"`, `p_opt = RoutingProblem(d.cost, time_matrix=d.time, split="optimal", **kw)` and `p_opt.evaluate(est.problem_.to_index_tour(est.tour_)) <= est.cost_ + 1e-9`. Comparing two independent heuristic fits (`fit(optimal).cost_ <= fit(greedy).cost_`) is NOT asserted — different searches return different tours; that inequality is asserted only for exact budget-aware solvers (BruteForce) in `test_exact.py`.
9. Nothing written to stdout/stderr (`capsys`); `verbose=1` emits at least one record on `logging.getLogger("skroute")` for iterative solvers.
10. Iterative: `len(history_) == n_iter_`, `history_` non-increasing, `history_[-1] == pytest.approx(cost_)`, `stop_reason_` in the solver's documented subset (§3.4 table); **if `"time_limit" in Solver._get_param_names()`**, `time_limit=1e-6` stops with `stop_reason_ == "time_limit"` after at most one outer iteration (TwoOpt/OrOpt/LocalSearch/SOM have no `time_limit` and skip this half).
11. Stochastic: two fits with `random_state=0` give `array_equal(tour_)`, equal `cost_`, `array_equal(history_)`; **seed difference is deterministic, not statistical**: seeds 0 and 1 give different `history_` arrays (or different `n_iter_`) on `small_euclidean`, or different `tour_` on `medium_euclidean` (n = 40), whichever the solver exposes — on n = 12 every metaheuristic finds the optimum with both seeds and `tour_` differs only by orientation, a coin flip; a passed `Generator` is advanced (`bit_generator.state` changes across the fit).
12. **Tolerances (in `test_common.py`, not in `check_router`)**, from `tests/tolerances.py` by class name and `RouterTags.kind`: exact solvers equal the tiny optimum (`rel=1e-9`); classes in `SEEDS_TO_OPTIMUM` equal it at seeds 0, 1, 2; every other class satisfies its `TINY` entry; `FAST` entries on `fast_instance`; `SLOW` entries in `tests/benchmarks/test_waterloo.py`.
13. **Smallest legal sizes**: every solver fits symmetric and asymmetric instances with n = 3 and n = 4 (built inline with coordinates), returns a valid tour, and exact solvers equal `reference.brute_force` there (covers the ILS kick below n = 8, Or-opt with L = 3 at n = 4, the Tabu neighbourhood, SA proposals and BruteForce's halving at their edges).

**Tolerance table** (`tests/tolerances.py`; gap = `cost_/optimum - 1`, defaults, `random_state=0`; `optimum <= cost_ + 1e-9` is asserted first — a violation is a reader/rounding/evaluator bug). Measured baselines per instance are recorded in `MEASURED` and `benchmarks.md`, so a regression is distinguishable from a tie-break difference:

| Solver | tiny (n ≤ 9) | fast tier (wi29, dj38) | slow (qa194, uy734, zi929, lu980) |
|---|---|---|---|
| BruteForce, HeldKarp (if shipped) | 0 | — (cap) | — |
| MILP | 0 | 0 | 0 on qa194 (`@slow`, 150 s) |
| NearestNeighbour | ≤ 0.50 | ≤ 0.50 (dj38 measures 46.4 %) | ≤ 0.35 |
| Insertion[farthest] / Insertion[cheapest] | ≤ 0.30 | ≤ 0.25 / 0.30 | ≤ 0.25 / 0.30 |
| ClarkeWright | ≤ 0.30 | ≤ 0.25 | ≤ 0.25 |
| NRBS | valid | ≤ 0.50 | ≤ 0.60 (qa194 only) |
| TwoOpt / OrOpt | ≤ 0.10 / 0.12 | ≤ 0.20 / 0.25 | ≤ 0.20 / 0.25 (OrOpt alone measures 21.7 % on lu980) |
| LocalSearch | ≤ 0.10 | ≤ 0.12 | ≤ 0.15 (measured 11.9 % qa194) |
| IteratedLocalSearch | 0 (3 seeds) | ≤ 0.03 | ≤ 0.06 (measured 4.16 % lu980) |
| SimulatedAnnealing | 0 (3 seeds) | ≤ 0.03 | ≤ 0.10 (measured 7.0 % lu980) |
| TabuSearch | 0 (3 seeds) | ≤ 0.08 | ≤ 0.15 |
| Genetic (plain) | 0 (3 seeds) | ≤ 0.15 (measured 8.6 % dj38) | ≤ 0.30, qa194 only (measured 18.8 %) |
| Genetic[memetic] (`local_search=("two_opt",)`) | 0 (3 seeds) | ≤ 0.05 | ≤ 0.08 on qa194 and lu980 (`@slow`, minutes) |
| AntColony (if shipped) | 0 (3 seeds) | ≤ 0.08 | ≤ 0.15 (n ≤ 1000) |
| SOM (+coords) | valid | ≤ 0.15 | ≤ 0.15 (qa194) |
| EnsembleGenetic / EnsembleSimulatedAnnealing | 0 (3 seeds) | ≤ the wrapped solver's entry | — |
| MultiStart(SA, 4) (`test_ensemble.py`) | 0 | ≤ min(SA) | ≤ SA, and `n_jobs=1 == n_jobs=2` |

Tightening a tolerance is a release-notes item; loosening one requires the lead's approval and a CHANGELOG line.

**Hypothesis** (`tests/test_core.py`, `settings(derandomize=True, deadline=None, max_examples=200)`): strategies generate `n in 3..12`, symmetric/asymmetric finite matrices, random permutations with the depot first, budgets in `[max round trip, 3 × max round trip]`. Properties: `tour_cost == reference`; `greedy_split_cost`/`optimal_split_cost` equal the reference and `optimal <= greedy`; every trip from `trip_starts` fits the budget and the trips cover positions `1..n-1`; `two_opt_delta == cost(after) - cost(before)` on symmetric matrices and `two_opt_delta_asym` on asymmetric; `or_opt_delta(reverse=False)` and `swap_delta` exact on asymmetric; applying a move then its inverse is the identity; `pos[tour[i]] == i` after every `_pos` move and every descent; `two_opt_descent`/`or_opt_descent`/`local_search_generic` never increase the cost, return a permutation, and the returned gain equals the cost difference; `double_bridge` is a permutation with the depot fixed; `from_dict_of_dicts(to_dict_of_dicts(M)) == M`; `read_tsplib` round-trips generated EUC_2D/EXPLICIT files; `nint` matches `floor(x+0.5)` on half-integers. Reproducibility is asserted, never assumed (check 11); no exact-float golden values across platforms (D19).

Example test (WP5):

```python
# tests/test_simulated_annealing.py
import numpy as np, pytest
import reference
from skroute.metaheuristics import SimulatedAnnealing
from skroute.metrics import route_cost

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reaches_optimum_on_tiny(tiny_instance, seed):
    C, opt = tiny_instance["C"], tiny_instance["optimum"]
    sa = SimulatedAnnealing(random_state=seed).fit(C)
    assert sa.cost_ == pytest.approx(opt, rel=1e-9)
    assert sa.cost_ == pytest.approx(route_cost(C, sa.route_))
    assert sa.history_[-1] == pytest.approx(sa.cost_) and np.all(np.diff(sa.history_) <= 1e-12)

def test_multi_trip_respects_budget_and_matches_reference(alicante):
    d, kw = alicante["bunch"], alicante["kwargs"]                    # kw carries labels= and the LABEL depot
    sa = SimulatedAnnealing(random_state=0).fit(d.cost, time_matrix=d.time, **kw)
    assert np.all(sa.trip_times_ <= kw["max_time_work"] + 1e-9)
    assert sa.cost_ == pytest.approx(alicante["optimum"]["greedy"], rel=1e-9)
    assert sa.cost_ == pytest.approx(reference.problem_cost(d.cost, d.time, sa.problem_.to_index_tour(sa.tour_),
                                                            kw["max_time_work"], 10.0 * 2, "greedy"))

def test_same_seed_is_bit_identical(small_euclidean):
    a, b = (SimulatedAnnealing(random_state=7).fit(small_euclidean["C"]) for _ in range(2))
    assert np.array_equal(a.tour_, b.tour_) and a.cost_ == b.cost_ and np.array_equal(a.history_, b.history_)
```

---

## 7. Documentation and community deliverables

**Docstrings**: numpydoc (`Parameters`, `Attributes`, `Notes` with the algorithm and complexity, `References`, `Examples` that run). Every public class documents its tags in a `Notes` line ("Supports: symmetric and asymmetric matrices, multi-trip objective; stochastic"). mkdocstrings renders them; `mkdocs build --strict` fails on broken cross-references.

**`mkdocs.yml`** (Material theme, `mkdocstrings` with `python` handler `docstring_style: numpy`, `include-markdown`, `search`):

```yaml
nav:
  - Home: index.md
  - Getting started: getting_started.md
  - Installation: installation.md
  - User guide:
      - The problem model: user_guide/problem_model.md
      - Choosing a solver: user_guide/choosing_a_solver.md
      - Multi-trip routing: user_guide/multi_trip.md
      - Warm starts and ensembles: user_guide/warm_starts_and_ensembles.md
  - API reference:
      - Base and problem: api/base.md
      - Exact solvers: api/exact.md
      - Construction heuristics: api/construction.md
      - Local search: api/local_search.md
      - Simulated annealing: api/simulated_annealing.md
      - Tabu search: api/tabu_search.md
      - Genetic algorithm: api/genetic.md
      - Ant colony: api/ant_colony.md        # removed by the deferral PR if AntColony is deferred (D26)
      - Self-organising map: api/som.md
      - Ensembles: api/ensemble.md
      - Datasets: api/datasets.md
      - Preprocessing: api/preprocessing.md
      - Utilities: api/utils.md
  - Benchmarks: benchmarks.md
  - Migration from 1.0: migration.md
  - Contributing: contributing.md
  - Changelog: changelog.md
  - About: about.md
```

**Every public symbol has a documentation home.** `api/base.md` (lead) carries exactly: `::: skroute.RoutingProblem`, `::: skroute.base.BaseRouter`, `::: skroute.base.RouterTags`, `::: skroute.clone`, `::: skroute.is_router`, `::: skroute.all_solvers`, `::: skroute.set_log_level`, `::: skroute.exceptions`, `::: skroute.metrics`; `api/utils.md` (lead) carries `::: skroute.utils` (Bunch, check_is_fitted, check_random_state, initial_tour) and `::: skroute.utils.estimator_checks.check_router`; each solver page carries one `:::` per class of its `__all__`; `api/ensemble.md` also documents `EnsembleGenetic`/`EnsembleSimulatedAnnealing`. `docs/check_api_coverage.py` (WP8, ≈15 lines, run in `docs.yml` before `mkdocs build`) fails if any name in any `__all__` under `skroute/` has no `:::` directive under `docs/api/`.

**Capability table, generated once, copied nowhere by hand.** `docs/gen_pages.py` (WP8, run by `mkdocs-gen-files`) imports `skroute.all_solvers()` and writes `docs/user_guide/_capability_table.md` with the columns solver · kind · exact · stochastic · multi-trip aware (`budget_aware`) · ATSP (`not requires_symmetric`) · needs coords · max n; `choosing_a_solver.md` includes it with `include-markdown`; the README carries the same table between `<!-- capability-table:start -->`/`<!-- capability-table:end -->` markers refreshed by `python docs/gen_pages.py --readme` in a pre-commit hook (the hook fails if the README is stale).

**README outline**: logo; badges (PyPI version, Python versions, CI, docs, license, coverage); one-paragraph pitch; install (`pip install scikit-route`, extras); the wi29 example with `IteratedLocalSearch` and the Barcelona multi-trip example **exactly as in §3.4** (`>>>` form, `labels=`, keyword `time_matrix=`; the README is doctested too); the generated capability table between its markers; links to docs, migration guide, contributing; citation ("Cite via `CITATION.cff` — GitHub → Cite this repository"); license.

**CHANGELOG.md** skeleton:

```
# Changelog
All notable changes to this project are documented here. Format: Keep a Changelog; versioning: SemVer.

## [2.0.0] - 2026-MM-DD
### Added
- Solvers: HeldKarp (if shipped), MILP, NearestNeighbour, Insertion, ClarkeWright, TwoOpt, OrOpt, LocalSearch,
  IteratedLocalSearch, AntColony (if shipped), MultiStart; RoutingProblem; skroute.metrics; check_router; set_log_level.
- Optimal split of the giant tour (`split="optimal"`), TSPLIB reader, distance_matrix with TSPLIB metrics.
### Changed
- fit() takes the cost matrix (numpy / DataFrame / dict-of-dicts) and returns self; time_matrix is keyword-only;
  results in route_, trips_, cost_ ...
- Objective: trips never exceed max_time_work including the return leg; people multiplies only extra_cost.
- Default hyper-parameters changed for every solver (table in the migration guide); results with defaults differ from 1.0.
- Genetic: real OX/PMX crossover; SimulatedAnnealing: aliasing bug fixed, moves and auto temperature;
  TabuSearch rewritten (edge attributes); NRBS: union-find cycle check, hard-coded 19 removed, distance_weigth -> distance_weight; SOM in numpy.
- Datasets return Bunch objects (feature_names dropped, DataFrame -> as_frame=True); pandas, googlemaps optional;
  tensorflow, scikit-learn, tqdm dropped; Python >= 3.11; numpy 2 compatible.
### Removed
- skroute.cluster (KMeansTruncate and the scikit-learn re-exports); preprocessing.df_to_tuple, matrix_parse, DataLossWarning;
  pickled datasets (CSV instead); the tuple return of fit(); route_example; CostScraper.to_pickle.
### Fixed
- load_costs_qatar loaded Valencia; matrix_to_dict undefined variable; datasets __all__; TSPLIB sep="\s".
### Deprecated
- skroute.heuristics.*, skroute.metaheuristics.<legacy subpackages>, CostScraper (wrapper), load_costs_qatar, mode= in loaders (removed in 3.0).
```

**Migration guide outline** (`docs/migration.md`, lead): (1) install and imports (the shim table of §4.6); (2) `fit` signature before/after for each of the eight classes with a runnable snippet, opened by a **boxed warning**: *"The matrix order is reversed and `time_matrix` is keyword-only in 2.0: 1.0's `fit(route, time, cost)` becomes `fit(cost, time_matrix=time, depot=route[0])`; passing the time matrix positionally raises `TypeError`"*; (3) the return value → attributes; (4) `route_example` → matrix rows + `depot=`/`init=`; (5) the three behaviour changes of the objective with the Barcelona worked example (numbers produced by the code); (6) the hyper-parameter rename table of §4.6 **and the "Defaults that changed" table** with the sentence "results with default parameters differ from 1.0"; (7) the "Every public name of 1.0.0a2" table of §4.6, verbatim; (8) datasets: Bunch fields, `as_frame`, `load_tsp`, `labels=b.labels`; (9) SOM with `coords=` **and the dense-matrix ceiling** (~20 000 nodes; coordinate-only fitting deferred to 2.1); (10) `Ensemble*` → `MultiStart`; (11) extras and dropped dependencies; (12) `CostScraper` → `GoogleDistanceMatrix` (or "returns in 2.1" if deferred).

**CI workflows** (lead):

- `ci.yml` — on push/PR. Job `lint` (ubuntu, 3.12): `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `cython-lint skroute`, `mypy skroute`. Job `test` matrix `os ∈ {ubuntu-latest, macos-latest, windows-latest} × python ∈ {3.11, 3.12, 3.13, 3.14}`: checkout, `setup-python`, `pip install -e ".[test]"` (builds the extension), `pytest tests --cov=skroute --cov-report=xml` (slow deselected by `addopts`), upload coverage with `codecov/codecov-action@v4` and the `CODECOV_TOKEN` repository secret (one job; the badge is dropped if the secret is not configured). Job `test-prerelease` (ubuntu, `3.15-dev`, `continue-on-error: true`): same steps. Job `sdist` (ubuntu, 3.12): `python -m build --sdist`, `pip install dist/*.tar.gz` in a fresh venv, `python -c "import skroute, sys; assert 'site-packages' in skroute.__file__, skroute.__file__"`, then `SKROUTE_EXPECT_WHEEL=1 pytest tests -x` — the `pytest` entry point, never `python -m pytest` (D16).
- `wheels.yml` — on tags `v*` and `workflow_dispatch`. `cibuildwheel` (≥ 3.0) matrix: Linux x86_64 and aarch64 on the default `manylinux_2_28` images (manylinux2014's CentOS 7 base is EOL; glibc ≥ 2.28 covers every distribution supported in 2026), macOS x86_64 and arm64, Windows AMD64; `build = "cp311-* cp312-* cp313-* cp314-*"`; `enable = ["cpython-prerelease"]` only in a separate `continue-on-error` job for cp315; `test-requires = ["pytest", "hypothesis", "pandas"]`, `test-command = "pytest {project}/tests -x -q"` (no `-m` clause — `cmd.exe` does not understand single quotes, and `addopts` already deselects `slow`), `test-environment = { SKROUTE_EXPECT_WHEEL = "1" }` so `test_base.py::test_runs_against_installed_copy` proves the wheel, not the checkout, is under test; sdist job; `pypa/gh-action-pypi-publish` with trusted publishing on tag (TestPyPI on `workflow_dispatch`). No abi3, no free-threaded wheels in 2.0.
- `docs.yml` — on push to `main` and `workflow_dispatch`: `pip install -e ".[docs,test]"`, `pytest --doctest-modules skroute docs README.md --doctest-glob="*.md" --ignore=docs/changelog.md --ignore=docs/contributing.md -q` (pytest collects doctests only under the paths given, so `docs` and `README.md` must be listed; `--doctest-glob="docs/**/*.md"` matched nothing — verified; the two ignored pages only include other files), `python docs/check_api_coverage.py`, `mkdocs build --strict`, `mkdocs gh-deploy --force` (Pages).
- `nightly.yml` — cron `0 3 * * *` and `workflow_dispatch`: ubuntu + macos, 3.13: `pytest tests -m slow -q --junitxml`, `python benchmarks/waterloo.py --out /tmp/benchmarks.md`, compared with the committed `docs/benchmarks.md`; when any gap moves by more than 0.5 percentage points the job opens or updates a PR "benchmarks: nightly update" with the regenerated page (`peter-evans/create-pull-request`). The committed page is regenerated by WP8 on the release candidate (release checklist item).

**Community files** (WP8, content fixed here so nobody has to ask): `CONTRIBUTING.md` (dev setup, `pre-commit install`, ownership rules of §8, how to add a solver: file, `__all__`, docs page, `check_router` green, a `tests/tolerances.py` entry, CHANGELOG line requested from the lead); `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1, enforcement contact `al.rubiales.b@gmail.com` — the address in `pyproject.toml`); `SECURITY.md` ("Report privately via GitHub → Security → Report a vulnerability, or e-mail al.rubiales.b@gmail.com; acknowledgement within 14 days; only the latest 2.x minor receives fixes; never open a public issue for a vulnerability"); `bug_report.yml` (fields: version, OS, Python, minimal matrix or dataset name, code, expected, actual, traceback); `feature_request.yml` (fields: problem, proposed API as a code block, alternatives considered, would you implement it); PR template (checklist: tests, docs page, CHANGELOG request, only my files, tolerances entry). `docs/about.md`: origin (2020, Alberto Rubiales, the Secoex routing problem), the 2026 rewrite, licence, how to cite (`CITATION.cff`), link to the benchmarks page. `docs/installation.md`: `pip install scikit-route` (wheels for CPython 3.11–3.14 on Linux x86_64/aarch64 `manylinux_2_28`, macOS x86_64/arm64, Windows AMD64), the extras table, source builds need a C compiler + Cython ≥ 3.1, no conda package yet (§10, #34). `docs/index.md`: the pitch, the two §3.4 examples, the capability table.

---

## 8. Work breakdown

**Dates (D26)**: week 0 starts 2026-09-07; core M1 2026-09-10, M2 2026-09-16; **feature freeze 2026-10-16** (P1 items ship or are deferred by written PR); release candidate 2026-10-23 (`docs/benchmarks.md` regenerated, nightly green on three OSes, every open issue dispositioned per §10); **2.0.0 on 2026-10-30**.

**Execution (D29)**: the work packages are executed by agents in parallel git worktrees merged by the lead, in two waves. Wave A: the spine (lead-owned files of §3 plus `utils/estimator_checks.py`, `tests/conftest.py`, `tests/tolerances.py`, `tests/test_common.py`), WP1 core, WP7 data (datasets + preprocessing) and WP8's community files, all at once; the lead merges, builds the core and runs the whole suite. Wave B: WP2–WP6, WP7 SOM and WP8's ensemble/shims/benchmarks/user guide against the compiled core. Nothing is developed against a stand-in: there is no `SKROUTE_REFERENCE` switch and no `_core/_reference.py`. Milestones M1/M2 collapse into 'WP1 complete before wave B'.

Rules: a PR touches only its owner's files (enforced by `CODEOWNERS` + required review); every `__init__.py` under `skroute/` except the subpackage ones listed as owned is lead-only and append-only; a new core primitive is requested by issue to WP1 (never edited in); a solver PR merges only when `check_router` is green for it, its `tests/tolerances.py` entry exists and passes on the tiny and fast tiers, its docs page exists, and its acceptance tests pass on the three OSes. With six implementers: merge WP5 into WP6 and WP7's SOM into WP4; never merge WP1 with anything.

| WP | Owner of (and only of) | Depends on | Definition of done |
|---|---|---|---|
| **L** lead | `pyproject.toml`, `setup.py`, `MANIFEST.in`, `README.md`, `CHANGELOG.md`, `CITATION.cff`, `mkdocs.yml`, `.pre-commit-config.yaml`, `.gitignore`, `.github/CODEOWNERS`, `.github/workflows/*`, `skroute/__init__.py`, `_version.py`, `py.typed`, `base.py`, `problem.py`, `exceptions.py`, `metrics.py`, `utils/**` **except `utils/estimator_checks.py`**, `metaheuristics/__init__.py`, `tests/test_base.py`, `docs/api/base.md`, `docs/api/utils.md`, `docs/migration.md` | — | week-0 spine merged; every export added on landing; release cut after nightly green on 3 OSes |
| **WP1** core | `skroute/_core/**` except `_core/__init__.py`, `tests/test_core.py`, `benchmarks/kernels.py` | `tests/reference.py` (WP8, day 1) | every function of §3.5 implemented (inline bodies in the `.pxd`) and `.pyi`-stubbed; hypothesis suite green; `benchmarks/kernels.py` reproduces the baseline (tour cost ≤ 30 µs at n = 10 639, 2-opt+Or-opt on fi10639 ≤ 150 ms) |
| **WP2** exact | `skroute/exact/**`, `tests/test_exact.py`, `docs/api/exact.md` | core M1 (M2 only for MILP's time-out fallback) | BruteForce/MILP per §4.1 (HeldKarp P1); MILP hits 27603/6656 fast and 9352 slow; `fit(optimal).cost_ <= fit(greedy).cost_` for BruteForce on alicante |
| **WP3** construction | `skroute/construction/**`, `tests/test_construction.py`, `tests/data/nrbs_barcelona_1_0.json`, `docs/api/construction.md` | core M1 | four classes per §4.2; NRBS 1.0 regression pinned by the stated procedure; ClarkeWright multi-trip test |
| **WP4** local search | `skroute/local_search/**`, `tests/test_local_search.py`, `docs/api/local_search.md` | core M2 | four classes per §4.3; ILS reaches optimum on tiny/alicante at 3 seeds; fast-tier gaps within table (slow-tier gaps run in WP8's `test_waterloo.py`, whose entries WP4 must make green) |
| **WP5** SA + Tabu | `skroute/metaheuristics/_simulated_annealing.py`, `_sa.pyx`, `_tabu_search.py`, `_tabu.pyx`, `tests/test_simulated_annealing.py`, `tests/test_tabu_search.py`, `docs/api/simulated_annealing.md`, `docs/api/tabu_search.md` | core M1 | both per §4.4; bit-identical reproducibility test; generic path covered by alicante and an asymmetric tiny |
| **WP6** GA + ACO | `skroute/metaheuristics/_genetic.py`, `_ga.pyx`, `_ant_colony.py`, `_aco.pyx`, `tests/test_genetic.py`, `tests/test_ant_colony.py`, `docs/api/genetic.md`, `docs/api/ant_colony.md` | core M1 (M2 for memetic) | Genetic per §4.4 with OX/PMX hypothesis tests; AntColony is P1 — ship or hand the lead a written deferral by the feature-freeze date |
| **WP7** SOM + data | `skroute/metaheuristics/_som.py`, `skroute/datasets/**`, `skroute/preprocessing/**`, `tests/test_som.py`, `tests/test_datasets.py`, `tests/test_preprocessing.py`, `tests/data/*` except `nrbs_barcelona_1_0.json`, `docs/api/som.md`, `docs/api/datasets.md`, `docs/api/preprocessing.md` | none (SOM needs core M1 only for evaluation) | loaders and reader per §5 (`GEO`/`ATT`/`GoogleDistanceMatrix` P1); `.pkl` deleted; `nint` test on qa194 (two half-integer entries); ulysses16 = 6859 / att48 = 10628 if `GEO`/`ATT` ship; Google client mocked; SOM per §4.4 (epochs, best-so-far history) |
| **WP8** ensemble, tests infra, docs | `skroute/ensemble/**`, `skroute/utils/estimator_checks.py`, all shim `__init__.py` under `heuristics/` and `metaheuristics/<legacy>/`, `tests/conftest.py`, `tests/reference.py`, `tests/tolerances.py`, `tests/test_common.py`, `tests/test_ensemble.py`, `tests/test_legacy_shims.py`, `tests/benchmarks/**`, `benchmarks/waterloo.py`, `docs/**` except `api/*` and `migration.md` (includes `gen_pages.py`, `check_api_coverage.py`, the committed `benchmarks.md`), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/ISSUE_TEMPLATE/*`, `.github/PULL_REQUEST_TEMPLATE.md` | base (day 0) | `reference.py`, `conftest.py`, `tolerances.py` and `check_router` on day 1 (structural checks 1–11, 13); `MultiStart` `n_jobs`-invariant; Ensemble wrappers per §4.5; shims warn with the exact text; site builds `--strict` with API coverage green; `benchmarks.md` committed on the release candidate |

Interfaces each package codes against: the fitted-attribute table (§3.4), `RoutingProblem` (§3.3), `_routing.pxd` (§3.5), `initial_tour` (§3.4), the glossary (§4.0), the fixtures (§6). Nothing else is shared.

---

## 9. Open risks and how to behave

- **R1 Core is late or incomplete.** Wave B does not start until WP1's hypothesis suite is green (D29). If a kernel of §3.5 turns out to be missing or wrong when a solver needs it, the solver owner reports it (never patches `_core/`), and the lead re-runs WP1 for that kernel before merging the solver; the `.pxd` signature does not change.
- **R2 A `.pxd` signature must change.** Open an issue titled `core: <function>`; WP1 and the lead approve; WP1 updates `.pxd/.pyx/.pyi` and every cimporting `.pyx` in the same PR (the only cross-ownership PR allowed), announced in the channel before merging.
- **R3 A tolerance fails on one OS.** Do not loosen it silently: post the three-OS gaps in the PR; the lead either accepts a documented loosening (CHANGELOG line) or asks for a default change. Never pin an exact float across platforms (libm `exp` differs by ulps and flips Metropolis decisions).
- **R4 Multi-trip metaheuristics are slow above ~2000 nodes** (no O(1) delta exists; full evaluation is 0.6–2.8 µs at n = 194–980). Documented ceiling; do not add Python-level shortcuts. If a user needs more, the answer is 2.1 (segment-based split deltas), not a hack.
- **R5 Cython compile surprises.** `cpdef` methods on `cdef class` with `noexcept nogil` and pointer arguments in `cpdef` signatures do not compile; the contract uses free functions and memoryviews only — do not introduce classes into `.pxd`. **`cdef inline` functions must be defined (with their bodies) in the `.pxd`**: a body-less inline declaration compiles in `_routing` itself but breaks every cimporting module in the C compiler (`'inline' can only appear on functions`) — there is no "it still links" path; only non-inline `cdef`/`cpdef` functions are shared through the module C-API. Verified with Cython 3.3.0: the bodies-in-`.pxd` variant compiles and `cimport`s cleanly with `noexcept nogil` and the `cpdef void` signatures. If a body must change, WP1 changes it in the `.pxd`; report, do not copy code.
- **R6 Wheel size** (~10.5 MB data, 3–4 MB compressed). **Decided for 2.0: the four instances > 20 000 nodes stay bundled** (readers and `n_nodes=` subsampling keep working offline; their DESCR says they cannot be solved whole in 2.0 — §4.4 SOM, §5.1). If PyPI or users complain, they move to a lazy download cache in 2.1 — keep `load_tsp` as the only entry point so that change is invisible.
- **R7 Hand-computed example numbers.** Two of the four proposals shipped examples that violated their own rules. Every docstring, docs page and README example is a `>>>` session collected by `pytest --doctest-modules skroute docs README.md --doctest-glob="*.md"` in `docs.yml` (the paths must be listed — pytest collects doctests only under the paths given). Doctest compares the written output on every CI run, so **stochastic solvers print only platform-stable facts** (`route_[0] == route_[-1] == depot_`, `n_iter_ == len(history_)`, `bool(np.all(trip_times_ <= 8.0))`, a gap below the tolerance) or carry `# doctest: +SKIP`; exact numbers appear only for deterministic solvers (BruteForce/HeldKarp/MILP), `optimal_tour_length` and dataset facts; floats use `NUMBER`; scalars are converted (`int()`, `float()`, `.tolist()`) so numpy 2's `np.int64(0)` repr never appears. `docs.yml` runs on ubuntu while authors paste from macOS: a float pinned from one machine is exactly the cross-platform pin D19 forbids (libm `exp` differs by ulps and flips Metropolis decisions). The example 3 arithmetic in §3.4 is a worked check for exactly this reason.
- **R8 `history_` semantics drift.** The base class asserts monotonicity only in tests (`check_router` 10); if a solver's natural trace is not monotone (SOM), record best-so-far per epoch, not current — as §4.4 now specifies — and say so in the docstring.
- **R9 Threads and Python-heavy solvers.** `MultiStart(prefer="threads")` gives near-linear speed-up for SA/ILS/Tabu (nogil kernels) and little for GA/SOM; the docs say so and `prefer="processes"` is one keyword away. Do not change the default per solver.
- **R10 Label dtype edge cases.** Mixed-type labels (ints and strings) become object arrays; `np.concatenate` of a 1-element label slice keeps the dtype (that is why `_set_results` slices `lab[d:d+1]` instead of wrapping the scalar). If a label type breaks `np.array_equal`, coerce in `coerce_labels`, never in a solver.
- **R11 Optimal split and BruteForce exactness.** Under `split="greedy"`, BruteForce is exact over greedy-decoded giant tours (a partition that closes a trip while the next node still fits is unrepresentable); under `split="optimal"` it is exact for the distance-constrained multi-trip problem. The user guide states this in one paragraph; tests check both.
- **R12 Scope creep.** Christofides with a greedy matching, MTZ "because it is simpler", a pure-Python fallback, `prange`, free-threaded wheels, on-the-fly distances: all are explicitly out of 2.0. Propose them for 2.1 in an issue; do not open a PR. P1 items (D26) are the only ones with a ship-or-defer switch.

---

## 10. Disposition of the 34 open GitHub issues (lead; release checklist: comment and close each one)

| Issues | Disposition |
|---|---|
| #1–#6, #8–#13, #16, #21 (documentation spelling/unification, datasets/preprocessing docs) | closed by 2.0: every page is rewritten (§7); mkdocstrings + `check_api_coverage.py` |
| #7 (file names to the Python convention) | closed by 2.0: layout of §2 (`_private.py` modules, public `__all__`) |
| #14, #15, #18 (cluster docs/tests) | closed won't-fix: `skroute.cluster` is removed (§4.6 table); use scikit-learn directly |
| #19, #20 (tests for datasets/preprocessing) | closed by 2.0: `tests/test_datasets.py`, `tests/test_preprocessing.py` (WP7) |
| #23 (euclidean distance) | closed by 2.0: `distance_matrix(metric="euclidean")` (§5.2) |
| #24 (Manhattan distance) | trivially added in 2.0: `metric="manhattan"` in `distance_matrix` plus TSPLIB `MAN_2D` (~5 lines, WP7) |
| #25 (cosine similarity) | closed won't-fix: not a routing metric; a user-supplied matrix is accepted by every solver |
| #26 (SOM distance in C++/Cython) | closed won't-fix: vectorised numpy is enough at the ≤ 20 000-node ceiling (§4.4) |
| #27–#28, #30–#32 (`distance_mode` per solver) | closed by 2.0: superseded by matrix input (numpy/DataFrame/dict) + `distance_matrix(metric=)` |
| #29 (Tabu speed/features) | closed by 2.0: TabuSearch rewritten in Cython (§4.4) |
| #33 (four tutorial notebooks) | deferred to 2.1: the user guide pages replace them for 2.0; an examples gallery is a 2.1 item |
| #34 (conda package) | post-release task (lead): conda-forge feedstock after the PyPI release; `installation.md` says "no conda package yet" |
| #37 (GA top-X % population kernel) | deferred to 2.1 as `Genetic(selection="truncation")`; noted in `docs/api/genetic.md` |
| #38 (unify `swap_route`/`mutate`) | closed by 2.0: one shared move set in `_routing.pxd` (§3.5) |

---

## Amendments during implementation (lead)

- 2026-09-03 GEO: the degree part of a DDD.MM coordinate is `int(x)` (truncation), not `nint(x)`; only truncation reproduces the published ulysses16 optimum 6859 (WP7 verified both variants).
- 2026-09-03 Core: the O(1) descents accept a move when `delta < -1e-9 * max(1, cost of the removed edges)` (a local scale — the current tour cost is not available inside the nogil kernel); `local_search_generic` uses the §4.0 test on the full cost. `problem_cost_py`/`trip_starts` release the GIL around the kernel call. `trip_starts` raises `ValueError` if the optimal split has no feasible partition (unreachable through `RoutingProblem` thanks to D5). The module also exposes `*_py` wrappers of every inline primitive (validated positions, `ValueError` on misuse) so the hypothesis suite can call them; solver `.pyx` files still `cimport` the inline bodies.
- 2026-09-03 Datasets: `read_tsplib` returns `coords`, `cost` (one of them `None`) and `labels`; the Spanish cost tables' `hours` column equals `(secs + 420) / 3600` off-diagonal (a fixed 7-minute stop per leg), so `time` is not `distance / speed` — documented in the DESCR. `GEO`, `ATT` and `GoogleDistanceMatrix` (P1) shipped in wave A.
- 2026-09-03 mypy runs with `python_version = "3.12"` because numpy >= 2.4 stubs use PEP 695 `type` statements; the code supports 3.11.

## Amendments after critique

One line per issue id of the three adversarial reviews (2026-09-03). "fixed" = applied as proposed or with the variant stated; "rejected" = kept the original decision, with the reason.

- ambiguity-01 / feasibility-01 — fixed: `cdef inline` primitives are defined with their bodies in `_routing.pxd`; non-inline functions declared there and defined in `.pyx`; frozen = signatures + semantics (D11, §3.5 header, R5, WP1 row).
- ambiguity-02 / feasibility-02 — fixed: `_SPLIT` uses `int(core.SplitRule.SPLIT_*)`; §3.5 documents the IntEnum exposure and the `.pyi` `class SplitRule(IntEnum)`.
- ambiguity-03 / feasibility-19 — fixed: `neighbours()` partitions a copy with the diagonal set to `inf`, stable ties by index; `test_base.py` case with 4 coincident points and a non-zero diagonal.
- ambiguity-04 / user-value-02 / feasibility-16 — fixed: `all_solvers()` defined (D27, §3.4): sorted by `__name__`, no-argument-constructible classes incl. the two Ensemble wrappers, never `MultiStart` (tested in `test_ensemble.py`); `check_router` takes an unfitted instance. The `include_ensembles`/`include_aliases` flags were not adopted: aliases no longer exist and the wrappers are ordinary solvers.
- ambiguity-05 / feasibility-17 / user-value-01 — fixed: alicante fixture carries `fit_kw` (labels + label depot) and `ref_kw` (`depot=0`, index); every loader-based example passes `labels=`; §5.1 states loader matrices carry no labels. `RoutingProblem.from_bunch` rejected: one keyword suffices and it would be a second entry point to document and test.
- ambiguity-06 / feasibility-05 / user-value-09 — fixed (feasibility-05 variant): `tests/` is not a package, `pythonpath = ["tests"]`, `import reference`; cibuildwheel `test-command = "pytest {project}/tests -x -q"` + `test-environment = { SKROUTE_EXPECT_WHEEL = "1" }`; `test_base.py` installed-copy guard; sdist job asserts `site-packages`; never `python -m pytest` (D16). cibuildwheel `test-sources` not used (the pythonpath route needs no copy).
- ambiguity-07 / feasibility-23 — fixed: check 10 runs the `time_limit` half only when the parameter exists; the allowed `stop_reason_` subsets are listed per solver (§3.4 table, §4.3, §4.4). `time_limit` was not added to the deterministic descents or SOM (they would then need a clock in a deterministic path).
- ambiguity-08 — fixed: SOM outer iteration = epoch of `max(1, n_iter // 100)` samples, `history_` best-so-far per epoch, `n_samples_`, `stop_reason_` in `{"converged", "max_iter"}`; the "documented exception" sentence is gone.
- ambiguity-09 / feasibility-14 — fixed: check 8 compares both decoders on the same fitted tour; `fit(optimal) <= fit(greedy)` asserted only for BruteForce in `test_exact.py`.
- ambiguity-10 / user-value-23 — fixed: `RouterTags.kind` (D28); tolerances moved out of `check_router` into `tests/test_common.py`, driven by `tests/tolerances.py` (single source, keyed by class name, `KeyError` for a missing class).
- ambiguity-11 / feasibility-18 / user-value-04 — fixed by removing `CheapestInsertion`/`FarthestInsertion` (D18): `Insertion(strategy=...)` is the only spelling; roster count restated as 15 (+AntColony; 14 without HeldKarp) + MultiStart + 2 wrappers.
- ambiguity-12 — fixed: `local_search` is `None` or a tuple ⊆ {"two_opt", "or_opt"}, a string normalised to a 1-tuple, `"both"` deleted; defaults ILS `("two_opt","or_opt")`, Genetic `None`, AntColony `("two_opt",)`.
- ambiguity-13 — fixed: position domains per move (2-opt/swap `1 <= i < j <= n-1`; Or-opt `1 <= i`, `i+L-1 <= n-1`, `0 <= j <= n-1`, `j ∉ [i-1, i+L-1]`), all ranges inclusive except `double_bridge`; `reverse_segment*` and `swap_delta` annotated.
- ambiguity-14 / feasibility-25 — fixed: kick positions drawn without replacement (`rng.choice(..., replace=False)`); for `n < 8` the kick is a random segment reversal (n = 3 → swap (1, 2)); new check 13 fits every solver at n = 3 and n = 4, symmetric and asymmetric.
- ambiguity-15 — fixed: one outer iteration = one call of each listed descent with `max_passes=1`, persistent buffers, `history_[k]` = cost after iteration k, `"converged"`/`"max_iter"` rule; `LocalSearch(moves=...)` accepts only `two_opt`/`or_opt`; ILS's inner descent uses the same accounting.
- ambiguity-16 — fixed: `utils/estimator_checks.py` is WP8's (the one `utils/` file the lead does not own); `check_router(estimator)` runs structural checks 1–11 and 13 on self-built instances; `check_router.checks` exposes `(name, fn)` pairs for `test_common.py`.
- ambiguity-17 — fixed: Ensemble wrappers' tags, `_solve` (MultiStart with `random_state=rng`, `prefer="threads"`, `n_jobs`/`verbose` propagated), copied attributes and return value spelled out (§4.5).
- ambiguity-18 / feasibility-07 / feasibility-08 / user-value-08 — fixed: `docs.yml` runs `pytest --doctest-modules skroute docs README.md --doctest-glob="*.md" --ignore=docs/changelog.md --ignore=docs/contributing.md`; `doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS NUMBER"`; §3.4 examples rewritten as `>>>` sessions printing platform-stable facts, exact numbers only for deterministic solvers; numpy 2 scalar reprs avoided via `int()/float()/.tolist()`; R7 rewritten.
- ambiguity-19 — fixed: `route_cost(depot=None)` means `route[0]`, a given depot must equal it; `split_trips(route, depot=None)` returns closed trips and treats an open tour as one trip; `reference.route_cost_from_labels` mirrors it; `labels=` added to `route_cost`.
- ambiguity-20 — fixed at the time (reference switch in `_core/__init__.py`), then SUPERSEDED by D29: the core is built before any solver work starts, so there is no `SKROUTE_REFERENCE` switch and no `_core/_reference.py`.
- ambiguity-21 / feasibility-09 — fixed: `lower_bound_` = largest valid bound seen (status-0 relaxation objectives, non-`None` `mip_dual_bound`), `res.fun` of a time-limited solve never used, `gap_` clipped at 0; time-out fallback is `core.nearest_neighbour_tour` + `core.two_opt_descent` (WP2 depends on M2 for that path only), never another WP's solver.
- ambiguity-22 — fixed: the SA draw → move mapping and the "invalid draw = rejected proposal that counts towards `n_moves`" rule (§4.4).
- ambiguity-23 — fixed: `coerce_labels(seq, n)` defined (int64 for int-like non-bool labels, object otherwise, `n` unique) and used by the dict and DataFrame paths of `coerce_matrix`; label dtype is always int64 or object.
- ambiguity-24 — fixed: `max_time_work` must be finite and > 0; `extra_cost`/`people`/`split` without `max_time_work` raise `ValueError("extra_cost, people and split have no effect without max_time_work")` (D3 spirit); `fit()`'s RoutingProblem passthrough check already covered these.
- ambiguity-25 — fixed: `tenure="auto"` = `rng.integers(ceil(sqrt(n)), 2*ceil(sqrt(n)) + 1, size=n_iter)` (inclusive both ends), int = fixed tenure with `Interval(Integral, 1, None, closed="left")`; generic path evaluates 2-opt and Or-opt L = 1, 2, 3 per candidate pair.
- ambiguity-26 — fixed: descents return `cost_after - cost_before <= 0`; callers add it.
- ambiguity-27 — fixed: `trip_starts` and `problem_cost_py` hold the GIL, malloc/free their scratch, may raise `MemoryError`, and are the only non-`noexcept nogil` functions (§3.5 header and listing).
- ambiguity-28 — fixed: `tiny_instance` everywhere; check 11 rewritten (see feasibility-15) with the new `medium_euclidean` (n = 40) fixture.
- ambiguity-29 — fixed: a merged Clarke–Wright trip keeps the smaller creation index, trips concatenated by increasing index and oriented by the endpoint nearer the depot; `shape` added to the glossary.
- ambiguity-30 — fixed: all slow-tier gap tests live in `tests/benchmarks/test_waterloo.py` (WP8), stated in §4.0, §6 and WP4's definition of done.
- ambiguity-31 — fixed: `TSPBunch(Bunch)` with a real `distance_matrix()` method cached in `_distance_matrix`; `keys()` lists data only.
- ambiguity-32 / feasibility-22 — fixed: `[tool.setuptools.package-data]` list in `pyproject.toml` (py.typed, .pxd, .pyi, .tsp, .csv, `_descr/*.md`); MANIFEST.in includes `*.md` and `py.typed`.
- ambiguity-33 — fixed: "the five exponents default to the value 1.0 (1.0.0a2 had no defaults)"; `distance_weigth → distance_weight` in §4.2, §4.6 and the CHANGELOG.
- ambiguity-34 — fixed: `n_simulateds` default change 20 → 10 stated; "Defaults that changed" table added to §4.6 and migration item (6).
- ambiguity-35 — fixed: check 4 uses `pytest.approx(..., rel=1e-12)`.
- ambiguity-36 / user-value-03 — fixed: `_euclid` returns `(C, coords)`; every instance dict carries `coords`; `fit_kwargs(Solver, inst)` passes them for `requires_coords` and `pytest.skip`s on asymmetric instances; `check_router` builds its instances with coordinates.
- feasibility-03 — fixed: SA `patience=None` by default (measured first-improvement levels 413–1023 quoted), patience counted only after the current cost first falls below the initial cost; `EnsembleSimulatedAnnealing` default updated; glossary updated.
- feasibility-04 — fixed: the optimal-split loop breaks on the monotone OPEN path, tests closed feasibility per `i`; `reference.optimal_split` uses the same rule.
- feasibility-06 — fixed: `test-command` without the `-m 'not slow'` clause (addopts deselects slow; `cmd.exe` quoting) in D16 and `wheels.yml`.
- feasibility-10 — fixed: BruteForce enumerates lexicographically with `next_permutation`; halving keeps `tour[1] < tour[n-1]`; ties match `itertools.permutations`.
- feasibility-11 — fixed: NearestNeighbour fast tier ≤ 0.50; measured baseline recorded (§4.2, tolerance table, `benchmarks.md`).
- feasibility-12 — fixed: Genetic plain ≤ 0.15 fast / ≤ 0.30 qa194 only; memetic `("two_opt",)` ≤ 0.05 / ≤ 0.08 (qa194, lu980, `@slow`, minutes); `patience` default 100 for Genetic and EnsembleGenetic.
- feasibility-13 — fixed: TwoOpt/OrOpt ≤ 0.20 / 0.25 on fast and slow tiers; `n_candidates=None` = full neighbourhood added as an option (default stays 10).
- feasibility-15 — fixed: check 11 asserts different `history_`/`n_iter_` on `small_euclidean` or different `tour_` on `medium_euclidean` (n = 40), and `bit_generator.state` advancement; "statistical" removed.
- feasibility-20 — fixed: `RoutingProblem._as_index` coerces `tour`/`starts` with `np.ascontiguousarray(..., dtype=np.int64)` in `evaluate`, `trip_starts`, `trip_costs`, `trip_times`.
- feasibility-21 — fixed: default `manylinux_2_28` images; manylinux2014 requirement dropped.
- feasibility-24 — fixed: `int max_segment` added to `local_search_generic` (after `moves`); generic path documented as first-improvement only; `LocalSearch` gains `max_segment=3`.
- feasibility-26 — fixed: GEO and ATT formulas spelled out (truncation + 1 for GEO); regression tests ulysses16 = 6859 and att48 = 10628 via bundled `.opt.tour` files (active when the P1 metrics ship).
- feasibility-27 — fixed: `tabu_until` is `int32`; practical ceiling ~5 000 nodes documented. The hash-table variant was not adopted (out of 2.0 scope; the ceiling is documented instead).
- feasibility-28 — fixed: NRBS connection score divides by `max(C[i,j], 1e-12)`, zero-distance pairs connected first.
- feasibility-29 — fixed: slow-tier SA ≤ 0.10, ILS ≤ 0.06, LocalSearch ≤ 0.15, and the measured baselines recorded in `tolerances.MEASURED`/`benchmarks.md`.
- feasibility-30 — fixed: `BaseRouter.__eq__` compares parameter by parameter with `np.array_equal` for ndarrays (`_param_equal`).
- feasibility-31 — fixed: `if TYPE_CHECKING:` eager imports of every public name in `skroute/__init__.py`; `all_solvers()` imports eagerly from the subpackage `__all__` lists.
- user-value-05 — fixed: "Every public name of 1.0.0a2" table and "Defaults that changed" table in §4.6, reproduced verbatim in `docs/migration.md` (items 6 and 7); CHANGELOG Removed/Changed lines updated (cluster, `df_to_tuple`, `matrix_parse`, `DataLossWarning`, datasets return shape, defaults).
- user-value-06 — fixed: `CostScraper` is a thin deprecated wrapper reproducing the 1.0 constructor, `scrap()` and `pandas()`; `to_pickle()` raises with a message; goes with `GoogleDistanceMatrix` if that is deferred.
- user-value-07 — fixed: the dense-matrix ceiling (~20 000 nodes) is stated in §4.4 SOM, §5.1 (DESCR of the four large instances with the `n_nodes=5000` hint), migration item (9) and `problem_model.md`; coordinate-only fitting deferred to 2.1 (D18); R6 decided: the four instances stay bundled.
- user-value-10 — fixed: §10 dispositions all 34 open issues (closed-by-2.0, trivially added #24 → `manhattan`/`MAN_2D`, won't-fix #25/#26, deferred #33/#37, post-release #34); release checklist item.
- user-value-11 — fixed: `NullHandler` on the `skroute` logger, `skroute.set_log_level()`, mandatory closing sentence of every `verbose` docstring, shown once in `getting_started.md` (D24).
- user-value-12 — fixed in part: D26 adds the P1 ship-or-defer rule with a feature-freeze date (2026-10-16), RC (2026-10-23) and release (2026-10-30) to HeldKarp, `GEO`/`ATT` in `read_tsplib`, and `GoogleDistanceMatrix`; nav/CHANGELOG/tolerance rows are conditional. Rejected for `EXPLICIT`/`CEIL_2D` (needed by the reader round-trip test and trivial) and for `SPLIT_OPTIMAL` inside `local_search_generic` (D1 promises the optimal split to every budget-aware solver; making the hottest kernel greedy-only would break that promise for LocalSearch/ILS/Tabu under `split="optimal"`).
- user-value-13 — fixed: the deferral procedure (files, nav, CHANGELOG, tolerances, exports, one cross-ownership PR) is written in §4.4 and applies to every P1 item.
- user-value-14 — fixed: `CITATION.cff` [L] in §2 and the L row; README cites via it; `codecov/codecov-action@v4` with `CODECOV_TOKEN` (badge dropped if the secret is absent).
- user-value-15 — fixed: SECURITY.md, CODE_OF_CONDUCT.md contact, issue-form fields, `about.md`, `installation.md`, `index.md` contents specified in §7.
- user-value-16 — fixed: `api/base.md` directives listed, `api/utils.md` added to the nav, `docs/check_api_coverage.py` in `docs.yml`.
- user-value-17 — fixed: `docs/gen_pages.py` (mkdocs-gen-files) writes the capability table; README carries it between markers refreshed by a pre-commit hook; `mkdocs-literate-nav` removed from the docs extra.
- user-value-18 — fixed: `docs/benchmarks.md` is committed on the RC with a provenance header; the nightly job opens a PR when a gap moves by > 0.5 pp.
- user-value-19 — fixed: `time_matrix` is keyword-only in `fit` and `RoutingProblem` (D7); every example, fixture and test updated; migration item (2) opens with the boxed warning.
- user-value-20 — fixed: exact shim contents (1.0 `__all__` per package, `SimmulatedAnnealing` typo noted), warning text with `stacklevel=2`, and what `test_legacy_shims.py` asserts (§4.6).
- user-value-21 — fixed: example 1 passes `labels=wi.labels` and shows the depot as id 1; `fast_instance`/`slow_instance` carry `labels`.
- user-value-22 — fixed: the NRBS pin procedure (worktree at `533f320`, Python ≤ 3.11, float64 `tour_cost` substitution, `tests/data/nrbs_barcelona_1_0.json` with provenance, sequence and cost pinned) replaces "the 1.0 checkout".
- user-value-24 — fixed in part: `budget_aware` defaults to `False` and the opt-in list is written next to `RouterTags` (D28). The proposed smoke check ("budget-aware `cost_` ≤ budget-unaware decode of its plain tour") is rejected: for a stochastic heuristic the two fits are different searches and the inequality is not guaranteed — it would be flaky by construction, exactly what checks 8 and 11 were just cured of.
- user-value-25 — fixed: one statement of the top-level surface in §3.4 ("Top-level surface"); §4.6 points to it.
