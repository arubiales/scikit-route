# Contributing to scikit-route

Thank you for taking the time. This guide covers the development setup, the checks a change must pass, who
owns which files, and the step-by-step recipe for adding a solver. The binding technical reference is the
[2.0 specification](https://github.com/arubiales/scikit-route/blob/main/docs/development/specification.md);
where this guide is shorter than the specification, the specification wins.

By participating you agree to the
[Code of Conduct](https://github.com/arubiales/scikit-route/blob/main/CODE_OF_CONDUCT.md). Security issues
go through [SECURITY.md](https://github.com/arubiales/scikit-route/blob/main/SECURITY.md), never through a
public issue.

## Development setup

scikit-route needs **Python 3.11 or newer**. The core is a Cython 3 extension, so installing from source
needs a **C compiler** (gcc or clang on Linux, the Xcode command line tools on macOS, the Visual Studio
Build Tools on Windows); `pip` fetches Cython itself in the isolated build.

```bash
git clone https://github.com/arubiales/scikit-route.git
cd scikit-route
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # compiles the Cython core in place and installs every dev tool
pre-commit install                 # ruff, cython-lint and whitespace hooks run on every commit
```

`pip install -e ".[dev]"` pulls in the `test` extra (pytest, pytest-cov, hypothesis, pandas) plus ruff, mypy,
cython-lint, pre-commit, build, cibuildwheel and Cython. The docs tools are a separate extra:
`pip install -e ".[docs]"`.

After editing a `.pyx` or `.pxd` file, rebuild the extensions: `pip install -e .` again, or the quicker
`python setup.py build_ext --inplace`. The `.so`/`.pyd` and `.c` files are git-ignored; never commit them.

## Running the tests

```bash
pytest                              # the fast suite: unit tests, check_router over every solver, tiny and fast tiers
pytest -m slow                      # the benchmarks: Waterloo instances with n >= 194 (minutes; nightly in CI)
pytest --doctest-modules skroute    # every Examples section of every docstring
pytest --cov=skroute --cov-report=term-missing   # what CI measures; target is 90 % of the .py lines
```

Conventions worth knowing before you write a test:

- `tests/` is **not a package** (no `__init__.py`), on purpose: wheels are tested from another working
  directory. `pyproject.toml` puts `tests/` on `sys.path`, so test modules import the pure-Python oracle
  as `import reference`.
- `pytest` deselects `slow` through `addopts`; `-m slow` on the command line selects it. Markers are strict:
  `slow` and `benchmark` are the only ones.
- Every slow-tier gap test lives in `tests/benchmarks/test_waterloo.py`; the per-package test files hold the
  tiny, fast-tier and multi-trip acceptance tests only.
- Tolerance numbers live in **one** place, `tests/tolerances.py` (`TINY`, `FAST`, `SLOW`, `SEEDS_TO_OPTIMUM`,
  `MEASURED`). Never write a gap number into a test module.
- Randomness is controlled by `random_state`; reproducibility is asserted as "same seed, same machine,
  bit-identical". Never pin an exact float across platforms: libm differs by ulps and flips Metropolis
  decisions. Compare against an integer optimum, a tolerance, or `pytest.approx`.
- Nothing in the suite touches the network. The Google client is tested with a mock.
- Every solver fixture and every example passes `labels=` when the matrix comes from a loader (loader
  matrices are plain ndarrays and carry no labels).

The docs examples are tested the way `docs.yml` does it:

```bash
pytest --doctest-modules skroute docs README.md --doctest-glob="*.md" \
       --ignore=docs/changelog.md --ignore=docs/contributing.md -q
```

## Lint, format and types

```bash
ruff check .                # lint (rules and per-file ignores in pyproject.toml)
ruff format .               # format; CI runs `ruff format --check .`
cython-lint skroute         # the .pyx/.pxd files
mypy skroute                # type-check the package (tests/ and docs/ are excluded)
```

All four run in the `lint` job of CI and must be clean. `pre-commit run --all-files` runs the first three
plus the whitespace, YAML and TOML hooks.

House rules that the linters cannot check:

- Type hints on every public signature; numpydoc docstrings on every public symbol (see "Docstrings" below).
- **No `print()` in library code.** `verbose` output goes to `logging.getLogger("skroute")` at INFO. Every
  `verbose` docstring ends with the sentence: *Records go to the `skroute` logger at INFO; enable them with
  `logging.basicConfig(level=logging.INFO)` or `skroute.set_log_level("INFO")`.*
- Money and costs are `float64`, indices `int64`; no `float32` matrices in 2.0.
- No numpy C-API in Cython (`cimport numpy` is forbidden): kernels take typed memoryviews, so wheels are
  numpy-ABI independent. Randomness is pre-drawn in Python with `numpy.random.default_rng(random_state)`
  and passed to the kernels as arrays.
- No pure-Python fallback of the core, no new hard dependencies (runtime is numpy, scipy and joblib).

## Repository layout

```
scikit-route/
  pyproject.toml, setup.py, MANIFEST.in     packaging; setup.py only cythonizes skroute/**/*.pyx
  README.md, CHANGELOG.md, CITATION.cff     lead-owned front matter
  CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
  .github/                                  CODEOWNERS, workflows (ci, wheels, docs, nightly), issue forms, PR template
  benchmarks/                               kernels.py (core micro-benchmarks), waterloo.py (gap table -> docs/benchmarks.md)
  skroute/
    __init__.py                             lazy public exports, all_solvers(), set_log_level()  [lead, append-only]
    base.py, problem.py, metrics.py         BaseRouter/RouterTags/clone, RoutingProblem, route_cost/split_trips  [lead]
    exceptions.py, utils/                   NotFittedError, InfeasibleProblemError; validation, Bunch, initial_tour  [lead]
    utils/estimator_checks.py               check_router(): the structural battery every solver must pass
    _core/_routing.pxd, _routing.pyx        the ONE Cython core: evaluation, split rules, move deltas, descents, construction
    exact/         BruteForce, HeldKarp, MILP
    construction/  NearestNeighbour, Insertion, ClarkeWright, NRBS
    local_search/  TwoOpt, OrOpt, LocalSearch, IteratedLocalSearch
    metaheuristics/ SimulatedAnnealing, TabuSearch, Genetic, AntColony, SOM  (__init__.py is lead-owned, append-only)
    ensemble/      MultiStart, EnsembleGenetic, EnsembleSimulatedAnnealing
    heuristics/, metaheuristics/<legacy>/   deprecated 1.0 import paths (shims that warn)
    datasets/      load_tsp + 27 TSPLIB country wrappers, 5 cost datasets, read_tsplib; data under _data/, DESCR under _descr/
    preprocessing/ distance_matrix (TSPLIB metrics), pairs_to_matrix, dict-of-dicts helpers, google.py (optional client)
  tests/                                    NOT a package; conftest.py, reference.py (oracles), tolerances.py, test_*.py
    benchmarks/test_waterloo.py             every @slow gap test
    data/                                   reader fixtures (.tsp, .tour, the pinned NRBS result)
  docs/                                     mkdocs-material site: user guide, api/*.md (one page per package), benchmarks.md,
                                            migration.md, gen_pages.py (capability table), check_api_coverage.py
```

A solver package is a directory with a public `__init__.py` (its `__all__` lists the classes), one private
module per solver (`_two_opt.py`) and, when it has a hot loop, a `.pyx` of the same stem next to it that
`cimport`s the inline primitives of `skroute._core._routing`. Public names live in `__all__`; everything
else is private.

## Who owns what

The code is developed by several people in parallel and merged by the lead, so **a pull request touches
the files of one package only**. `.github/CODEOWNERS` enforces it with required reviews.

- **The lead owns the spine**: `pyproject.toml`, `setup.py`, `MANIFEST.in`, `README.md`, `CHANGELOG.md`,
  `CITATION.cff`, `mkdocs.yml`, `.pre-commit-config.yaml`, `.gitignore`, `.github/CODEOWNERS`,
  `.github/workflows/*`, `skroute/__init__.py`, `_version.py`, `py.typed`, `base.py`, `problem.py`,
  `exceptions.py`, `metrics.py`, `utils/**` (except `utils/estimator_checks.py`),
  `metaheuristics/__init__.py`, `tests/test_base.py`, `docs/api/base.md`, `docs/api/utils.md` and
  `docs/migration.md`.
- Every `__init__.py` under `skroute/` that is not the public `__init__.py` of a solver package you own is
  **lead-only and append-only** (one export per line, added at the end).
- The core (`skroute/_core/**`) has one owner. A missing or wrong kernel is **requested**, never patched in
  from a solver PR: open an issue titled `core: <function>`; the core owner changes `.pxd`/`.pyx`/`.pyi` and
  every `cimport`ing `.pyx` in one PR. The `.pxd` signatures are frozen; their semantics are the contract.
- `tests/reference.py` (the pure-Python oracles), `tests/conftest.py`, `tests/tolerances.py`,
  `tests/test_common.py` and `tests/benchmarks/**` belong to the test-infrastructure owner.
- When you need a change in a file you do not own, write it in the PR description under **For the lead**
  (the PR template has the section): the export to add, the CHANGELOG line, the nav entry, the tolerance row.

## Adding a solver, step by step

1. **Read first**: the specification's public API contract (`RoutingProblem`, `BaseRouter`, the fitted
   attributes table), the core `.pxd` contract and the per-solver section closest to yours. Names are not
   free: `n_iter`, `max_passes`, `patience`, `time_limit`, `init`, `n_candidates`, `random_state`, `verbose`,
   `max_nodes`, `local_search` (`None` or a tuple of move names among `"two_opt"`/`"or_opt"`) mean one thing
   everywhere; reuse them.

2. **Create the module** `skroute/<package>/_my_solver.py` and, if it has a hot loop,
   `skroute/<package>/_my_solver.pyx` next to it (`setup.py` picks it up automatically; `cimport` from
   `skroute._core._routing`, never copy a kernel).

3. **Subclass `BaseRouter`**:
   - `__init__` stores every argument verbatim under the same name and does nothing else (no validation,
     no derived attributes). Knobs go here; data goes to `fit`.
   - `_parameter_constraints`: a dict `{"param": [Interval(...) | Options(...) | "boolean" | "random_state" | ...]}`
     from `skroute.utils._param_validation`; the base class validates at fit time with scikit-learn's message
     format.
   - `_get_tags()` returns a `RouterTags(kind=..., exact=..., stochastic=..., iterative=..., budget_aware=...,
     requires_symmetric=..., requires_coords=..., max_nodes=...)`. `budget_aware` defaults to `False`: opt in
     only when the search itself sees the multi-trip objective.
   - `_solve(problem, rng) -> np.ndarray`: an `int64` permutation of `range(problem.n)` with `problem.depot`
     at position 0, in **index space** (labels are the base class's business). Never compute or report a cost:
     `BaseRouter.fit` recomputes `cost_` from your tour with the problem's own decoder and raises
     `RuntimeError` on an invalid tour. Iterative solvers set `history_` (best-so-far per outer iteration,
     non-increasing), `n_iter_ == len(history_)` and `stop_reason_` (from the documented subset) inside
     `_solve`; exact solvers set `is_optimal_`. `rng` is a `numpy.random.Generator` for stochastic solvers
     and `None` otherwise; pre-draw your random numbers and hand them to the kernels as arrays.
   - Improvement test everywhere: `new < best - 1e-9 * max(1.0, abs(best))`.

4. **Docstring** (numpydoc): one-line summary, `Parameters`, `Attributes` (the fitted ones you add plus a
   pointer to the common ones), `Notes` with the algorithm, its complexity and the tags line
   ("Supports: symmetric and asymmetric matrices, multi-trip objective; stochastic"), `References`, and
   `Examples` that run deterministically (see "Docstrings and examples" below).

5. **Export it in your package**: add the class to `skroute/<package>/__init__.py` and its `__all__`.
   The top-level name (`skroute.MySolver`: PEP 562 lazy export, `TYPE_CHECKING` import and `__all__` entry
   in `skroute/__init__.py`) is added **by the lead**; request it in the PR. `all_solvers()` picks the class
   up from your package's `__all__` once it is no-argument constructible.

6. **Docs page**: add `::: skroute.<package>.MySolver` to `docs/api/<package>.md`.
   `docs/check_api_coverage.py` fails the docs build when a name in any `__all__` has no directive.

7. **`check_router` must pass**: `from skroute import check_router; check_router(MySolver())` runs the
   structural battery (parameter protocol, fitted attributes, input coercion, error messages, tags honoured,
   multi-trip decoding, no stdout, history monotonicity, reproducibility, n = 3 and n = 4). It is also
   parametrised over `all_solvers()` in `tests/test_common.py`, the merge gate of every solver PR.

8. **Acceptance tests** in `tests/test_<package>.py`: the tiny instances (`tiny_instance`, n = 5..9, optimum
   by brute force), the fast tier (`fast_instance`: wi29, dj38) and, for budget-aware solvers, the `alicante`
   multi-trip fixture. Gaps are `cost_ / optimum - 1` with `random_state=0` and default parameters; the
   number itself comes from the tolerance table, which is data in `tests/tolerances.py` (`TINY`, `FAST`,
   `SLOW`, `SEEDS_TO_OPTIMUM`, `MEASURED`). Request the entry for your class in the PR, together with the
   measured gaps on the three CI operating systems; a class without an entry fails `test_common.py` with
   `KeyError`. Slow-tier gaps run only in `tests/benchmarks/test_waterloo.py`. Loosening a tolerance needs
   the lead's approval and a CHANGELOG line; tightening one is a release-notes item.

9. **CHANGELOG line**: put the line you want under `### Added` in the PR description; the lead edits
   `CHANGELOG.md`.

10. Run everything before opening the PR: `pytest`, `pytest --doctest-modules skroute`, `ruff check .`,
    `ruff format --check .`, `cython-lint skroute`, `mypy skroute`.

Out of scope for 2.0 (propose in an issue, do not open a PR): Christofides with a real blossom matching,
Or-3opt/Lin-Kernighan, VNS, a multi-trip MILP, `float32` matrices, on-the-fly distances and coordinate-only
fitting, `prange`, free-threaded wheels, a pure-Python fallback.

## Docstrings and examples

Every `Examples` section is executed by `pytest --doctest-modules` in CI (Linux), while authors write on
macOS or Windows, so an example must print only **platform-stable facts**:

- Stochastic solvers print booleans and structural equalities: `route_[0] == route_[-1] == depot_`,
  `n_iter_ == len(history_)`, `bool(np.all(trip_times_ <= 8.0))`, `cost_ / optimum < tolerance`.
- Exact numbers appear only for deterministic solvers (`BruteForce`, `HeldKarp`, `MILP`), dataset
  constants (`optimal_tour_length`) and integer-valued matrices.
- Convert numpy scalars before printing (`int()`, `float()`, `.item()`, `.tolist()`); numpy 2 prints
  `np.int64(0)` otherwise. The doctest flags `NORMALIZE_WHITESPACE ELLIPSIS NUMBER` are on.
- A line whose output is run-dependent carries `# doctest: +SKIP`.
- Pass `labels=b.labels` whenever the matrix comes from a loader.

```python
>>> from skroute import IteratedLocalSearch
>>> from skroute.datasets import load_tsp
>>> wi = load_tsp("wi29")
>>> ils = IteratedLocalSearch(random_state=0).fit(wi.distance_matrix(), labels=wi.labels)
>>> ils.cost_ / wi.optimal_tour_length < 1.03
True
>>> int(ils.route_[0]) == int(ils.route_[-1]) == int(ils.depot_) == 1
True
```

## Branches, commits and pull requests

- Branch from `main`; never commit to `main` directly. Work-package branches are `wp/<package>`
  (`wp/local_search`); any other change is `<package>/<short-description>` (`datasets/manhattan-metric`,
  `docs/multi-trip-guide`).
- Commit messages in **English, imperative mood**, one line of at most 72 characters that says what the
  commit does ("Add Or-opt segment length parameter", "Fix nint rounding on half-integers"); details in the
  body when the title is not enough.
- One PR per package, one topic per PR; fill in the template's checklist honestly and the **For the lead**
  section when you need a shared file changed. A PR with a migration of behaviour (a default changed, a
  tolerance moved) says so in its description and in the requested CHANGELOG line.
- CI must be green on the three operating systems and Python 3.11 to 3.14 before review. A tolerance that
  fails on one OS is not loosened silently: post the three-OS gaps in the PR and let the lead decide.

## Building the documentation

```bash
pip install -e ".[docs]"
python docs/check_api_coverage.py     # every public name has a documentation home
mkdocs serve                          # live preview at http://127.0.0.1:8000
mkdocs build --strict                 # what docs.yml runs; broken cross-references fail the build
```

Docstrings are rendered by mkdocstrings (numpy style). The capability table in the user guide is generated
by `docs/gen_pages.py` from `all_solvers()` and copied nowhere by hand; the README carries the same table
between markers refreshed by `python docs/gen_pages.py --readme`. `docs/benchmarks.md` is produced by
`benchmarks/waterloo.py` on the release candidate and committed with its provenance header.

## Questions

Open a [discussion](https://github.com/arubiales/scikit-route/discussions) for design questions and usage
help, an issue (with the forms) for bugs and feature requests. Thank you for contributing.
