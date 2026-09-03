# Installation

## From PyPI

```bash
pip install scikit-route
```

2.0.0 is the first release of scikit-route published on PyPI. Binary wheels are provided for **CPython 3.11,
3.12, 3.13 and 3.14** on:

| Platform | Architectures | Wheel tag |
|---|---|---|
| Linux | x86_64, aarch64 | `manylinux_2_28` (glibc 2.28 or newer: every distribution supported in 2026) |
| macOS | x86_64 (Intel), arm64 (Apple silicon) | `macosx` |
| Windows | AMD64 | `win_amd64` |

With a wheel you need no compiler. The runtime dependencies are `numpy>=1.26`, `scipy>=1.11` and
`joblib>=1.3`; numpy 2 is supported.

Check the installation:

```bash
python -c "import skroute; print(skroute.__version__)"
```

## Extras

| Extra | Installs | When you need it |
|---|---|---|
| `pandas` | `pandas>=2.2` | `fit` on a `DataFrame` cost matrix (labels from the index) and `load_*(as_frame=True)` |
| `google` | `googlemaps>=4.10` | `skroute.preprocessing.google.GoogleDistanceMatrix`, the optional client that fetches driving distances and times |
| `test` | pytest, pytest-cov, hypothesis, pandas | running the test suite against your installation |
| `dev` | `test` + ruff, mypy, cython-lint, pre-commit, build, cibuildwheel, Cython | contributing (see the [contributing guide](contributing.md)) |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-gen-files, mkdocs-include-markdown-plugin | building this site |

```bash
pip install "scikit-route[pandas]"
pip install "scikit-route[pandas,google]"
```

DataFrames are recognised by duck typing, so a plain `pip install scikit-route` never imports pandas.

## From source

The core of scikit-route is a Cython 3 extension. Building it needs a **C compiler** and **Cython 3.1 or
newer**; `pip` fetches Cython and setuptools itself in an isolated build environment, so you only provide
the compiler:

- Linux: `gcc` or `clang` (`sudo apt install build-essential`, `sudo dnf groupinstall "Development Tools"`).
- macOS: the Xcode command line tools (`xcode-select --install`).
- Windows: the "Desktop development with C++" workload of the Visual Studio Build Tools.

```bash
# the released source distribution
pip install --no-binary scikit-route scikit-route

# the development version
git clone https://github.com/arubiales/scikit-route.git
cd scikit-route
pip install .
```

No numpy headers are needed at build time: the kernels work on typed memoryviews, so a wheel built against
one numpy version runs with any supported numpy.

If `import skroute` fails with *"scikit-route's compiled core is missing"*, the extension was not built:
install a wheel, or install a compiler and run `pip install .` again from the source tree.

## Supported Python versions

| Python | Status |
|---|---|
| 3.11, 3.12, 3.13, 3.14 | supported; wheels published; tested on Linux, macOS and Windows in CI |
| 3.15 pre-release | tested as an allowed failure; no wheel until the final release |
| 3.10 and older | not supported (`requires-python = ">=3.11"`) |

## conda

There is no conda package yet. Install with `pip` inside your conda environment; a conda-forge feedstock is
planned after the PyPI release.

## Upgrading from 1.0

2.0 changes the `fit` signature, the return value and the objective. Read the
[migration guide](migration.md) before upgrading: the one-line summary is that 1.0's
`fit(route, time, cost)` becomes `fit(cost, time_matrix=time, depot=route[0])`, `fit` returns the estimator,
and the results are in `route_`, `trips_` and `cost_`.
