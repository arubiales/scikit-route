"""The deprecated 1.0 import paths (SPEC §4.6, "Shim contents"): every legacy ``__init__.py`` re-exports
exactly the 1.0 ``__all__`` of its package, the re-exported objects ARE the 2.0 classes, and the module
warns once at import with the exact text and ``stacklevel=2``.

``pyproject.toml`` turns ``DeprecationWarning``s attributed to ``skroute.*`` into errors, so the shims are
observed through ``pytest.warns`` after evicting them from ``sys.modules`` (a module body runs once)."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import skroute

REMOVAL = "is deprecated since 2.0 and will be removed in 3.0; "

# (legacy module, its __all__, {name: 2.0 home}, expected warning text)
SHIMS = [
    (
        "skroute.heuristics.brute",
        ["BruteForce"],
        {"BruteForce": "skroute.exact"},
        "skroute.heuristics.brute " + REMOVAL + "import BruteForce from skroute.exact",
    ),
    (
        "skroute.heuristics.NRBS",
        ["NRBS"],
        {"NRBS": "skroute.construction"},
        "skroute.heuristics.NRBS " + REMOVAL + "import NRBS from skroute.construction",
    ),
    (
        "skroute.metaheuristics.genetics",
        ["Genetic", "EnsembleGenetic"],
        {"Genetic": "skroute.metaheuristics", "EnsembleGenetic": "skroute.ensemble"},
        "skroute.metaheuristics.genetics "
        + REMOVAL
        + "import Genetic from skroute.metaheuristics and EnsembleGenetic from skroute.ensemble",
    ),
    (
        "skroute.metaheuristics.simulated_annealing",
        ["SimulatedAnnealing", "EnsembleSimulatedAnnealing"],
        {"SimulatedAnnealing": "skroute.metaheuristics", "EnsembleSimulatedAnnealing": "skroute.ensemble"},
        "skroute.metaheuristics.simulated_annealing "
        + REMOVAL
        + "import SimulatedAnnealing from skroute.metaheuristics and "
        "EnsembleSimulatedAnnealing from skroute.ensemble",
    ),
    (
        "skroute.metaheuristics.tabu_search",
        ["TabuSearch"],
        {"TabuSearch": "skroute.metaheuristics"},
        "skroute.metaheuristics.tabu_search " + REMOVAL + "import TabuSearch from skroute.metaheuristics",
    ),
    (
        "skroute.metaheuristics.som",
        ["SOM"],
        {"SOM": "skroute.metaheuristics"},
        "skroute.metaheuristics.som " + REMOVAL + "import SOM from skroute.metaheuristics",
    ),
]
SHIM_IDS = [path for path, *_ in SHIMS]
PARENT_MESSAGE = (
    "skroute.heuristics "
    + REMOVAL
    + "import BruteForce from skroute.exact and NRBS from skroute.construction"
)


def _evict(path):
    """Drop ``path`` (and only it) from ``sys.modules`` so that its body runs again on import.

    The parent's attribute is removed through ``vars()``: ``hasattr`` would trigger the lazy ``__getattr__``
    of ``skroute.heuristics`` and re-import the very module being evicted."""
    sys.modules.pop(path, None)
    parent, _, child = path.rpartition(".")
    if parent in sys.modules:
        vars(sys.modules[parent]).pop(child, None)


def _import_with_warning(path, message):
    with warnings.catch_warnings():  # the parent is in place (its own warning is another test's business)
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module(path.rpartition(".")[0])
    _evict(path)
    with pytest.warns(DeprecationWarning) as record:
        module = importlib.import_module(path)
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, [str(w.message) for w in deprecations]
    assert str(deprecations[0].message) == message
    return module


@pytest.mark.parametrize(("path", "names", "homes", "message"), SHIMS, ids=SHIM_IDS)
def test_shim_reexports_the_1_0_names_and_warns_with_the_exact_text(path, names, homes, message):
    module = _import_with_warning(path, message)
    assert module.__all__ == names
    for name in names:
        home = importlib.import_module(homes[name])
        assert getattr(module, name) is getattr(home, name), f"{path}.{name} must BE {homes[name]}.{name}"
        assert getattr(module, name) is getattr(skroute, name)  # and the top-level export


def test_simulated_annealing_shim_exports_the_real_name_not_the_1_0_typo():
    module = _import_with_warning("skroute.metaheuristics.simulated_annealing", SHIMS[3][3])
    assert "SimulatedAnnealing" in module.__all__ and "SimmulatedAnnealing" not in module.__all__


def test_heuristics_package_warns_and_exposes_its_subpackages():
    _evict("skroute.heuristics.brute")
    _evict("skroute.heuristics.NRBS")
    _evict("skroute.heuristics")
    with pytest.warns(DeprecationWarning, match=re.escape(PARENT_MESSAGE)) as record:
        heuristics = importlib.import_module("skroute.heuristics")
    assert len([w for w in record if issubclass(w.category, DeprecationWarning)]) == 1
    assert heuristics.__all__ == ["NRBS", "brute"]
    with pytest.warns(DeprecationWarning, match="skroute.heuristics.brute is deprecated"):
        brute = heuristics.brute  # lazy: the subpackage warns on its own, once
    assert brute.BruteForce is skroute.exact.BruteForce
    with pytest.warns(DeprecationWarning, match="skroute.heuristics.NRBS is deprecated"):
        nrbs = heuristics.NRBS
    assert nrbs.NRBS is skroute.construction.NRBS
    with pytest.raises(AttributeError, match="has no attribute 'cluster'"):
        heuristics.cluster  # noqa: B018


def test_warning_is_attributed_to_the_importer(tmp_path):
    """``stacklevel=2``: the warning points at the line that wrote the legacy import, not at the shim."""
    for path in ("skroute.metaheuristics.tabu_search",):
        _evict(path)
    with pytest.warns(DeprecationWarning) as record:
        import skroute.metaheuristics.tabu_search  # noqa: F401
    (w,) = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert w.filename == __file__


def test_a_second_import_does_not_warn_again(recwarn):
    importlib.import_module("skroute.metaheuristics.som")  # may warn if evicted by an earlier test
    recwarn.clear()
    module = importlib.import_module("skroute.metaheuristics.som")
    assert module.SOM is skroute.metaheuristics.SOM
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_importing_skroute_itself_never_warns():
    """In a fresh interpreter (evicting ``skroute`` from ``sys.modules`` in-process would hand every later
    test new class objects): importing the package and building the roster must not touch a shim."""
    root = Path(skroute.__file__).resolve().parents[1]  # the tree the session imports skroute from
    code = "import skroute, skroute.ensemble; skroute.all_solvers(); print(len(skroute.all_solvers()))"
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == len(skroute.all_solvers())
