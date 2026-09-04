"""Fail when a public name of ``skroute`` has no ``:::`` directive under ``docs/api/``.

Every name listed in an ``__all__`` of the package (top level and sub-packages) must be
rendered by mkdocstrings somewhere in ``docs/api/*.md``, either directly
(``::: skroute.exact.BruteForce``) or through its module/package page
(``::: skroute.exact``). Run in ``docs.yml`` before ``mkdocs build --strict``.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if importlib.util.find_spec("skroute") is None:  # development checkout without an installed package (D29)
    sys.path.insert(0, str(ROOT))
DIRECTIVE = re.compile(r"^:::\s+([\w.]+)", re.MULTILINE)
SKIP_PACKAGES = {
    "skroute._core",
    "skroute.heuristics",
    "skroute.metaheuristics.genetics",
    "skroute.metaheuristics.simulated_annealing",
    "skroute.metaheuristics.tabu_search",
    "skroute.metaheuristics.som",
}


def documented_targets() -> set[str]:
    targets: set[str] = set()
    for page in (ROOT / "docs" / "api").glob("*.md"):
        targets.update(DIRECTIVE.findall(page.read_text(encoding="utf-8")))
    return targets


def public_names() -> dict[str, list[str]]:
    import skroute

    names = {"skroute": list(getattr(skroute, "__all__", []))}
    for info in pkgutil.walk_packages(skroute.__path__, prefix="skroute."):
        # Public surface = the top-level names plus the __all__ of every sub-PACKAGE (skroute.exact,
        # skroute.datasets, ...); plain modules (skroute.base, skroute.utils.validation) are reached
        # through the names their package re-exports.
        if not info.ispkg or info.name.split(".")[-1].startswith("_"):
            continue
        if any(info.name.startswith(s) for s in SKIP_PACKAGES):
            continue
        module = importlib.import_module(info.name)
        exported = list(getattr(module, "__all__", []))
        if exported:
            names[info.name] = exported
    return names


def missing() -> list[str]:
    targets = documented_targets()
    gaps = []
    for module, exported in public_names().items():
        for name in exported:
            candidates = {f"{module}.{name}", module}
            if module == "skroute":
                if name.startswith("__"):
                    continue  # dunders such as __version__ are not API pages
                skroute = importlib.import_module("skroute")
                registered = getattr(skroute, "_EXPORTS", {}).get(name)
                if registered:  # the public home of a re-exported name, e.g. skroute.exact.BruteForce
                    candidates.add(f"{registered}.{name}")
                    candidates.add(registered)
                obj = getattr(skroute, name, None)
                if obj is not None and hasattr(obj, "__module__") and isinstance(obj.__module__, str):
                    candidates.add(f"{obj.__module__}.{name}")
                    candidates.add(obj.__module__)
            if not candidates & targets:
                gaps.append(f"{module}.{name}")
    return gaps


if __name__ == "__main__":
    gaps = missing()
    if gaps:
        print("Public names without a docs/api directive:")
        for gap in gaps:
            print("  ", gap)
        sys.exit(1)
    print("API coverage OK: every public name has a documentation page")
