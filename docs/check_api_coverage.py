"""Fail when a public name of ``skroute`` has no ``:::`` directive under ``docs/api/``.

Every name listed in an ``__all__`` of the package (top level and sub-packages) must be
rendered by mkdocstrings somewhere in ``docs/api/*.md``, either directly
(``::: skroute.exact.BruteForce``) or through its module/package page
(``::: skroute.exact``). Run in ``docs.yml`` before ``mkdocs build --strict``.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
        if info.name.split(".")[-1].startswith("_") or any(info.name.startswith(s) for s in SKIP_PACKAGES):
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
                obj = getattr(importlib.import_module("skroute"), name, None)
                if obj is not None and hasattr(obj, "__module__"):
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
