"""Generate the solver capability table from the routers' tags.

Two entry points:

* as an ``mkdocs-gen-files`` script (configured in ``mkdocs.yml``) it writes
  ``user_guide/_capability_table.md``, which ``choosing_a_solver.md`` includes;
* ``python docs/gen_pages.py --readme`` rewrites the table in ``README.md`` between the
  ``<!-- capability-table:start -->`` and ``<!-- capability-table:end -->`` markers
  (a pre-commit hook fails if the README is stale).

The table has one row per class returned by :func:`skroute.all_solvers`, so it can
never drift from the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- capability-table:start -->"
END = "<!-- capability-table:end -->"
HEADER = (
    "| Solver | Kind | Exact | Stochastic | Multi-trip aware | Asymmetric | Needs coordinates | Max nodes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def capability_rows() -> list[str]:
    """One markdown table row per solver, in ``all_solvers()`` order."""
    import skroute

    rows = []
    for cls in skroute.all_solvers():
        tags = cls()._get_tags()
        max_nodes = tags.max_nodes if tags.max_nodes is not None else "—"
        cells = [
            f"`{cls.__name__}`",
            tags.kind.replace("_", " "),
            _yes(tags.exact),
            _yes(tags.stochastic),
            _yes(tags.budget_aware),
            _yes(not tags.requires_symmetric),
            _yes(tags.requires_coords),
            str(max_nodes),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def capability_table() -> str:
    return HEADER + "\n".join(capability_rows()) + "\n"


def refresh_readme(path: Path = ROOT / "README.md", *, check: bool = False) -> bool:
    """Rewrite the table between the markers. Returns True when the file was (or would be) changed."""
    text = path.read_text(encoding="utf-8")
    start, end = text.index(START), text.index(END)
    new_text = text[: start + len(START)] + "\n" + capability_table() + text[end:]
    changed = new_text != text
    if changed and not check:
        path.write_text(new_text, encoding="utf-8")
    return changed


def _mkdocs_hook() -> None:
    import mkdocs_gen_files

    with mkdocs_gen_files.open("user_guide/_capability_table.md", "w") as handle:
        handle.write(capability_table())


if __name__ == "__main__":
    if "--readme" in sys.argv:
        stale = refresh_readme(check="--check" in sys.argv)
        if "--check" in sys.argv and stale:
            print("README.md capability table is stale: run python docs/gen_pages.py --readme")
            sys.exit(1)
        print("README.md capability table refreshed" if stale else "README.md capability table up to date")
    else:
        sys.stdout.write(capability_table())
else:  # imported by mkdocs-gen-files
    _mkdocs_hook()
