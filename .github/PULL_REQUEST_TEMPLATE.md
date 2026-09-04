<!-- Title: imperative English, one line ("Add Or-3opt move to LocalSearch"). Link the issue below. -->

## What and why

Closes #

## Checklist

- [ ] The PR touches the files of **one** package only (see "Who owns what" in CONTRIBUTING.md). Changes I need in
      shared files (`skroute/__init__.py`, `pyproject.toml`, CI, `CHANGELOG.md`, `mkdocs.yml`) are listed below for the lead.
- [ ] Tests added or updated, and `pytest` is green locally (`pytest -m slow` too if I touched a solver's defaults).
- [ ] New solver only: `check_router(MySolver())` passes, the `tests/tolerances.py` entry exists (or is requested below
      with the measured gaps) and the acceptance tests follow the tolerance table.
- [ ] Docstrings are numpydoc, every public symbol has a `:::` directive on a page under `docs/api/`, and every
      example is deterministic (`pytest --doctest-modules skroute` passes; no floats pinned from one machine).
- [ ] `ruff check .`, `ruff format --check .`, `cython-lint skroute` and `mypy skroute` are clean.
- [ ] No `print()` in library code; `verbose` goes through `logging.getLogger("skroute")`.
- [ ] A CHANGELOG line is requested below (the lead edits `CHANGELOG.md`).

## For the lead

<!-- Exports to add, registry entries, dependency changes, the CHANGELOG line, tolerance entries. Delete if none. -->

- CHANGELOG: `### Added|Changed|Fixed` — "..."
