"""Build script for the Cython extensions only; all metadata lives in pyproject.toml.

Every ``.pyx`` under ``skroute/`` becomes an extension module of the same dotted
name. The compiler directives below apply to all of them (SPEC §3.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent
PACKAGE = ROOT / "skroute"

DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "embedsignature": True,
}

if sys.platform == "win32":
    EXTRA_COMPILE_ARGS = ["/O2"]
else:
    EXTRA_COMPILE_ARGS = ["-O3"]


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


extensions = [
    Extension(
        _module_name(pyx),
        sources=[str(pyx.relative_to(ROOT))],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )
    for pyx in sorted(PACKAGE.rglob("*.pyx"))
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives=DIRECTIVES,
        include_path=[str(ROOT)],
        annotate=False,
    ),
)
