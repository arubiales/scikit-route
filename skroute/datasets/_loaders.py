"""Loaders of the bundled data sets: 27 Waterloo TSP instances and five road-cost tables."""

from __future__ import annotations

import csv
import inspect
import string
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from skroute.preprocessing import distance_matrix, pairs_to_matrix
from skroute.utils import Bunch

from ._tsplib import read_tsplib

__all__ = [
    "TSPBunch",
    "list_tsp",
    "load_alicante_murcia",
    "load_argentina",
    "load_barcelona",
    "load_burma",
    "load_canada",
    "load_china",
    "load_costs_qatar",
    "load_djibouti",
    "load_egypt",
    "load_finland",
    "load_greece",
    "load_honduras",
    "load_ireland",
    "load_italy",
    "load_japan",
    "load_kazakhstan",
    "load_luxembourg",
    "load_madrid",
    "load_morocco",
    "load_nicaragua",
    "load_oman",
    "load_panama",
    "load_qatar",
    "load_qatar_costs",
    "load_rwanda",
    "load_sahara",
    "load_sweden",
    "load_tanzania",
    "load_tsp",
    "load_uruguay",
    "load_valencia",
    "load_vietnam",
    "load_yemen",
    "load_zimbabwe",
]

_HERE = Path(__file__).resolve().parent
_TSPLIB_DIR = _HERE / "_data" / "tsplib"
_COSTS_DIR = _HERE / "_data" / "costs"
_DESCR_DIR = _HERE / "_descr"

#: Above this many nodes ``TSPBunch.distance_matrix`` refuses to build the dense matrix without ``force``.
_MAX_DENSE_N = 20_000


class _Instance(NamedTuple):
    country: str
    #: Tour length published by Waterloo under ``EUC_2D``: the proven optimum unless ``gap`` is set.
    tour_length: int
    #: For the two instances still open (``bm33708``, ``ch71009``): the gap between the best-known tour
    #: and Waterloo's lower bound. ``None`` when ``tour_length`` is proven optimal.
    gap: str | None = None


#: name -> instance, ordered by size. Values from Waterloo's status table
#: https://www.math.uwaterloo.ca/tsp/world/summary.html (last updated 2022-07-31).
_INSTANCES: dict[str, _Instance] = {
    "wi29": _Instance("Western Sahara", 27603),
    "dj38": _Instance("Djibouti", 6656),
    "qa194": _Instance("Qatar", 9352),
    "uy734": _Instance("Uruguay", 79114),
    "zi929": _Instance("Zimbabwe", 95345),
    "lu980": _Instance("Luxembourg", 11340),
    "rw1621": _Instance("Rwanda", 26051),
    "mu1979": _Instance("Oman", 86891),
    "nu3496": _Instance("Nicaragua", 96132),
    "ca4663": _Instance("Canada", 1290319),
    "tz6117": _Instance("Tanzania", 394718),
    "eg7146": _Instance("Egypt", 172386),
    "ym7663": _Instance("Yemen", 238314),
    "pm8079": _Instance("Panama", 114855),
    "ei8246": _Instance("Ireland", 206171),
    "ar9152": _Instance("Argentina", 837479),
    "ja9847": _Instance("Japan", 491924),
    "gr9882": _Instance("Greece", 300899),
    "kz9976": _Instance("Kazakhstan", 1061881),
    "fi10639": _Instance("Finland", 520527),
    "mo14185": _Instance("Morocco", 427377),
    "ho14473": _Instance("Honduras", 177092),
    "it16862": _Instance("Italy", 557315),
    "vm22775": _Instance("Vietnam", 569288),
    "sw24978": _Instance("Sweden", 855597),
    "bm33708": _Instance("Burma", 959289, gap="0.031 %"),
    "ch71009": _Instance("China", 4566506, gap="0.024 %"),
}

_EWT_TO_METRIC = {
    "EUC_2D": "tsplib_euc_2d",
    "CEIL_2D": "tsplib_ceil_2d",
    "MAN_2D": "tsplib_man_2d",
    "ATT": "tsplib_att",
    "GEO": "tsplib_geo",
}

_LARGE_NOTE = """
## This instance cannot be solved whole in scikit-route 2.0

scikit-route 2.0 evaluates every solution on a **dense** cost matrix, and the
`(n, n)` `float64` matrix of $name would take about $gb GB. `distance_matrix()`
therefore refuses to build it (pass `force=True` if you really have the memory);
work on a subsample instead, whose optimum is unknown (`optimal_tour_length is
None`):

    b = load_tsp("$name", n_nodes=5000)
"""


def _read_descr(family: str, **fields: Any) -> str:
    template = string.Template((_DESCR_DIR / f"{family}.md").read_text(encoding="utf-8"))
    return template.safe_substitute(**{k: str(v) for k, v in fields.items()})


def _warn_deprecated(message: str) -> None:
    """Emit a ``DeprecationWarning`` attributed to the first frame outside this module."""
    level = 2
    frame = inspect.currentframe()
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
        level += 1
    warnings.warn(message, DeprecationWarning, stacklevel=level - 1)


class TSPBunch(Bunch):
    """A :class:`~skroute.utils.Bunch` of a coordinate instance with a cached ``distance_matrix()``.

    Behaves like a ``dict`` with attribute access (``b.coords`` or ``b["coords"]``);
    ``keys()`` lists only the data fields. The distance matrix is a *method*, so it
    is never built by the loader and is cached under the private attribute
    ``_distance_matrix`` (not a key).

    Examples
    --------
    >>> from skroute.datasets import load_tsp
    >>> b = load_tsp("wi29")
    >>> sorted(b)
    ['DESCR', 'coords', 'depot', 'edge_weight_type', 'labels', 'name', 'optimal_tour_length']
    >>> C = b.distance_matrix()
    >>> C.shape, C is b.distance_matrix()  # (n, n) and cached
    ((29, 29), True)
    """

    _distance_matrix: NDArray[np.float64] | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_distance_matrix", None)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(sorted(self))})"

    def distance_matrix(self, *, force: bool = False) -> NDArray[np.float64]:
        """The dense ``(n, n)`` matrix under the instance's TSPLIB metric, computed once and cached.

        Parameters
        ----------
        force : bool, default=False
            Build the matrix even above 20 000 nodes (a ``float64`` matrix of
            ``ch71009`` needs about 40 GB). Without it such instances raise
            ``ValueError`` and point to ``load_tsp(name, n_nodes=...)``.

        Returns
        -------
        ndarray of shape (n, n), dtype float64
            ``skroute.preprocessing.distance_matrix(coords, metric="tsplib_euc_2d")``
            for ``EUC_2D`` instances (the other TSPLIB coordinate types map to their
            ``tsplib_*`` metric). The same array object is returned on every call:
            copy it before modifying it.

        Raises
        ------
        ValueError
            Above 20 000 nodes without ``force``, or when ``edge_weight_type`` is not
            one of ``EUC_2D``, ``CEIL_2D``, ``MAN_2D``, ``ATT``, ``GEO`` (never a
            silent fall-back to the planar metric).
        """
        cached = self._distance_matrix
        if cached is not None:
            return cached
        n = int(np.asarray(self["coords"]).shape[0])
        if n > _MAX_DENSE_N and not force:
            gb = n * n * 8 / 1e9  # decimal GB, as everywhere in scikit-route (ch71009: 40 GB)
            raise ValueError(
                f"{self['name']} has {n} nodes: its dense float64 distance matrix needs {gb:.1f} GB. "
                f"Subsample with load_tsp({self['name']!r}, n_nodes=...) or pass force=True."
            )
        ewt = str(self["edge_weight_type"]).upper()
        if ewt not in _EWT_TO_METRIC:
            raise ValueError(
                f"edge_weight_type {self['edge_weight_type']!r} has no tsplib_* metric; "
                f"TSPBunch.distance_matrix supports {list(_EWT_TO_METRIC)}"
            )
        matrix = distance_matrix(self["coords"], metric=_EWT_TO_METRIC[ewt])
        self._distance_matrix = matrix
        return matrix


def list_tsp() -> list[str]:
    """Names of the 27 bundled Waterloo instances, from the smallest (``wi29``) to the largest (``ch71009``).

    Returns
    -------
    list of str

    Examples
    --------
    >>> from skroute.datasets import list_tsp
    >>> names = list_tsp()
    >>> len(names), names[:3]
    (27, ['wi29', 'dj38', 'qa194'])
    """
    return list(_INSTANCES)


def load_tsp(
    name: str,
    *,
    n_nodes: int | None = None,
    random_state: int | None = 2019,
    mode: str | None = None,
) -> TSPBunch:
    """Load one of the 27 Waterloo national TSP instances bundled with scikit-route.

    Parameters
    ----------
    name : str
        Instance name, e.g. ``"wi29"``; see :func:`list_tsp`.
    n_nodes : int, optional
        Keep only ``n_nodes`` of the cities, sampled without replacement with
        ``numpy.random.default_rng(random_state)``; the first city of the file is
        always kept and the file order is preserved. ``optimal_tour_length`` is
        then ``None``.
    random_state : int or None, default=2019
        Seed of the subsampling (only used with ``n_nodes``).
    mode : {"small", "medium", "big"}, optional
        Deprecated 1.0 spelling of ``n_nodes``: ``"small"`` keeps
        ``max(10, round(0.005 n))`` cities, ``"medium"`` ``round(0.2 n)``, ``"big"``
        all of them. Emits ``DeprecationWarning``; removed in 3.0.

    Returns
    -------
    TSPBunch
        With ``name``, ``coords`` (``float64 (n, 2)``, ``x``/``y`` as in the file),
        ``labels`` (``int64``, the file's 1-based ids), ``depot`` (the first label,
        ``1``), ``edge_weight_type`` (``"EUC_2D"`` for all 27), ``optimal_tour_length``
        (the tour length published by Waterloo: the proven optimum, except for
        ``bm33708`` and ``ch71009`` where it is the best-known tour; ``None`` when
        subsampled), ``DESCR``, and the cached method ``distance_matrix(*, force=False)``.
        The matrix is never built by the loader.

    Notes
    -----
    The four instances above 20 000 nodes (``vm22775``, ``sw24978``, ``bm33708``,
    ``ch71009``) cannot be solved whole in 2.0 (dense matrices only); their
    ``DESCR`` says so and shows ``load_tsp(name, n_nodes=5000)``. ``bm33708`` and
    ``ch71009`` are still open problems: their ``optimal_tour_length`` is the
    best-known tour, within 0.031 % and 0.024 % of Waterloo's lower bound, and the
    ``DESCR`` says so too.

    References
    ----------
    W. Cook et al., *National Traveling Salesman Problems*, University of Waterloo,
    https://www.math.uwaterloo.ca/tsp/world/countries.html; status table at
    https://www.math.uwaterloo.ca/tsp/world/summary.html

    Examples
    --------
    >>> from skroute.datasets import load_tsp
    >>> wi = load_tsp("wi29")
    >>> wi.coords.shape, wi.labels[:3].tolist(), wi.depot, wi.optimal_tour_length
    ((29, 2), [1, 2, 3], 1, 27603)
    >>> small = load_tsp("qa194", n_nodes=20, random_state=0)
    >>> small.coords.shape, int(small.labels[0]), small.optimal_tour_length
    ((20, 2), 1, None)
    """
    if name not in _INSTANCES:
        raise ValueError(f"unknown instance {name!r}; available: {', '.join(_INSTANCES)}")
    instance = _INSTANCES[name]
    country, optimum = instance.country, instance.tour_length
    raw = read_tsplib(_TSPLIB_DIR / f"{name}.tsp")
    coords: NDArray[np.float64] = raw.coords
    labels: NDArray[np.int64] = raw.labels
    n = coords.shape[0]

    if mode is not None:
        _warn_deprecated("mode= is deprecated since 2.0 and will be removed in 3.0; use n_nodes= instead")
        if n_nodes is not None:
            raise ValueError("pass either n_nodes= or the deprecated mode=, not both")
        if mode == "small":
            n_nodes = max(10, round(0.005 * n))
        elif mode == "medium":
            n_nodes = round(0.2 * n)
        elif mode == "big":
            n_nodes = None
        else:
            raise ValueError(f"mode must be 'small', 'medium' or 'big'; got {mode!r}")

    subsampled = False
    if n_nodes is not None:
        if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)):
            raise ValueError(f"n_nodes must be an integer; got {n_nodes!r}")
        if not 1 <= n_nodes <= n:
            raise ValueError(f"n_nodes must be in [1, {n}] for {name}; got {n_nodes}")
        if n_nodes < n:
            rng = np.random.default_rng(random_state)
            chosen = rng.choice(np.arange(1, n), size=int(n_nodes) - 1, replace=False)
            keep = np.concatenate(([0], np.sort(chosen)))
            coords = np.ascontiguousarray(coords[keep])
            labels = labels[keep]
            subsampled = True

    if n > _MAX_DENSE_N:
        note = string.Template(_LARGE_NOTE).safe_substitute(name=name, gb=f"{n * n * 8 / 1e9:.0f}")
    else:
        note = ""
    if instance.gap is None:
        optimality = "the proven optimal tour length published by the University of Waterloo"
    else:
        optimality = (
            "the length of the best-known tour published by the University of Waterloo, within "
            f"{instance.gap} of its lower bound and **not proven optimal**"
        )
    descr = _read_descr(
        "tsplib", name=name, country=country, n=n, optimum=optimum, optimality=optimality, note=note
    )
    if subsampled:
        descr += (
            f"\n## Subsample\n\nThis Bunch holds {coords.shape[0]} of the {n} cities (`n_nodes=`); "
            "its optimum is unknown.\n"
        )

    return TSPBunch(
        name=name,
        coords=coords,
        labels=labels,
        depot=int(labels[0]),
        edge_weight_type=raw.edge_weight_type,
        optimal_tour_length=None if subsampled else optimum,
        DESCR=descr,
    )


# --------------------------------------------------------------------------- country wrappers
# One-line wrappers of load_tsp with the exact 1.0 names; **kwargs = n_nodes, random_state. ``mode`` is
# spelled out so that it may also be positional, as in 1.0 (``load_qatar("small")``): that call then
# reaches load_tsp's DeprecationWarning instead of dying with an opaque TypeError.


def load_sahara(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Western Sahara, ``wi29`` (29 cities, optimum 27603); ``load_tsp("wi29", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_sahara
    >>> b = load_sahara()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('wi29', 29, 27603)
    """
    return load_tsp("wi29", mode=mode, **kwargs)


def load_djibouti(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Djibouti, ``dj38`` (38 cities, optimum 6656); ``load_tsp("dj38", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_djibouti
    >>> b = load_djibouti()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('dj38', 38, 6656)
    """
    return load_tsp("dj38", mode=mode, **kwargs)


def load_qatar(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Qatar, ``qa194`` (194 cities, optimum 9352); ``load_tsp("qa194", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_qatar
    >>> b = load_qatar()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('qa194', 194, 9352)
    """
    return load_tsp("qa194", mode=mode, **kwargs)


def load_uruguay(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Uruguay, ``uy734`` (734 cities, optimum 79114); ``load_tsp("uy734", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_uruguay
    >>> b = load_uruguay()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('uy734', 734, 79114)
    """
    return load_tsp("uy734", mode=mode, **kwargs)


def load_zimbabwe(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Zimbabwe, ``zi929`` (929 cities, optimum 95345); ``load_tsp("zi929", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_zimbabwe
    >>> b = load_zimbabwe()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('zi929', 929, 95345)
    """
    return load_tsp("zi929", mode=mode, **kwargs)


def load_luxembourg(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Luxembourg, ``lu980`` (980 cities, optimum 11340); ``load_tsp("lu980", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_luxembourg
    >>> b = load_luxembourg()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('lu980', 980, 11340)
    """
    return load_tsp("lu980", mode=mode, **kwargs)


def load_rwanda(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Rwanda, ``rw1621`` (1621 cities, optimum 26051); ``load_tsp("rw1621", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_rwanda
    >>> b = load_rwanda()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('rw1621', 1621, 26051)
    """
    return load_tsp("rw1621", mode=mode, **kwargs)


def load_oman(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Oman, ``mu1979`` (1979 cities, optimum 86891); ``load_tsp("mu1979", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_oman
    >>> b = load_oman()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('mu1979', 1979, 86891)
    """
    return load_tsp("mu1979", mode=mode, **kwargs)


def load_nicaragua(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Nicaragua, ``nu3496`` (3496 cities, optimum 96132); ``load_tsp("nu3496", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_nicaragua
    >>> b = load_nicaragua()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('nu3496', 3496, 96132)
    """
    return load_tsp("nu3496", mode=mode, **kwargs)


def load_canada(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Canada, ``ca4663`` (4663 cities, optimum 1290319); ``load_tsp("ca4663", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_canada
    >>> b = load_canada()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ca4663', 4663, 1290319)
    """
    return load_tsp("ca4663", mode=mode, **kwargs)


def load_tanzania(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Tanzania, ``tz6117`` (6117 cities, optimum 394718); ``load_tsp("tz6117", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_tanzania
    >>> b = load_tanzania()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('tz6117', 6117, 394718)
    """
    return load_tsp("tz6117", mode=mode, **kwargs)


def load_egypt(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Egypt, ``eg7146`` (7146 cities, optimum 172386); ``load_tsp("eg7146", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_egypt
    >>> b = load_egypt()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('eg7146', 7146, 172386)
    """
    return load_tsp("eg7146", mode=mode, **kwargs)


def load_yemen(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Yemen, ``ym7663`` (7663 cities, optimum 238314); ``load_tsp("ym7663", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_yemen
    >>> b = load_yemen()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ym7663', 7663, 238314)
    """
    return load_tsp("ym7663", mode=mode, **kwargs)


def load_panama(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Panama, ``pm8079`` (8079 cities, optimum 114855); ``load_tsp("pm8079", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_panama
    >>> b = load_panama()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('pm8079', 8079, 114855)
    """
    return load_tsp("pm8079", mode=mode, **kwargs)


def load_ireland(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Ireland, ``ei8246`` (8246 cities, optimum 206171); ``load_tsp("ei8246", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_ireland
    >>> b = load_ireland()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ei8246', 8246, 206171)
    """
    return load_tsp("ei8246", mode=mode, **kwargs)


def load_argentina(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Argentina, ``ar9152`` (9152 cities, optimum 837479); ``load_tsp("ar9152", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_argentina
    >>> b = load_argentina()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ar9152', 9152, 837479)
    """
    return load_tsp("ar9152", mode=mode, **kwargs)


def load_japan(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Japan, ``ja9847`` (9847 cities, optimum 491924); ``load_tsp("ja9847", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_japan
    >>> b = load_japan()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ja9847', 9847, 491924)
    """
    return load_tsp("ja9847", mode=mode, **kwargs)


def load_greece(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Greece, ``gr9882`` (9882 cities, optimum 300899); ``load_tsp("gr9882", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_greece
    >>> b = load_greece()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('gr9882', 9882, 300899)
    """
    return load_tsp("gr9882", mode=mode, **kwargs)


def load_kazakhstan(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Kazakhstan, ``kz9976`` (9976 cities, optimum 1061881); ``load_tsp("kz9976", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_kazakhstan
    >>> b = load_kazakhstan()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('kz9976', 9976, 1061881)
    """
    return load_tsp("kz9976", mode=mode, **kwargs)


def load_finland(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Finland, ``fi10639`` (10639 cities, optimum 520527); ``load_tsp("fi10639", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_finland
    >>> b = load_finland()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('fi10639', 10639, 520527)
    """
    return load_tsp("fi10639", mode=mode, **kwargs)


def load_morocco(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Morocco, ``mo14185`` (14185 cities, optimum 427377); ``load_tsp("mo14185", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_morocco
    >>> b = load_morocco()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('mo14185', 14185, 427377)
    """
    return load_tsp("mo14185", mode=mode, **kwargs)


def load_honduras(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Honduras, ``ho14473`` (14473 cities, optimum 177092); ``load_tsp("ho14473", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_honduras
    >>> b = load_honduras()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ho14473', 14473, 177092)
    """
    return load_tsp("ho14473", mode=mode, **kwargs)


def load_italy(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Italy, ``it16862`` (16862 cities, optimum 557315); ``load_tsp("it16862", **kwargs)``.

    Examples
    --------
    >>> from skroute.datasets import load_italy
    >>> b = load_italy()
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('it16862', 16862, 557315)
    """
    return load_tsp("it16862", mode=mode, **kwargs)


def load_vietnam(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Vietnam, ``vm22775`` (22775 cities, optimum 569288); ``load_tsp("vm22775", **kwargs)``.

    Above the 20 000-node dense-matrix ceiling of 2.0: subsample with ``n_nodes=``.

    Examples
    --------
    >>> from skroute.datasets import load_vietnam
    >>> b = load_vietnam(n_nodes=5000)
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('vm22775', 5000, None)
    """
    return load_tsp("vm22775", mode=mode, **kwargs)


def load_sweden(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Sweden, ``sw24978`` (24978 cities, optimum 855597); ``load_tsp("sw24978", **kwargs)``.

    Above the 20 000-node dense-matrix ceiling of 2.0: subsample with ``n_nodes=``.

    Examples
    --------
    >>> from skroute.datasets import load_sweden
    >>> b = load_sweden(n_nodes=5000)
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('sw24978', 5000, None)
    """
    return load_tsp("sw24978", mode=mode, **kwargs)


def load_burma(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """Burma, ``bm33708`` (33708 cities, best-known tour 959289); ``load_tsp("bm33708", **kwargs)``.

    Above the 20 000-node dense-matrix ceiling of 2.0: subsample with ``n_nodes=``. Still an
    open problem: ``optimal_tour_length`` is the best-known tour (within 0.031 % of the lower
    bound), not a proven optimum.

    Examples
    --------
    >>> from skroute.datasets import load_burma
    >>> b = load_burma(n_nodes=5000)
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('bm33708', 5000, None)
    """
    return load_tsp("bm33708", mode=mode, **kwargs)


def load_china(mode: str | None = None, **kwargs: Any) -> TSPBunch:
    """China, ``ch71009`` (71009 cities, best-known tour 4566506); ``load_tsp("ch71009", **kwargs)``.

    Above the 20 000-node dense-matrix ceiling of 2.0: subsample with ``n_nodes=``. Still an
    open problem: ``optimal_tour_length`` is the best-known tour (within 0.024 % of the lower
    bound), not a proven optimum.

    Examples
    --------
    >>> from skroute.datasets import load_china
    >>> b = load_china(n_nodes=5000)
    >>> b.name, len(b.labels), b.optimal_tour_length
    ('ch71009', 5000, None)
    """
    return load_tsp("ch71009", mode=mode, **kwargs)


# --------------------------------------------------------------------------- cost data sets

_SPANISH_SCHEMA: dict[str, type] = {
    "id_origin": int,
    "id_destinity": int,
    "lat_origin": float,
    "lon_origin": float,
    "lat_destiniy": float,  # sic: spelled like this in the original files
    "lon_destinity": float,
    "cluster": int,
    "origin": str,
    "destinity": str,
    "meters": float,
    "secs": float,
    "hours": float,
    "kilometers": float,
    "cost": float,
}

_QATAR_SCHEMA: dict[str, type] = {
    "id_origin": int,
    "lat_origin": float,
    "lon_origin": float,
    "address_origin": str,
    "id_destinity": int,
    "lat_destinity": float,
    "lon_destinity": float,
    "address_destinity": str,
    "meters": float,
    "seconds": float,
}

_PANDAS_MESSAGE = "pandas is required for as_frame=True: pip install scikit-route[pandas]"


def _read_csv(path: Path, schema: dict[str, type]) -> dict[str, list[Any]]:
    """Read a long table with the stdlib ``csv`` module into typed columns."""
    columns: dict[str, list[Any]] = {name: [] for name in schema}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for name, kind in schema.items():
                raw = row[name]
                columns[name].append(kind(float(raw)) if kind is int else kind(raw))
    return columns


def _coords_by_label(
    labels: NDArray[Any],
    origin: list[int],
    destination: list[int],
    lat_o: list[float],
    lon_o: list[float],
    lat_d: list[float],
    lon_d: list[float],
) -> NDArray[np.float64]:
    coords: dict[int, tuple[float, float]] = {}
    for o, d, lo, lno, ld, lnd in zip(origin, destination, lat_o, lon_o, lat_d, lon_d, strict=True):
        coords.setdefault(o, (lo, lno))
        coords.setdefault(d, (ld, lnd))
    return np.asarray([coords[int(lab)] for lab in labels], dtype=np.float64)


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_PANDAS_MESSAGE) from exc
    return pd


def _cost_bunch(
    *,
    columns: dict[str, list[Any]],
    cost: list[float],
    time: list[float],
    distance: list[float],
    lat_d: str,
    units: dict[str, str],
    descr: str,
    as_frame: bool,
) -> Bunch:
    origin, destination = columns["id_origin"], columns["id_destinity"]
    cost_m, labels = pairs_to_matrix(origin, destination, cost, symmetric=True)
    time_m, _ = pairs_to_matrix(origin, destination, time, symmetric=True, labels=labels)
    dist_m, _ = pairs_to_matrix(origin, destination, distance, symmetric=True, labels=labels)
    labels = labels.astype(np.int64)
    coords = _coords_by_label(
        labels,
        origin,
        destination,
        columns["lat_origin"],
        columns["lon_origin"],
        columns[lat_d],
        columns["lon_destinity"],
    )
    frame = None
    cost_out: Any = cost_m
    time_out: Any = time_m
    dist_out: Any = dist_m
    if as_frame:
        pd = _import_pandas()
        idx = pd.Index(labels, name="id")
        cost_out = pd.DataFrame(cost_m, index=idx, columns=idx)
        time_out = pd.DataFrame(time_m, index=idx, columns=idx)
        dist_out = pd.DataFrame(dist_m, index=idx, columns=idx)
        frame = pd.DataFrame(columns)
    return Bunch(
        cost=cost_out,
        time=time_out,
        distance=dist_out,
        coords=coords,
        labels=labels,
        depot=int(labels[0]),
        units=units,
        DESCR=descr,
        frame=frame,
    )


def _load_spanish(file: str, city: str, as_frame: bool) -> Bunch:
    columns = _read_csv(_COSTS_DIR / f"{file}.csv", _SPANISH_SCHEMA)
    n = len(dict.fromkeys(columns["id_origin"] + columns["id_destinity"]))
    descr = _read_descr("spanish_costs", city=city, n=n, depot=columns["id_origin"][0], file=f"{file}.csv")
    return _cost_bunch(
        columns=columns,
        cost=columns["cost"],
        time=columns["hours"],
        distance=columns["meters"],
        lat_d="lat_destiniy",
        units={"cost": "EUR", "time": "h", "distance": "m"},
        descr=descr,
        as_frame=as_frame,
    )


def load_alicante_murcia(*, as_frame: bool = False) -> Bunch:
    """Road costs between 8 places of Alicante and Murcia (Spain); depot ``10000002``.

    Parameters
    ----------
    as_frame : bool, default=False
        Return ``cost``, ``time`` and ``distance`` as labelled ``pandas.DataFrame``
        objects and the long table under ``frame``. Requires pandas
        (``pip install scikit-route[pandas]``).

    Returns
    -------
    Bunch
        ``cost`` (EUR), ``time`` (hours), ``distance`` (metres) as symmetric
        ``float64 (8, 8)`` matrices with a zero diagonal; ``coords`` as
        ``(latitude, longitude)``; ``labels`` (``int64`` ids in first-appearance
        order); ``depot`` (``10000002``, the first id); ``units``; ``DESCR``;
        ``frame`` (``None`` unless ``as_frame=True``).

    Notes
    -----
    The matrices carry no labels: pass ``labels=b.labels, depot=b.depot`` to ``fit``.
    Parsed with the stdlib ``csv`` module (the addresses contain commas).

    Examples
    --------
    >>> from skroute.datasets import load_alicante_murcia
    >>> b = load_alicante_murcia()
    >>> b.cost.shape, b.depot, b.units
    ((8, 8), 10000002, {'cost': 'EUR', 'time': 'h', 'distance': 'm'})
    >>> bool((b.cost == b.cost.T).all()) and float(b.cost.diagonal().max()) == 0.0
    True
    """
    return _load_spanish("alicante_murcia", "Alicante and Murcia", as_frame)


def load_barcelona(*, as_frame: bool = False) -> Bunch:
    """Road costs between 19 places of Barcelona (Spain); depot ``10000007``.

    Parameters
    ----------
    as_frame : bool, default=False
        Return ``cost``, ``time`` and ``distance`` as labelled ``pandas.DataFrame``
        objects and the long table under ``frame``. Requires pandas.

    Returns
    -------
    Bunch
        ``cost`` (EUR), ``time`` (hours), ``distance`` (metres) as symmetric
        ``float64 (19, 19)`` matrices; ``coords`` ``(latitude, longitude)``;
        ``labels``; ``depot == 10000007``; ``units``; ``DESCR``; ``frame``.

    Examples
    --------
    >>> from skroute.datasets import load_barcelona
    >>> bcn = load_barcelona()
    >>> bcn.time.shape, bcn.depot, bcn.units["time"]
    ((19, 19), 10000007, 'h')
    >>> int(bcn.labels[0]) == bcn.depot
    True
    """
    return _load_spanish("barcelona", "Barcelona", as_frame)


def load_madrid(*, as_frame: bool = False) -> Bunch:
    """Road costs between 18 places of Madrid (Spain); depot ``10000016``.

    Parameters
    ----------
    as_frame : bool, default=False
        Return labelled ``pandas.DataFrame`` matrices and the long table. Requires pandas.

    Returns
    -------
    Bunch
        ``cost`` (EUR), ``time`` (hours), ``distance`` (metres) as ``float64 (18, 18)``
        matrices; ``coords``; ``labels``; ``depot == 10000016``; ``units``; ``DESCR``; ``frame``.

    Examples
    --------
    >>> from skroute.datasets import load_madrid
    >>> b = load_madrid()
    >>> b.cost.shape, b.depot
    ((18, 18), 10000016)
    """
    return _load_spanish("madrid", "Madrid", as_frame)


def load_valencia(*, as_frame: bool = False) -> Bunch:
    """Road costs between 14 places of Valencia (Spain); depot ``10000022``.

    Parameters
    ----------
    as_frame : bool, default=False
        Return labelled ``pandas.DataFrame`` matrices and the long table. Requires pandas.

    Returns
    -------
    Bunch
        ``cost`` (EUR), ``time`` (hours), ``distance`` (metres) as ``float64 (14, 14)``
        matrices; ``coords``; ``labels``; ``depot == 10000022``; ``units``; ``DESCR``; ``frame``.

    Examples
    --------
    >>> from skroute.datasets import load_valencia
    >>> b = load_valencia()
    >>> b.cost.shape, b.depot
    ((14, 14), 10000022)
    """
    return _load_spanish("valencia", "Valencia", as_frame)


def load_qatar_costs(*, as_frame: bool = False) -> Bunch:
    """Road distances and driving times between 192 places of Qatar; depot ``1``.

    Parameters
    ----------
    as_frame : bool, default=False
        Return labelled ``pandas.DataFrame`` matrices and the long table. Requires pandas.

    Returns
    -------
    Bunch
        ``cost`` in **kilometres** (the table has no money column), ``time`` in hours
        (``seconds / 3600``), ``distance`` in metres, as symmetric ``float64 (192, 192)``
        matrices; ``coords`` ``(latitude, longitude)``; ``labels`` (ids ``1..193`` with one
        absent, first-appearance order); ``depot == 1``;
        ``units == {"cost": "km", "time": "h", "distance": "m"}``; ``DESCR``; ``frame``.

    Notes
    -----
    In 1.0 ``load_costs_qatar()`` returned the Valencia table; that name survives as a
    deprecated alias of this function.

    The source table records the pair ``(104, 111)`` as 0 m / 0 s although the two
    places are about 4 km apart, so ``cost``, ``time`` and ``distance`` each hold one
    zero off the diagonal (the Spanish tables have none). It is kept as in the source;
    fill it (e.g. with the haversine distance) if your use needs positive legs.

    Examples
    --------
    >>> from skroute.datasets import load_qatar_costs
    >>> q = load_qatar_costs()
    >>> q.cost.shape, q.depot, q.units["cost"]
    ((192, 192), 1, 'km')
    >>> bool((q.cost == q.distance / 1000.0).all())  # cost is the road distance in km
    True
    """
    columns = _read_csv(_COSTS_DIR / "qatar_costs.csv", _QATAR_SCHEMA)
    metres = columns["meters"]
    return _cost_bunch(
        columns=columns,
        cost=[m / 1000.0 for m in metres],
        time=[s / 3600.0 for s in columns["seconds"]],
        distance=metres,
        lat_d="lat_destinity",
        units={"cost": "km", "time": "h", "distance": "m"},
        descr=_read_descr("qatar_costs"),
        as_frame=as_frame,
    )


def load_costs_qatar(*, as_frame: bool = False) -> Bunch:
    """Deprecated alias of :func:`load_qatar_costs` (1.0 loaded Valencia under this name).

    Emits ``DeprecationWarning``; removed in 3.0.

    Examples
    --------
    >>> from skroute.datasets import load_costs_qatar
    >>> load_costs_qatar().cost.shape  # doctest: +SKIP
    (192, 192)
    """
    _warn_deprecated(
        "load_costs_qatar is deprecated since 2.0 and will be removed in 3.0; use load_qatar_costs "
        "(note: 1.0's load_costs_qatar returned the Valencia table)"
    )
    return load_qatar_costs(as_frame=as_frame)
