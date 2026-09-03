"""Build cost matrices for the solvers: distances from coordinates, pivots of long tables, conversions.

Every solver in scikit-route consumes a dense ``(n, n)`` cost matrix. This package
produces such matrices from coordinates (:func:`distance_matrix` and its shorthands
:func:`euclidean_matrix` / :func:`haversine_matrix`), from long tables of
``(origin, destination, value)`` rows (:func:`pairs_to_matrix`) and from the legacy
dict-of-dicts format (:func:`from_dict_of_dicts` / :func:`to_dict_of_dicts`);
:func:`normalize_coords` rescales coordinates into the unit square. Road distances
from the Google Distance Matrix API live in :mod:`skroute.preprocessing.google`
(optional ``googlemaps`` dependency).
"""

from ._convert import from_dict_of_dicts, normalize_coords, pairs_to_matrix, to_dict_of_dicts
from ._distances import distance_matrix, euclidean_matrix, haversine_matrix, tsplib_nint

__all__ = [
    "distance_matrix",
    "euclidean_matrix",
    "from_dict_of_dicts",
    "haversine_matrix",
    "normalize_coords",
    "pairs_to_matrix",
    "to_dict_of_dicts",
    "tsplib_nint",
]
