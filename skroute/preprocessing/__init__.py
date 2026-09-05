"""Build cost matrices for the solvers: distances from coordinates, pivots of long tables, conversions.

Every solver in scikit-route consumes a dense ``(n, n)`` cost matrix. This package
produces such matrices from coordinates (`distance_matrix` and its shorthands
`euclidean_matrix` / `haversine_matrix`), from long tables of
``(origin, destination, value)`` rows (`pairs_to_matrix`) and from the legacy
dict-of-dicts format (`from_dict_of_dicts` / `to_dict_of_dicts`);
`normalize_coords` rescales coordinates into the unit square. Real-world inputs
come from map services: `travel_time_matrix` (road travel times and distances from
OSRM or Google), `geocode` (an address to coordinates, Nominatim or Google) and
`fetch_pois` (OpenStreetMap points of interest through Overpass) live in
`skroute.preprocessing.maps`; the Google Distance Matrix client itself is
`skroute.preprocessing.google.GoogleDistanceMatrix` (optional ``googlemaps``
dependency).
"""

from ._convert import from_dict_of_dicts, normalize_coords, pairs_to_matrix, to_dict_of_dicts
from ._distances import distance_matrix, euclidean_matrix, haversine_matrix, tsplib_nint
from .maps import fetch_pois, geocode, travel_time_matrix

__all__ = [
    "distance_matrix",
    "euclidean_matrix",
    "fetch_pois",
    "from_dict_of_dicts",
    "geocode",
    "haversine_matrix",
    "normalize_coords",
    "pairs_to_matrix",
    "to_dict_of_dicts",
    "travel_time_matrix",
    "tsplib_nint",
]
