# Preprocessing

Every solver in scikit-route consumes a dense `(n, n)` cost matrix.
`skroute.preprocessing` builds such matrices from what you usually have:

* **coordinates** — [`distance_matrix`][skroute.preprocessing.distance_matrix]
  with a planar, TSPLIB 95 or great-circle metric (and the shorthands
  `euclidean_matrix` / `haversine_matrix`);
* **a long table** of `(origin, destination, value)` rows —
  [`pairs_to_matrix`][skroute.preprocessing.pairs_to_matrix], the order-independent
  replacement of 1.0's `dfcolumn_to_dict`;
* **a dict of dicts** `{i: {j: value}}` — `from_dict_of_dicts` and
  `to_dict_of_dicts`;
* **the road network** — `skroute.preprocessing.google.GoogleDistanceMatrix`,
  which batches Google Distance Matrix API requests (optional `googlemaps`
  extra, billed to your account).

`normalize_coords` rescales planar coordinates into the unit square while
preserving their aspect ratio (what the SOM feeds its ring with).

```python
>>> from skroute.preprocessing import distance_matrix, pairs_to_matrix
>>> xy = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
>>> distance_matrix(xy).tolist()
[[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]]
>>> M, labels = pairs_to_matrix([1, 1, 2], [2, 3, 3], [5.0, 9.0, 4.0])
>>> labels.tolist(), M[0].tolist()
([1, 2, 3], [0.0, 5.0, 9.0])

```

## TSPLIB metrics

The `tsplib_*` metrics reproduce the TSPLIB 95 edge-weight types exactly, with
`nint(x) = floor(x + 0.5)` (never `numpy.rint`, which rounds half to even and
evaluates the published optimum of `qa194` to 9351 instead of 9352):

| Metric | TSPLIB type | Distance |
|---|---|---|
| `tsplib_euc_2d` | `EUC_2D` | `nint(sqrt(dx² + dy²))` |
| `tsplib_ceil_2d` | `CEIL_2D` | `ceil(sqrt(dx² + dy²))` |
| `tsplib_man_2d` | `MAN_2D` | `nint(abs(dx) + abs(dy))` |
| `tsplib_att` | `ATT` | `r = sqrt((dx² + dy²) / 10); t = nint(r); t + 1 if t < r else t` |
| `tsplib_geo` | `GEO` | great circle on a sphere of radius 6378.388 km from `DDD.MM` coordinates, truncated plus one |
| `haversine` | — | great circle in km on a sphere of radius 6371.0088 km from decimal degrees |

`GEO` converts each `DDD.MM` coordinate with `deg = int(x); m = x - deg;
rad = PI * (deg + 5 m / 3) / 180` where `PI = 3.141592` and `int` truncates
(the convention of Concorde and of the published optima: `ulysses16.opt.tour`
evaluates to 6859 this way, 6917 with `nint`), then
`d = int(RRR * acos(0.5 * ((1 + q1) q2 - (1 - q1) q3)) + 1)` with
`RRR = 6378.388`, `q1 = cos(lon_i - lon_j)`, `q2 = cos(lat_i - lat_j)`,
`q3 = cos(lat_i + lat_j)`.

## Distances from coordinates

::: skroute.preprocessing.distance_matrix
    options:
      show_root_heading: true

::: skroute.preprocessing.euclidean_matrix
    options:
      show_root_heading: true

::: skroute.preprocessing.haversine_matrix
    options:
      show_root_heading: true

::: skroute.preprocessing.tsplib_nint
    options:
      show_root_heading: true

## Long tables and dicts

::: skroute.preprocessing.pairs_to_matrix
    options:
      show_root_heading: true

::: skroute.preprocessing.to_dict_of_dicts
    options:
      show_root_heading: true

::: skroute.preprocessing.from_dict_of_dicts
    options:
      show_root_heading: true

## Coordinates

::: skroute.preprocessing.normalize_coords
    options:
      show_root_heading: true

## Google Distance Matrix API

Install the extra with `pip install scikit-route[google]`. Requests are batched
`batch_size × batch_size` origins × destinations (the API caps a request at 100
elements); the 1.0 `CostScraper` issued one request per pair.

::: skroute.preprocessing.google.GoogleDistanceMatrix
    options:
      show_root_heading: true

::: skroute.preprocessing.google.CostScraper
    options:
      show_root_heading: true
