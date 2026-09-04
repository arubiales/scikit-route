# $name -- $country ($n cities)

One of the 27 **national TSP instances** of the University of Waterloo
(William Cook and collaborators): the $n populated places of $country as
recorded by the National Imagery and Mapping Agency, in the TSPLIB 95 format
with `EDGE_WEIGHT_TYPE: EUC_2D`.

## Fields

- `coords`: `float64 (n, 2)`, the `x`/`y` values exactly as written in the file
  (they derive from latitude/longitude but are *not* degrees: use the planar
  TSPLIB metric, never haversine).
- `labels`: `int64 (n,)`, the file's 1-based ids; `depot` is the first label.
- `edge_weight_type`: `"EUC_2D"`.
- `optimal_tour_length`: $optimum -- $optimality, under the TSPLIB `EUC_2D`
  metric (`nint(x) = floor(x + 0.5)` of the Euclidean distance); `None` when
  the instance is subsampled with `n_nodes=` or `mode=`.
- `distance_matrix(*, force=False)`: builds (once, cached) the `(n, n)`
  `float64` matrix with `skroute.preprocessing.distance_matrix(coords,
  metric="tsplib_euc_2d")`; the benchmarks compare every solver against
  `optimal_tour_length` on it.

## Usage

    from skroute.datasets import load_tsp
    b = load_tsp("$name")
    C = b.distance_matrix()             # plain ndarray, no labels
    est.fit(C, labels=b.labels)         # pass labels= so route_ uses the ids
$note
## Source and licence

Data compiled by the University of Waterloo for the "National Traveling
Salesman Problems" collection (https://www.math.uwaterloo.ca/tsp/world/countries.html;
tour lengths and status from https://www.math.uwaterloo.ca/tsp/world/summary.html),
derived from the National Imagery and Mapping Agency database of geographic
feature names. Redistributed unchanged for research and teaching.
