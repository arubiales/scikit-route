# $city -- road costs between $n places

A small **multi-trip routing** instance: $n addresses in $city (Spain) with the
cost of driving between every pair, priced for the two-people, per-extra-trip
objective of scikit-route (`fit(cost, time_matrix=time, max_time_work=8,
extra_cost=..., people=2)`). The depot is the first id, `$depot`.

## Fields

- `cost`: `float64 (n, n)`, EUR -- fuel plus driver time, as the 1.0 data set
  priced it. Symmetric with a zero diagonal.
- `time`: `float64 (n, n)`, hours of driving (the `hours` column of the table).
- `distance`: `float64 (n, n)`, metres by road.
- `coords`: `float64 (n, 2)`, `(latitude, longitude)` in decimal degrees.
- `labels`: `int64 (n,)`, the place ids in order of first appearance;
  `depot == labels[0] == $depot`.
- `units == {"cost": "EUR", "time": "h", "distance": "m"}`.
- `frame`: the long table as a `pandas.DataFrame` when `as_frame=True`
  (columns `id_origin, id_destinity, lat_origin, lon_origin, lat_destiniy,
  lon_destinity, cluster, origin, destinity, meters, secs, hours, kilometers,
  cost`, spelled as in the original file), otherwise `None`.

The matrices are plain arrays without labels: pass `labels=b.labels` and
`depot=b.depot` to `fit`, or use `as_frame=True` to get labelled DataFrames.

## Source

Distances and durations were retrieved from the Google Distance Matrix API in
2020 for the upper triangle of the pairs (including the diagonal) and mirrored;
the file `$file` is that table verbatim (CSV, addresses quoted).
