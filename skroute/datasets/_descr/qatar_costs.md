# Qatar -- road distances between 192 places

Road distances and driving times between 192 populated places of Qatar (the
nodes of the Waterloo instance `qa194` that the Google Distance Matrix API
could route), retrieved in 2020. Unlike the Spanish data sets this table has
no money column: `cost` is the driving distance in kilometres.

## Fields

- `cost`: `float64 (192, 192)`, kilometres by road (`meters / 1000`).
- `time`: `float64 (192, 192)`, hours of driving (`seconds / 3600`).
- `distance`: `float64 (192, 192)`, metres by road.
- `coords`: `float64 (192, 2)`, `(latitude, longitude)` in decimal degrees.
- `labels`: `int64 (192,)`, ids `1..193` in order of first appearance (one id
  is absent: the API could not route it); `depot == labels[0] == 1`.
- `units == {"cost": "km", "time": "h", "distance": "m"}`.
- `frame`: the long table as a `pandas.DataFrame` when `as_frame=True`
  (columns `id_origin, lat_origin, lon_origin, address_origin, id_destinity,
  lat_destinity, lon_destinity, address_destinity, meters, seconds`),
  otherwise `None`.

The table holds each unordered pair once (18 336 rows, no diagonal); the
matrices are mirrored from it and are therefore symmetric.

One pair is recorded as zeros: the row `(104, 111)` has `meters = 0` and
`seconds = 0` although the two places (25.533 N 50.977 E and 25.568 N 50.959 E)
are about 4 km apart, so `cost`, `time` and `distance` each hold one zero off
the diagonal (the Spanish tables have none). It is kept as in the source; fill
it yourself (for example with the haversine distance) if your use needs
strictly positive legs.

## Note

In 1.0 `load_costs_qatar()` returned the Valencia table by mistake; the 2.0
name is `load_qatar_costs()` and `load_costs_qatar` is a deprecated alias that
now loads this data set.
