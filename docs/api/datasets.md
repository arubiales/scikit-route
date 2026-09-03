# Datasets

`skroute.datasets` bundles two families of routing instances and a reader for the
TSPLIB 95 file format. Nothing is downloaded: every file ships inside the wheel.

* **27 Waterloo national TSP instances** — the populated places of a country as
  planar coordinates with a published optimal tour length under the TSPLIB
  `EUC_2D` metric. Load one with `load_tsp("wi29")` or its country wrapper
  `load_sahara()`.
* **Five road-cost tables** — small multi-trip instances (Alicante–Murcia,
  Barcelona, Madrid, Valencia and Qatar) with cost, time and distance matrices
  retrieved from the Google Distance Matrix API.

Loaders return a [`Bunch`][skroute.utils.Bunch]: a `dict` whose keys are also
attributes. **The matrices carry no labels** — they are plain `float64` arrays —
so pass `labels=b.labels` (and `depot=b.depot`) to `fit` or to
`RoutingProblem`, or ask for `as_frame=True` and get labelled DataFrames.

```python
>>> from skroute.datasets import load_tsp, load_barcelona
>>> wi = load_tsp("wi29")
>>> wi.coords.shape, wi.depot, wi.optimal_tour_length
((29, 2), 1, 27603)
>>> C = wi.distance_matrix()  # built once, cached; never built by the loader
>>> C.shape
(29, 29)
>>> bcn = load_barcelona()
>>> bcn.cost.shape, bcn.depot, bcn.units
((19, 19), 10000007, {'cost': 'EUR', 'time': 'h', 'distance': 'm'})

```

## The 27 Waterloo instances

All 27 use `EDGE_WEIGHT_TYPE: EUC_2D`, so the distance between two cities is
`nint(sqrt(dx² + dy²))` with `nint(x) = floor(x + 0.5)` — what
`TSPBunch.distance_matrix()` computes. Optima are those published by the
University of Waterloo. The four instances above 20 000 nodes cannot be solved
whole in scikit-route 2.0 (every solver works on a dense matrix; the one of
`ch71009` would take 40 GB): `distance_matrix()` refuses them unless
`force=True`, and `load_tsp(name, n_nodes=5000)` gives a subsample whose optimum
is unknown.

| Name | Country | Wrapper | Cities | Optimal tour |
|---|---|---|---|---|
| `wi29` | Western Sahara | `load_sahara` | 29 | 27 603 |
| `dj38` | Djibouti | `load_djibouti` | 38 | 6 656 |
| `qa194` | Qatar | `load_qatar` | 194 | 9 352 |
| `uy734` | Uruguay | `load_uruguay` | 734 | 79 114 |
| `zi929` | Zimbabwe | `load_zimbabwe` | 929 | 95 345 |
| `lu980` | Luxembourg | `load_luxembourg` | 980 | 11 340 |
| `rw1621` | Rwanda | `load_rwanda` | 1 621 | 26 051 |
| `mu1979` | Oman | `load_oman` | 1 979 | 86 891 |
| `nu3496` | Nicaragua | `load_nicaragua` | 3 496 | 96 132 |
| `ca4663` | Canada | `load_canada` | 4 663 | 1 290 319 |
| `tz6117` | Tanzania | `load_tanzania` | 6 117 | 394 718 |
| `eg7146` | Egypt | `load_egypt` | 7 146 | 172 386 |
| `ym7663` | Yemen | `load_yemen` | 7 663 | 238 314 |
| `pm8079` | Panama | `load_panama` | 8 079 | 114 855 |
| `ei8246` | Ireland | `load_ireland` | 8 246 | 206 171 |
| `ar9152` | Argentina | `load_argentina` | 9 152 | 837 479 |
| `ja9847` | Japan | `load_japan` | 9 847 | 491 924 |
| `gr9882` | Greece | `load_greece` | 9 882 | 300 899 |
| `kz9976` | Kazakhstan | `load_kazakhstan` | 9 976 | 1 061 882 |
| `fi10639` | Finland | `load_finland` | 10 639 | 520 527 |
| `mo14185` | Morocco | `load_morocco` | 14 185 | 427 377 |
| `ho14473` | Honduras | `load_honduras` | 14 473 | 177 092 |
| `it16862` | Italy | `load_italy` | 16 862 | 557 315 |
| `vm22775` | Vietnam | `load_vietnam` | 22 775 | 569 288 |
| `sw24978` | Sweden | `load_sweden` | 24 978 | 855 597 |
| `bm33708` | Burma | `load_burma` | 33 708 | 959 289 |
| `ch71009` | China | `load_china` | 71 009 | 4 566 563 |

Source: W. Cook et al., *National Traveling Salesman Problems*, University of
Waterloo, <https://www.math.uwaterloo.ca/tsp/world/countries.html>.

::: skroute.datasets.load_tsp
    options:
      show_root_heading: true

::: skroute.datasets.list_tsp
    options:
      show_root_heading: true

::: skroute.datasets.TSPBunch
    options:
      show_root_heading: true

### Country wrappers

Each one is `load_tsp("<name>", **kwargs)` under its 1.0 name and accepts the
same `n_nodes=`, `random_state=` and (deprecated) `mode=` keywords.

::: skroute.datasets.load_sahara
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_djibouti
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_qatar
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_uruguay
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_zimbabwe
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_luxembourg
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_rwanda
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_oman
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_nicaragua
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_canada
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_tanzania
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_egypt
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_yemen
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_panama
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_ireland
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_argentina
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_japan
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_greece
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_kazakhstan
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_finland
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_morocco
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_honduras
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_italy
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_vietnam
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_sweden
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_burma
    options:
      show_root_heading: true
      show_docstring_examples: false

::: skroute.datasets.load_china
    options:
      show_root_heading: true
      show_docstring_examples: false

## Road-cost tables

Five long tables of `(origin, destination, cost, time, distance)` rows pivoted
into symmetric matrices with
[`pairs_to_matrix`][skroute.preprocessing.pairs_to_matrix]. `cost` is in EUR
(the Spanish files) or kilometres (Qatar, which has no money column), `time` in
hours, `distance` in metres; `coords` are `(latitude, longitude)` in decimal
degrees and `depot` is the first id of each file. The Spanish `time` matrices
come from the `hours` column of the tables, which adds a fixed 7-minute stop to
every leg (`hours = (secs + 420) / 3600`).

| Loader | Places | Depot | `cost` unit | Rows in the table |
|---|---|---|---|---|
| `load_alicante_murcia` | 8 | `10000002` | EUR | 36 |
| `load_barcelona` | 19 | `10000007` | EUR | 190 |
| `load_madrid` | 18 | `10000016` | EUR | 171 |
| `load_valencia` | 14 | `10000022` | EUR | 105 |
| `load_qatar_costs` | 192 | `1` | km | 18 336 |

::: skroute.datasets.load_alicante_murcia
    options:
      show_root_heading: true

::: skroute.datasets.load_barcelona
    options:
      show_root_heading: true

::: skroute.datasets.load_madrid
    options:
      show_root_heading: true

::: skroute.datasets.load_valencia
    options:
      show_root_heading: true

::: skroute.datasets.load_qatar_costs
    options:
      show_root_heading: true

::: skroute.datasets.load_costs_qatar
    options:
      show_root_heading: true

## TSPLIB reader

Pure-Python readers for the TSPLIB 95 formats. `read_tsplib` handles the
coordinate types `EUC_2D`, `CEIL_2D`, `MAN_2D`, `ATT` and `GEO` (coordinates are
returned raw; convert them with
[`distance_matrix`][skroute.preprocessing.distance_matrix] and the matching
`tsplib_*` metric) and `EXPLICIT` matrices in the formats `FULL_MATRIX`,
`UPPER_ROW`, `LOWER_ROW`, `UPPER_DIAG_ROW` and `LOWER_DIAG_ROW`.

::: skroute.datasets.read_tsplib
    options:
      show_root_heading: true

::: skroute.datasets.read_tsplib_tour
    options:
      show_root_heading: true
