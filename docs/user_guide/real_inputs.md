# Real inputs from map services

The solvers want a dense matrix; the world gives you addresses, a brand name and a
road network. `skroute.preprocessing.maps` bridges the two with three functions that
talk to public map services through the standard library alone — no extra
dependency, no key for the OpenStreetMap ones — and return
[`Bunch`][skroute.utils.Bunch] objects ready for `fit`:

| Function | Question it answers | Default service | Key |
|---|---|---|---|
| [`fetch_pois`][skroute.preprocessing.fetch_pois] | *Where are all the X of this area?* | Overpass (OpenStreetMap) | none |
| [`geocode`][skroute.preprocessing.geocode] | *Where is this address?* | Nominatim (OpenStreetMap) | none |
| [`travel_time_matrix`][skroute.preprocessing.travel_time_matrix] | *How long does it take to drive from every stop to every other?* | OSRM demo server | none |

Each one takes `provider="google"` as well (Geocoding API, Distance Matrix API — an
API key, billed to your account). This page walks through the typical sequence on the
case of a maintenance technician who must visit every Burger King of the Madrid region
from an office in Leganés; the examples call the live services, so they are marked
`# doctest: +SKIP` and their outputs are the ones recorded on 2026-09-05.

!!! warning "Coordinates are `(lat, lon)`"
    Everywhere in scikit-route a point is a `(latitude, longitude)` pair — the order
    Google Maps shows and the one `geocode`/`fetch_pois` return. OSRM wants `lon,lat`
    in its URLs; `travel_time_matrix` swaps them for you. Passing `(lon, lat)` by
    mistake is only caught when a "latitude" exceeds 90°, so check the order once.

## 1. The stops: `fetch_pois`

One Overpass QL query selects an administrative area by its OpenStreetMap name and
every element (`nwr`: nodes, ways, relations) matching the filters inside it. The most
reliable brand filter is the Wikidata id, immune to spelling variants (`Q177054` is
Burger King; look yours up on wikidata.org):

```python
>>> from skroute.preprocessing import fetch_pois
>>> bk = fetch_pois("Comunidad de Madrid", amenity="fast_food", wikidata="Q177054")  # doctest: +SKIP
>>> bk  # doctest: +SKIP
Bunch(DESCR, addresses, coords, labels, names, tags)
>>> bk.coords.shape, bk.labels[:2].tolist()  # doctest: +SKIP
((183, 2), ['node/26289763', 'node/178821228'])
>>> bk.names[1], bk.addresses[1]  # doctest: +SKIP
('Burger King', 'Calle de Esparteros 3, 28012 Madrid')
>>> print(bk.DESCR.splitlines()[-1])  # doctest: +SKIP
Data © OpenStreetMap contributors, licensed under the Open Database License (ODbL): https://www.openstreetmap.org/copyright

```

`coords` is a float64 `(n, 2)` array; `labels` is an object array of the OSM ids
(`node/123`, `way/123`, `relation/123` — ways and relations, e.g. a restaurant mapped
as a building, are placed at the centre of their bounding box thanks to `out center`);
`names` and `addresses` are lists from the `name` and `addr:*` tags (`""` when unknown)
and `tags` keeps everything else. The query itself is in `DESCR`, together with the ODbL
attribution you must show with the data.

The filters are `amenity=` (exact tag), `brand=` (exact), `name=` (case-insensitive
regular expression) and `wikidata=` (`brand:wikidata`, exact); give at least one. The
area name must match the `name` tag of an administrative boundary exactly
(`"Comunidad de Madrid"`, `"Leganés"`, `"Catalunya"`) — when nothing matches you get an
empty result and a `UserWarning`, not an error.

!!! note "Near-duplicates are kept"
    OpenStreetMap sometimes has the same shop twice: as the building (a way) and as a
    node inside it. `fetch_pois` does not guess which one you want. Keep only nodes
    (`mask = [l.startswith("node/") for l in bk.labels]`, then `bk.coords[mask]` and
    `bk.labels[mask]`) or drop anything within a few metres of an earlier point with
    the great-circle distances of
    [`haversine_matrix`][skroute.preprocessing.haversine_matrix] (kilometres):

    ```python
    >>> import numpy as np
    >>> from skroute.preprocessing import haversine_matrix
    >>> coords = np.array([[40.3366, -3.7690], [40.3366, -3.7690], [40.3065, -3.8100]])
    >>> D = haversine_matrix(coords)
    >>> keep = [i for i in range(len(coords)) if not (D[i, :i] < 0.02).any()]  # 20 m
    >>> keep
    [0, 2]

    ```

## 2. The depot: `geocode`

```python
>>> from skroute.preprocessing import geocode
>>> office = geocode("Calle Ramón y Cajal 18, Leganés")  # doctest: +SKIP
>>> office  # doctest: +SKIP
Bunch(display_name, lat, lon, raw)
>>> round(office.lat, 4), round(office.lon, 4)  # doctest: +SKIP
(40.3296, -3.7373)
>>> office.display_name  # doctest: +SKIP
'18, Calle de Ramón y Cajal, Polígono Industrial de Nuestra Señora de Butarque, El Carrascal, Leganés, Comunidad de Madrid, 28916, España'

```

Nominatim returns its best match (`limit=1`); read `display_name` to confirm it is
the place you meant, and raise a clearer question if it is not (add the postcode or
the town). A query with no match raises `ValueError("no result for ...")`. Nominatim's
usage policy — a descriptive `User-Agent` and at most one request per second — is
enforced for you: the default agent is `scikit-route/<version> (+repository URL)`
(pass `user_agent=` to name your own application), and a second call within a second
of the first simply waits — so does a retry after a `429`.

## 3. The matrix: `travel_time_matrix`

Put the depot first and ask for the table. The default provider is the public OSRM
demo server (`https://router.project-osrm.org`, car profile), which returns
durations in seconds and distances in metres for every ordered pair — asymmetric when
the roads are:

```python
>>> import numpy as np
>>> from skroute.preprocessing import travel_time_matrix
>>> coords = np.vstack([[office.lat, office.lon], bk.coords])  # doctest: +SKIP
>>> res = travel_time_matrix(coords)  # doctest: +SKIP
>>> res  # doctest: +SKIP
Bunch(coords, distance, provider, time, units)
>>> res.units, res.time.shape  # doctest: +SKIP
({'time': 'min', 'distance': 'm'}, (184, 184))
>>> res.time[0, 1:4].round(1), res.time[1:4, 0].round(1)  # doctest: +SKIP
(array([20.3, 18.8, 20.8]), array([21.7, 17.5, 18.6]))

```

`time` is in minutes by default (`units="s"` or `"h"` to change it), `distance` always
in metres, the diagonal is `0`. Pairs the server cannot route become `nan` in both
matrices with a single `RuntimeWarning` naming how many — fill them (a large value, or
the great-circle distance at a plausible speed) before solving, since every solver
needs finite matrices. An OSRM server that returns no distances (a build without
`annotations=distance`) also gets one `RuntimeWarning`: `time` is complete, `distance`
is `nan` off the diagonal.

The demo server caps the size of a table request, so the points are tiled in blocks of
`chunk_size` (default 50): a request carries the row block as `sources` and the column
block as `destinations` — at most 100 coordinates — and `pause` seconds (default 1)
separate consecutive requests. The 184 points of the case are `4 × 4 = 16` requests,
twenty seconds with the default pause; 300 points would be 36. Your own OSRM server takes `base_url=`, `mode=` as
the profile name in its URLs (`car`, `bike`, `foot`… — the demo only serves
`driving`), `pause=0` and a larger `chunk_size`.

With `provider="google"` the same call delegates to
[`GoogleDistanceMatrix`][skroute.preprocessing.google.GoogleDistanceMatrix] (install
`scikit-route[google]`; requests are billed to the account behind `api_key`) and
converts its hours to the requested unit; `departure_time="now"` (or a `datetime`)
asks for traffic-aware durations and `timeout=` reaches the client, while `base_url`,
`user_agent` and `pause` only concern OSRM. OSRM has no traffic model — a
`departure_time` given with `provider="osrm"` is ignored with a `UserWarning`.

```python
>>> res = travel_time_matrix(coords, provider="google", api_key="<your key>", departure_time="now")  # doctest: +SKIP
>>> res.provider, res.units  # doctest: +SKIP
('google', {'time': 'min', 'distance': 'm'})

```

## 4. Solve

The matrix is the `time_matrix` of the [multi-trip model](multi_trip.md): one working
day per trip, `max_time_work` in the same minutes, and — to minimise driving — the
time itself as the cost:

```python
>>> from skroute import MultiStart, IteratedLocalSearch
>>> labels = ["office", *bk.labels]  # doctest: +SKIP
>>> est = MultiStart(IteratedLocalSearch(time_limit=15), n_restarts=4, random_state=0)  # doctest: +SKIP
>>> est.fit(res.time, time_matrix=res.time, labels=labels, depot="office",
...         max_time_work=8 * 60, extra_cost=8 * 60, split="optimal")  # doctest: +SKIP
MultiStart(estimator=IteratedLocalSearch(time_limit=15), n_restarts=4, random_state=0)
>>> est.n_trips_, round(float(est.trip_costs_.sum()))  # days, minutes of driving  # doctest: +SKIP
(3, 1164)

```

Three days of pure driving — but every stop also takes time: half an hour of
maintenance per restaurant here. Pass it as `service_time=30` (minutes, the unit of the
matrix; the same at every customer, nothing at the depot) and the budget accounts for it
— see [Service times](multi_trip.md#service-times) for the model. The 183 half-hours
alone are 91.5 hours of work, so no plan fits in fewer than twelve eight-hour days:

```python
>>> est.fit(res.time, time_matrix=res.time, labels=labels, depot="office",
...         max_time_work=8 * 60, extra_cost=8 * 60, split="optimal", service_time=30)  # doctest: +SKIP
MultiStart(estimator=IteratedLocalSearch(time_limit=15), n_restarts=4, random_state=0)
>>> est.n_trips_ >= 12  # doctest: +SKIP
True

```

## Being a good citizen

The OpenStreetMap services are run by volunteers and donations. Keep the default
`User-Agent` or, better, name your application; keep the pauses; cache what you
fetched (`numpy.save` the matrix, write the POIs to a CSV) instead of fetching it on
every run; and display "© OpenStreetMap contributors" wherever the data appears — the
`DESCR` of `fetch_pois` carries the exact sentence. The usage policies are at
[operations.osmfoundation.org](https://operations.osmfoundation.org/policies/nominatim/)
(Nominatim), [github.com/Project-OSRM](https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server)
(the OSRM demo) and [wiki.openstreetmap.org](https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances)
(Overpass). When a service is down or throttles you, the functions retry three times
with back-off and then raise
[`MapServiceError`][skroute.preprocessing.maps.MapServiceError] with the HTTP status,
the request URL (API keys redacted) and the first 200 characters of the answer.

Contributors: the tests of this module never touch the network (recorded answers under
`tests/data/maps/`); the live checks carry the `network` marker and run in the nightly
(`pytest -m network`).
