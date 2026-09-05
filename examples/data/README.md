# Data of the worked case: the Burger King restaurants of the Madrid region

Three CSV files, captured on **2026-09-05** for `examples/technician_madrid.py` (the
maintenance technician's plan, D35) so that the example — and its test — run offline.

| File | Content |
|---|---|
| `madrid_burger_king.csv` | 183 rows: the office first (label `office`), then 182 Burger King restaurants of the Comunidad de Madrid. Columns `label` (the OpenStreetMap id, `node/123` or `way/123`), `name`, `lat`, `lon` (decimal degrees), `city`, `street`, `housenumber`, `postcode`, `opening_hours` (the `addr:*` and `opening_hours` tags; many are empty in OpenStreetMap) |
| `madrid_burger_king_times_min.csv` | 183 × 183 driving times in **minutes**, row = origin, column = destination, labelled header and first column; asymmetric (one-way streets, motorway exits) |
| `madrid_burger_king_dist_km.csv` | the same shape in **kilometres** along the road |

## Provenance

- **Restaurants** — OpenStreetMap through the Overpass API
  (`skroute.preprocessing.fetch_pois`): the elements (`nwr`, i.e. nodes, ways and relations)
  tagged `amenity=fast_food` and `brand:wikidata=Q177054` (Burger King) inside the
  administrative area named `Comunidad de Madrid`. Ways and relations are placed at the
  centre of their bounding box (`out center`). One near-duplicate — the same restaurant
  mapped twice within 60 m — was dropped, keeping the first element in the order the query
  returned them.
- **Office** — `Calle Ramón y Cajal 18, Leganés, Madrid, España`, geocoded with Nominatim
  (`skroute.preprocessing.geocode`): (40.329559, −3.737270).
- **Matrices** — the table service of the public OSRM demo server
  (`skroute.preprocessing.travel_time_matrix`, car profile, no traffic model), requested in
  50 × 50 blocks with a one-second pause between requests. No pair was unroutable.

**Attribution** — Data © OpenStreetMap contributors (ODbL); routing by OSRM
(router.project-osrm.org). Show it wherever these numbers or the pictures made from them
appear; the ODbL text is at <https://www.openstreetmap.org/copyright>.

## Regenerating the files

```bash
python examples/technician_madrid.py --refresh                                     # OSRM (free, no key)
python examples/technician_madrid.py --refresh --provider google --google-key AIza... # Google, with traffic
```

`--refresh` runs the three calls above again — Overpass, Nominatim, then OSRM (16 table
requests, about twenty seconds with the pauses) or the Google Distance Matrix API with
`departure_time="now"` (billed to the key's project) — drops the near-duplicates within
60 m and rewrites the three files in `--data` (default: this directory). Expect the counts to
change: restaurants open and close, mappers add and correct them, and the road network
moves on. A pair the router cannot connect is filled with the great-circle time at 30 km/h
and reported, so the matrices stay finite.
