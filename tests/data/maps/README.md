# Recorded map-service answers (tests/test_maps.py)

Served offline through the fake opener of `tests/test_maps.py`; no test touches the network.

| File | Origin |
|---|---|
| `osrm_table_madrid.json` | `GET https://router.project-osrm.org/table/v1/driving/-3.7635,40.3272;-3.7038,40.4168;-3.3635,40.4819?sources=0;1;2&destinations=0;1;2&annotations=duration,distance`, captured once on 2026-09-05 (Leganés office, Puerta del Sol, Alcalá de Henares). |
| `nominatim_leganes.json` | `GET https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=Calle+Ramón+y+Cajal+18,+Leganés`, captured once on 2026-09-05. |
| `overpass_leganes.json` | Overpass answer for the Burger King restaurants of Leganés (`nwr["amenity"="fast_food"]["brand:wikidata"="Q177054"]`, `out center`), captured once on 2026-09-05; the ways' member lists were dropped and three things were added by hand: `addr:*` tags on three nodes, a `relation` with a `center` and no `name`, and a `relation` without coordinates (skipped by `fetch_pois`). The elements are stored in reverse order to exercise the sort. |
| `google_geocode_leganes.json` | Written by hand after the documented shape of the Geocoding API (https://developers.google.com/maps/documentation/geocoding/requests-geocoding). |

OpenStreetMap data © OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright).
