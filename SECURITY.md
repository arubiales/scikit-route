# Security policy

## Supported versions

| Version | Supported |
|---|---|
| latest 2.x minor | yes |
| older 2.x minors | no: upgrade to the latest 2.x |
| 1.0.0a2 and earlier | no |

Only the latest 2.x minor receives security fixes. A fix ships as a patch release of that minor.

## Reporting a vulnerability

**Never open a public issue for a vulnerability.** Report it privately, in one of two ways:

1. GitHub: *Security* tab of the repository → *Report a vulnerability* (private advisory).
2. E-mail: al.rubiales.b@gmail.com — put `scikit-route security` in the subject.

Include the scikit-route version (`python -c "import skroute; print(skroute.__version__)"`), the Python
version and operating system, a minimal reproducer and the impact you foresee.

## What to expect

- Acknowledgement of the report within 14 days.
- A private assessment and, when confirmed, a fix in the latest 2.x minor with a release note that
  credits you (unless you prefer to stay anonymous).
- Public disclosure only after the fixed release is on PyPI.

## Scope

scikit-route is a pure computation library: it reads cost matrices you already hold in memory, runs
solvers on them and returns arrays. It performs **no network access**, with a single opt-in exception —
the optional Google Distance Matrix client in `skroute.preprocessing.google` (extra
`scikit-route[google]`), which contacts Google's API only when you call it with your own API key.
Nothing is written to disk and nothing is deserialised with `pickle`. The bundled datasets are plain
text (TSPLIB `.tsp` files and CSV tables) parsed with the standard library.

Issues in third-party dependencies (numpy, scipy, joblib, googlemaps, pandas) should be reported to
those projects; tell us as well if scikit-route's usage of them makes an issue exploitable.
