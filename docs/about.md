# About

## History

scikit-route was started in 2020 by Alberto Rubiales as a route-optimisation library with the spirit of
scikit-learn: estimators you configure in `__init__`, fit on your data and read results from. It grew out of
a concrete routing problem at Secoex — planning the daily rounds of a sales team around Spanish cities,
where a working day must not exceed eight hours and every extra day has a price — which is why the library
has always carried, next to the plain travelling salesman problem, a multi-trip objective with a per-trip
working-time budget and a fixed charge per additional trip. The bundled cost datasets (Alicante–Murcia,
Barcelona, Madrid, Valencia) date from that period.

The first design reached 1.0.0a2 in 2021. Version 2.0 (2026) is a full rewrite with the same aim and a stricter
contract:

- Python 3.11 or newer, numpy 2 compatible, three runtime dependencies (numpy, scipy, joblib); pandas and
  the Google client are optional extras.
- One Cython 3 core over typed memoryviews shared by every solver: cost evaluation, the two split rules
  (greedy and Prins' optimal split), move deltas, local-search descents and constructions.
- Every solver is an estimator: `fit(X, ...)` returns `self`, results live in trailing-underscore attributes
  (`route_`, `trips_`, `cost_`, `history_`, ...), and the base class recomputes `cost_` from the returned
  tour, so a route can never disagree with its reported cost.
- The multi-trip objective is available to every budget-aware solver through the same kernels; plain TSP
  is `est.fit(C)`.
- Deterministic tests through `random_state`, exact results checked on small instances, and benchmarks
  against the Waterloo national instances with measured tolerances — see the [benchmarks](benchmarks.md).

The [changelog](changelog.md) lists every change and the [migration guide](migration.md) walks a 1.0 user
through them.

## Licence

scikit-route is released under the [MIT License](https://github.com/arubiales/scikit-route/blob/main/LICENSE),
copyright 2020 Alberto Rubiales. The bundled TSPLIB-format instances and cost tables are redistributed as
data for testing and examples; see the acknowledgements below for their origin.

## Citing

If scikit-route is useful in your work, please cite it. The repository ships a
[`CITATION.cff`](https://github.com/arubiales/scikit-route/blob/main/CITATION.cff) file, so GitHub's
*Cite this repository* button gives you APA and BibTeX entries for the current version. In BibTeX:

```bibtex
@software{rubiales_scikit_route_2026,
  author  = {Rubiales, Alberto},
  title   = {scikit-route: route optimisation with a scikit-learn flavoured API},
  year    = {2026},
  version = {2.0.0},
  license = {MIT},
  url     = {https://github.com/arubiales/scikit-route},
}
```

Replace `version` and `year` with those of the release you used (`skroute.__version__`).

## Acknowledgements

- The 27 national TSP instances bundled in `skroute.datasets` (`wi29`, `dj38`, `qa194`, ..., `ch71009`)
  come from the **National Traveling Salesman Problems** collection assembled by William Cook and his
  colleagues at the University of Waterloo (<https://www.math.uwaterloo.ca/tsp/world/countries.html>), which
  also publishes the optimal or best-known tour lengths that the benchmarks page compares against.
- The `.tsp` file format, the `EUC_2D`/`CEIL_2D`/`ATT`/`GEO` distance definitions implemented in
  `skroute.preprocessing.distance_matrix`, and the `ulysses16` and `att48` instances used as reader tests are
  from **TSPLIB** by Gerhard Reinelt (Heidelberg University,
  <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>).
- The estimator protocol (`get_params`/`set_params`, `clone`, trailing-underscore attributes, `random_state`)
  follows the conventions of **scikit-learn**, whose API design this library has borrowed since its first
  version.
- The optimal split of a giant tour into trips implements the procedure of C. Prins, *A simple and effective
  evolutionary algorithm for the vehicle routing problem*, Computers & Operations Research 31 (2004).

## Links

- Source code and issue tracker: <https://github.com/arubiales/scikit-route>
- Documentation: <https://arubiales.github.io/scikit-route/>
- Package on PyPI: <https://pypi.org/project/scikit-route/>
- [Installation](installation.md) · [Benchmarks](benchmarks.md) · [Contributing](contributing.md) ·
  [Changelog](changelog.md)
