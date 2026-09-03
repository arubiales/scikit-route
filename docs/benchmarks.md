# Benchmarks

Gap of every solver to the published optimum of the bundled Waterloo instances, produced by
`python benchmarks/waterloo.py` on the release candidate and committed here (SPEC §7). The
test-suite asserts the **Tolerance** column on every run (`tests/test_common.py` for the fast
tier, `tests/benchmarks/test_waterloo.py` for the slow one); this page records the gaps actually
measured, with their provenance. Gaps are `cost_ / optimum - 1` with `random_state=0`; a
different machine may move a stochastic solver's gap by a tie-break (libm `exp` differs by
ulps and flips Metropolis decisions), never below the optimum.

- **Date:** 2026-09-03
- **Commit:** `26cb1f3` (branch `wp/ensemble`), scikit-route 2.0.0
- **CPU:** Apple M1 Max
- **OS:** macOS-26.6.2-arm64-arm-64bit-Mach-O
- **Python:** 3.13.3 · numpy 2.5.2 · scipy 1.18.1
- **Seed:** `random_state=0` for every stochastic solver; one fit per cell, wall time in seconds
- **Parameters:** the defaults of every class, except `MILP(time_limit=150)` on the slow instances (as in the tolerance table); bracketed keys apply the `set_params` of `tests/tolerances.py` (`Insertion[farthest]` = `strategy="farthest"`, `Genetic[memetic]` = `local_search=("two_opt",)`)
- **Instances:** wi29 (optimum 27603), dj38 (optimum 6656), qa194 (optimum 9352), lu980 (optimum 11340) — TSPLIB `EUC_2D` with `nint` rounding (D15)
- **Tolerance:** the maximum gap asserted by the test-suite (`tests/tolerances.py`: `FAST` on wi29/dj38, `SLOW` elsewhere; `—` = not asserted on that instance). **Baseline:** the gap measured while the specification was written (`tolerances.MEASURED`, SPEC §4), so a regression is distinguishable from a tie-break difference.

| Solver | Instance | n | Optimum | Cost | Gap | Tolerance | Baseline | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `AntColony` | wi29 | 29 | 27603 | 27603 | 0.00 % | 8.00 % | — | 0.01 |
| `AntColony` | dj38 | 38 | 6656 | 6656 | 0.00 % | 8.00 % | — | 0.01 |
| `AntColony` | qa194 | 194 | 9352 | 9512 | 1.71 % | 15.00 % | — | 0.21 |
| `AntColony` | lu980 | 980 | 11340 | 11982 | 5.66 % | 15.00 % | — | 1.14 |
| `BruteForce` | wi29 | 29 | 27603 | capped | — | — | — | — |
| `BruteForce` | dj38 | 38 | 6656 | capped | — | — | — | — |
| `BruteForce` | qa194 | 194 | 9352 | capped | — | — | — | — |
| `BruteForce` | lu980 | 980 | 11340 | capped | — | — | — | — |
| `ClarkeWright` | wi29 | 29 | 27603 | 28953 | 4.89 % | 25.00 % | — | 0.00 |
| `ClarkeWright` | dj38 | 38 | 6656 | 6664 | 0.12 % | 25.00 % | — | 0.00 |
| `ClarkeWright` | qa194 | 194 | 9352 | 10498 | 12.25 % | 25.00 % | — | 0.00 |
| `ClarkeWright` | lu980 | 980 | 11340 | 12324 | 8.68 % | 25.00 % | — | 0.07 |
| `EnsembleGenetic` | wi29 | 29 | 27603 | 27603 | 0.00 % | 15.00 % | — | 0.15 |
| `EnsembleGenetic` | dj38 | 38 | 6656 | 6656 | 0.00 % | 15.00 % | — | 0.18 |
| `EnsembleGenetic` | qa194 | 194 | 9352 | 10935 | 16.93 % | — | — | 0.55 |
| `EnsembleGenetic` | lu980 | 980 | 11340 | 14368 | 26.70 % | — | — | 0.63 |
| `EnsembleSimulatedAnnealing` | wi29 | 29 | 27603 | 27603 | 0.00 % | 3.00 % | — | 0.35 |
| `EnsembleSimulatedAnnealing` | dj38 | 38 | 6656 | 6656 | 0.00 % | 3.00 % | — | 0.41 |
| `EnsembleSimulatedAnnealing` | qa194 | 194 | 9352 | 9456 | 1.11 % | — | — | 1.46 |
| `EnsembleSimulatedAnnealing` | lu980 | 980 | 11340 | 12030 | 6.08 % | — | — | 10.37 |
| `Genetic` | wi29 | 29 | 27603 | 27750 | 0.53 % | 15.00 % | 4.90 % | 0.02 |
| `Genetic` | dj38 | 38 | 6656 | 7258 | 9.04 % | 15.00 % | 8.60 % | 0.02 |
| `Genetic` | qa194 | 194 | 9352 | 10894 | 16.49 % | 30.00 % | 18.80 % | 0.06 |
| `Genetic` | lu980 | 980 | 11340 | 14370 | 26.72 % | — | — | 0.08 |
| `Genetic[memetic]` | wi29 | 29 | 27603 | 27603 | 0.00 % | 5.00 % | — | 0.01 |
| `Genetic[memetic]` | dj38 | 38 | 6656 | 6656 | 0.00 % | 5.00 % | — | 0.01 |
| `Genetic[memetic]` | qa194 | 194 | 9352 | 9382 | 0.32 % | 8.00 % | — | 0.05 |
| `Genetic[memetic]` | lu980 | 980 | 11340 | 11416 | 0.67 % | 8.00 % | — | 0.81 |
| `HeldKarp` | wi29 | 29 | 27603 | capped | — | — | — | — |
| `HeldKarp` | dj38 | 38 | 6656 | capped | — | — | — | — |
| `HeldKarp` | qa194 | 194 | 9352 | capped | — | — | — | — |
| `HeldKarp` | lu980 | 980 | 11340 | capped | — | — | — | — |
| `Insertion[farthest]` | wi29 | 29 | 27603 | 28138 | 1.94 % | 25.00 % | — | 0.00 |
| `Insertion[farthest]` | dj38 | 38 | 6656 | 6656 | 0.00 % | 25.00 % | — | 0.00 |
| `Insertion[farthest]` | qa194 | 194 | 9352 | 9972 | 6.63 % | 25.00 % | — | 0.00 |
| `Insertion[farthest]` | lu980 | 980 | 11340 | 12494 | 10.18 % | 25.00 % | — | 0.00 |
| `Insertion[cheapest]` | wi29 | 29 | 27603 | 30360 | 9.99 % | 30.00 % | — | 0.00 |
| `Insertion[cheapest]` | dj38 | 38 | 6656 | 7835 | 17.71 % | 30.00 % | — | 0.00 |
| `Insertion[cheapest]` | qa194 | 194 | 9352 | 11021 | 17.85 % | 30.00 % | — | 0.00 |
| `Insertion[cheapest]` | lu980 | 980 | 11340 | 13364 | 17.85 % | 30.00 % | — | 0.02 |
| `IteratedLocalSearch` | wi29 | 29 | 27603 | 27603 | 0.00 % | 3.00 % | — | 0.01 |
| `IteratedLocalSearch` | dj38 | 38 | 6656 | 6656 | 0.00 % | 3.00 % | — | 0.01 |
| `IteratedLocalSearch` | qa194 | 194 | 9352 | 9377 | 0.27 % | 6.00 % | — | 0.06 |
| `IteratedLocalSearch` | lu980 | 980 | 11340 | 11475 | 1.19 % | 6.00 % | 4.16 % | 0.25 |
| `LocalSearch` | wi29 | 29 | 27603 | 27603 | 0.00 % | 12.00 % | — | 0.00 |
| `LocalSearch` | dj38 | 38 | 6656 | 6808 | 2.28 % | 12.00 % | — | 0.00 |
| `LocalSearch` | qa194 | 194 | 9352 | 9939 | 6.28 % | 15.00 % | 11.90 % | 0.00 |
| `LocalSearch` | lu980 | 980 | 11340 | 11878 | 4.74 % | 15.00 % | 7.80 % | 0.02 |
| `MILP` | wi29 | 29 | 27603 | 27603 | 0.00 % | 0.00 % | — | 0.03 |
| `MILP` | dj38 | 38 | 6656 | 6656 | 0.00 % | 0.00 % | — | 0.04 |
| `MILP` | qa194 | 194 | 9352 | 9352 | 0.00 % | 0.00 % | — | 39.58 |
| `MILP` | lu980 | 980 | 11340 | capped | — | — | — | — |
| `NRBS` | wi29 | 29 | 27603 | 32658 | 18.31 % | 50.00 % | — | 0.00 |
| `NRBS` | dj38 | 38 | 6656 | 7175 | 7.80 % | 50.00 % | — | 0.00 |
| `NRBS` | qa194 | 194 | 9352 | 10678 | 14.18 % | 60.00 % | — | 0.00 |
| `NRBS` | lu980 | 980 | 11340 | 13607 | 19.99 % | — | — | 0.08 |
| `NearestNeighbour` | wi29 | 29 | 27603 | 36388 | 31.83 % | 50.00 % | 31.80 % | 0.00 |
| `NearestNeighbour` | dj38 | 38 | 6656 | 9745 | 46.41 % | 50.00 % | 46.40 % | 0.00 |
| `NearestNeighbour` | qa194 | 194 | 9352 | 11640 | 24.47 % | 35.00 % | 24.50 % | 0.00 |
| `NearestNeighbour` | lu980 | 980 | 11340 | 14370 | 26.72 % | 35.00 % | 26.70 % | 0.00 |
| `OrOpt` | wi29 | 29 | 27603 | 28947 | 4.87 % | 25.00 % | — | 0.00 |
| `OrOpt` | dj38 | 38 | 6656 | 7972 | 19.77 % | 25.00 % | — | 0.00 |
| `OrOpt` | qa194 | 194 | 9352 | 10817 | 15.67 % | 25.00 % | — | 0.00 |
| `OrOpt` | lu980 | 980 | 11340 | 13336 | 17.60 % | 25.00 % | 21.70 % | 0.01 |
| `SOM` | wi29 | 29 | 27603 | 27603 | 0.00 % | 15.00 % | — | 0.12 |
| `SOM` | dj38 | 38 | 6656 | 6660 | 0.06 % | 15.00 % | — | 0.14 |
| `SOM` | qa194 | 194 | 9352 | 9954 | 6.44 % | 15.00 % | — | 0.48 |
| `SOM` | lu980 | 980 | 11340 | 12499 | 10.22 % | — | — | 2.77 |
| `SimulatedAnnealing` | wi29 | 29 | 27603 | 27603 | 0.00 % | 3.00 % | 0.00 % | 0.04 |
| `SimulatedAnnealing` | dj38 | 38 | 6656 | 6656 | 0.00 % | 3.00 % | 0.00 % | 0.04 |
| `SimulatedAnnealing` | qa194 | 194 | 9352 | 9580 | 2.44 % | 10.00 % | 2.44 % | 0.15 |
| `SimulatedAnnealing` | lu980 | 980 | 11340 | 12136 | 7.02 % | 10.00 % | 7.02 % | 1.02 |
| `TabuSearch` | wi29 | 29 | 27603 | 27603 | 0.00 % | 8.00 % | 0.00 % | 0.03 |
| `TabuSearch` | dj38 | 38 | 6656 | 6656 | 0.00 % | 8.00 % | 0.00 % | 0.03 |
| `TabuSearch` | qa194 | 194 | 9352 | 9851 | 5.34 % | 15.00 % | 5.34 % | 0.37 |
| `TabuSearch` | lu980 | 980 | 11340 | 11823 | 4.26 % | 15.00 % | 4.26 % | 1.36 |
| `TwoOpt` | wi29 | 29 | 27603 | 28512 | 3.29 % | 20.00 % | — | 0.00 |
| `TwoOpt` | dj38 | 38 | 6656 | 6660 | 0.06 % | 20.00 % | — | 0.00 |
| `TwoOpt` | qa194 | 194 | 9352 | 10076 | 7.74 % | 20.00 % | — | 0.00 |
| `TwoOpt` | lu980 | 980 | 11340 | 12037 | 6.15 % | 20.00 % | 16.20 % | 0.01 |
| `MultiStart[SimulatedAnnealing, 4]` | wi29 | 29 | 27603 | 27603 | 0.00 % | 3.00 % | — | 0.14 |
| `MultiStart[SimulatedAnnealing, 4]` | dj38 | 38 | 6656 | 6656 | 0.00 % | 3.00 % | — | 0.16 |
| `MultiStart[SimulatedAnnealing, 4]` | qa194 | 194 | 9352 | 9641 | 3.09 % | 10.00 % | — | 0.59 |
| `MultiStart[SimulatedAnnealing, 4]` | lu980 | 980 | 11340 | 12030 | 6.08 % | 10.00 % | — | 4.62 |

Regenerate with `python benchmarks/waterloo.py` (a few minutes on a laptop: the `MILP` proof
of qa194 dominates). Tightening a tolerance is a release-notes item; loosening one requires
the lead's approval and a CHANGELOG line.
