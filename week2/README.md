# Week 2 — Lennard-Jones fluid

Part 5 first-stage baseline, measured before any cell-list optimization. The
force calculation is still the naive O(N²) pair loop.

## Timing

Contract run: N = 100, rho = 0.8, T = 0.5, dt = 0.01, eq-steps = 2000 +
production steps = 10000 (12 000 velocity-Verlet steps in total). Wall-clock
time, 3 runs each.

| Program | Median (s) | Range: min-max (s) |
| --- | ---: | ---: |
| NumPy week2-sim.py | 17.833 | 17.762–18.136 |
| Rust debug | 5.926 | 5.922–5.934 |
| Rust release | 0.914 | 0.906–0.920 |

## Profile

Where the force stage spends its time at N = 400, 1000 steps (samply,
release build with debug info, 5 kHz sampling, same workload for both rows).
Force share = wall-clock fraction inside the force stage; elapsed time = total
profiled run.

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 98.37 | 0.7631 |
| Cell list | 93.85 | 0.1758 |

The cell-list run is ~4.3× faster in wall time than the naive run at the same
N = 400 workload (0.1758 s vs 0.7631 s).

Profiler evidence: `profile-naive.png` (3,773 samples) and `profile-cells.png`
(886 samples), both flame graphs rendered from the samply profiles.
