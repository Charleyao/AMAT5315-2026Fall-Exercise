# Week 2 — Lennard-Jones fluid

Part 5 first-stage baseline, measured before any cell-list optimization. The
force calculation is still the naive O(N²) pair loop.

## Timing

Contract run: N = 100, rho = 0.8, T = 0.5, dt = 0.01, eq-steps = 2000 +
production steps = 10000 (12 000 velocity-Verlet steps in total). Wall-clock
time, 3 runs each.

|Program|Median (s)|Range: min-max (s)|
|-|-:|-:|
|NumPy week2-sim.py|17.833|17.762–18.136|
|Rust debug|5.926|5.922–5.934|
|Rust release|0.914|0.906–0.920|

## Profile

Where the force stage spends its time at N = 400, 1000 steps (samply,
release build with debug info, 5 kHz sampling, same workload for both rows).
Force share = wall-clock fraction inside the force stage; elapsed time = total
profiled run.

|Version|Force share (%)|Elapsed time (s)|
|-|-:|-:|
|Naive|98.37|0.7631|
|Cell list|93.85|0.1758|

The cell-list run is \~4.3× faster in wall time than the naive run at the same
N = 400 workload (0.1758 s vs 0.7631 s).

Profiler evidence: `profile-naive.png` (3,773 samples) and `profile-cells.png`
(886 samples), both flame graphs rendered from the samply profiles.

## Benchmark

Scaling of `md run` at fixed density (rho = 0.8), `--force naive` vs
`--force cells`: wall time of the full run (release build, `--steps 500`,
`--eq-steps 100`, all other defaults unchanged), 3 runs each.
`speedup = naive median / cells median`.

|N|Naive (s)|Cells (s)|Speedup|
|-:|-:|-:|-:|
|100|0.0397 (0.0392–0.0406)|0.0266 (0.0265–0.0270)|1.50|
|400|0.5542 (0.5534–0.5867)|0.1227 (0.0980–0.1243)|4.52|
|1600|8.4077 (8.2445–8.5603)|0.3879 (0.3808–0.4008)|21.68|

Median and min–max range are shown. Speedup grows with N and reaches \~22× at
N = 1600. The two curves in `scaling.png` (seconds per step vs N, log–log)
have different slopes because the naive path searches all O(N²) pairs, while
the cell-list path keeps the candidate count per atom bounded at fixed density
and cutoff. Reproduce with:

```sh
python3 benchmark\_scaling.py   # needs matplotlib for scaling.png
```

## GitHub Pages

The heating trajectory (T: 0.2 → 1.2, N = 400, 200 frames) is published with
the official trajectory viewer from the repository's `docs/` folder:

[**https://charleyao.github.io/AMAT5315-2026Fall-Exercise/**](https://charleyao.github.io/AMAT5315-2026Fall-Exercise/)

`docs/index.html` is a copy of `viewer/index.html`, sitting next to
`docs/run.json` and `docs/traj.jsonl` (the viewer loads `./traj.jsonl` +
`./run.json` relative to the page). GitHub Pages is configured to publish
folder `/docs` on branch `main` (repo Settings → Pages).



**## GitHub Pages**



Viewer:

https://charleyao.github.io/AMAT5315-2026Fall-Exercise/



Final recording:

https://github.com/Charleyao/AMAT5315-2026Fall-Exercise/releases/tag/week2-final





## Reproduce

Commands below assume the repository root; `MD` is the release binary.

```sh
cargo build --release --manifest-path week2/md/Cargo.toml --bin md
MD=week2/md/target/release/md
```

* **Timing** (contract run, NumPy vs Rust debug/release):
`python3 week2/data/benchmark.py` → `week2/data/timing.json` (median/min/max).
* **Profile** (samply at N = 400, 1000 steps — the README table's numbers):

```sh
  cd week2/data/profile \&\& cargo build --release
  samply record --save-only --rate 5000 -o naive-n400.json.gz \\
    ./target/release/md-profile 400 1000 naive
  samply record --save-only --rate 5000 -o cells-n400.json.gz \\
    ./target/release/md-profile 400 1000 cells
  ```

  Force share and elapsed time print on the run's stdout; the flame graphs
`profile-naive.png` / `profile-cells.png` are rendered from those two files.

* **scaling.png**: `python3 week2/benchmark\_scaling.py` (matplotlib needed)
→ `week2/scaling.png` + `week2/data/scaling-md.json`.
* **cold.mp4 / hot.mp4**: fixed-temperature reference runs and videos:

```sh
  $MD run --temperature 0.2 --out /tmp/cold
  $MD video /tmp/cold --out week2/cold.mp4
  $MD run --temperature 1.0 --out /tmp/hot
  $MD video /tmp/hot --out week2/hot.mp4
  ```

* **Heating trajectory** (regenerates `docs/`):

```sh
  $MD run --n 400 --temperature 0.2 --ramp-to 1.2 \\
    --steps 20000 --sample-every 100 --out docs
  cp week2/viewer/index.html docs/index.html
  ```

