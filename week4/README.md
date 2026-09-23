# Week 4 — 2D incompressible flow (vorticity–streamfunction, Fourier pseudospectral)

This week builds, test by test, a periodic 2D incompressible-flow solver and its
evidence:

* **Part 1** — a 1D advection–diffusion toy: Euler / midpoint / classical RK4,
  periodic centred differences, Fourier spectral derivatives, stability and
  accuracy figures.
* **Part 2** — the 2D `field`/`fluid` CLIs, Taylor-Green t = 0 → 1 verification.
* **Part 3** — seeded random flow, stability-limit scan, sensitivity vs blow-up.
* **Part 4** — Taylor-Green RK4 order and random-flow self-convergence.

Every number quoted in the evidence is produced by the code here; artifacts are
regenerated locally (see below).

## 0. Prerequisites and installation

* Rust toolchain (edition 2024; verified with `rustc 1.98`).
* Python with `numpy` and `matplotlib`. The course environment is
  `/home/yao_yiyi/.venvs/amat5315/bin/python`.

Install the CLIs (and the small Rust data generators) from `week4/`:

```sh
cd week4
cargo install --path . --quiet
```

This installs `field`, `fluid`, `line-stability-data`, `line-accuracy-data` and
`sensitivity-prep` into `~/.cargo/bin`. The shell scripts fall back to
`cargo run --quiet --bin ...` when the binaries are not on `PATH`.

> `week4/artifacts/` is **gitignored** (the repository ignores `artifacts/`).
> It is regenerated locally by the commands below; nothing under it is tracked.

## 1. The two CLIs (contracts in the design files)

`week4/field.design.toml` and `week4/fluid.design.toml` are the frozen
contracts. They are unchanged from the starter kit.

```sh
# field: write a velocity field to stdout as one JSON object
field taylor-green --n 64 --nu 0.1 --t 1
field random --n 128 --seed 2026 --k-min 2 --k-max 6

# fluid: read field JSON on stdin, integrate, write snapshots
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
          --out artifacts/taylor-green
```

`fluid` writes `t<TAB>E<TAB>Z` to stdout, `<out>/run.json` and
`<out>/fields.jsonl` (values to 6 decimals; integrated at full precision).
A non-finite energy prints the offending line and exits 1.

## 2. Main pipelines (exact parameters)

**Taylor-Green, t = 0 → 1**

```sh
field taylor-green --n 64 \
  | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
          --out artifacts/taylor-green \
  > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
```

**Seeded random flow, t = 0 → 10**

```sh
field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
          --out artifacts/random \
  > artifacts/random.tsv
```

All experiments use **fixed** timesteps and never shorten a step for a
snapshot: snapshots are taken at step 0 and every `round(every/dt)` steps.

## 3. Evidence files and how to regenerate them

Run the Python scripts from `week4/scripts/` with the course interpreter, e.g.
`PY=/home/yao_yiyi/.venvs/amat5315/bin/python`. The Rust generators are launched
automatically by the Python scripts.

| Evidence | Generator | Prerequisite |
|---|---|---|
| `evidence/line-stability.png` (+ `.txt`) | `$PY line_stability.py` | none (builds/runs the `line-stability-data` bin) |
| `evidence/line-accuracy.png` (+ `.txt`) | `$PY line_accuracy.py` | none (runs the `line-accuracy-data` bin) |
| `evidence/taylor-green.png` (+ `.txt`) | `bash run_taylor_green.sh` then `$PY taylor_green_figure.py` | `artifacts/taylor-green/` |
| `evidence/random.png` (+ `.txt`) | `bash run_random.sh` then `$PY random_figure.py` | `artifacts/random/` |
| `evidence/blowup.png` (+ `.txt`) | `bash run_scan.sh` then `$PY blowup_figure.py` | `artifacts/scan/`, `artifacts/unstable/` |
| `evidence/sensitivity.png` (+ `.txt`) | `bash run_sensitivity.sh` then `$PY sensitivity_plot.py` | `artifacts/sensitivity/` |
| `evidence/order.png` (+ `.txt`) | `bash run_order.sh` then `$PY order_plot.py` | `artifacts/order/` |
| `evidence/convergence.png`, `evidence/convergence.json` (+ `.txt`) | `bash run_convergence.sh` then `$PY convergence_plot.py` | `artifacts/convergence/` |

Read-only consistency check (no simulation): `$PY check_convergence_selection.py`.
`probe_random_rk4.sh` is the exploration helper that located the random RK4
stability boundary; it is not needed to regenerate the evidence.

One-shot regeneration, in order:

```sh
cd week4 && cargo install --path . --quiet
cd scripts
PY=/home/yao_yiyi/.venvs/amat5315/bin/python
$PY line_stability.py
$PY line_accuracy.py
bash run_taylor_green.sh && $PY taylor_green_figure.py
bash run_random.sh        && $PY random_figure.py
bash run_scan.sh          && $PY blowup_figure.py
bash run_sensitivity.sh   && $PY sensitivity_plot.py
bash run_order.sh         && $PY order_plot.py
bash run_convergence.sh   && $PY convergence_plot.py
$PY check_convergence_selection.py
```

## 4. Tests and checker

```sh
cd week4
cargo test
```

64 tests pass (43 library, 10 `field`, 4 `fluid`, 3 `field_cli`, 4 `fluid_cli`).

The provided physics gate is run from the repository root (it is stdlib-only):

```sh
python3 week4/checker/check week4
```

It checks the contract artifacts (`artifacts/taylor-green`, `artifacts/random`,
`artifacts/order/<dir>` ×3, `artifacts/unstable/taylor-green`), so those must be
regenerated first. Last run: **PASS**, exit 0.

## 5. Measured results worth keeping honest

* Taylor-Green t = 1: `E = 0.167580`, `Z = 0.335160` (exact `0.25 e^{-0.4}`,
  `0.5 e^{-0.4}`); relative velocity error `7.04e-07`.
* Taylor-Green RK4 order: fitted `p = 4.1044`.
* Stability boundaries: predicted diffusive limit `0.03158` for Taylor-Green
  (bracketed by `0.032`/`0.033`); predicted random advective bound `0.02075`,
  measured random boundary `0.030` (stable) / `0.032` (unstable).
* Sensitivity at `dt = 0.01`: random perturbation grows `≈ 63x` by `t = 20`
  while both runs stay numerically stable.
* **Random-flow self-convergence:** fitted slope `q ≈ 4.0356`; the Richardson
  rule (predicted error `< 5e-6`) selects **`dt = 0.01`** for this random
  realization, because `dt = 0.0125` has predicted error `5.66e-6`, just over
  the threshold. The answer-key realization selects `0.0125`; the result here is
  kept as measured and **not** replaced by `0.0125`.
