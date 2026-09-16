# Week 3 — 2D Ising reproducibility guide

This guide regenerates every committed Week 3 result from a clean clone, in
execution order. All shell commands are run from the `week3/` directory unless
stated otherwise.

Simulation data lives in `week3/artifacts/` and `week3/runs/`. Both folders are
generated locally and are intentionally not committed (see section 6), so they
must be recreated with the `ising` commands below before the plotting scripts
will run.

---

## 1. Install `ising`

`ising` is a Rust binary. Build and install it from the crate root:

```bash
cd week3
cargo install --path .
```

Afterwards `ising` is available on `PATH` (normally at `~/.cargo/bin/ising`).
The committed `Cargo.lock` pins the dependency versions.

---

## 2. Generate the simulation data

Every command below writes a `run.json` and a `series.jsonl` into the `--out`
folder. The Metropolis runs cover:

* the heating ramp (also writes `spins.jsonl`),
* the four Part 2 runs (coarse and window ramps for L = 32 and L = 64),
* the two single-temperature runs used by `plot_boltzmann.py`.

The two Part 4 Wolff runs are listed in section 3.

### 2.1 Heating ramp → `spins.jsonl`

```bash
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 \
      --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

The `--every 20` flag records spin frames at measured steps 20, 40, … at each
temperature. The committed `week3/spins.jsonl` is a copy of
`runs/ramp/spins.jsonl` and is what the course viewer reads for the manual
screenshots in section 5.

### 2.2 Four Part 2 Metropolis runs

```bash
# coarse ramp, L = 32
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
      --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32

# coarse ramp, L = 64
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 \
      --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64

# critical-window ramp, L = 32
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
      --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32

# critical-window ramp, L = 64
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
      --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

### 2.3 Two Boltzmann single-temperature Metropolis runs

These are required only by `plot_boltzmann.py`. For a single temperature the
`--t-step` value is irrelevant; `0.1` is used here.

```bash
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 \
      --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0

ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 \
      --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
```

---

## 3. Two Part 4 Wolff runs

The two Part 4 Wolff runs are:

```bash
# Wolff critical-window ramp, L = 32
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
      --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32

# Wolff critical-window ramp, L = 64
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 \
      --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
```

---

## 4. Analysis and plotting scripts

All Python scripts use the course virtual environment. Define the interpreter
once:

```bash
PY=/home/yao_yiyi/.venvs/amat5315/bin/python
```

Run the scripts in this order (all from `week3/`):

| Command | Evidence produced |
|---|---|
| `$PY plot_boltzmann.py` | `evidence/boltzmann.png` |
| `$PY plot_thermo.py` | `evidence/magnetization.png`, `evidence/susceptibility.png` |
| `$PY scripts/peaks.py` | `evidence/peaks.txt` |
| `$PY scripts/errors.py` | `evidence/errors.txt` |
| `$PY scripts/trace.py` | `evidence/trace.png` |
| `$PY scripts/acf_binning.py` | `evidence/acf-binning.png` |
| `$PY scripts/tau.py` | `evidence/tau.png` |
| `$PY scripts/bootstrap.py` | `evidence/chi-bootstrap.png`, `evidence/chi-bootstrap.txt` |
| `$PY scripts/compare_metropolis_wolff.py` | `evidence/magnetization-compare.png`, `evidence/magnetization-compare.txt` |
| `$PY scripts/compare.py` | `evidence/tau-compare.png`, `evidence/tau-compare.txt` |

The exact commands to reproduce each committed evidence file:

```bash
# boltzmann.png (needs runs/T3.0 and runs/T3.1 from section 2.3)
$PY plot_boltzmann.py

# magnetization.png and susceptibility.png
# (needs artifacts/coarse-* and artifacts/window-* from section 2.2)
$PY plot_thermo.py

# peaks.txt (needs artifacts/window-l32 and artifacts/window-l64)
$PY scripts/peaks.py

# errors.txt (needs artifacts/coarse-* and artifacts/window-*)
$PY scripts/errors.py

# trace.png (needs artifacts/window-l64 and artifacts/coarse-l64)
$PY scripts/trace.py

# acf-binning.png (needs artifacts/window-l64)
$PY scripts/acf_binning.py

# tau.png (needs artifacts/coarse-* and artifacts/window-*)
$PY scripts/tau.py

# chi-bootstrap.png and chi-bootstrap.txt (needs artifacts/window-l32/l64)
$PY scripts/bootstrap.py

# magnetization-compare.png and magnetization-compare.txt
# (needs artifacts/window-l64 and artifacts/wolff-l32/l64)
$PY scripts/compare_metropolis_wolff.py

# tau-compare.png and tau-compare.txt
# (needs artifacts/window-l64 and artifacts/wolff-l64)
$PY scripts/compare.py
```

Run all tests with:

```bash
$PY -m pytest -q
```

---

## 5. Viewer screenshots (manual)

The three viewer screenshots are saved by hand from the course viewer; they
cannot be regenerated by a script:

* `evidence/viewer-T1.8.png` — viewer frame at T = 1.8,
* `evidence/viewer-T2.3.png` — viewer frame at T = 2.3,
* `evidence/viewer-T3.0.png` — viewer frame at T = 3.0.

The viewer reads the committed `week3/spins.jsonl`, which is produced by the
heating-ramp command in section 2.1. To reproduce a screenshot, open that file
in the course viewer, navigate to the requested temperature, and save the
image manually with the name above.

---

## 6. Committed vs generated files

* `week3/artifacts/` and `week3/runs/` are generated locally and are
  intentionally **not committed**. They are recreated by the `ising` commands
  in sections 2–3.
* `week3/spins.jsonl` **is committed**; it is a copy of the generated
  `runs/ramp/spins.jsonl` and exists so the viewer screenshots are reproducible
  without keeping the full `runs/` tree.
* All `week3/scripts/*.py`, `week3/plot_*.py`, `week3/test_*.py`, and
  `week3/evidence/*` result files are committed.

---

## 7. Uncertainty conclusions

* **Metropolis sampling error near Tc is unresolved.** The block-bootstrap
  standard error of mean |M| depends on the block length near the critical
  point, so the error is block-length sensitive and the Metropolis-vs-Wolff
  agreement is only provisional there.
* **The Part 2 Tc sampling uncertainty is roughly 0.008–0.010.** Across block
  lengths 2000/4000/8000 the bootstrap standard error of the five-point Tc
  estimate is in that range, but the three block lengths disagree by more than
  10%, so the sampling error is unresolved.
* **Wolff strongly reduces critical slowing down.** At T = 2.3 the
  work-normalized autocorrelation time is about 550 sweeps for Metropolis and
  about 1.0 sweep-equivalent for Wolff.
* **Systematic errors are not included in the bootstrap errors.** Finite-size
  effects and the quadratic five-point peak fit contribute systematic
  uncertainty that the bootstrap sampling error does not capture.
