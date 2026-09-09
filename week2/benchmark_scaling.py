#!/usr/bin/env python3
"""Part 5 scaling benchmark: naive vs cell-list force methods of `md run`.

Times the release `md` binary for N = 100, 400, 1600 with --eq-steps 100 and
--steps 500 (all other parameters at their defaults) for --force naive and
--force cells, three repeats each, and writes:

    data/scaling-md.json   per-N medians, min/max, speedup = naive/cells
    scaling.png            seconds per step vs N (log-log) for both methods

Usage:
    python3 benchmark_scaling.py

The plot needs matplotlib; the benchmark itself is stdlib-only.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "md", "target", "release", "md")
DATA = os.path.join(HERE, "data", "scaling-md.json")
PNG = os.path.join(HERE, "scaling.png")

NS = [100, 400, 1600]
EQ_STEPS = 100
STEPS = 500
TOTAL_STEPS = EQ_STEPS + STEPS  # seconds-per-step divides by this
REPEATS = 3
METHODS = ["naive", "cells"]


def ensure_binary():
    if not os.path.isfile(MD):
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path",
             os.path.join(HERE, "md", "Cargo.toml"), "--bin", "md"],
            cwd=HERE, check=True,
        )


def wall_time(n, method, tmpdir):
    """One wall-clock run of `md run` in seconds."""
    t0 = time.perf_counter()
    subprocess.run(
        [MD, "run", "--n", str(n), "--eq-steps", str(EQ_STEPS),
         "--steps", str(STEPS), "--force", method, "--out", tmpdir],
        check=True, capture_output=True,
    )
    return time.perf_counter() - t0


def measure(n, method, tmpdir):
    runs = [wall_time(n, method, tmpdir) for _ in range(REPEATS)]
    return {
        "median": statistics.median(runs),
        "min": min(runs),
        "max": max(runs),
        "runs": runs,
    }


def main():
    ensure_binary()
    rows = []
    with tempfile.TemporaryDirectory(prefix="md-scaling-") as tmpdir:
        for n in NS:
            naive = measure(n, "naive", tmpdir)
            cells = measure(n, "cells", tmpdir)
            speedup = naive["median"] / cells["median"]
            rows.append({"n": n, "naive": naive, "cells": cells, "speedup": speedup})
            print(f"N={n:5d}  naive {naive['median']:.4f} s "
                  f"({naive['min']:.4f}-{naive['max']:.4f})   "
                  f"cells {cells['median']:.4f} s "
                  f"({cells['min']:.4f}-{cells['max']:.4f})   "
                  f"speedup {speedup:.2f}")

    payload = {
        "eq_steps": EQ_STEPS,
        "steps": STEPS,
        "total_steps": TOTAL_STEPS,
        "repeats": REPEATS,
        "force": ["naive", "cells"],
        "rows": rows,
    }
    with open(DATA, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", DATA)

    # ---- plot: seconds per step vs N (log-log) ------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, marker, color in [("naive", "o", "#c0392b"), ("cells", "s", "#1e6f9f")]:
        xs = [r["n"] for r in rows]
        ys = [r[method]["median"] / TOTAL_STEPS for r in rows]
        ylo = [r[method]["min"] / TOTAL_STEPS for r in rows]
        yhi = [r[method]["max"] / TOTAL_STEPS for r in rows]
        label = "naive (all pairs)" if method == "naive" else "cells (cell list)"
        ax.plot(xs, ys, marker=marker, color=color, ls="-", lw=1.5,
                label=label)
        lower = [y - lo for y, lo in zip(ys, ylo)]
        upper = [hi - y for y, hi in zip(ys, yhi)]
        ax.errorbar(xs, ys, yerr=[lower, upper], fmt="none", ecolor=color,
                    capsize=3, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of atoms  N")
    ax.set_ylabel(f"seconds per step (median, {TOTAL_STEPS} steps)")
    ax.set_title("md run: seconds per step vs N (naive vs cell list)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
