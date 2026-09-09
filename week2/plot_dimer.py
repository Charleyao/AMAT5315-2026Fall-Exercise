#!/usr/bin/env python3
"""Plot the two-atom Lennard-Jones dimer relative total-energy error.

The error histories come from the actual Rust integrators in the `md` crate
(`week2/md`). This script never re-implements the physics: it builds the
`dimer` helper binary (`md/src/bin/dimer.rs`), runs it, and plots the output.

Initial state (shared by every run):
    positions  = [[0.0, 0.0], [1.2, 0.0]]
    velocities = [[0.0, 0.0], [0.0, 0.0]]
    dt         = 0.01

Panels:
    left  — relative total-energy error (E(t) - E0) / |E0| for both
            ForwardEuler and VelocityVerlet over 500 steps.
    right — VelocityVerlet only over 5000 steps, with the relative energy
            error multiplied by 1000.

Exact command used to generate the figure
-----------------------------------------
    cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise/week2 && \
        /home/yao_yiyi/.venvs/amat5315/bin/python plot_dimer.py

The script first builds the Rust sampler/runner
(`cargo build --release --bin dimer`) and then runs it. The output is saved
to `week2/dimer.png`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK2 = Path(__file__).resolve().parent
MD_DIR = WEEK2 / "md"
RUNNER = MD_DIR / "target" / "release" / "dimer"

DT = 0.01


def build_runner() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "dimer"],
        cwd=MD_DIR,
        check=True,
    )


def run_dimer() -> dict[tuple[str, int], np.ndarray]:
    """Run the Rust helper and parse its three error-history blocks.

    Each block is:
        <method> <steps>
        <index> <relative_error>
        ...
    Returns a mapping {(method, steps): error_array}.
    """
    proc = subprocess.run(
        [str(RUNNER)],
        cwd=MD_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = proc.stdout.splitlines()
    data: dict[tuple[str, int], np.ndarray] = {}
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if not header:
            i += 1
            continue
        method, steps = header[0], int(header[1])
        i += 1
        values = []
        for _ in range(steps + 1):
            parts = lines[i].split()
            values.append(float(parts[1]))
            i += 1
        data[(method, steps)] = np.asarray(values)
    return data


def main() -> None:
    build_runner()
    data = run_dimer()

    euler_500 = data[("ForwardEuler", 500)]
    vv_500 = data[("VelocityVerlet", 500)]
    vv_5000 = data[("VelocityVerlet", 5000)]

    steps_500 = np.arange(euler_500.size)
    steps_5000 = np.arange(vv_5000.size)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # ---- Left: both integrators, 500 steps --------------------------------
    ax_left.plot(steps_500, euler_500, label="ForwardEuler", linewidth=1.4)
    ax_left.plot(
        steps_500,
        vv_500,
        label="VelocityVerlet",
        linewidth=1.4,
    )
    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Relative total-energy error\n$(E(t) - E_0) / |E_0|$")
    ax_left.set_title("Both integrators, 500 steps")
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    # ---- Right: VelocityVerlet only, 5000 steps, x1000 --------------------
    ax_right.plot(
        steps_5000,
        vv_5000 * 1000.0,
        label="VelocityVerlet",
        linewidth=1.4,
    )
    ax_right.set_xlabel("Step")
    ax_right.set_ylabel(r"$1000 \cdot (E(t) - E_0) / |E_0|$")
    ax_right.set_title("VelocityVerlet, 5000 steps (error × 1000)")
    ax_right.legend()
    ax_right.grid(True, alpha=0.3)

    fig.suptitle("Two-atom Lennard-Jones dimer: relative total-energy error", y=1.02)
    fig.tight_layout()

    out = WEEK2 / "dimer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
