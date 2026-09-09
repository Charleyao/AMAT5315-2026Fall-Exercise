#!/usr/bin/env python3
"""Plot the Lennard-Jones pair field around one atom.

Energy U(r) is drawn as a color map and the radial force F(r) is drawn as
arrows.  The energy/force values come from the actual `md::energy` and
`md::force` functions in the Rust crate `week2/md` (see
`md/src/bin/field_grid.rs`), so this script never re-implements the physics.

Exact command used to generate the figure
-----------------------------------------
    cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise/week2 && \
        /home/yao_yiyi/.venvs/amat5315/bin/python plot_field.py

The script first builds the Rust sampler
(`cargo build --release --bin field_grid`) and then runs it to sample the
field on a grid.  The output is saved to `week2/field.png`.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK2 = Path(__file__).resolve().parent
MD_DIR = WEEK2 / "md"
SAMPLER = MD_DIR / "target" / "release" / "field_grid"

# Domain and grid sizes.  Even point counts are used on purpose so that the
# grid never contains r = 0 (where U and F diverge).
DOMAIN = (-2.0, 2.0)
COLOR_GRID = 400   # fine grid for the energy color map
ARROW_GRID = 24    # coarse grid for the force arrows


def build_sampler() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "field_grid"],
        cwd=MD_DIR,
        check=True,
    )


def sample_field(nx: int, ny: int) -> np.ndarray:
    """Run the Rust sampler and return an array with columns [x, y, U, Fx, Fy]."""
    xmin, xmax = DOMAIN
    proc = subprocess.run(
        [str(SAMPLER), str(xmin), str(xmax), str(nx), str(xmin), str(xmax), str(ny)],
        cwd=MD_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.loadtxt(io.StringIO(proc.stdout))


def main() -> None:
    build_sampler()

    # ---- Energy color field (fine grid) ---------------------------------
    fine = sample_field(COLOR_GRID, COLOR_GRID)
    nx = ny = COLOR_GRID
    X = fine[:, 0].reshape(ny, nx)
    Y = fine[:, 1].reshape(ny, nx)
    U = fine[:, 2].reshape(ny, nx)

    # Cap the *displayed* potential energy at U = 1 for readability (the true
    # Lennard-Jones potential diverges to +infinity as r -> 0).
    U_display = np.clip(U, -1.0, 1.0)

    # ---- Force arrows (coarse grid) -------------------------------------
    coarse = sample_field(ARROW_GRID, ARROW_GRID)
    nqx = nqy = ARROW_GRID
    Xq = coarse[:, 0].reshape(nqy, nqx)
    Yq = coarse[:, 1].reshape(nqy, nqx)
    Fx = coarse[:, 3].reshape(nqy, nqx)
    Fy = coarse[:, 4].reshape(nqy, nqx)

    # The Rust `force(r)` is a radial scalar (positive = outward for r < r0,
    # negative = inward for r > r0).  The sampler already resolves it into
    # Cartesian components.  Normalize the arrow lengths so the sign/direction
    # is easy to read everywhere; the magnitude varies over many orders of
    # magnitude near the origin.
    magnitude = np.hypot(Fx, Fy)
    safe = magnitude > 0.0
    Fx_unit = np.zeros_like(Fx)
    Fy_unit = np.zeros_like(Fy)
    Fx_unit[safe] = Fx[safe] / magnitude[safe]
    Fy_unit[safe] = Fy[safe] / magnitude[safe]

    arrow_length = 0.25  # in data units

    # ---- Draw ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    im = ax.pcolormesh(
        X,
        Y,
        U_display,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, label="Pair potential energy U (capped at 1)")

    # Equilibrium radius r0 = 2^(1/6), where U(r0) = -1 and F(r0) = 0.
    r0 = 2.0 ** (1.0 / 6.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        r0 * np.cos(theta),
        r0 * np.sin(theta),
        linestyle="--",
        color="white",
        linewidth=1.6,
        label=r"$r_0 = 2^{1/6}$",
    )

    ax.quiver(
        Xq,
        Yq,
        Fx_unit * arrow_length,
        Fy_unit * arrow_length,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        width=0.0035,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=3.5,
        label="Force (direction, normalized)",
    )

    # The central atom.
    ax.plot(0.0, 0.0, "o", color="black", markersize=6, label="Atom")

    ax.set_xlim(DOMAIN)
    ax.set_ylim(DOMAIN)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Lennard-Jones pair field around one atom")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out = WEEK2 / "field.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
