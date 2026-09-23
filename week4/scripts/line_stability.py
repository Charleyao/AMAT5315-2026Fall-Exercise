#!/usr/bin/env python3
"""Part 1 stability evidence: ``week4/evidence/line-stability.png``.

Left  : RK4 growth map measured from the library RK4 (2-D real embedding of
        ``y' = lambda y``), the analytic ``|R(z)| = 1`` boundaries of Euler,
        midpoint and RK4, and the line spectrum ``z_k = lambda_k dt`` for
        ``dt = 0.045`` and ``dt = 0.056``.
Middle: periodised-Gaussian pulse integrated with the library Fourier
        advection-diffusion RHS and RK4 at ``dt = 0.045`` (below the limit).
Right : the same at ``dt = 0.056`` (above the limit), showing the growing
        short-wavelength instability.

The heavy lifting (measured growth map, axis crossings, ``dt_crit``, pulse
histories) is done by the crate binary ``scripts/line_stability_data.rs``, which
links the real library. This script only drives it and draws.

Run from ``week4/scripts/`` with the course environment::

    /home/yao_yiyi/.venvs/amat5315/bin/python line_stability.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent  # week4/scripts
WEEK4 = HERE.parent                      # week4
EVID = WEEK4 / "evidence"
DATA = WEEK4 / "artifacts" / "line-stability"  # generated, gitignored

DTS = ("0.045", "0.056")


def stability_function(name: str, z: np.ndarray) -> np.ndarray:
    """Analytic R(z) for the contour overlay only; the colour map uses the library."""
    if name == "euler":
        return 1.0 + z
    if name == "midpoint":
        return 1.0 + z + z**2 / 2.0
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def generate_data() -> None:
    """Build and run the Rust data generator (links the library)."""
    subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--bin", "line-stability-data"],
        cwd=WEEK4,
        check=True,
    )


def main() -> None:
    generate_data()
    data = json.loads((DATA / "line-stability-data.json").read_text())

    nx, ny = data["nx"], data["ny"]
    growth = np.fromfile(DATA / "line-stability-growth.bin", dtype="<f8").reshape(ny, nx)
    re = np.linspace(data["re_min"], data["re_max"], nx)
    im = np.linspace(data["im_min"], data["im_max"], ny)
    re_grid, im_grid = np.meshgrid(re, im)
    z_grid = re_grid + 1j * im_grid

    n = data["n"]
    x = 2.0 * np.pi * np.arange(n) / n

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.28, 1, 1])
    )

    # ---- left: measured growth map + boundaries + spectrum --------------------
    ax = axes[0]
    mesh = ax.pcolormesh(
        re, im, growth, norm=LogNorm(vmin=0.3, vmax=3.0), cmap="RdBu_r",
        shading="auto", rasterized=True,
    )
    ax.contour(re, im, np.abs(stability_function("rk4", z_grid)), levels=[1.0],
               colors="k", linewidths=1.2)
    for name, colour in (("euler", "0.55"), ("midpoint", "0.30")):
        ax.contour(re, im, np.abs(stability_function(name, z_grid)), levels=[1.0],
                   colors=colour, linewidths=0.8, linestyles="--")

    spectrum = data["spectrum"]
    spec_markers = {"0.045": ("tab:green", "o"), "0.056": ("tab:red", "^")}
    for dt in DTS:
        pts = np.asarray(spectrum[dt])
        colour, marker = spec_markers[dt]
        ax.plot(pts[:, 0], pts[:, 1], marker, color=colour, ms=3.6, mec="k", mew=0.3,
                ls="none")

    ax.axhline(0.0, color="k", lw=0.4)
    ax.axvline(0.0, color="k", lw=0.4)
    ax.set_xlabel(r"Re $\lambda h$")
    ax.set_ylabel(r"Im $\lambda h$")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    handles = [
        Line2D([], [], color="k", lw=1.2, label=r"RK4 $|R|=1$"),
        Line2D([], [], color="0.55", lw=0.8, ls="--", label=r"Euler $|R|=1$"),
        Line2D([], [], color="0.30", lw=0.8, ls="--", label=r"midpoint $|R|=1$"),
        Line2D([], [], color="tab:green", marker="o", ls="none", ms=4,
               label=r"$h$ = 0.045"),
        Line2D([], [], color="tab:red", marker="^", ls="none", ms=4,
               label=r"$h$ = 0.056"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.6,
              title=r"line $z_k = \lambda_k h$", title_fontsize=7.6)

    ax.text(
        0.03, 0.03,
        "computed $h_{crit}$ = %.4f\nRK4 real crossing = %.3f\nRK4 imag crossing = %.3f"
        % (data["dt_crit"], data["rk4_real_crossing"], data["rk4_imag_crossing"]),
        transform=ax.transAxes, fontsize=7.4, va="bottom", ha="left",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.4, alpha=0.92),
    )

    cb = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cb.set_label("measured RK4 growth per step", size=9)
    cb.ax.tick_params(labelsize=8)

    # ---- middle/right: Gaussian pulse below and above the limit ---------------
    for ax, dt in zip(axes[1:], DTS):
        run = data["pulse"][dt]
        nt = run["nt"]
        hist = np.fromfile(DATA / f"line-pulse-{dt}.bin", dtype="<f8").reshape(nt, n)
        times = np.asarray(run["times"])
        ax.pcolormesh(x, times, np.clip(hist, -1.0, 1.0), cmap="RdBu_r",
                      vmin=-1.0, vmax=1.0, shading="auto", rasterized=True)
        ax.invert_yaxis()  # t increasing downwards
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.text(
            0.03, 0.96, r"$h$ = %s" % dt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left",
            bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
        )

    fig.tight_layout()
    out = EVID / "line-stability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ---- evidence text --------------------------------------------------------
    lines = [
        "Part 1 line stability",
        f"n = {n}, c = {data['c']}, nu = {data['nu']}",
        f"RK4 real-axis crossing  = {data['rk4_real_crossing']:.6f}",
        f"RK4 imaginary crossing  = {data['rk4_imag_crossing']:.6f}",
        f"RK4 line dt_crit        = {data['dt_crit']:.6f}",
    ]
    for dt in DTS:
        mg = data["max_growth"][dt]
        lines.append(
            f"dt = {dt}: max_k |R(lambda_k dt)| = {mg['max']:.6f}, "
            f"dominant k = {mg['dominant_k']:.0f}"
        )
    lines.append(
        f"Nyquist fourier_d1 max (library check) = {data['nyquist_d1_max']:.3e}"
    )
    lines.append("figure: evidence/line-stability.png")
    (EVID / "line-stability.txt").write_text("\n".join(lines) + "\n")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
