#!/usr/bin/env python3
"""Week 5 Part 3 VERIFY (1): three-panel RTM image.

Reads the already-computed adjoint output

    week5/artifacts/adjoint/image.npy   (float64 [z, x])
    week5/inputs/reflector.json         (velocity perturbation)

and draws:

  1. the known reflector (velocity perturbation, km/s),
  2. the raw signed RTM image (arbitrary units),
  3. the row L2 norm of the image over x = 7..33 for z = 10..33.

All three panels share the depth axis and the same spatial window
x = 7..33, z = 10..33 (inclusive). Grid index ``i`` is drawn at
``i * dx * length_unit_m / 1000`` km, so z = 21 is 2.1 km.

No solver is rerun and no data is synthesised.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_rtm_image.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK5 = Path(__file__).resolve().parents[1]
ADJOINT = WEEK5 / "artifacts" / "adjoint"
IMAGE_NPY = ADJOINT / "image.npy"
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"
OUT_PNG = ADJOINT / "image.png"

# Spatial window shared by the three panels (inclusive grid indices).
X0, X1 = 7, 33
Z0, Z1 = 10, 33


def cell_edges(dx_km, first, last):
    """Physical edges of grid cells ``first..last`` (inclusive)."""
    return (first - 0.5) * dx_km, (last + 0.5) * dx_km


def main():
    image = np.load(IMAGE_NPY, allow_pickle=False)
    experiment = json.loads(EXPERIMENT.read_text())

    nx, nz = int(experiment["nx"]), int(experiment["nz"])
    if image.dtype != np.float64 or image.shape != (nz, nx):
        print(f"FAIL: image.npy is {image.dtype} {image.shape}, "
              f"expected float64 ({nz}, {nx})", file=sys.stderr)
        return 1
    if not np.isfinite(image).all():
        print("FAIL: image.npy contains non-finite values", file=sys.stderr)
        return 1

    dx_km = float(experiment["dx"]) * float(experiment["length_unit_m"]) / 1000.0
    perturbation = np.asarray(experiment["perturbation"], dtype=float)

    # True reflector depth from the largest perturbation row (index 21 -> 2.1 km).
    z_reflector = int(np.argmax(perturbation.max(axis=1)))
    true_km = z_reflector * dx_km

    # Row L2 norm over the restricted x window, one value per depth row.
    window = image[Z0:Z1 + 1, X0:X1 + 1]
    profile = np.linalg.norm(window, axis=1)
    peak_z = Z0 + int(np.argmax(profile))
    peak_km = peak_z * dx_km
    depth_diff_cells = abs(peak_z - z_reflector)
    depth_diff_km = abs(peak_km - true_km)

    x_lo, x_hi = cell_edges(dx_km, X0, X1)
    z_lo, z_hi = cell_edges(dx_km, Z0, Z1)
    extent = (x_lo, x_hi, z_hi, z_lo)  # (left, right, bottom, top), origin upper
    depths_km = np.arange(Z0, Z1 + 1) * dx_km

    pert_window = perturbation[Z0:Z1 + 1, X0:X1 + 1]
    image_limit = float(np.abs(image).max())
    pert_limit = float(np.abs(pert_window).max())

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True,
                             layout="constrained")

    # Panel 1: known reflector.
    pert_artist = axes[0].imshow(
        pert_window, cmap="seismic", vmin=-pert_limit, vmax=pert_limit,
        extent=extent, origin="upper", interpolation="nearest", aspect="auto")
    axes[0].axhline(true_km, color="black", lw=1.2, ls="--")
    axes[0].text(x_lo + 0.04, true_km - 0.06, f"reflector {true_km:.1f} km",
                 color="black", fontsize=8.5, va="bottom", ha="left")
    axes[0].set_title("1. Known reflector")
    axes[0].set_xlabel("Horizontal position (km)")
    axes[0].set_ylabel("Depth (km)")
    fig.colorbar(pert_artist, ax=axes[0], fraction=0.046, pad=0.04,
                 label="Velocity perturbation (km/s)")

    # Panel 2: raw signed RTM image, same window and its own symmetric scale.
    image_artist = axes[1].imshow(
        window, cmap="seismic", vmin=-image_limit, vmax=image_limit,
        extent=extent, origin="upper", interpolation="nearest", aspect="auto")
    axes[1].axhline(true_km, color="black", lw=1.2, ls="--")
    axes[1].set_title("2. Raw signed RTM image")
    axes[1].set_xlabel("Horizontal position (km)")
    fig.colorbar(image_artist, ax=axes[1], fraction=0.046, pad=0.04,
                 label="RTM image amplitude (arbitrary units)")

    # Panel 3: row L2 depth profile.
    axes[2].plot(profile, depths_km, color="#1f4e9c", lw=1.9, label="row L2 norm")
    axes[2].axhline(true_km, color="#00a35a", lw=1.5, ls="--",
                    label=f"true reflector {true_km:.1f} km")
    axes[2].axhline(peak_km, color="#d1495b", lw=1.6, ls=":",
                    label=f"profile peak {peak_km:.1f} km")
    axes[2].plot([profile.max()], [peak_km], marker="o", color="#d1495b", ms=5,
                 zorder=5)
    axes[2].set_title("3. Depth profile")
    axes[2].set_xlabel("Row L2 norm (arbitrary units)")
    axes[2].set_ylabel("Depth (km)")
    axes[2].legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    axes[2].grid(True, color="#d9d9d9", lw=0.6)

    axes[0].set_ylim(z_hi, z_lo)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- numerical verification -------------------------------------------
    left = float(np.sum(np.load(WEEK5 / "artifacts" / "born" / "born_data.npy",
                                allow_pickle=False) ** 2))
    right = float(np.sum(perturbation * image))
    transpose_error = abs(left - right) / max(abs(left), abs(right), 1e-30)

    print("VERIFY (1) — RTM image")
    print(f"  1. image shape / dtype        : {image.shape} / {image.dtype}")
    print(f"  2. image min / max            : {image.min():.6f} / {image.max():.6f}")
    print(f"  3. true reflector depth       : z = {z_reflector} = {true_km:.2f} km")
    print(f"  4. profile peak depth         : z = {peak_z} = {peak_km:.2f} km")
    print(f"  5. |peak - true|              : {depth_diff_cells} cells = "
          f"{depth_diff_km:.3f} km")
    print(f"     transpose identity         : left = {left:.15f}")
    print(f"                                  right = {right:.15f}")
    print(f"                                  relative error = {transpose_error:.3e}"
          f" ({'PASS' if transpose_error < 1e-9 else 'FAIL'})")
    print(f"  wrote {OUT_PNG}")

    checks = [
        ("image is finite float64 [41, 41]",
         image.dtype == np.float64 and image.shape == (41, 41)
         and np.isfinite(image).all()),
        ("image is not identically zero", bool((image != 0).any())),
        ("profile peak within one cell of the reflector",
         depth_diff_cells <= 1),
        ("transpose identity < 1e-9", transpose_error < 1e-9),
    ]
    print("\nVERIFY (1) image checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall RTM image checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
