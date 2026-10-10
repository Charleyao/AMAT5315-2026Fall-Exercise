#!/usr/bin/env python3
"""Week 5 Part 3 VERIFY (1): adjoint wavefield snapshot at step 132.

Reads the already-computed reverse-time recording

    week5/artifacts/adjoint/wavefield.npy   (float32 [frame, z, x])
    week5/artifacts/adjoint/run.json        (recording step/time metadata)
    week5/inputs/reflector.json             (survey geometry)

locates the frame whose recorded step is 132 (t = 2.64 s) from ``run.json``
rather than assuming its index, and draws the full 4 km x 4 km domain with a
symmetric blue-white-red scale. Receivers and the true reflector depth are
marked.

The official viewer needs a browser; this environment has no Playwright or
Chromium, so the figure is generated directly from the recorded array. Manual
viewer instructions are printed at the end.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_adjoint_wavefield.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WEEK5 = Path(__file__).resolve().parents[1]
ADJOINT = WEEK5 / "artifacts" / "adjoint"
WAVEFIELD_NPY = ADJOINT / "wavefield.npy"
RUN_JSON = ADJOINT / "run.json"
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"
OUT_PNG = ADJOINT / "wavefield.png"

TARGET_STEP = 132
TARGET_TIME_S = 2.64


def main():
    for path in (WAVEFIELD_NPY, RUN_JSON, EXPERIMENT):
        if not path.is_file():
            print(f"FAIL: {path} does not exist", file=sys.stderr)
            return 1

    wavefield = np.load(WAVEFIELD_NPY, allow_pickle=False)
    run = json.loads(RUN_JSON.read_text())
    experiment = json.loads(EXPERIMENT.read_text())
    recording = run["recording"]

    nx, nz = int(experiment["nx"]), int(experiment["nz"])
    if wavefield.dtype != np.float32 or wavefield.shape[1:] != (nz, nx):
        print(f"FAIL: wavefield.npy is {wavefield.dtype} {wavefield.shape}, "
              f"expected float32 [frame, {nz}, {nx}]", file=sys.stderr)
        return 1

    if TARGET_STEP not in recording["steps"]:
        print(f"FAIL: step {TARGET_STEP} is not recorded", file=sys.stderr)
        return 1
    frame = recording["steps"].index(TARGET_STEP)
    time_reduced = recording["times"][frame]
    time_s = time_reduced * float(experiment["time_unit_s"])

    field = np.asarray(wavefield[frame], dtype=np.float64)
    limit = float(np.abs(field).max())

    dx_km = float(experiment["dx"]) * float(experiment["length_unit_m"]) / 1000.0
    full_extent = (-0.5 * dx_km, (nx - 0.5) * dx_km,
                   (nz - 0.5) * dx_km, -0.5 * dx_km)
    perturbation = np.asarray(experiment["perturbation"], dtype=float)
    z_reflector = int(np.argmax(perturbation.max(axis=1)))
    true_km = z_reflector * dx_km

    receivers = np.asarray(experiment["receivers"], dtype=float)
    receiver_x = receivers[:, 0] * dx_km
    receiver_z = receivers[:, 1] * dx_km

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6.8, 6.2), layout="constrained")
    artist = ax.imshow(field, cmap="bwr", vmin=-limit, vmax=limit,
                       extent=full_extent, origin="upper", interpolation="nearest",
                       aspect="equal")
    ax.plot(receiver_x, receiver_z, marker="v", color="black", ms=6,
            linestyle="none", label="receivers", zorder=5)
    ax.axhline(true_km, color="#00a35a", lw=1.5, ls="--", zorder=4)
    ax.text((nx - 1) * dx_km, true_km - 0.06, f"reflector {true_km:.1f} km",
            color="#00713c", ha="right", va="bottom", fontsize=8.5, zorder=6)

    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"Adjoint field at step {TARGET_STEP}; {time_s:.2f} s")
    ax.set_xticks(np.arange(0.0, 4.01, 0.5))
    ax.set_yticks(np.arange(0.0, 4.01, 0.5))
    ax.set_xlim(full_extent[0], full_extent[1])
    ax.set_ylim(full_extent[2], full_extent[3])
    handles = [
        Line2D([], [], marker="v", color="black", linestyle="none",
               markersize=6, label="receivers"),
        Line2D([], [], color="#00a35a", ls="--", label="true reflector"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04,
                 label="Pressure adjoint (arbitrary units)")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- reverse-propagation sanity ---------------------------------------
    steps = recording["steps"]
    ordered = all(steps[i] > steps[i + 1] for i in range(len(steps) - 1))
    receiver_row = int(np.asarray(experiment["receivers"], dtype=float)[0, 1])
    receiver_energy = float(np.abs(wavefield[:, receiver_row, :]).max())
    frame_change = float(np.abs(np.diff(
        np.asarray(wavefield, dtype=np.float64), axis=0)).mean())

    print("VERIFY (1) — adjoint wavefield")
    print(f"  wavefield shape / dtype     : {wavefield.shape} / {wavefield.dtype}")
    print(f"  frame for step {TARGET_STEP}         : index {frame}")
    print(f"  recorded time               : {time_reduced:g} reduced units = "
          f"{time_s:.3f} s")
    print(f"  step-132 min / max          : {field.min():.6f} / {field.max():.6f}")
    print(f"  recorded frames             : {len(steps)} (step {steps[0]}..{steps[-1]})")
    print(f"  frames ordered decreasing   : {ordered}")
    print(f"  max |field| on receiver row : {receiver_energy:.3e}")
    print(f"  mean frame-to-frame change  : {frame_change:.3e}")
    print("  frame   step    t(s)     max|field|   depth of max (km)")
    for index in range(0, len(steps), 10):
        snapshot = np.asarray(wavefield[index], dtype=np.float64)
        row = int(np.unravel_index(np.argmax(np.abs(snapshot)), snapshot.shape)[0])
        print(f"  {index:>5}   {steps[index]:>4}   "
              f"{recording['times'][index] * float(experiment['time_unit_s']):>5.2f}   "
              f"{np.abs(snapshot).max():>10.3e}   {row * dx_km:>8.2f}")
    print(f"  wrote {OUT_PNG}")

    print("\nmanual course-viewer instructions (browser required):")
    print("  cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise")
    print("  python3 -m http.server 8000")
    print("  open http://localhost:8000/week5/viewer/index.html"
          "?src=../artifacts/adjoint/wavefield.npy")
    print(f"  run.json is read from the same folder; set the Frame slider to "
          f"{frame} (step {TARGET_STEP}, {time_s:.2f} s), then Save PNG")

    checks = [
        ("step 132 is recorded", TARGET_STEP in recording["steps"]),
        ("physical time is 2.64 s", abs(time_s - TARGET_TIME_S) < 1e-9),
        ("frames are in decreasing time", ordered),
        ("receiver row has injected energy", receiver_energy > 0.0),
        ("adjoint field changes between frames", frame_change > 0.0),
        ("step-132 field is finite and non-zero",
         np.isfinite(field).all() and bool((field != 0).any())),
    ]
    print("\nVERIFY (1) wavefield checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall adjoint-wavefield checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
