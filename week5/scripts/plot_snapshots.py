#!/usr/bin/env python3
"""Week 5 Part 2 VERIFY (3): forward wavefield and reflector echo snapshots.

Reads the Rust recordings

    week5/artifacts/forward/wavefield.npy   (float32 [frame, z, x])
    week5/artifacts/forward/echo.npy        (float32 [frame, z, x])
    week5/artifacts/forward/run.json        (recording step/time metadata)

and draws the first shot at simulation step 150 (t = 3.00 s):

    week5/artifacts/forward/wavefield.png
    week5/artifacts/forward/echo.png

The frame index is looked up in ``run.json`` rather than assumed. The echo is
the already-recorded difference "background + perturbation" minus "background".
No solver is re-implemented and no data is synthesised.

The official viewer needs a browser; this environment has neither Playwright
nor Chromium, so the two figures are generated directly from the recorded
arrays. Launch instructions for the viewer are printed for manual inspection.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_snapshots.py
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
FORWARD = WEEK5 / "artifacts" / "forward"
WAVEFIELD_NPY = FORWARD / "wavefield.npy"
ECHO_NPY = FORWARD / "echo.npy"
RUN_JSON = FORWARD / "run.json"
WAVEFIELD_PNG = FORWARD / "wavefield.png"
ECHO_PNG = FORWARD / "echo.png"

TARGET_STEP = 150
TARGET_TIME_S = 3.00

# Approximate visual reference values from the learning sheet.
REFERENCE_WAVEFIELD = 0.26
REFERENCE_ECHO = 0.006
REFERENCE_RATIO = 0.02
ECHO_EARLY_FRACTION = 0.01


def draw_snapshot(field, experiment, title, out_path, color_limit,
                  reflector_km=None):
    nx, nz = experiment["nx"], experiment["nz"]
    dx_km = experiment["dx"] * experiment["length_unit_m"] / 1000.0
    width_km = (nx - 1) * dx_km
    depth_km = (nz - 1) * dx_km
    length_km = experiment["length_unit_m"] / 1000.0

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6.6, 5.8), layout="constrained")
    image = ax.imshow(field, cmap="bwr", vmin=-color_limit, vmax=color_limit,
                      extent=(0.0, width_km, depth_km, 0.0), origin="upper",
                      interpolation="nearest", aspect="equal")

    source_x, source_z = (value * length_km for value in experiment["shots"][0])
    receiver_x = [value[0] * length_km for value in experiment["receivers"]]
    receiver_z = [value[1] * length_km for value in experiment["receivers"]]
    ax.plot([source_x], [source_z], marker="x", color="black", ms=13,
            mew=3.5, linestyle="none", zorder=5)
    ax.plot([source_x], [source_z], marker="x", color="#ffe16e", ms=8,
            mew=2.0, linestyle="none", zorder=6)
    ax.plot(receiver_x, receiver_z, marker="v", color="white",
            markeredgecolor="black", markeredgewidth=0.6, ms=5,
            linestyle="none", zorder=5)

    if reflector_km is not None:
        ax.axhline(reflector_km, color="#00d26a", lw=1.4, ls="--", zorder=4)
        ax.text(width_km * 0.99, reflector_km - 0.06,
                f"reflector z ≈ {reflector_km:.1f} km", color="#007a3d",
                ha="right", va="bottom", fontsize=8.5, zorder=6)

    ax.set_xlabel("x (km)")
    ax.set_ylabel("depth z (km)")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, width_km + 0.1, 1.0))
    ax.set_yticks(np.arange(0, depth_km + 0.1, 1.0))

    handles = [
        Line2D([], [], marker="x", color="#ffe16e", markeredgecolor="black",
               linestyle="none", markersize=8, label="source (shot 0)"),
        Line2D([], [], marker="v", color="white", markeredgecolor="black",
               linestyle="none", markersize=6, label="receivers"),
    ]
    if reflector_km is not None:
        handles.append(Line2D([], [], color="#00d26a", ls="--",
                              label="reflector"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.85,
              fontsize=8.5)

    fig.colorbar(image, ax=ax, label="pressure u (reduced units)",
                 shrink=0.9, pad=0.02)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    for path in (WAVEFIELD_NPY, ECHO_NPY, RUN_JSON):
        if not path.is_file():
            print(f"FAIL: {path} does not exist", file=sys.stderr)
            return 1

    run = json.loads(RUN_JSON.read_text())
    recording = run["recording"]
    experiment = run["experiment"]
    wavefield = np.load(WAVEFIELD_NPY, allow_pickle=False)
    echo = np.load(ECHO_NPY, allow_pickle=False)

    if TARGET_STEP not in recording["steps"]:
        print(f"FAIL: step {TARGET_STEP} is not recorded", file=sys.stderr)
        return 1
    frame = recording["steps"].index(TARGET_STEP)
    time_reduced = recording["times"][frame]
    time_s = time_reduced * experiment["time_unit_s"]

    print("existing recordings:")
    print(f"  wavefield.npy : {wavefield.dtype} {wavefield.shape}")
    print(f"  echo.npy      : {echo.dtype} {echo.shape}")
    print(f"  run.json      : every {recording['every']} steps, "
          f"{len(recording['steps'])} frames, "
          f"steps {recording['steps'][0]}..{recording['steps'][-1]}")
    print(f"  step {TARGET_STEP} -> frame index {frame}, "
          f"t = {time_reduced:g} reduced units = {time_s:.3f} s")

    forward_frame = np.asarray(wavefield[frame], dtype=np.float64)
    echo_frame = np.asarray(echo[frame], dtype=np.float64)
    max_forward = float(np.abs(forward_frame).max())
    max_echo = float(np.abs(echo_frame).max())
    ratio = max_echo / max_forward

    print("\nnumerical verification at step 150:")
    print(f"  max |wavefield| : {max_forward:.6f}  "
          f"(reference ~{REFERENCE_WAVEFIELD})")
    print(f"  max |echo|      : {max_echo:.6e}  "
          f"(reference ~{REFERENCE_ECHO})")
    print(f"  echo ratio      : {ratio:.6f} ({100 * ratio:.2f}% of the direct "
          f"wave, reference ~{100 * REFERENCE_RATIO:.0f}%)")
    peak_z = int(np.unravel_index(np.argmax(np.abs(echo_frame)),
                                  echo_frame.shape)[0])
    dx_km = experiment["dx"] * experiment["length_unit_m"] / 1000.0
    print(f"  echo peak depth : {peak_z * dx_km:.2f} km "
          f"(reflector at ~2.1 km, receivers at 0.8 km)")

    print("\necho growth over the recording "
          "(max|echo| and its fraction of max|wavefield|):")
    for step in (3, 30, 60, 75, 90, 105, 120, 150, 180, 240):
        index = recording["steps"].index(step)
        echo_max = float(np.abs(echo[index]).max())
        field_max = float(np.abs(wavefield[index]).max())
        fraction = echo_max / field_max if field_max else 0.0
        t_s = recording["times"][index] * experiment["time_unit_s"]
        print(f"  step {step:3d}  t={t_s:4.2f} s  max|echo|={echo_max:.3e}  "
              f"max|wavefield|={field_max:.3e}  echo/wavefield={fraction:.2e}")

    draw_snapshot(forward_frame, experiment,
                  "Forward wavefield; shot 0, t = 3.00 s",
                  WAVEFIELD_PNG, color_limit=max_forward)
    draw_snapshot(echo_frame, experiment,
                  "Reflector echo; shot 0, t = 3.00 s",
                  ECHO_PNG, color_limit=max_echo, reflector_km=2.1)
    print(f"\nwrote {WAVEFIELD_PNG}")
    print(f"wrote {ECHO_PNG}")

    print("\nmanual course-viewer instructions (browser required):")
    print("  cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise")
    print("  python3 -m http.server 8000")
    print("  open http://localhost:8000/week5/viewer/index.html"
          "?src=../artifacts/forward/wavefield.npy")
    print("  set the Frame slider to 49 (step 150), click 'Save PNG' or press S")
    print("  then open the same address with echo.npy and save again")

    # --- verification -------------------------------------------------------
    early_step = 75  # t = 1.50 s
    early_index = recording["steps"].index(early_step)
    early_fraction = (float(np.abs(echo[early_index]).max())
                      / float(np.abs(wavefield[early_index]).max()))

    checks = [
        ("frame step is 150", recording["steps"][frame] == TARGET_STEP),
        ("frame time is 3.00 s", abs(time_s - TARGET_TIME_S) < 1e-9),
        ("max forward pressure near 0.26",
         abs(max_forward - REFERENCE_WAVEFIELD) < 0.05),
        ("max echo pressure near 0.006",
         abs(max_echo - REFERENCE_ECHO) < 0.004),
        ("echo ratio near 2%",
         abs(ratio - REFERENCE_RATIO) < 0.02),
        ("echo nearly blank through t = 1.5 s",
         early_fraction < ECHO_EARLY_FRACTION),
        ("echo peak sits between receivers and reflector",
         0.8 <= peak_z * dx_km <= 2.1),
    ]
    print("\nVERIFY (3) checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall VERIFY (3) checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
