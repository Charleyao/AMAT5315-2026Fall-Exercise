#!/usr/bin/env python3
"""Week 5 Part 4 VERIFY (3): Marmousi imaging figure and verification.

Reads the official input and the already-computed outputs

    week5/inputs/marmousi.json
    week5/artifacts/marmousi-born/born_data.npy
    week5/artifacts/marmousi-image/image.npy
    week5/artifacts/marmousi-image/result.json
    week5/artifacts/marmousi-image/actions-<shot>.json

and draws four panels: the smoothed background, the short-wavelength
perturbation, the Born gather for the source at x = 10 km, and the raw
checkpointed RTM image. It also rechecks the image norm, the checkpoint
schedule audits and the saved-state budget. No solver is rerun and no data is
changed.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_marmousi.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK5 = Path(__file__).resolve().parents[1]
EXPERIMENT = WEEK5 / "inputs" / "marmousi.json"
BORN = WEEK5 / "artifacts" / "marmousi-born" / "born_data.npy"
IMAGE = WEEK5 / "artifacts" / "marmousi-image" / "image.npy"
RESULT = WEEK5 / "artifacts" / "marmousi-image" / "result.json"
OUT_PNG = WEEK5 / "artifacts" / "marmousi.png"

REFERENCE_NORM = 6.7037741e-4
TOLERANCE = 1e-4
BUDGET = 5
TARGET_SOURCE_KM = 10.0
COMPARE_KM = (8.0, 14.0)


def mark_region(ax, x0, x1, label=True):
    ax.axvline(x0, color="#00d26a", ls="--", lw=1.2, alpha=0.95)
    ax.axvline(x1, color="#00d26a", ls="--", lw=1.2, alpha=0.95)
    if label:
        low, high = ax.get_ylim()  # inverted depth axis: (depth, 0)
        y = high + 0.25 * (low - high)
        ax.text((x0 + x1) / 2, y,
                f"comparison {x0:g}-{x1:g} km", color="#007a3d", ha="center",
                va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="#00d26a", alpha=0.85))


def audit_actions(actions, steps, delta):
    """Mirror checker/check; return (calls, peak, error counts)."""
    available, working = {0}, None
    reverse, peak, calls = [], 1, 0
    grad_errors = restore_errors = overrun_errors = 0
    for event in actions:
        action, step = event["action"], event["step"]
        if action == "restore":
            if step not in available:
                restore_errors += 1
            working = step
        elif action == "call":
            if working != step or step >= steps:
                grad_errors += 1
            working = step + 1
            calls += 1
        elif action == "store":
            if working != step or step in available:
                grad_errors += 1
            available.add(step)
            peak = max(peak, len(available))
            if len(available) > delta + 1:
                overrun_errors += 1
        elif action == "grad":
            if step not in available:
                grad_errors += 1
            reverse.append(step)
        elif action == "fetch":
            if step not in available or step == 0:
                restore_errors += 1
            available.discard(step)
        else:
            raise ValueError(f"unknown action {action!r}")
        if event["saved_states"] != len(available):
            grad_errors += 1
    if reverse != list(range(steps - 1, -1, -1)):
        grad_errors += 1
    if available != {0}:
        grad_errors += 1
    return calls, peak, (grad_errors, restore_errors, overrun_errors)


def main():
    experiment = json.loads(EXPERIMENT.read_text())
    nx, nz = int(experiment["nx"]), int(experiment["nz"])
    dx_km = float(experiment["dx"]) * float(experiment["length_unit_m"]) / 1000.0
    dt_s = float(experiment["dt"]) * float(experiment["time_unit_s"])
    steps = int(experiment["steps"])

    background = np.asarray(experiment["background"], dtype=float)
    perturbation = np.asarray(experiment["perturbation"], dtype=float)
    shots = np.asarray(experiment["shots"], dtype=float)
    receivers = np.asarray(experiment["receivers"], dtype=float)

    born = np.load(BORN, allow_pickle=False)
    image = np.load(IMAGE, allow_pickle=False)
    stats = json.loads(RESULT.read_text())["statistics"]

    source_km = shots[:, 0] * dx_km
    shot_index = int(np.argmin(np.abs(source_km - TARGET_SOURCE_KM)))
    assert abs(source_km[shot_index] - TARGET_SOURCE_KM) < 1e-9

    # Full-domain cell-centre coordinates.
    x_lo, x_hi = -0.5 * dx_km, (nx - 0.5) * dx_km
    z_lo, z_hi = -0.5 * dx_km, (nz - 0.5) * dx_km
    full_extent = (x_lo, x_hi, z_hi, z_lo)
    width_km = (nx - 1) * dx_km
    depth_km = (nz - 1) * dx_km

    # --- image norm and audits -------------------------------------------
    image_l2 = float(np.linalg.norm(image))
    norm_rel = abs(image_l2 - REFERENCE_NORM) / abs(REFERENCE_NORM)

    audit_totals = [0, 0, 0]
    calls_total = 0
    peak_max = 1
    for shot in range(len(shots)):
        actions = json.loads(
            (WEEK5 / "artifacts" / "marmousi-image" / f"actions-{shot}.json").read_text())
        calls, peak, audit = audit_actions(actions, steps, BUDGET)
        calls_total += calls
        peak_max = max(peak_max, peak)
        audit_totals = [audit_totals[i] + audit[i] for i in range(3)]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.2), layout="constrained")

    # Panel 1: smoothed background.
    ax = axes[0, 0]
    bg_artist = ax.imshow(background, cmap="turbo", extent=full_extent,
                          origin="upper", interpolation="nearest", aspect="auto")
    ax.set_title("Smoothed Marmousi background")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(bg_artist, ax=ax, fraction=0.046, pad=0.02,
                 label="Wave speed (km/s)")

    # Panel 2: perturbation.
    ax = axes[0, 1]
    pert_limit = float(np.abs(perturbation).max())
    pert_artist = ax.imshow(perturbation, cmap="seismic", vmin=-pert_limit,
                            vmax=pert_limit, extent=full_extent, origin="upper",
                            interpolation="nearest", aspect="auto")
    ax.set_title("Short-wavelength perturbation")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(pert_artist, ax=ax, fraction=0.046, pad=0.02,
                 label="Velocity perturbation (km/s)")
    ax.set_xlim(0.0, width_km)
    ax.set_ylim(depth_km, 0.0)
    mark_region(ax, *COMPARE_KM)

    # Panel 3: Born gather for the shot at x = 10 km.
    ax = axes[1, 0]
    gather = born[shot_index]
    receiver_km = receivers[:, 0] * dx_km
    spacing = float(np.diff(receiver_km).mean())
    x_edges = np.concatenate(([receiver_km[0] - spacing / 2],
                              receiver_km + spacing / 2))
    times_s = (np.arange(steps) + 1) * dt_s
    time_edges = np.concatenate(([times_s[0] - dt_s / 2], times_s + dt_s / 2))
    gather_limit = float(np.abs(gather).max())
    gather_artist = ax.pcolormesh(x_edges, time_edges, gather, cmap="seismic",
                                  vmin=-gather_limit, vmax=gather_limit,
                                  shading="flat", rasterized=True)
    ax.invert_yaxis()
    ax.set_title(f"Born gather; source x = {source_km[shot_index]:.1f} km")
    ax.set_xlabel("Receiver position (km)")
    ax.set_ylabel("Time (s)")
    ax.set_xlim(x_edges[0], x_edges[-1])
    fig.colorbar(gather_artist, ax=ax, fraction=0.046, pad=0.02,
                 label="Born scattered pressure")

    # Panel 4: raw Treeverse RTM image, one fixed amplitude scale.
    ax = axes[1, 1]
    image_limit = float(np.abs(image).max())
    image_artist = ax.imshow(image, cmap="seismic", vmin=-image_limit,
                             vmax=image_limit, extent=full_extent,
                             origin="upper", interpolation="nearest", aspect="auto")
    ax.set_title("Checkpointed migration image")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.02,
                 label="RTM image amplitude (arbitrary units)")
    ax.set_xlim(0.0, width_km)
    ax.set_ylim(depth_km, 0.0)
    mark_region(ax, *COMPARE_KM)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- reporting --------------------------------------------------------
    print("VERIFY (3) — Marmousi imaging")
    print(f"  grid                    : nx = {nx}, nz = {nz}, dx = {dx_km * 1000:.0f} m")
    print(f"  steps / dt              : {steps}, {dt_s * 1000:.0f} ms")
    print(f"  shots / receivers       : {len(shots)} / {len(receivers)}")
    print(f"  source frequency        : "
          f"{float(experiment['source_frequency']) / float(experiment['time_unit_s']):.1f} Hz")
    print(f"  Born data               : {born.shape} {born.dtype}, "
          f"L2 = {np.linalg.norm(born):.12e}, nonzero = {bool((born != 0).any())}")
    print(f"  shot for x = 10 km      : index {shot_index} "
          f"(source x = {source_km[shot_index]:.1f} km)")
    print(f"  image                   : {image.shape} {image.dtype}, "
          f"finite = {bool(np.isfinite(image).all())}, "
          f"nonzero = {bool((image != 0).any())}")
    print(f"  image min / max         : {image.min():.6e} / {image.max():.6e}")
    print(f"  image L2                : {image_l2:.12e}")
    print(f"  reference L2            : {REFERENCE_NORM:.12e}")
    print(f"  abs difference          : {abs(image_l2 - REFERENCE_NORM):.6e}")
    print(f"  relative error          : {norm_rel:.6e} "
          f"({'PASS' if norm_rel < TOLERANCE else 'FAIL'})")
    print(f"  storage / checkpoints   : {stats['storage']} / {stats['checkpoints']}")
    print(f"  reverse / forward calls : {stats['reverse_calls']} / {stats['scheduler_forward_calls']}")
    print(f"  peak saved states       : {peak_max} (expected 6)")
    print(f"  peak saved bytes        : {stats['peak_saved_bytes']} "
          f"(expected 20788320)")
    print(f"  audit grad/restore/overrun: {tuple(audit_totals)}")
    print(f"  wrote {OUT_PNG}")

    checks = [
        ("Born data is float64 [9, 1200, 91] and non-zero",
         born.dtype == np.float64 and born.shape == (9, 1200, 91)
         and np.isfinite(born).all() and bool((born != 0).any())),
        ("image is finite float64 [269, 805] and non-zero",
         image.dtype == np.float64 and image.shape == (nz, nx)
         and np.isfinite(image).all() and bool((image != 0).any())),
        ("all nine shots processed", len(stats["per_shot"]) == 9),
        ("image norm relative error < 1e-4", norm_rel < TOLERANCE),
        ("Treeverse with 6 saved states / 20,788,320 bytes",
         stats["storage"] == "treeverse" and stats["checkpoints"] == BUDGET
         and peak_max == 6 and stats["peak_saved_bytes"] == 20788320),
        ("audit counts all zero", all(v == 0 for v in audit_totals)),
        ("forward calls match the scheduler",
         stats["scheduler_forward_calls"] == calls_total),
    ]
    print("\nVERIFY (3) checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall Marmousi checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
