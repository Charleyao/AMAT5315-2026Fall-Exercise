#!/usr/bin/env python3
"""Week 5 Part 4 VERIFY (2): Treeverse work and memory trade-off.

Reads the recorded statistics

    week5/artifacts/checkpoint-{1,3,5,10}/result.json and image.npy
    week5/artifacts/checkpoint-{1,3,5,10}/actions-<shot>.json
    week5/artifacts/adjoint/result.json and image.npy

and plots two panels:

  1. forward recomputation per shot versus checkpoint budget (log scale),
  2. peak saved-state bytes versus checkpoint budget, with the full-history
     reference shown by an annotation and a log-scale inset.

It also re-audits every schedule and rechecks the image errors. No solver is
rerun and no recorded number is changed.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_checkpoint_work.py
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
BUDGETS = (1, 3, 5, 10)
OUT_PNG = WEEK5 / "artifacts" / "checkpoint-work.png"

# Learning Sheet reference values.
REFERENCE_FORWARD = {1: 28680, 3: 1695, 5: 990, 10: 642}
REFERENCE_PEAK_STATES = {1: 2, 3: 4, 5: 6, 10: 11}
REFERENCE_PEAK_BYTES = {1: 53792, 3: 107584, 5: 161376, 10: 295856}
FULL_FORWARD = 240
FULL_PEAK_STATES = 241
FULL_PEAK_BYTES = 6481936
TOLERANCE = 1e-9


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
    experiment = json.loads((WEEK5 / "inputs" / "reflector.json").read_text())
    steps = int(experiment["steps"])
    n_shots = len(experiment["shots"])
    state_bytes = 2 * int(experiment["nx"]) * int(experiment["nz"]) * 8

    full_stats = json.loads((ADJOINT / "result.json").read_text())["statistics"]
    full_image = np.load(ADJOINT / "image.npy", allow_pickle=False)

    forward, peak_states, peak_bytes, errors = [], [], [], []
    totals = [0, 0, 0]
    rows = []
    for budget in BUDGETS:
        folder = WEEK5 / "artifacts" / f"checkpoint-{budget}"
        stats = json.loads((folder / "result.json").read_text())["statistics"]
        image = np.load(folder / "image.npy", allow_pickle=False)
        relerr = float(np.linalg.norm(image - full_image) / np.linalg.norm(full_image))

        calls_total = 0
        peak_max = 1
        for shot in range(n_shots):
            actions = json.loads((folder / f"actions-{shot}.json").read_text())
            calls, peak, audit = audit_actions(actions, steps, budget)
            totals[0] += audit[0]
            totals[1] += audit[1]
            totals[2] += audit[2]
            calls_total += calls
            peak_max = max(peak_max, peak)
        per_shot = calls_total // n_shots

        rows.append((budget, per_shot, peak_max, stats["peak_saved_bytes"], relerr))
        forward.append(per_shot)
        peak_states.append(peak_max)
        peak_bytes.append(stats["peak_saved_bytes"])
        errors.append((per_shot, peak_max, stats["peak_saved_bytes"]))

    # --- Panel 1: recomputation cost --------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.3),
                                   layout="constrained")

    ax1.semilogy(BUDGETS, forward, "o-", color="#1f77b4", lw=1.8, ms=8,
                 label="Treeverse")
    for budget, value in zip(BUDGETS, forward):
        ax1.annotate(f"{value}", (budget, value), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5, color="#1f4e9c")
    ax1.axhline(FULL_FORWARD, color="#d62728", ls="--", lw=1.6)
    ax1.annotate(f"full history ({FULL_FORWARD} steps/shot)", (10, FULL_FORWARD),
                 textcoords="offset points", xytext=(-4, 6), ha="right",
                 fontsize=8.5, color="#a01c1c")
    ax1.set_xlabel("Additional checkpoint slots $\\delta$")
    ax1.set_ylabel("Forward steps per shot (log scale)")
    ax1.set_title("Recomputation cost")
    ax1.set_xticks(list(BUDGETS))
    ax1.set_xlim(0, 11)
    ax1.grid(True, which="both", color="#e6e6e6", lw=0.6)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Panel 2: saved-state memory --------------------------------------
    ax2.plot(BUDGETS, peak_bytes, "s-", color="#2ca02c", lw=1.8, ms=8)
    for budget, bytes_value, states in zip(BUDGETS, peak_bytes, peak_states):
        ax2.annotate(f"{states} states", (budget, bytes_value),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=8.5, color="#1b6b1b")
    ax2.set_xlabel("Additional checkpoint slots $\\delta$")
    ax2.set_ylabel("Peak saved-state bytes")
    ax2.set_title("Two wavefields per saved state")
    ax2.set_xticks(list(BUDGETS))
    ax2.set_xlim(0.4, 10.6)
    ax2.grid(True, color="#e6e6e6", lw=0.6)
    ax2.annotate(
        f"full history: {FULL_PEAK_STATES} states\n"
        f"{FULL_PEAK_BYTES:,} bytes\n"
        f"({FULL_PEAK_BYTES / peak_bytes[-1]:.1f}x budget 10, off scale)",
        xy=(0.03, 0.62), xycoords="axes fraction", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800", alpha=0.95))

    # Log-scale inset so the full-history bar is not hidden by the linear axis.
    inset = ax2.inset_axes([0.55, 0.10, 0.42, 0.34])
    positions = list(BUDGETS) + [12.0]
    heights = list(peak_bytes) + [FULL_PEAK_BYTES]
    colors = ["#2ca02c"] * len(BUDGETS) + ["#d62728"]
    inset.bar(positions, heights, color=colors, width=0.9)
    inset.set_yscale("log")
    inset.set_xticks(positions)
    inset.set_xticklabels([str(b) for b in BUDGETS] + ["full"], fontsize=7.5)
    inset.set_title("all budgets (log)", fontsize=8)
    inset.tick_params(axis="y", labelsize=7)
    inset.grid(True, axis="y", which="both", color="#eeeeee", lw=0.5)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- reporting --------------------------------------------------------
    print("VERIFY (2) — Treeverse work and memory")
    print(f"  {'budget':>6} {'fwd/shot':>9} {'peak states':>12} "
          f"{'peak bytes':>11} {'relerr':>10}")
    for budget, per_shot, states, bytes_value, relerr in rows:
        print(f"  {budget:>6} {per_shot:>9} {states:>12} {bytes_value:>11} "
              f"{relerr:>10.3e}")
    print(f"  {'full':>6} {FULL_FORWARD:>9} {FULL_PEAK_STATES:>12} "
          f"{FULL_PEAK_BYTES:>11}")
    print(f"\n  audit counts (grad/restore/overrun): {tuple(totals)}")
    print(f"  wrote {OUT_PNG}")

    decreasing = all(forward[i] > forward[i + 1] for i in range(len(forward) - 1))
    increasing = all(peak_bytes[i] < peak_bytes[i + 1] for i in range(len(peak_bytes) - 1))
    audit_ok = all(value == 0 for value in totals)
    counts_match = all(
        per_shot == REFERENCE_FORWARD[b] and states == REFERENCE_PEAK_STATES[b]
        and bytes_value == REFERENCE_PEAK_BYTES[b]
        for b, per_shot, states, bytes_value, _ in rows
    )
    errors_ok = all(relerr < TOLERANCE for *_, relerr in rows)
    full_ok = (full_stats["storage"] == "full"
               and full_stats["peak_saved_states"] == FULL_PEAK_STATES
               and full_stats["peak_saved_bytes"] == FULL_PEAK_BYTES
               and full_stats["scheduler_forward_calls"] // n_shots == FULL_FORWARD)

    checks = [
        ("forward work decreases with budget", decreasing),
        ("peak memory increases with budget", increasing),
        ("all forward/peak counts match the reference", counts_match),
        ("relative image errors < 1e-9", errors_ok),
        ("audit counts all zero", audit_ok),
        ("full history is 241 states / 6,481,936 bytes / 240 steps per shot", full_ok),
    ]
    print("\nVERIFY (2) checks:")
    failed = not all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall work/memory checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
