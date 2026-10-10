#!/usr/bin/env python3
"""Week 5 Part 4 VERIFY (1): Treeverse scheduler action plot.

Reads the actual schedule

    week5/artifacts/checkpoint-5/actions-0.json
    week5/artifacts/checkpoint-5/result.json

and plots the complete ordered action sequence (operation index versus
timestep), coloured by operation type. No solver is rerun and no action is
dropped or changed.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_checkpoint_actions.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK5 = Path(__file__).resolve().parents[1]
FOLDER = WEEK5 / "artifacts" / "checkpoint-5"
ACTIONS = FOLDER / "actions-0.json"
RESULT = FOLDER / "result.json"
OUT_PNG = WEEK5 / "artifacts" / "checkpoint-actions.png"

BUDGET = 5
SHOT = 0
STEPS = 240

# Colour and marker per operation; drawn in this legend order.
STYLE = {
    "call": ("#1f77b4", ".", 14),
    "restore": ("#d62728", "v", 26),
    "store": ("#2ca02c", "s", 26),
    "grad": ("#ff7f0e", "x", 22),
    "fetch": ("#9467bd", "P", 24),
}


def audit(actions, steps, delta):
    """Same rules as checker/check: return (counts, calls, peak, errors)."""
    counts = {name: 0 for name in STYLE}
    available, working = {0}, None
    reverse, peak, calls = [], 1, 0
    grad_errors = restore_errors = overrun_errors = 0
    for event in actions:
        action, step = event["action"], event["step"]
        counts[action] = counts.get(action, 0) + 1
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
        if event["saved_states"] != len(available):
            grad_errors += 1
    if reverse != list(range(steps - 1, -1, -1)):
        grad_errors += 1
    if available != {0}:
        grad_errors += 1
    return counts, calls, peak, (grad_errors, restore_errors, overrun_errors)


def main():
    actions = json.loads(ACTIONS.read_text())
    stats = json.loads(RESULT.read_text())["statistics"]
    order = np.arange(len(actions))
    steps = np.asarray([event["step"] for event in actions])

    counts, calls, peak, errors = audit(actions, STEPS, BUDGET)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(11.5, 5.2), layout="constrained")

    # Light trajectory so the sawtooth jumps are visible.
    ax.plot(order, steps, color="#c9c9c9", lw=0.6, zorder=1)

    for name, (color, marker, size) in STYLE.items():
        mask = np.asarray([event["action"] == name for event in actions])
        ax.scatter(order[mask], steps[mask], s=size, c=color, marker=marker,
                   label=f"{name} ({counts[name]})", zorder=3 if name != "call" else 2,
                   linewidths=0.8)

    ax.set_xlabel("Operation index")
    ax.set_ylabel("Timestep")
    ax.set_title(f"Treeverse schedule; reflector shot {SHOT}, budget {BUDGET}")
    ax.set_xlim(-10, len(actions) + 10)
    ax.set_ylim(-5, STEPS + 5)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, ncols=2,
              markerscale=1.4)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result_calls = stats["per_shot"][SHOT]["scheduler_forward_calls"]
    result_peak = stats["per_shot"][SHOT]["peak_saved_states"]

    print("VERIFY (1) — Treeverse schedule")
    print(f"  source                  : {ACTIONS.relative_to(WEEK5)}")
    print(f"  action records          : {len(actions)}")
    print(f"  operation counts        : {counts}")
    print(f"  all five types present  : {set(counts) == set(STYLE)}")
    print(f"  forward calls (calls)   : {calls}  (result.json {result_calls})")
    print(f"  reverse grads           : {counts['grad']} (expected {STEPS})")
    print(f"  peak saved states       : {peak}  (result.json {result_peak})")
    print(f"  audit errors (grad/restore/overrun): {errors}")
    print(f"  wrote {OUT_PNG}")

    checks = [
        ("uses budget-5 shot-0 actions", ACTIONS.name == "actions-0.json"),
        ("all five operation types", set(counts) == set(STYLE)),
        ("forward calls match result.json", calls == result_calls),
        ("peak matches result.json", peak == result_peak),
        ("240 reverse grads", counts["grad"] == STEPS),
        ("audit counts all zero", all(value == 0 for value in errors)),
    ]
    print("\nVERIFY (1) checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall schedule checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
