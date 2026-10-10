#!/usr/bin/env python3
"""Week 5 Part 4: verify Treeverse checkpointing against full history.

Reads the full-history adjoint image

    week5/artifacts/adjoint/image.npy

and every Treeverse run

    week5/artifacts/checkpoint-<b>/image.npy
    week5/artifacts/checkpoint-<b>/result.json
    week5/artifacts/checkpoint-<b>/actions-<shot>.json

For each budget it reports the relative image error, forward/reverse call
counts, peak saved states/bytes, and audits the action schedules with the same
rules as ``week5/checker/check``. No solver is rerun and no data is changed.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/verify_treeverse.py
"""

from pathlib import Path
import json
import sys

import numpy as np

WEEK5 = Path(__file__).resolve().parents[1]
FULL_IMAGE = WEEK5 / "artifacts" / "adjoint" / "image.npy"
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"
BUDGETS = (1, 3, 5, 10)

# Reference scheduler counts per shot from the Learning Sheet.
EXPECTED_FORWARD = {1: 28680, 3: 1695, 5: 990, 10: 642}
EXPECTED_PEAK = {1: 2, 3: 4, 5: 6, 10: 11}
TOLERANCE = 1e-9


def validate_actions(actions, steps, delta):
    """Replay a schedule; return (calls, peak, audit counters)."""
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
            assert working == step and step < steps, "replay starts from wrong time"
            working = step + 1
            calls += 1
        elif action == "store":
            assert working == step and step not in available, "invalid save"
            available.add(step)
            peak = max(peak, len(available))
            if len(available) > delta + 1:
                overrun_errors += 1
        elif action == "grad":
            if step not in available:
                grad_errors += 1
            reverse.append(step)
        elif action == "fetch":
            assert step in available and step != 0, "invalid checkpoint discard"
            available.remove(step)
        else:
            raise ValueError(f"unknown action {action!r}")
        assert event["saved_states"] == len(available), "stored-state counter"
    if reverse != list(range(steps - 1, -1, -1)):
        grad_errors += 1
    if available != {0}:
        grad_errors += 1
    return calls, peak, (grad_errors, restore_errors, overrun_errors)


def main():
    full = np.load(FULL_IMAGE, allow_pickle=False)
    experiment = json.loads(EXPERIMENT.read_text())
    steps = int(experiment["steps"])
    n_shots = len(experiment["shots"])
    state_bytes = 2 * int(experiment["nx"]) * int(experiment["nz"]) * 8

    totals = [0, 0, 0]
    failed = False
    print(f"reference full-history image: {full.shape} {full.dtype}, "
          f"norm {np.linalg.norm(full):.12f}\n")

    header = (f"{'budget':>6} {'relerr':>12} {'fwd/shot':>9} {'fwd total':>10} "
              f"{'peak':>5} {'bytes':>10} {'actions':>8}")
    print(header)
    print("-" * len(header))
    for budget in BUDGETS:
        folder = WEEK5 / "artifacts" / f"checkpoint-{budget}"
        image = np.load(folder / "image.npy", allow_pickle=False)
        result = json.loads((folder / "result.json").read_text())
        stats = result["statistics"]

        if image.dtype != np.float64 or image.shape != full.shape:
            print(f"FAIL: budget {budget} image {image.dtype} {image.shape}")
            failed = True
            continue
        relerr = float(np.linalg.norm(image - full) / np.linalg.norm(full))

        calls_total = 0
        peak_max = 1
        action_count = 0
        for shot in range(n_shots):
            path = folder / f"actions-{shot}.json"
            actions = json.loads(path.read_text())
            action_count += len(actions)
            calls, peak, audit = validate_actions(actions, steps, budget)
            totals[0] += audit[0]
            totals[1] += audit[1]
            totals[2] += audit[2]
            calls_total += calls
            peak_max = max(peak_max, peak)
            assert stats["per_shot"][shot]["actions_file"] == path.name
            assert stats["per_shot"][shot]["scheduler_forward_calls"] == calls
            assert stats["per_shot"][shot]["reverse_calls"] == steps
            assert stats["per_shot"][shot]["peak_saved_states"] == peak

        checks = [
            stats["storage"] == "treeverse",
            stats["checkpoints"] == budget,
            stats["reverse_calls"] == steps * n_shots,
            stats["scheduler_forward_calls"] == calls_total,
            stats["peak_saved_states"] == peak_max == EXPECTED_PEAK[budget],
            stats["peak_saved_bytes"] == peak_max * state_bytes,
            calls_total // n_shots == EXPECTED_FORWARD[budget],
            relerr < TOLERANCE,
            all(t == 0 for t in (totals[0], totals[1], totals[2])),
        ]
        failed = failed or not all(checks)
        print(f"{budget:>6} {relerr:>12.3e} {calls_total // n_shots:>9} "
              f"{calls_total:>10} {peak_max:>5} "
              f"{stats['peak_saved_bytes']:>10} {action_count:>8}")

    print("\naudit counts across every shot and budget:")
    print(f"  missing/duplicate/out-of-order grad steps : {totals[0]}")
    print(f"  invalid restore operations                : {totals[1]}")
    print(f"  checkpoint budget overruns                : {totals[2]}")

    print("\nchecks:")
    for budget in BUDGETS:
        folder = WEEK5 / "artifacts" / f"checkpoint-{budget}"
        image = np.load(folder / "image.npy", allow_pickle=False)
        relerr = float(np.linalg.norm(image - full) / np.linalg.norm(full))
        status = "PASS" if relerr < TOLERANCE else "FAIL"
        print(f"  [{status}] budget {budget:>2}: relative image error {relerr:.3e} "
              f"< {TOLERANCE:g}")
    print(f"  [{'PASS' if all(t == 0 for t in totals) else 'FAIL'}] all audit counts zero")

    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall Treeverse checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
