#!/usr/bin/env python3
"""Week 5 Part 2 VERIFY (2): check the Rust forward receiver traces.

Reads ``week5/artifacts/forward/traces.npy`` and compares the L2 norm, the
per-shot peak pressure, its time index and receiver against the instructor's
reference values. All comparisons must be below 1e-4 relative error.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/verify_forward.py
"""

from pathlib import Path
import json
import sys

import numpy as np

WEEK5 = Path(__file__).resolve().parents[1]
TRACES = WEEK5 / "artifacts" / "forward" / "traces.npy"
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"

TOTAL_L2_REFERENCE = 11.574770
PEAK_REFERENCES = (0.60809514, 0.59271397, 0.60809514)
STEP_REFERENCE = 83
TOLERANCE = 1e-4


def relative(actual, expected):
    return abs(actual - expected) / max(abs(expected), 1e-30)


def main():
    if not TRACES.is_file():
        print(f"FAIL: {TRACES} does not exist", file=sys.stderr)
        return 1

    traces = np.load(TRACES, allow_pickle=False)
    experiment = json.loads(EXPERIMENT.read_text())
    shots = np.asarray(experiment["shots"], dtype=float)
    receivers = np.asarray(experiment["receivers"], dtype=float)
    dt, time_unit = float(experiment["dt"]), float(experiment["time_unit_s"])

    expected_shape = (len(shots), int(experiment["steps"]), len(receivers))
    if traces.dtype != np.float64 or traces.shape != expected_shape:
        print(f"FAIL: traces {traces.dtype} {traces.shape}, "
              f"expected float64 {expected_shape}", file=sys.stderr)
        return 1

    total_norm = float(np.linalg.norm(traces))
    print(f"traces: {traces.dtype} shape {traces.shape}")
    print(f"total trace L2 norm : {total_norm:.9f} "
          f"(reference {TOTAL_L2_REFERENCE:.6f}, "
          f"relative error {relative(total_norm, TOTAL_L2_REFERENCE):.3e})")

    print("\nper-shot peaks:")
    print(f"  {'shot':>4} {'peak':>12} {'step':>5} {'time(s)':>8} "
          f"{'recv':>5} {'relerr':>10} {'dist^2':>7}")
    failures = []
    for shot in range(traces.shape[0]):
        step, receiver = np.unravel_index(np.argmax(np.abs(traces[shot])),
                                          traces[shot].shape)
        peak = float(abs(traces[shot, step, receiver]))
        error = relative(peak, PEAK_REFERENCES[shot])
        displacement = receivers[receiver] - shots[shot]
        distance2 = float(displacement @ displacement)
        time_s = (step + 1) * dt * time_unit
        print(f"  {shot:>4} {peak:>12.9f} {step:>5d} {time_s:>8.3f} "
              f"{receiver:>5d} {error:>10.3e} {distance2:>7.1f}")
        if error >= TOLERANCE:
            failures.append(f"shot {shot} peak relative error {error:.3e}")
        if step != STEP_REFERENCE:
            failures.append(f"shot {shot} peak step {step} != {STEP_REFERENCE}")
        if distance2 != 1.0:
            failures.append(f"shot {shot} peak receiver is not one cell away")

    total_error = relative(total_norm, TOTAL_L2_REFERENCE)
    if total_error >= TOLERANCE:
        failures.append(f"total L2 relative error {total_error:.3e}")

    print(f"\ntime of step {STEP_REFERENCE}: "
          f"t = {(STEP_REFERENCE + 1) * dt:g} reduced units = "
          f"{(STEP_REFERENCE + 1) * dt * time_unit:g} s")

    if failures:
        print("\nVERIFY (2) FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nall VERIFY (2) numerical checks PASS "
          f"(relative error < {TOLERANCE:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
