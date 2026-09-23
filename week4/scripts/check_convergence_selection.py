#!/usr/bin/env python3
"""Read-only check of the Part 4 convergence data and timestep selection.

Recomputes everything from the stored artifacts (no simulation) and prints the
measured errors, the direct Richardson estimate from omega_0.02 and omega_0.01,
the h^4 predictions, and the selection rule outcome.

    /home/yao_yiyi/.venvs/amat5315/bin/python check_convergence_selection.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEEK4 = HERE.parent
ART = WEEK4 / "artifacts" / "convergence"
EVID = WEEK4 / "evidence"

CANDIDATES = (0.02, 0.0125, 0.01)          # as used by the selection
REFERENCE_DT = 0.0025
THRESHOLD = 5e-6


def omega_at_t2(dt: float) -> np.ndarray:
    path = ART / f"rk4-dt{dt}" / "fields.jsonl"
    frames = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return np.asarray(frames[-1]["omega"])


def norm(x: np.ndarray) -> float:
    return math.sqrt(float(np.sum(x * x)))


def main() -> None:
    omega = {dt: omega_at_t2(dt) for dt in (*CANDIDATES, REFERENCE_DT)}

    # 1. measured relative error against the dt = 0.0025 reference.
    measured = {
        dt: norm(omega[dt] - omega[REFERENCE_DT]) / norm(omega[REFERENCE_DT])
        for dt in CANDIDATES
    }
    print("1. measured relative omega error (vs dt = 0.0025 reference)")
    for dt in CANDIDATES:
        print(f"   dt = {dt:<7g} {measured[dt]:.10e}")

    # 2. fitted slope over the three candidates.
    dts = np.asarray(CANDIDATES)
    errs = np.asarray([measured[dt] for dt in CANDIDATES])
    slope, _ = np.polyfit(np.log(dts), np.log(errs), 1)
    print(f"\n2. measured fitted slope q = {slope:.10f}")

    # 3. direct Richardson estimate from omega_0.02 and omega_0.01.
    e_R = norm(omega[0.02] - omega[0.01]) / (15.0 * norm(omega[0.01]))
    print(f"\n3. e_R(0.01) = ||omega_0.02 - omega_0.01|| / (15 ||omega_0.01||)")
    print(f"            = {e_R:.10e}")

    # 4. strict fourth-order scaling.
    predicted = {dt: e_R * (dt / 0.01) ** 4 for dt in CANDIDATES}
    print("\n4. e_pred(h') = e_R(0.01) * (h'/0.01)^4")
    for dt in CANDIDATES:
        print(f"   h' = {dt:<7g} {predicted[dt]:.10e}")

    # 5. side-by-side.
    print("\n5. dt        predicted       measured")
    for dt in CANDIDATES:
        print(f"   {dt:<8g} {predicted[dt]:.6e}    {measured[dt]:.6e}")

    # 6/7. selection rule: largest candidate with predicted < threshold.
    eligible = [dt for dt in CANDIDATES if predicted[dt] < THRESHOLD]
    chosen = max(eligible) if eligible else None
    print(f"\n6. threshold = {THRESHOLD:g}")
    for dt in CANDIDATES:
        verdict = "PASS" if predicted[dt] < THRESHOLD else "FAIL"
        print(f"   predicted({dt:<7g}) = {predicted[dt]:.6e}  {verdict}")
    print(f"\n7. selection rule: largest candidate with predicted < {THRESHOLD:g}")
    print(f"   eligible = {eligible}")
    print(f"   chosen   = {chosen}")
    print(f"   reason 0.0125 was not chosen: predicted {predicted[0.0125]:.6e} "
          f"{'>' if predicted[0.0125] >= THRESHOLD else '<'} {THRESHOLD:g}")

    # Cross-check against the stored evidence JSON (must match, nothing rewritten).
    stored = json.loads((EVID / "convergence.json").read_text())
    print("\n   stored convergence.json: chosen_dt = "
          f"{stored['chosen_dt']}, fitted_slope = {stored['fitted_slope']:.10f}, "
          f"richardson.estimate = {stored['richardson']['estimate']:.10e}")


if __name__ == "__main__":
    main()
