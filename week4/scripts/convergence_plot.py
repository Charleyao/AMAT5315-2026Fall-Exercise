#!/usr/bin/env python3
"""Part 4 random-flow convergence evidence.

Computes the relative omega error of three RK4 steps against the dt = 0.0025
reference, fits the log-log slope, forms the Richardson estimate at dt = 0.01,
predicts the candidate errors with h^4 scaling, and picks the largest step below
the 5e-6 threshold.

Writes ``week4/evidence/convergence.json``, ``convergence.txt`` and
``convergence.png``. Run from ``week4/scripts/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python convergence_plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEK4 = HERE.parent
ART = WEEK4 / "artifacts" / "convergence"
EVID = WEEK4 / "evidence"

CANDIDATES = (0.02, 0.0125, 0.01)
REFERENCE_DT = 0.0025
THRESHOLD = 5e-6


def omega_at_t2(dt: float) -> np.ndarray:
    path = ART / f"rk4-dt{dt}" / "fields.jsonl"
    frames = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return np.asarray(frames[-1]["omega"])


def rel_error(omega: np.ndarray, reference: np.ndarray) -> float:
    return math.sqrt(float(np.sum((omega - reference) ** 2))) / math.sqrt(
        float(np.sum(reference**2))
    )


def main() -> None:
    omegas = {dt: omega_at_t2(dt) for dt in (*CANDIDATES, REFERENCE_DT)}
    reference = omegas[REFERENCE_DT]
    errors = {dt: rel_error(omegas[dt], reference) for dt in CANDIDATES}

    dts = np.asarray(CANDIDATES)
    errs = np.asarray([errors[dt] for dt in CANDIDATES])
    slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)

    # Richardson estimate at h = 0.01 from the 2h = 0.02 and h = 0.01 fields.
    h = 0.01
    omega_h = omegas[h]
    omega_2h = omegas[2 * h]
    e_rich = math.sqrt(float(np.sum((omega_2h - omega_h) ** 2))) / (
        (2.0**4 - 1.0) * math.sqrt(float(np.sum(omega_h**2)))
    )

    predicted = {dt: e_rich * (dt / h) ** 4 for dt in CANDIDATES}

    # Largest candidate whose predicted error is under the threshold.
    eligible = [dt for dt in CANDIDATES if predicted[dt] < THRESHOLD]
    chosen = max(eligible) if eligible else None

    # ---- convergence.json -----------------------------------------------------
    payload = {
        "case": "random",
        "n": 128,
        "seed": 2026,
        "k_band": [2, 6],
        "nu": 0.004,
        "t_end": 2.0,
        "method": "rk4",
        "reference_dt": REFERENCE_DT,
        "quantity": "final vorticity at t = 2, relative L2 error",
        "runs": [
            {"dt": dt, "relative_omega_error": errors[dt],
             "predicted_error": predicted[dt]}
            for dt in CANDIDATES
        ],
        "fitted_slope": float(slope),
        "error_ratio_0.02_over_0.01": errors[0.02] / errors[0.01],
        "richardson": {
            "dt": h,
            "assumed_order": 4,
            "denominator": 2.0**4 - 1.0,
            "estimate": e_rich,
        },
        "target_relative_error": THRESHOLD,
        "chosen_dt": chosen,
    }
    (EVID / "convergence.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---- plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.9), layout="constrained")
    ax.loglog(dts, errs, "o", color="tab:blue", ms=6, label="measured")
    fit = np.exp(intercept) * dts**slope
    ax.loglog(dts, fit, "-", color="tab:blue", lw=1.2, label=r"fit, $q$ = %.2f" % slope)
    if chosen is not None:
        ax.axvline(chosen, color="tab:green", ls=":", lw=1.1)
        ax.annotate(f"chosen $h$ = {chosen:g}", (chosen, errs.min()),
                    xytext=(4, 6), textcoords="offset points", fontsize=8,
                    color="tab:green")
    ax.axhline(THRESHOLD, color="0.5", ls="--", lw=1.0,
               label=r"threshold $5\times10^{-6}$")
    ax.set_xticks(dts)
    ax.set_xticklabels(["%g" % d for d in dts])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"relative $\omega$ error at $t = 2$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(r"random flow, RK4, $n$ = 128, $\nu$ = 0.004", fontsize=10)
    fig.savefig(EVID / "convergence.png", dpi=150)
    plt.close(fig)

    # ---- text -----------------------------------------------------------------
    lines = [
        "Part 4 random-flow convergence",
        "n = 128, seed = 2026, k in [2,6], nu = 0.004, t_end = 2, RK4",
        f"reference: dt = {REFERENCE_DT} (not a fit point)",
        "",
        f"{'dt':>9}{'measured error':>18}{'predicted error':>18}",
    ]
    for dt in CANDIDATES:
        lines.append(f"{dt:>9g}{errors[dt]:>18.6e}{predicted[dt]:>18.6e}")
    lines += [
        "",
        f"fitted slope q = {slope:.4f}",
        f"error(0.02)/error(0.01) = {errors[0.02] / errors[0.01]:.4f} "
        "(fourth order predicts ~16)",
        f"Richardson estimate at dt = 0.01 = {e_rich:.6e}",
        f"target relative error < {THRESHOLD:g}",
        f"chosen dt = {chosen}",
    ]
    if chosen is not None:
        lines.append(f"  predicted error = {predicted[chosen]:.6e}")
        lines.append(f"  measured  error = {errors[chosen]:.6e}")
    text = "\n".join(lines) + "\n"
    (EVID / "convergence.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
