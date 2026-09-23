#!/usr/bin/env python3
"""Part 4 Taylor-Green RK4 order evidence: ``week4/evidence/order.png``.

Reads the three fixed-step runs and the exact t = 2 field from
``artifacts/order/`` (produced by ``scripts/run_order.sh``), computes the
relative velocity error at t = 2, fits its log-log slope, and draws the measured
points with the fitted line.

Run from ``week4/scripts/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python order_plot.py
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
ART = WEEK4 / "artifacts" / "order"
EVID = WEEK4 / "evidence"

DTS = (0.4, 0.25, 0.2)


def last_frame(path: Path) -> dict:
    frames = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return frames[-1]


def relative_velocity_error(frame: dict, exact: dict) -> float:
    u = np.asarray(frame["u"]) - np.asarray(exact["u"])
    v = np.asarray(frame["v"]) - np.asarray(exact["v"])
    ue = np.asarray(exact["u"])
    ve = np.asarray(exact["v"])
    return math.sqrt(float(np.sum(u * u + v * v)) / float(np.sum(ue * ue + ve * ve)))


def main() -> None:
    exact = json.loads((ART / "exact-t2.json").read_text())
    errors = np.array(
        [relative_velocity_error(last_frame(ART / f"rk4-dt{dt}" / "fields.jsonl"), exact)
         for dt in DTS]
    )
    dts = np.asarray(DTS)
    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)
    ratio = errors[0] / errors[-1]

    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    ax.loglog(dts, errors, "o", color="tab:blue", ms=6, label="measured")
    fit = np.exp(intercept) * dts**slope
    ax.loglog(dts, fit, "-", color="tab:blue", lw=1.2,
              label=r"fit, $p$ = %.2f" % slope)
    ref = errors[-1] * (dts / dts[-1]) ** 4
    ax.loglog(dts, ref, "--", color="0.5", lw=1.0, label=r"slope 4 reference")
    ax.set_xticks(dts)
    ax.set_xticklabels(["%g" % d for d in dts])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"relative velocity error at $t = 2$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(r"Taylor-Green, RK4, $n$ = 8, $\nu$ = 0.5", fontsize=10)
    fig.savefig(EVID / "order.png", dpi=150)
    plt.close(fig)

    lines = [
        "Part 4 Taylor-Green RK4 order",
        "n = 8, nu = 0.5, t_end = 2, relative velocity error at t = 2",
        "",
        f"{'dt':>8}{'relative error':>18}",
    ]
    for dt, err in zip(DTS, errors):
        lines.append(f"{dt:>8g}{err:>18.6e}")
    lines += [
        "",
        f"fitted RK4 order p = {slope:.4f}",
        f"error(dt=0.4)/error(dt=0.2) = {ratio:.4f}  (fourth order predicts 16)",
        f"within +/-15% of 4: {'yes' if 3.4 <= slope <= 4.6 else 'no'}",
        "figure: evidence/order.png",
    ]
    text = "\n".join(lines) + "\n"
    (EVID / "order.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
