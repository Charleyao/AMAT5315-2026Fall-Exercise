#!/usr/bin/env python3
"""Part 3 stability-limit evidence: ``week4/evidence/blowup.png``.

Left  : Taylor-Green energy for RK4 dt = 0.032 (stable), dt = 0.033 (unstable)
        and dt = 0.04 (far over the limit), against the exact decay.
Right : random-flow energy for a stable RK4 step, an unstable RK4 step and
        forward Euler, all from the same saved initial field.

The two stability predictions are computed here (not hard-coded): the RK4
axis crossings are found by bisection on |R(z)| = 1, then the diffusive and
advective step bounds follow from the measured spectra and U_max.

Run from ``week4/scripts/`` after ``scripts/run_scan.sh``::

    /home/yao_yiyi/.venvs/amat5315/bin/python blowup_figure.py
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
ART = WEEK4 / "artifacts"
EVID = WEEK4 / "evidence"


# ---- RK4 stability function and axis crossings ------------------------------

def rk4_R(z: complex) -> complex:
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def _bisect(f, lo: float, hi: float) -> float:
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0.0) == (flo > 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def real_axis_crossing() -> float:
    return abs(_bisect(lambda x: abs(rk4_R(x + 0j)) - 1.0, -2.0, -3.0))


def imaginary_axis_crossing() -> float:
    return _bisect(lambda y: abs(rk4_R(1j * y)) - 1.0, 2.0, 3.5)


# ---- TSV loading -------------------------------------------------------------

def load_tsv(path: Path):
    ts, es = [], []
    first_nonfinite = None
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        t, e = float(parts[0]), float(parts[1])
        if math.isfinite(e) and e > 0.0:
            ts.append(t)
            es.append(e)
        elif first_nonfinite is None:
            first_nonfinite = t
    return np.asarray(ts), np.asarray(es), first_nonfinite


def main() -> None:
    real_cross = real_axis_crossing()
    imag_cross = imaginary_axis_crossing()

    # Taylor-Green diffusive prediction.
    n_tg, nu_tg, k_cut_tg = 64, 0.1, 64 // 3
    k2max_tg = 2 * k_cut_tg**2
    dt_diff_pred = real_cross / (nu_tg * k2max_tg)

    # Random advective bound from the actually measured initial U_max.
    initial = json.loads((ART / "scan" / "random-initial.json").read_text())
    u0 = np.asarray(initial["u"])
    v0 = np.asarray(initial["v"])
    u_max = float(np.sqrt(u0 * u0 + v0 * v0).max())
    k_cut_r = 128 // 3
    k_max_r = math.sqrt(2.0) * k_cut_r
    dt_adv_bound = imag_cross / (u_max * k_max_r)

    # ---- curves ---------------------------------------------------------------
    tg = {
        "0.032": load_tsv(ART / "scan" / "tg-rk4-dt0.032.tsv"),
        "0.033": load_tsv(ART / "scan" / "tg-rk4-dt0.033.tsv"),
        "0.040": load_tsv(ART / "unstable" / "taylor-green.tsv"),
    }
    rnd = {
        "rk4 0.030": load_tsv(ART / "scan" / "random-rk4-dt0.030.tsv"),
        "rk4 0.032": load_tsv(ART / "scan" / "random-rk4-dt0.032.tsv"),
        "euler 0.010": load_tsv(ART / "scan" / "random-euler-dt0.01.tsv"),
    }

    fig, (ax_tg, ax_rnd) = plt.subplots(1, 2, figsize=(12.4, 4.8), layout="constrained")

    # ---- left: Taylor-Green ---------------------------------------------------
    t_exact = np.linspace(0.0, 8.0, 400)
    ax_tg.semilogy(t_exact, 0.25 * np.exp(-0.4 * t_exact), "k--", lw=1.2,
                   label=r"exact $0.25\,e^{-0.4t}$")
    style = {"0.032": "tab:blue", "0.033": "tab:red", "0.040": "tab:orange"}
    for dt, (ts, es, stop) in tg.items():
        ax_tg.semilogy(ts, es, "-", color=style[dt], lw=1.4, label=f"RK4 $h$ = {dt}")
        if stop is not None:
            ax_tg.axvline(stop, color=style[dt], ls=":", lw=1.0)
            ax_tg.annotate(f"stop $t$ = {stop:g}", (stop, es.min()),
                           xytext=(4, 6), textcoords="offset points",
                           fontsize=7.5, color=style[dt])
    ax_tg.set_xlabel(r"$t$")
    ax_tg.set_ylabel(r"energy $E$")
    ax_tg.set_title("Taylor-Green", fontsize=10)
    ax_tg.grid(True, which="both", lw=0.3, alpha=0.35)
    ax_tg.legend(fontsize=7.6, loc="upper right")
    ax_tg.text(
        0.03, 0.06,
        "predicted diffusive limit\n$h_{diff}$ = %.5f\n$k_{cut}$ = %d, "
        r"$|k|^2_{max}$ = %d" % (dt_diff_pred, k_cut_tg, k2max_tg),
        transform=ax_tg.transAxes, fontsize=7.6, va="bottom", ha="left",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
    )

    # ---- right: random flow ---------------------------------------------------
    rstyle = {"rk4 0.030": "tab:blue", "rk4 0.032": "tab:red",
              "euler 0.010": "tab:green"}
    for name, (ts, es, stop) in rnd.items():
        ax_rnd.semilogy(ts, es, "-", color=rstyle[name], lw=1.4, label=name)
        if stop is not None:
            ax_rnd.axvline(stop, color=rstyle[name], ls=":", lw=1.0)
            ax_rnd.annotate(f"stop $t$ = {stop:g}", (stop, es.min()),
                            xytext=(4, 6), textcoords="offset points",
                            fontsize=7.5, color=rstyle[name])
    ax_rnd.set_xlabel(r"$t$")
    ax_rnd.set_ylabel(r"energy $E$")
    ax_rnd.set_title("random flow (same initial field)", fontsize=10)
    ax_rnd.grid(True, which="both", lw=0.3, alpha=0.35)
    ax_rnd.legend(fontsize=7.6, loc="upper right")
    ax_rnd.text(
        0.03, 0.06,
        "predicted advective bound\n$h_{adv}$ = %.5f\n$U_{max}$ = %.4f, "
        r"$|k|_{max}$ = %.2f" % (dt_adv_bound, u_max, k_max_r),
        transform=ax_rnd.transAxes, fontsize=7.6, va="bottom", ha="left",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
    )

    fig.savefig(EVID / "blowup.png", dpi=150)
    plt.close(fig)

    # ---- evidence text --------------------------------------------------------
    def describe(ts, es, stop):
        last_finite = float(ts[-1]) if len(ts) else float("nan")
        status = "reached the end" if stop is None else f"stopped at t = {stop:g}"
        return last_finite, status

    lines = [
        "Part 3 stability limits",
        "",
        f"RK4 real-axis crossing   = {real_cross:.6f}",
        f"RK4 imaginary crossing   = {imag_cross:.6f}",
        f"Taylor-Green: k_cut = {k_cut_tg}, |k|^2_max = {k2max_tg}, "
        f"predicted diffusive dt limit = {dt_diff_pred:.6f}",
        f"random: U_max = {u_max:.4f}, k_cut = {k_cut_r}, |k|_max = {k_max_r:.3f}, "
        f"predicted advective bound = {dt_adv_bound:.6f}",
        f"measured random boundary ratio (0.030..0.032 / bound) = "
        f"{0.030 / dt_adv_bound:.2f} .. {0.032 / dt_adv_bound:.2f}",
        "",
        "Taylor-Green runs:",
    ]
    for dt, (ts, es, stop) in tg.items():
        last, status = describe(ts, es, stop)
        lines.append(f"  dt = {dt}: last finite t = {last:g}, {status}")
    lines.append("")
    lines.append("random runs:")
    for name, (ts, es, stop) in rnd.items():
        last, status = describe(ts, es, stop)
        lines.append(f"  {name}: last finite t = {last:g}, {status}")
    lines.append("")
    lines.append("figure: evidence/blowup.png")
    text = "\n".join(lines) + "\n"
    (EVID / "blowup.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
