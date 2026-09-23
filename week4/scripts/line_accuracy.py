#!/usr/bin/env python3
"""Part 1 accuracy evidence: ``week4/evidence/line-accuracy.png``.

Left panel : the Gaussian pulse after one lap ``t ~= 2*pi`` for the exact
             periodic advection-diffusion solution, RK4 + Fourier, RK4 + centred
             finite differences and Euler + Fourier.
Right panel: temporal convergence of Euler, midpoint, classical RK4 and the
             equal-weight four-stage control, Fourier derivatives only, with the
             fitted order ``p`` from the actual errors.

The numerical profiles, errors and fitted slopes come from the crate binary
``scripts/line_accuracy_data.rs`` (it links the real library); this script only
reads the JSON and draws.

Run from ``week4/scripts/`` with the course environment::

    /home/yao_yiyi/.venvs/amat5315/bin/python line_accuracy.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent       # week4/scripts
WEEK4 = HERE.parent                           # week4
EVID = WEEK4 / "evidence"
DATA = WEEK4 / "artifacts" / "line-accuracy"

PROP_STYLE = {
    "RK4 + Fourier": ("tab:blue", "-"),
    "RK4 + centred finite diff": ("tab:orange", "-"),
    "Euler + Fourier": ("tab:red", "-"),
}
CONV_STYLE = {
    "Euler": ("tab:red", "o"),
    "Midpoint": ("tab:orange", "s"),
    "RK4": ("tab:blue", "^"),
    "Equal-weight RK": ("0.40", "x"),
}


def generate_data() -> None:
    subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--bin", "line-accuracy-data"],
        cwd=WEEK4,
        check=True,
    )


def main() -> None:
    generate_data()
    data = json.loads((DATA / "line-accuracy-data.json").read_text())
    prop = data["propagation"]
    conv = data["convergence"]
    x = np.asarray(prop["x"])

    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.4), gridspec_kw=dict(width_ratios=[1.4, 1])
    )

    # ---- left: propagation ----------------------------------------------------
    ax = axes[0]
    ax.plot(x, np.asarray(prop["initial"]), ":", color="0.55", lw=1.0, label="start")
    ax.plot(x, np.asarray(prop["exact"]), "-", color="k", lw=3.0, alpha=0.28,
            label="exact, after one lap")
    for case in prop["cases"]:
        colour, ls = PROP_STYLE[case["label"]]
        ax.plot(x, np.asarray(case["u"]), ls, color=colour, lw=1.5,
                label=case["label"] + r",  $h$ = %g" % case["dt"])
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(-0.6, 1.3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.97, 0.04,
        r"$N$ = %d, $\nu$ = %g, $\sigma$ = %g, $t$ = %.4f"
        % (prop["n"], prop["nu"], prop["sigma"], prop["t_final"]),
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
    )

    # ---- right: temporal convergence -----------------------------------------
    ax = axes[1]
    dts = np.asarray(conv["dts"])
    for method in conv["methods"]:
        colour, marker = CONV_STYLE[method["label"]]
        ax.loglog(dts, np.asarray(method["errors"]), "-", marker=marker, ms=5.0,
                  lw=1.1, color=colour,
                  label="%s,  $p$ = %.2f" % (method["label"], method["slope"]))
    ax.set_xticks(dts)
    ax.set_xticklabels(["%g" % d for d in dts])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"max $|u(T) - u_{\rm exact}(T)|$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.03, 0.96,
        r"$N$ = %d, $\nu$ = %g, $\sigma$ = %g, $T$ = %g"
        % (conv["n"], conv["nu"], conv["sigma"], conv["t_end"]),
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
    )

    for a in axes:
        a.tick_params(labelsize=8)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)

    fig.tight_layout()
    fig.savefig(EVID / "line-accuracy.png", dpi=150)
    plt.close(fig)

    # ---- evidence text --------------------------------------------------------
    lines = [
        "Part 1 line accuracy",
        "",
        "propagation (one lap):",
        f"  n = {prop['n']}, c = {prop['c']}, nu = {prop['nu']}, "
        f"sigma = {prop['sigma']}, t_final = {prop['t_final']:.6f} "
        f"(2*pi ~= {prop['t_end']:.6f})",
        f"  {'method':<28}{'dt':>8}{'max abs error':>16}",
    ]
    for case in prop["cases"]:
        lines.append(
            f"  {case['label']:<28}{case['dt']:>8g}{case['max_abs_error']:>16.6e}"
        )

    lines += [
        "",
        "temporal convergence (Fourier only):",
        f"  n = {conv['n']}, c = {conv['c']}, nu = {conv['nu']}, "
        f"sigma = {conv['sigma']}, t_end = {conv['t_end']}",
        f"  {'method':<18}{'dt':>8}{'max error':>16}",
    ]
    for method in conv["methods"]:
        for dt, err in zip(conv["dts"], method["errors"]):
            lines.append(f"  {method['label']:<18}{dt:>8g}{err:>16.6e}")
    lines += [""]
    for method in conv["methods"]:
        lines.append(f"  fitted slope, {method['label']:<18} p = {method['slope']:.4f}")

    lines += ["", "figure: evidence/line-accuracy.png"]
    text = "\n".join(lines) + "\n"
    (EVID / "line-accuracy.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
