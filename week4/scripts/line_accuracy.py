#!/usr/bin/env python3
"""Part 1 propagation-accuracy evidence: ``week4/evidence/line-accuracy.png``.

One panel showing the Gaussian pulse after one lap ``t ~= 2*pi`` for

* the exact periodic advection-diffusion solution,
* RK4 with Fourier derivatives,
* RK4 with centred finite differences,
* forward Euler with Fourier derivatives,

plus the shared initial condition as a faint reference.

The numerical profiles and the error table come from the crate binary
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

STYLE = {
    "RK4 + Fourier": ("tab:blue", "-"),
    "RK4 + centred finite diff": ("tab:orange", "-"),
    "Euler + Fourier": ("tab:red", "-"),
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
    x = np.asarray(data["x"])
    exact = np.asarray(data["exact"])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(x, np.asarray(data["initial"]), ":", color="0.55", lw=1.0, label="start")
    ax.plot(x, exact, "-", color="k", lw=3.0, alpha=0.28,
            label="exact, after one lap")
    for case in data["cases"]:
        colour, ls = STYLE[case["label"]]
        ax.plot(x, np.asarray(case["u"]), ls, color=colour, lw=1.5,
                label=case["label"] + r",  $h$ = %g" % case["dt"])

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(-0.6, 1.3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.97, 0.04,
        r"$N$ = %d, $\nu$ = %g, $\sigma$ = %g, $t$ = %.4f"
        % (data["n"], data["nu"], data["sigma"], data["t_final"]),
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.2, alpha=0.92),
    )

    fig.tight_layout()
    fig.savefig(EVID / "line-accuracy.png", dpi=150)
    plt.close(fig)

    # ---- evidence text --------------------------------------------------------
    lines = [
        "Part 1 line propagation accuracy (one lap)",
        f"n = {data['n']}, c = {data['c']}, nu = {data['nu']}, "
        f"sigma = {data['sigma']}, t_final = {data['t_final']:.6f} (2*pi ~= "
        f"{data['t_end']:.6f})",
        "",
        f"{'method':<28}{'dt':>8}{'max abs error':>16}",
    ]
    for case in data["cases"]:
        lines.append(
            f"{case['label']:<28}{case['dt']:>8g}{case['max_abs_error']:>16.6e}"
        )
    lines.append("")
    lines.append("figure: evidence/line-accuracy.png")
    text = "\n".join(lines) + "\n"
    (EVID / "line-accuracy.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
