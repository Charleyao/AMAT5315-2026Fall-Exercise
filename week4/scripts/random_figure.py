#!/usr/bin/env python3
"""Part 3 random-flow evidence: ``week4/evidence/random.png``.

Four snapshots (t = 0, 2, 5, 10) of the vorticity field of the seeded random run,
all drawn with *one* shared colour scale so the viscous decay is visible rather
than renormalised away. Each panel is labelled with its ``t``, ``E`` and ``Z``.

Reads ``artifacts/random/`` produced by ``scripts/run_random.sh``; run from
``week4/scripts/`` with the course environment::

    /home/yao_yiyi/.venvs/amat5315/bin/python random_figure.py
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
ART = WEEK4 / "artifacts" / "random"
EVID = WEEK4 / "evidence"

TIMES = (0.0, 2.0, 5.0, 10.0)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_tsv(path: Path) -> dict[float, tuple[float, float]]:
    out: dict[float, tuple[float, float]] = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 3:
            out[round(float(parts[0]), 6)] = (float(parts[1]), float(parts[2]))
    return out


def main() -> None:
    run = json.loads((ART / "run.json").read_text())
    frames = read_jsonl(ART / "fields.jsonl")
    diag = read_tsv(WEEK4 / "artifacts" / "random.tsv")

    n = run["n"]
    by_t = {round(fr["t"], 6): fr for fr in frames}
    chosen = [by_t[t] for t in TIMES]
    omegas = [np.asarray(fr["omega"]).reshape(n, n) for fr in chosen]
    vmax = float(max(np.abs(w).max() for w in omegas))

    x = 2.0 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x)

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6), layout="constrained")
    mesh = None
    for ax, t, omega in zip(axes, TIMES, omegas):
        mesh = ax.pcolormesh(X, Y, omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="auto")
        e, z = diag[round(t, 6)]
        ax.set_title(f"t = {t:g}\nE = {e:.4f}   Z = {z:.4f}", fontsize=9)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"vorticity $\omega$  (shared scale)", size=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        r"Random flow, RK4, $n$ = %d, seed = %d, $k \in [%d, %d]$, "
        r"$\nu$ = %g, $h$ = %g"
        % (n, run["seed"], run["k_band"][0], run["k_band"][1], run["nu"], run["dt"]),
        fontsize=10,
    )
    fig.savefig(EVID / "random.png", dpi=150)
    plt.close(fig)

    # ---- diagnostics ----------------------------------------------------------
    e0, z0 = diag[0.0]
    e10, z10 = diag[10.0]
    u0 = np.asarray(chosen[0]["u"])
    v0 = np.asarray(chosen[0]["v"])
    u_max = float(np.sqrt(u0 * u0 + v0 * v0).max())

    lines = [
        "Part 3 standard random flow",
        f"n = {n}, seed = {run['seed']}, k_band = {run['k_band']}, "
        f"method = {run['method']}, nu = {run['nu']}, dt = {run['dt']}, "
        f"t_end = {run['t_end']}, snapshot_every = {run['snapshot_every']}",
        f"snapshots stored: {len(frames)}",
        "",
        "t        E            Z",
    ]
    for t in TIMES:
        e, z = diag[round(t, 6)]
        lines.append(f"{t:<8g} {e:<12.6f} {z:<12.6f}")
    lines += [
        "",
        f"energy reduction factor   E(0)/E(10) = {e0 / e10:.3f}  (E(10)/E(0) = {e10 / e0:.3f})",
        f"enstrophy reduction factor Z(0)/Z(10) = {z0 / z10:.3f}  (Z(10)/Z(0) = {z10 / z0:.3f})",
        f"initial U_max = {u_max:.4f}",
        f"shared vorticity scale: +/-{vmax:.4f}",
        "figure: evidence/random.png",
    ]
    text = "\n".join(lines) + "\n"
    (EVID / "random.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
