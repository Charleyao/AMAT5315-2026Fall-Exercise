#!/usr/bin/env python3
"""Part 2 Taylor-Green evidence: ``week4/evidence/taylor-green.png``.

Two panels (t = 0 and t = 1) of the vorticity field with the recovered velocity
overlaid as arrows. Both panels share one vorticity colour scale, so the t = 1
decay is visible instead of being renormalised away. It also prints and stores
the relative velocity error of the last stored frame against
``artifacts/taylor-green/exact-t1.json``.

Reads the artifacts produced by ``scripts/run_taylor_green.sh``; run from
``week4/scripts/`` with the course environment::

    /home/yao_yiyi/.venvs/amat5315/bin/python taylor_green_figure.py
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
ART = WEEK4 / "artifacts" / "taylor-green"
EVID = WEEK4 / "evidence"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def relative_velocity_error(frame: dict, exact: dict) -> float:
    u = np.asarray(frame["u"]) - np.asarray(exact["u"])
    v = np.asarray(frame["v"]) - np.asarray(exact["v"])
    ue = np.asarray(exact["u"])
    ve = np.asarray(exact["v"])
    return math.sqrt(float(np.sum(u * u + v * v)) / float(np.sum(ue * ue + ve * ve)))


def main() -> None:
    run = json.loads((ART / "run.json").read_text())
    frames = read_jsonl(ART / "fields.jsonl")
    exact = json.loads((ART / "exact-t1.json").read_text())

    n = run["n"]
    x = 2.0 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x)  # rows: y (iy), columns: x (ix)

    first, last = frames[0], frames[-1]
    omega0 = np.asarray(first["omega"]).reshape(n, n)
    omega1 = np.asarray(last["omega"]).reshape(n, n)
    # One shared colour scale, taken over both frames.
    vmax = float(max(np.abs(omega0).max(), np.abs(omega1).max()))

    rel_err = relative_velocity_error(last, exact)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), layout="constrained")
    step = max(1, n // 16)
    for ax, frame, omega, title in (
        (axes[0], first, omega0, r"$t$ = 0"),
        (axes[1], last, omega1, r"$t$ = 1"),
    ):
        mesh = ax.pcolormesh(X, Y, omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="auto")
        u = np.asarray(frame["u"]).reshape(n, n)
        v = np.asarray(frame["v"]).reshape(n, n)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  u[::step, ::step], v[::step, ::step],
                  color="k", pivot="mid", scale=18, width=0.0035, alpha=0.75)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(0.0, 2.0 * np.pi)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"vorticity $\omega$", size=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        r"Taylor-Green, RK4, $n$ = %d, $\nu$ = %g, $h$ = %g: "
        r"E = %.6f $\to$ %.6f at $t$ = 1"
        % (n, run["nu"], run["dt"], 0.25, last_E(frames)),
        fontsize=10,
    )
    fig.savefig(EVID / "taylor-green.png", dpi=150)
    plt.close(fig)

    # ---- evidence text --------------------------------------------------------
    theory_E = 0.25 * math.exp(-4.0 * run["nu"] * run["t_end"])
    theory_Z = 0.5 * math.exp(-4.0 * run["nu"] * run["t_end"])
    lines = [
        "Part 2 Taylor-Green t = 0 -> t = 1",
        f"case = {run['case']}, n = {n}, method = {run['method']}, "
        f"nu = {run['nu']}, dt = {run['dt']}, t_end = {run['t_end']}, "
        f"snapshot_every = {run['snapshot_every']}",
        f"snapshots stored: {len(frames)}",
        f"theory t = 1: E = {theory_E:.6f}, Z = {theory_Z:.6f}",
        f"relative velocity error (last frame vs exact t = 1) = {rel_err:.3e}",
        "figure: evidence/taylor-green.png",
    ]
    text = "\n".join(lines) + "\n"
    (EVID / "taylor-green.txt").write_text(text)
    print(text, end="")


def last_E(frames: list[dict]) -> float:
    u = np.asarray(frames[-1]["u"])
    v = np.asarray(frames[-1]["v"])
    return float(0.5 * np.mean(u * u + v * v))


if __name__ == "__main__":
    main()
