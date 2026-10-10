#!/usr/bin/env python3
"""Week 5 Part 2 VERIFY (2): plot the three forward shot gathers.

Reads the Rust forward traces from ``week5/artifacts/forward/traces.npy`` and
draws one panel per shot with a shared colour scale, time increasing downward,
so the direct arrival near each source and the later reflection events are
visible.

Run with the week-5 environment:

    week5/.venv/bin/python week5/scripts/plot_gathers.py
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEEK5 = Path(__file__).resolve().parents[1]
TRACES = WEEK5 / "artifacts" / "forward" / "traces.npy"
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"
OUT = WEEK5 / "artifacts" / "forward" / "gathers.png"


def main():
    traces = np.load(TRACES, allow_pickle=False)
    experiment = json.loads(EXPERIMENT.read_text())
    shots = np.asarray(experiment["shots"], dtype=float)
    receivers = np.asarray(experiment["receivers"], dtype=float)

    length_km = float(experiment["length_unit_m"]) / 1000.0
    dt_s = float(experiment["dt"]) * float(experiment["time_unit_s"])
    n_steps = traces.shape[1]

    receiver_km = receivers[:, 0] * length_km
    spacing_km = float(np.diff(receiver_km).mean())
    x_edges = np.concatenate(([receiver_km[0] - spacing_km / 2],
                              receiver_km + spacing_km / 2))
    times_s = (np.arange(n_steps) + 1) * dt_s
    time_edges = np.concatenate(([times_s[0] - dt_s / 2],
                                 times_s + dt_s / 2))

    limit = float(np.max(np.abs(traces)))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, traces.shape[0], figsize=(13.0, 4.6),
                             sharey=True, layout="constrained")
    image = None
    for shot, ax in enumerate(np.atleast_1d(axes)):
        image = ax.pcolormesh(x_edges, time_edges, traces[shot],
                              cmap="seismic", vmin=-limit, vmax=limit,
                              shading="flat", rasterized=True)
        source_km = shots[shot, 0] * length_km
        ax.axvline(source_km, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.plot([source_km], [time_edges[0] + 0.02], marker="*", color="black",
                markersize=10, clip_on=False)
        ax.set_title(f"Shot {shot}: source x = {source_km:.1f} km")
        ax.set_xlabel("Receiver position (km)")
        ax.set_xticks(receiver_km)
        ax.set_xticklabels([f"{value:.1f}" for value in receiver_km],
                           rotation=90, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(x_edges[0], x_edges[-1])

    axes[0].set_ylabel("Time (s)")
    fig.colorbar(image, ax=list(np.atleast_1d(axes)), label="Pressure",
                 fraction=0.025)
    fig.suptitle("Forward shot gathers: background velocity model", fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"  panels: {traces.shape[0]}, receivers: {traces.shape[2]}, "
          f"time: {times_s[0]:.3f}..{times_s[-1]:.3f} s, "
          f"shared scale +/-{limit:.4f}")


if __name__ == "__main__":
    main()
