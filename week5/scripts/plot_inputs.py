#!/usr/bin/env python3
"""Week 5 Part 2 VERIFY (1): inspect and plot the reflector experiment.

Reads the official ``week5/inputs/reflector.json`` and produces a two-panel
figure:

  * left  -- the seismic survey geometry: wave-speed model, thin reflector,
             sponge border, shot and receiver positions;
  * right -- the Ricker source pulse sampled at the start of every time step.

Physical display units follow the experiment header:
one length unit = 100 m = 0.1 km, one time unit = 0.1 s, speeds in km/s.

The experiment arrays are indexed [z, x] and shot/receiver coordinates are
[x, z]. The script only reads the official input; it never rewrites it.

Run from anywhere with the week-5 JAX environment:

    week5/.venv/bin/python week5/scripts/plot_inputs.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

WEEK5 = Path(__file__).resolve().parents[1]
EXPERIMENT = WEEK5 / "inputs" / "reflector.json"
OUT = WEEK5 / "artifacts" / "inputs.png"

# Tolerances for the numerical verification of the sampled pulse.
PULSE_PEAK_TOL = 1e-12
PULSE_TROUGH_VALUE_TOL = 0.01
PULSE_TROUGH_TIME_TOL = 0.02


def ricker(t_s, f0_hz, t0_s):
    """Ricker pulse g(t) = (1 - 2*theta**2) * exp(-theta**2)."""
    theta = np.pi * f0_hz * (t_s - t0_s)
    return (1.0 - 2.0 * theta**2) * np.exp(-theta**2)


def load_experiment(path):
    data = json.loads(Path(path).read_text())
    background = np.asarray(data["background"], dtype=float)
    perturbation = np.asarray(data["perturbation"], dtype=float)
    if background.shape != (data["nz"], data["nx"]):
        raise ValueError("background shape does not match nz x nx")
    if perturbation.shape != background.shape:
        raise ValueError("perturbation shape does not match background")
    return data, background, perturbation


def main():
    if not EXPERIMENT.is_file():
        print(f"FAIL: {EXPERIMENT} does not exist", file=sys.stderr)
        return 1

    data, background, perturbation = load_experiment(EXPERIMENT)
    nx, nz = int(data["nx"]), int(data["nz"])
    dx, dt, steps = float(data["dx"]), float(data["dt"]), int(data["steps"])
    length_km = float(data["length_unit_m"]) / 1000.0   # km per length unit
    time_s = float(data["time_unit_s"])                 # s per time unit

    shots = np.asarray(data["shots"], dtype=float)
    receivers = np.asarray(data["receivers"], dtype=float)

    velocity = background + perturbation
    c_min, c_max = float(background.min()), float(background.max())

    # Reflector location: the row/column with the largest perturbation.
    z_profile = perturbation.max(axis=1)
    x_profile = perturbation.max(axis=0)
    z_reflector = int(np.argmax(z_profile))
    x_reflector = int(np.argmax(x_profile))

    # Sponge: sigma > 0 for cells closer than sponge_width to any edge.
    width = int(data["sponge_width"])
    x_index, z_index = np.meshgrid(np.arange(nx), np.arange(nz))
    distance = np.minimum.reduce([x_index, z_index,
                                  nx - 1 - x_index, nz - 1 - z_index])
    sponge = distance < width

    # Ricker pulse at the start of every step, in seconds.
    f0_hz = float(data["source_frequency"]) / time_s
    t0_s = float(data["source_peak_time"]) * time_s
    t_s = np.arange(steps) * dt * time_s
    pulse = ricker(t_s, f0_hz, t0_s)

    peak_index = int(np.argmax(pulse))
    trough_indices = [i for i in range(1, len(pulse) - 1)
                      if pulse[i] <= pulse[i - 1] and pulse[i] <= pulse[i + 1]
                      and pulse[i] < 0.0]
    peak_value, peak_time = float(pulse[peak_index]), float(t_s[peak_index])
    trough_values = [float(pulse[i]) for i in trough_indices]
    trough_times = [float(t_s[i]) for i in trough_indices]

    # Cell-centre coordinates in km for the geometry panel.
    x_centres = (np.arange(nx) + 0.5) * dx * length_km
    z_centres = (np.arange(nz) + 0.5) * dx * length_km
    x_edges = (np.arange(nx + 1)) * dx * length_km
    z_edges = (np.arange(nz + 1)) * dx * length_km

    # --- figure -------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                      layout="constrained")

    mesh = left.pcolormesh(x_edges, z_edges, velocity, cmap="magma",
                           vmin=c_min, vmax=float(velocity.max()),
                           shading="flat")
    contours = left.contour(x_centres, z_centres, perturbation,
                            levels=[0.02, 0.05, 0.10], colors="cyan",
                            linewidths=0.8)
    left.clabel(contours, fmt="%.2f", fontsize=7, colors="cyan")

    # Sponge border as a hatched band.
    left.contourf(x_centres, z_centres, sponge.astype(float),
                  levels=[0.5, 1.5], colors=["white"], alpha=0.10,
                  hatches=["////"])
    inner = width * dx * length_km
    outer = (nx - width) * dx * length_km
    left.add_patch(Rectangle((inner, inner), outer - inner, outer - inner,
                             fill=False, edgecolor="white", lw=1.0, ls="--"))

    left.plot(shots[:, 0] * length_km, shots[:, 1] * length_km, "*",
              color="red", markersize=17, markeredgecolor="black",
              markeredgewidth=0.7, linestyle="none", label="Shots (3)")
    left.plot(receivers[:, 0] * length_km, receivers[:, 1] * length_km, "v",
              color="#4fc3f7", markersize=7, markeredgecolor="black",
              markeredgewidth=0.5, linestyle="none",
              label="Receivers (14)")

    left.set_xlabel("Horizontal position (km)")
    left.set_ylabel("Depth (km)")
    left.set_title(f"Survey geometry (background $c_0 = {c_min:.1f}$ km/s)")
    left.set_aspect("equal")
    left.invert_yaxis()
    left.legend(loc="lower left", framealpha=0.85)
    left.text(0.98, 0.02, f"sponge width = {width} cells",
              transform=left.transAxes, ha="right", va="bottom",
              fontsize=8.5, color="white")
    fig.colorbar(mesh, ax=left, label="Wave speed (km/s)", fraction=0.046)

    # Ricker panel.
    right.plot(t_s, pulse, color="#1f77b4", lw=2.0)
    right.axhline(0.0, color="0.7", lw=0.8)
    right.plot(peak_time, peak_value, "o", color="#2ca02c", ms=7, zorder=4)
    right.annotate(f"peak {peak_value:.3f} at {peak_time:.2f} s",
                   xy=(peak_time, peak_value), xytext=(peak_time + 0.35, 0.55),
                   arrowprops=dict(arrowstyle="->", color="0.4"),
                   color="#2f7d32")
    for value, time, offset in zip(trough_values, trough_times,
                                   ((0.45, -0.82), (2.55, -0.82))):
        right.plot(time, value, "s", color="#d62728", ms=7, zorder=4)
        right.annotate(f"{value:.3f} at {time:.2f} s", xy=(time, value),
                       xytext=offset,
                       arrowprops=dict(arrowstyle="->", color="0.4"),
                       color="#a83232", ha="center")
    right.set_xlabel("Time (s)")
    right.set_ylabel("Source pulse $g(t)$")
    right.set_title(f"Ricker pulse ($f_0 = {f0_hz:.1f}$ Hz, "
                    f"$t_0 = {t0_s:.1f}$ s)")
    right.set_xlim(t_s[0], t_s[-1])
    right.set_ylim(pulse.min() - 0.6, pulse.max() + 0.35)
    right.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Week 5 seismic input: reflector experiment", fontsize=13)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- extracted values ---------------------------------------------------
    print("extracted reflector experiment:")
    print(f"  schema              : {data['schema']}")
    print(f"  grid (nx x nz)      : {nx} x {nz}")
    print(f"  dx, dt              : {dx:g} length units ({dx*length_km:g} km), "
          f"{dt:g} time units ({dt*time_s:g} s)")
    print(f"  steps               : {steps} (total {steps*dt*time_s:g} s)")
    print(f"  background speed    : {c_min:g} .. {c_max:g} km/s (uniform)")
    print(f"  shots ({len(shots)})          : "
          f"{[(round(float(x)*length_km, 2), round(float(z)*length_km, 2)) for x, z in shots]} km")
    print(f"  receivers ({len(receivers)})      : x = "
          f"{receivers[:, 0].min()*length_km:g} .. "
          f"{receivers[:, 0].max()*length_km:g} km, z = "
          f"{receivers[0, 1]*length_km:g} km")
    print(f"  max perturbation    : {perturbation.max():g} "
          f"at (z, x) = ({z_reflector}, {x_reflector}) -> "
          f"depth {z_reflector*length_km:.2f} km")
    print(f"  sponge              : {width} cells, strength "
          f"{data['sponge_strength']:g}")
    print(f"  Ricker peak         : {peak_value:.6f} at {peak_time:.4f} s")
    print(f"  Ricker troughs      : "
          f"{[(round(v, 6), round(tt, 4)) for v, tt in zip(trough_values, trough_times)]}")
    print(f"  wrote {OUT}")

    # --- verification -------------------------------------------------------
    checks = [
        ("reflector.json exists", True),
        ("grid is 41 x 41", (nx, nz) == (41, 41)),
        ("dx = 1 length unit", dx == 1.0),
        ("dt = 0.2 time units", dt == 0.2),
        ("240 time steps", steps == 240),
        ("uniform background 1.8 km/s", c_min == c_max == 1.8),
        ("three shots at 0.8 km depth",
         len(shots) == 3 and np.allclose(shots[:, 1], 8.0)),
        ("fourteen receivers at 0.8 km depth",
         len(receivers) == 14 and np.allclose(receivers[:, 1], 8.0)),
        ("reflector near 2.1 km depth", abs(z_reflector * length_km - 2.1) < 0.05),
        ("max perturbation is 0.15", abs(float(perturbation.max()) - 0.15) < 1e-12),
        ("Ricker peak ~1 at ~1.5 s",
         abs(peak_value - 1.0) < PULSE_PEAK_TOL
         and abs(peak_time - 1.5) < PULSE_TROUGH_TIME_TOL),
        ("Ricker has two troughs near -0.446",
         len(trough_values) == 2
         and all(abs(v - (-0.44626)) < PULSE_TROUGH_VALUE_TOL
                 for v in trough_values)),
        ("trough times near 1.02 s and 1.98 s",
         len(trough_times) == 2
         and abs(trough_times[0] - 1.02) < PULSE_TROUGH_TIME_TOL
         and abs(trough_times[1] - 1.98) < PULSE_TROUGH_TIME_TOL),
    ]
    print("\nVERIFY (1) checks:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall VERIFY (1) checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
