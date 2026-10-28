"""Part 3 VERIFY (2) sampling-metric plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_METHOD_STYLES = {
    "DDPM": {"color": "#ff7f0e", "marker": "o", "label": "DDPM"},
    "DDIM": {"color": "#1f77b4", "marker": "s", "label": "DDIM"},
    "Flow": {"color": "#2ca02c", "marker": "^", "label": "Flow Matching"},
}
_EXACT_COLOR = "#d62728"


def _series(results: dict, method: str, metric: str) -> tuple[list[int], list[float]]:
    method_results = results[method]
    steps = sorted(int(step) for step in method_results)
    values = [float(method_results[step][metric]) for step in steps]
    return steps, values


def plot_metrics_vs_steps(
    results: dict[str, dict[int, dict[str, float]]],
    null_q95: float,
    output_path: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot energy distance, population error, and width error vs steps."""
    figure, axes = plt.subplots(
        1, 3, figsize=(16.0, 4.8), layout="constrained"
    )

    energy_axis, population_axis, width_axis = axes
    for method in ("DDPM", "DDIM", "Flow"):
        if method not in results:
            continue
        style = _METHOD_STYLES[method]
        steps, values = _series(results, method, "energy_distance")
        energy_axis.plot(
            steps,
            values,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=style["label"],
        )

    energy_axis.axhline(
        null_q95, color="0.35", linestyle="--", linewidth=1.0
    )
    energy_axis.set_ylabel("energy distance")
    energy_axis.set_title("Energy distance vs sampling steps")
    energy_axis.legend(fontsize=8)

    for method in ("DDPM", "DDIM", "Flow"):
        if method not in results:
            continue
        style = _METHOD_STYLES[method]
        steps, values = _series(results, method, "max_population_error")
        population_axis.plot(
            steps,
            values,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=style["label"],
        )
    population_axis.set_ylabel("max |cluster fraction - 0.25|")
    population_axis.set_title("Maximum cluster population error")
    population_axis.legend(fontsize=8)

    for method in ("DDPM", "DDIM", "Flow"):
        if method not in results:
            continue
        style = _METHOD_STYLES[method]
        steps, values = _series(results, method, "max_width_error")
        width_axis.plot(
            steps,
            values,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=style["label"],
        )
    width_axis.axhline(
        0.03,
        color=_EXACT_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="0.03 check",
    )
    width_axis.set_ylabel("max |cluster width - 0.30|")
    width_axis.set_title("Maximum cluster width error")
    width_axis.legend(fontsize=8)

    # Exact-predictor DDIM points isolate discretization error at coarse grids.
    if "DDIM_exact" in results:
        exact_results = results["DDIM_exact"]
        for axis, metric in (
            (energy_axis, "energy_distance"),
            (population_axis, "max_population_error"),
            (width_axis, "max_width_error"),
        ):
            steps = sorted(int(step) for step in exact_results)
            values = [
                float(exact_results[step][metric]) for step in steps
            ]
            axis.plot(
                steps,
                values,
                color=_EXACT_COLOR,
                marker="*",
                markersize=10,
                linestyle="none",
                label="DDIM with exact predictor",
                zorder=6,
            )
        for axis in axes:
            axis.legend(fontsize=8)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(bottom=1e-4)
        axis.set_xlabel("sampling steps")
        axis.grid(alpha=0.25, which="both")
        all_steps = sorted(
            {
                int(step)
                for method in results.values()
                for step in method
            }
        )
        axis.set_xticks(all_steps)
        axis.set_xticklabels([str(step) for step in all_steps], rotation=45)

    energy_axis.axhspan(
        1e-4,
        null_q95,
        color="0.75",
        alpha=0.45,
        label="exact-exact null 95%",
    )
    energy_axis.legend(fontsize=8)

    figure.suptitle(
        "Part 3 VERIFY (2): DDPM, DDIM, and Flow Matching sampling",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=160)
    return figure, axes


__all__ = ["plot_metrics_vs_steps"]
