"""Part 3 VERIFY (1) velocity-field and training-loss figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _ema(values: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    updates = np.arange(window, len(values) + 1)
    return updates, smoothed


def plot_velocity_fields(
    fields_result: dict[str, object], output_path: str
) -> tuple[plt.Figure, np.ndarray]:
    """Plot learned and exact velocity fields on the same spatial grid."""
    times = fields_result["times"]

    figure, axes = plt.subplots(
        len(times),
        2,
        figsize=(10.5, 4.0 * len(times)),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for row, time in enumerate(times):
        key = str(time)
        grid = np.asarray(fields_result["grids"][key])
        learned = np.asarray(fields_result["learned_fields"][key])
        exact = np.asarray(fields_result["exact_fields"][key])
        error = fields_result["relative_rms"][key]
        pass_text = "PASS" if fields_result["pass"][key] else "FAIL"
        max_magnitude = max(
            float(np.sqrt((field**2).sum(axis=-1)).max())
            for field in (learned, exact)
        )

        for column, (field, label) in enumerate(
            ((learned, "Learned"), (exact, "Exact"))
        ):
            axis = axes[row, column]
            u = field[:, 0]
            v = field[:, 1]
            magnitude = np.sqrt(u**2 + v**2)
            quiver = axis.quiver(
                grid[:, 0],
                grid[:, 1],
                u,
                v,
                magnitude,
                cmap="viridis",
                angles="xy",
                scale_units="xy",
                scale=8.0,
                width=0.004,
            )
            quiver.set_clim(0.0, max_magnitude)
            axis.set_xlim(-4.0, 4.0)
            axis.set_ylim(-4.0, 4.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{label}, s={key}"
                + (f" | rel RMS={error:.4f} {pass_text}" if column == 0 else "")
            )

        figure.colorbar(
            quiver,
            ax=list(axes[row, :]),
            shrink=0.85,
            pad=0.02,
            label="|v|",
        )

    figure.suptitle(
        "Part 3 VERIFY (1): learned vs exact marginal velocity fields",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=155)
    return figure, axes


def plot_flow_training_loss(
    losses: np.ndarray,
    checkpoints: list[int] | tuple[int, ...],
    output_path: str,
) -> None:
    """Plot the full Flow Matching training-loss history."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or len(losses) == 0:
        raise ValueError("losses must be a non-empty 1-D array")

    updates = np.arange(1, len(losses) + 1)
    smoothed_updates, smoothed_losses = _ema(losses)

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(updates, losses, color="0.7", alpha=0.35, linewidth=0.7)
    axis.plot(
        smoothed_updates,
        smoothed_losses,
        color="#1f77b4",
        linewidth=2.0,
        label="100-update mean",
    )
    for checkpoint in checkpoints:
        axis.axvline(
            float(checkpoint), color="#d62728", linestyle=":", linewidth=0.8
        )
    axis.set_xlim(0, len(losses) + 1)
    axis.set_yscale("log")
    axis.set_xlabel("training update")
    axis.set_ylabel("velocity MSE")
    axis.set_title("Part 3: Flow Matching velocity-network training")
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.04,
        f"final loss (last update) = {losses[-1]:.6f}\n"
        f"final 50-update mean = {losses[-50:].mean():.6f}",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


__all__ = ["plot_flow_training_loss", "plot_velocity_fields"]
