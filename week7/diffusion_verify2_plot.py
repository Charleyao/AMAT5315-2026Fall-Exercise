"""VERIFY (2) figures: training loss and checkpoint endpoint evolution."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

VERIFY2_AXIS_LIMIT = 5.0
_TRAJECTORY_COLORS = plt.get_cmap("tab10").colors


def _ema(values: np.ndarray, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    updates = np.arange(window, len(values) + 1)
    return updates, smoothed


def _loss_at_checkpoint(losses: np.ndarray, checkpoint: int) -> float:
    index = max(int(checkpoint) - 1, 0)
    return float(losses[min(index, len(losses) - 1)])


def _clamp_to_axis(points: np.ndarray, limit: float) -> np.ndarray:
    """Place off-screen points on the axis boundary for visualization."""
    points = np.asarray(points, dtype=np.float64)
    radii = np.max(np.abs(points), axis=1)
    scales = np.where(radii > limit, 0.97 * limit / np.maximum(radii, 1e-12), 1.0)
    return points * scales[:, None]


def _outside_axis(points: np.ndarray, limit: float) -> np.ndarray:
    return np.max(np.abs(np.asarray(points)), axis=1) > limit


def create_endpoint_evolution_animation(
    checkpoints: Sequence[int],
    endpoints: np.ndarray,
    losses: np.ndarray,
    interval_ms: int = 900,
    axis_limit: float = VERIFY2_AXIS_LIMIT,
) -> tuple[FuncAnimation, plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Animate how the 10 DDPM endpoints change across training checkpoints.

    The right panel contains the full training-loss history, with the current
    checkpoint and its loss marked.
    """
    checkpoints = np.asarray(list(checkpoints), dtype=np.int64)
    endpoints = np.asarray(endpoints, dtype=np.float32)
    losses = np.asarray(losses, dtype=np.float64)
    if endpoints.ndim != 3 or endpoints.shape[2] != 2:
        raise ValueError("endpoints must have shape (checkpoints, trajectories, 2)")
    if endpoints.shape[0] != len(checkpoints):
        raise ValueError("one endpoint set is required per checkpoint")
    if losses.ndim != 1 or len(losses) == 0:
        raise ValueError("losses must be a non-empty 1-D array")

    loss_updates = np.arange(1, len(losses) + 1)
    smoothed_updates, smoothed_losses = _ema(losses)

    figure, (scatter_axis, loss_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.4), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    figure.suptitle(
        "VERIFY (2): DDPM endpoints across training checkpoints",
        fontsize=13,
    )

    def update(frame_index: int) -> tuple:
        checkpoint = int(checkpoints[frame_index])
        current_endpoints = endpoints[frame_index]

        scatter_axis.clear()
        outside = _outside_axis(current_endpoints, axis_limit)
        boundary_points = _clamp_to_axis(current_endpoints, axis_limit)
        for trajectory_index in range(endpoints.shape[1]):
            color = _TRAJECTORY_COLORS[
                trajectory_index % len(_TRAJECTORY_COLORS)
            ]
            trail = endpoints[: frame_index + 1, trajectory_index]
            if len(trail) > 1:
                scatter_axis.plot(
                    trail[:, 0],
                    trail[:, 1],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
            if outside[trajectory_index]:
                scatter_axis.scatter(
                    boundary_points[trajectory_index, 0],
                    boundary_points[trajectory_index, 1],
                    color=color,
                    marker="x",
                    s=38,
                    alpha=0.85,
                    zorder=4,
                )
            else:
                scatter_axis.scatter(
                    current_endpoints[trajectory_index, 0],
                    current_endpoints[trajectory_index, 1],
                    color=color,
                    s=48,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
        offscreen_count = int(outside.sum())
        if offscreen_count:
            visibility_text = (
                f"{offscreen_count}/{len(outside)} endpoints outside "
                f"fixed axes (x marks at boundary)"
            )
        else:
            visibility_text = f"all {len(outside)} endpoints in view"
        scatter_axis.text(
            0.02,
            0.98,
            visibility_text,
            transform=scatter_axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        scatter_axis.set_xlim(-axis_limit, axis_limit)
        scatter_axis.set_ylim(-axis_limit, axis_limit)
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.set_xlabel("x")
        scatter_axis.set_ylabel("y")
        scatter_axis.grid(alpha=0.25)
        scatter_axis.set_title(
            f"10 generated endpoints | checkpoint {frame_index + 1}/"
            f"{len(checkpoints)} | update = {checkpoint}",
            fontsize=10,
        )

        loss_axis.clear()
        loss_axis.plot(
            loss_updates, losses, color="0.65", alpha=0.35, linewidth=0.7
        )
        loss_axis.plot(
            smoothed_updates,
            smoothed_losses,
            color="#1f77b4",
            linewidth=1.8,
            label="training MSE (100-update mean)",
        )
        current_loss = _loss_at_checkpoint(losses, checkpoint)
        loss_axis.axvline(
            checkpoint, color="#d62728", linestyle="--", linewidth=1.0
        )
        loss_axis.scatter(
            [checkpoint],
            [current_loss],
            color="#d62728",
            s=45,
            zorder=4,
            label="current checkpoint",
        )
        loss_axis.set_yscale("log")
        loss_axis.set_xlim(0, len(losses) + 1)
        loss_axis.set_xlabel("training update")
        loss_axis.set_ylabel("MSE loss")
        loss_axis.grid(alpha=0.25, which="both")
        loss_axis.legend(loc="upper right", fontsize=8)
        loss_axis.set_title(
            f"training loss at update = {checkpoint} | "
            f"{'first-batch loss' if checkpoint == 0 else 'loss'} = "
            f"{current_loss:.6f}",
            fontsize=10,
        )

        return tuple(scatter_axis.collections) + tuple(loss_axis.lines)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(checkpoints),
        interval=interval_ms,
        blit=False,
    )
    figure.tight_layout()
    return animation, figure, (scatter_axis, loss_axis)


def render_checkpoint_grid(
    checkpoints: Sequence[int],
    trajectories: np.ndarray,
    output_path: str,
    axis_limit: float = VERIFY2_AXIS_LIMIT,
) -> None:
    """Draw all 10 reverse trajectories at every checkpoint."""
    checkpoints = np.asarray(list(checkpoints), dtype=np.int64)
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError(
            "trajectories must have shape (checkpoints, reverse_steps, trajectories, 2)"
        )
    if trajectories.shape[0] != len(checkpoints):
        raise ValueError("one trajectory block is required per checkpoint")

    num_columns = 3
    num_rows = math.ceil(len(checkpoints) / num_columns)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.3 * num_columns, 4.0 * num_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for panel_index, axis in enumerate(axes):
        if panel_index >= len(checkpoints):
            axis.set_visible(False)
            continue
        checkpoint_trajectories = trajectories[panel_index]
        final_endpoints = checkpoint_trajectories[-1]
        outside = _outside_axis(final_endpoints, axis_limit)
        boundary_points = _clamp_to_axis(final_endpoints, axis_limit)
        for trajectory_index in range(checkpoint_trajectories.shape[1]):
            color = _TRAJECTORY_COLORS[
                trajectory_index % len(_TRAJECTORY_COLORS)
            ]
            path = checkpoint_trajectories[:, trajectory_index]
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                alpha=0.48,
                linewidth=0.8,
            )
            if outside[trajectory_index]:
                axis.scatter(
                    boundary_points[trajectory_index, 0],
                    boundary_points[trajectory_index, 1],
                    color=color,
                    marker="x",
                    s=28,
                    alpha=0.8,
                    zorder=4,
                )
            else:
                axis.scatter(
                    path[-1, 0],
                    path[-1, 1],
                    color=color,
                    s=24,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=3,
                )
            axis.scatter(
                path[0, 0],
                path[0, 1],
                color=color,
                marker="x",
                s=18,
                alpha=0.55,
                zorder=2,
            )
        offscreen_count = int(outside.sum())
        axis.text(
            0.02,
            0.02,
            (
                f"{offscreen_count}/10 endpoints off-screen"
                if offscreen_count
                else "all endpoints in view"
            ),
            transform=axis.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.set_title(f"update = {int(checkpoints[panel_index])}")

    figure.suptitle(
        "VERIFY (2): 10 DDPM trajectories at every training checkpoint\n"
        "(x marks x_T; dots mark generated endpoints)",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_training_loss(
    losses: np.ndarray,
    checkpoints: Sequence[int],
    output_path: str,
) -> None:
    """Plot the full training loss history and the checkpoint positions."""
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
    axis.set_ylabel("MSE loss")
    axis.set_title("Week 7 Part 1: time-conditioned noise-predictor training")
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
