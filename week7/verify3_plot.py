"""Plots and animation for the Week 7 VERIFY (3) sampling study."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from forward_diffusion import CENTERS
from verify3 import assign_to_nearest_center

REVERSE_AXIS_LIMIT = 4.5
_CLUSTER_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def _colors_from_labels(labels: np.ndarray | None, num_samples: int) -> np.ndarray:
    if labels is None:
        return np.full(num_samples, _CLUSTER_COLORS[0])
    colors = np.asarray(_CLUSTER_COLORS)
    return colors[np.asarray(labels) % len(_CLUSTER_COLORS)]


def create_reverse_diffusion_animation(
    trajectory: np.ndarray,
    frame_steps: Sequence[int],
    labels: np.ndarray | None = None,
    interval_ms: int = 80,
) -> tuple[FuncAnimation, plt.Figure, plt.Axes]:
    """Animate the reverse process from x_T to the generated samples.

    ``trajectory`` is indexed by reverse step: index 0 is x_T and the final
    index is x_0. Axes are fixed for the whole animation.
    """
    trajectory = np.asarray(trajectory)
    frame_steps = list(frame_steps)
    if trajectory.ndim != 3 or trajectory.shape[2] != 2:
        raise ValueError("trajectory must have shape (steps, samples, 2)")
    if max(frame_steps) >= trajectory.shape[0] or min(frame_steps) < 0:
        raise ValueError("frame_steps must index into trajectory")

    colors = _colors_from_labels(labels, trajectory.shape[1])
    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    scatter = axis.scatter(
        trajectory[frame_steps[0], :, 0],
        trajectory[frame_steps[0], :, 1],
        s=5,
        c=colors,
        alpha=0.5,
        linewidths=0,
    )
    axis.set_xlim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
    axis.set_ylim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)
    title = axis.set_title(
        f"Reverse diffusion step = {frame_steps[0]} / 1000"
    )

    def update(frame_index: int) -> tuple:
        step = frame_steps[frame_index]
        scatter.set_offsets(trajectory[step])
        title.set_text(f"Reverse diffusion step = {step} / 1000")
        return (scatter, title)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_steps),
        interval=interval_ms,
        blit=False,
    )
    return animation, figure, axis


def render_reverse_keyframes(
    trajectory: np.ndarray,
    frame_steps: Sequence[int],
    output_path: str,
) -> None:
    """Save selected reverse-diffusion frames, including x_T and x_0."""
    trajectory = np.asarray(trajectory)
    frame_steps = list(frame_steps)
    num_columns = 3
    num_rows = math.ceil(len(frame_steps) / num_columns)
    figure, axes = plt.subplots(
        num_rows, num_columns, figsize=(4.3 * num_columns, 4.0 * num_rows)
    )
    axes = np.atleast_1d(axes).ravel()

    for panel_index, axis in enumerate(axes):
        if panel_index >= len(frame_steps):
            axis.set_visible(False)
            continue
        step = frame_steps[panel_index]
        points = trajectory[step]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=4,
            color=_CLUSTER_COLORS[0],
            alpha=0.45,
            linewidths=0,
        )
        axis.set_xlim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        axis.set_ylim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.set_title(f"Reverse step = {step}")

    figure.suptitle("Reverse diffusion from Gaussian noise to four clusters")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_final_comparison(
    generated: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, object],
    output_path: str,
) -> None:
    """Plot final generated samples beside exact Gaussian-mixture samples."""
    generated = np.asarray(generated)
    target = np.asarray(target)
    labels = np.asarray(labels)
    target_labels, _ = assign_to_nearest_center(target)
    target_labels = np.asarray(target_labels)
    centers = np.asarray(CENTERS)

    figure, (generated_axis, target_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.8), sharex=True, sharey=True
    )
    for axis, points, point_labels, title in (
        (
            generated_axis,
            generated,
            labels,
            "Generated: 2000 DDPM samples",
        ),
        (
            target_axis,
            target,
            target_labels,
            "Target: 2000 exact Gaussian-mixture samples",
        ),
    ):
        colors = np.asarray(_CLUSTER_COLORS)[point_labels % 4]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=6,
            c=colors,
            alpha=0.5,
            linewidths=0,
        )
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=80,
            c="black",
            linewidths=2.0,
            label="target centers",
            zorder=5,
        )
        axis.set_xlim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        axis.set_ylim(-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.set_title(title)

    overall = "PASS" if metrics["passed"] else "FAIL"
    figure.suptitle(
        "VERIFY (3): final generated samples vs exact target samples | "
        f"overall {overall}",
        fontsize=13,
    )
    generated_axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


__all__ = [
    "REVERSE_AXIS_LIMIT",
    "create_reverse_diffusion_animation",
    "render_final_comparison",
    "render_reverse_keyframes",
]
