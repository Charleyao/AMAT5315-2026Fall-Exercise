"""Figures 19-21 for Week 8 Part 6 scaling laws."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .scaling_laws import (
    CHINCHILLA_COMPUTE,
    CHINCHILLA_D,
    CHINCHILLA_N,
    PARAMETER_NAMES,
    PUBLISHED_PARAMS,
    compute_optimal,
    log_residuals,
    predict_loss,
    ratio_at_compute,
)


def plot_figure19(
    N: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    D: np.ndarray,
    params: tuple,
    output_path: str | Path,
) -> Path:
    """Figure 19: loss versus model size and measured versus fitted loss."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.0, 6.0))

    scatter = left.scatter(
        N,
        L,
        c=C,
        cmap="viridis",
        norm=LogNorm(vmin=C.min(), vmax=C.max()),
        s=18,
        alpha=0.75,
        edgecolors="none",
        label="measured runs",
    )
    cbar = fig.colorbar(scatter, ax=left, pad=0.015)
    cbar.set_label("training compute C (FLOP)")

    n_grid = np.logspace(7.5, 10.5, 400)
    budgets = (1e19, 1e20, 1e21)
    budget_colors = ("#1f5fa8", "#d08b1f", "#b03a2e")
    for budget, color in zip(budgets, budget_colors):
        d_grid = budget / (6.0 * n_grid)
        loss_curve = predict_loss(params, n_grid, d_grid)
        left.plot(
            n_grid,
            loss_curve,
            color=color,
            lw=2.0,
            label=f"C = {budget:.0e}",
        )
        n_star, d_star = compute_optimal(params, np.array([budget]))
        loss_star = predict_loss(params, n_star, d_star)[0]
        left.plot(
            n_star,
            loss_star,
            marker="*",
            color=color,
            ms=12,
            markeredgecolor="white",
        )

    left.set_xscale("log")
    left.set_xlabel("model size N (parameters)")
    left.set_ylabel("loss L")
    left.set_ylim(1.75, 3.6)
    left.set_xlim(5e7, 3e10)
    left.set_title("Loss versus model size at fixed compute")
    left.grid(alpha=0.22, which="both")
    left.legend(frameon=False, fontsize=8.5, loc="upper right")

    fitted = predict_loss(params, N, D)
    right.scatter(
        fitted,
        L,
        s=18,
        color="#1f5fa8",
        alpha=0.75,
        edgecolors="none",
    )
    limits = [
        min(fitted.min(), L.min()) - 0.03,
        max(fitted.max(), L.max()) + 0.03,
    ]
    right.plot(limits, limits, color="0.3", ls="--", lw=1.0)
    right.set_xlabel("fitted loss")
    right.set_ylabel("measured loss")
    rmse = float(np.sqrt(np.mean((L - fitted) ** 2)))
    right.set_title(f"Measured versus fitted loss (RMSE = {rmse:.4f})")
    right.grid(alpha=0.22)

    fig.suptitle("Figure 19 reproduction: Chinchilla scaling-law fit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_figure20(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    params: tuple,
    published: tuple,
    output_path: str | Path,
) -> Path:
    """Figure 20: log-loss residuals for our fit and the published law."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ours = log_residuals(params, N, D, L)
    pub = log_residuals(published, N, D, L)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    left.axhline(0.0, color="0.3", lw=1.0)
    left.scatter(
        N,
        ours,
        s=16,
        alpha=0.7,
        color="#1f5fa8",
        label=f"our fit (mean {ours.mean():+.4f})",
    )
    left.scatter(
        N,
        pub,
        s=16,
        alpha=0.7,
        color="#d08b1f",
        label=f"published (mean {pub.mean():+.4f})",
    )
    left.set_xscale("log")
    left.set_xlabel("model size N")
    left.set_ylabel("log-loss residual (predicted - observed)")
    left.set_title("Residuals versus model size")
    left.grid(alpha=0.22, which="both")
    left.legend(frameon=False, fontsize=8.5)

    bins = np.linspace(-0.05, 0.06, 35)
    right.hist(
        ours,
        bins=bins,
        alpha=0.55,
        color="#1f5fa8",
        label="our fit",
        density=True,
    )
    right.hist(
        pub,
        bins=bins,
        alpha=0.55,
        color="#d08b1f",
        label="published",
        density=True,
    )
    right.axvline(
        ours.mean(), color="#1f5fa8", ls="--", lw=1.5, label="our mean"
    )
    right.axvline(
        pub.mean(), color="#d08b1f", ls="--", lw=1.5, label="published mean"
    )
    right.axvline(0.0, color="0.3", lw=1.0)
    right.set_xlabel("log-loss residual")
    right.set_ylabel("density")
    right.set_title("Residual histograms")
    right.grid(alpha=0.22)
    right.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Figure 20 reproduction: residual comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_figure21(
    params: tuple,
    bootstrap_draws: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Figure 21: compute-optimal N* and D* versus compute."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compute_grid = np.logspace(18, 25, 120)
    published_tuple = tuple(
        float(PUBLISHED_PARAMS[name]) for name in PARAMETER_NAMES
    )
    n_ours, d_ours = compute_optimal(params, compute_grid)
    n_pub, d_pub = compute_optimal(published_tuple, compute_grid)

    n_draws = np.empty((len(bootstrap_draws), len(compute_grid)))
    d_draws = np.empty_like(n_draws)
    for index, draw in enumerate(bootstrap_draws):
        n_draws[index], d_draws[index] = compute_optimal(
            draw, compute_grid
        )
    n_lower, n_upper = np.percentile(n_draws, [2.5, 97.5], axis=0)
    d_lower, d_upper = np.percentile(d_draws, [2.5, 97.5], axis=0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.0, 6.2))
    left.fill_between(
        compute_grid,
        n_lower,
        n_upper,
        color="#1f5fa8",
        alpha=0.18,
        label="our fit 95% bootstrap",
    )
    left.plot(
        compute_grid,
        n_ours,
        color="#1f5fa8",
        lw=2.0,
        label="our fit",
    )
    left.plot(
        compute_grid,
        n_pub,
        color="#d08b1f",
        lw=2.0,
        ls="--",
        label="published parameters",
    )
    left.scatter(
        [CHINCHILLA_COMPUTE],
        [CHINCHILLA_N],
        marker="*",
        s=180,
        color="#b03a2e",
        edgecolors="white",
        zorder=5,
        label="Chinchilla (70B)",
    )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("training compute C (FLOP)")
    left.set_ylabel("compute-optimal N* (parameters)")
    left.set_title("Compute-optimal model size")
    left.grid(alpha=0.22, which="both")
    left.legend(frameon=False, fontsize=8.3)

    right.fill_between(
        compute_grid,
        d_lower,
        d_upper,
        color="#1f5fa8",
        alpha=0.18,
        label="our fit 95% bootstrap",
    )
    right.plot(
        compute_grid,
        d_ours,
        color="#1f5fa8",
        lw=2.0,
        label="our fit",
    )
    right.plot(
        compute_grid,
        d_pub,
        color="#d08b1f",
        lw=2.0,
        ls="--",
        label="published parameters",
    )
    right.scatter(
        [CHINCHILLA_COMPUTE],
        [CHINCHILLA_D],
        marker="*",
        s=180,
        color="#b03a2e",
        edgecolors="white",
        zorder=5,
        label="Chinchilla (1.4T)",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("training compute C (FLOP)")
    right.set_ylabel("compute-optimal D* (tokens)")
    right.set_title("Compute-optimal data size")
    right.grid(alpha=0.22, which="both")
    right.legend(frameon=False, fontsize=8.3)

    ratio = ratio_at_compute(params)
    fig.suptitle(
        "Figure 21 reproduction: compute-optimal allocation\n"
        f"at C=5.76e23: N*={ratio['N_star']/1e9:.1f}B, "
        f"D*={ratio['D_star']/1e12:.2f}T, "
        f"D*/N*={ratio['D_over_N']:.2f} (Chinchilla: 20)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
