"""Part 3 VERIFY (2): coarse DDPM jumps, energy distance, sampling metrics."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ddim import normalize_ddim_levels
from forward_diffusion import CENTERS, NUM_DIFFUSION_STEPS
from verify3 import assign_to_nearest_center

NUM_SAMPLES = 4000
NULL_PAIRS = 20
SAMPLING_STEPS = (1, 2, 5, 10, 20, 50, 100)
FULL_DIFFUSION_STEPS = 1000
WIDTH_TARGET = 0.30
POPULATION_TARGET = 0.25


def evenly_spaced_levels(
    num_steps: int, total_steps: int = NUM_DIFFUSION_STEPS
) -> tuple[int, ...]:
    """Return evenly spaced positive noise levels from T down to T/num_steps."""
    if num_steps <= 0 or num_steps > total_steps:
        raise ValueError("num_steps must lie in 1, ..., total_steps")
    levels = np.rint(
        np.linspace(total_steps, total_steps / num_steps, num_steps)
    ).astype(int)
    levels = tuple(int(level) for level in levels)
    if any(
        current <= following
        for current, following in zip(levels, levels[1:])
    ):
        raise ValueError("rounded levels must be strictly decreasing")
    if levels[-1] < 1:
        raise ValueError("levels must be positive")
    return levels


def coarse_ddpm_step(
    x_t: jax.Array,
    t: int | jax.Array,
    previous_t: int | jax.Array,
    predicted_noise: jax.Array,
    noise: jax.Array,
    alpha_bars: jax.Array,
) -> jax.Array:
    """One DDPM jump from t to previous_t using Equation (17).

    The jump posterior is

        q(x_prev | x_t, x0) = N(mean, jump_variance I)

    with

        jump_variance = (1 - ab_prev) / (1 - ab_t)
                        * (1 - ab_t / ab_prev).

    The final jump to previous_t=0 has zero variance.
    """
    alpha_bar_t = alpha_bars[t]
    alpha_bar_previous = alpha_bars[previous_t]
    predicted_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)

    alpha_ratio = alpha_bar_t / alpha_bar_previous
    jump_beta = 1.0 - alpha_ratio
    posterior_mean = (
        jnp.sqrt(alpha_bar_previous)
        * jump_beta
        / (1.0 - alpha_bar_t)
    ) * predicted_x0 + (
        jnp.sqrt(alpha_ratio)
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    ) * x_t
    jump_variance = (
        (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
        * jump_beta
    )
    return posterior_mean + jnp.sqrt(jump_variance) * noise


def make_coarse_ddpm_sampler(
    predictor,
    alpha_bars: jax.Array,
    levels: Sequence[int],
):
    """Compile a stochastic DDPM sampler with coarse Equation (17) jumps."""
    levels = normalize_ddim_levels(levels)
    current_levels = jnp.asarray(levels, dtype=jnp.int32)
    previous_levels = jnp.asarray(levels[1:] + (0,), dtype=jnp.int32)

    @jax.jit
    def sample(x_T: jax.Array, reverse_noise: jax.Array) -> jax.Array:
        if reverse_noise.shape[0] != len(levels):
            raise ValueError("reverse_noise must have one draw per jump")

        def body(x_t, inputs):
            t, previous_t, noise = inputs
            predicted_noise = predictor(x_t, t)
            x_previous = coarse_ddpm_step(
                x_t,
                t,
                previous_t,
                predicted_noise,
                noise,
                alpha_bars,
            )
            return x_previous, x_previous

        _, sampled_steps = jax.lax.scan(
            body, x_T, (current_levels, previous_levels, reverse_noise)
        )
        return sampled_steps[-1]

    return sample


def mean_pairwise_distance(
    first: jax.Array, second: jax.Array, chunk_size: int = 512
) -> jax.Array:
    """Mean Euclidean distance over all pairs, computed in chunks."""
    first = jnp.asarray(first, dtype=jnp.float32)
    second = jnp.asarray(second, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for start in range(0, first.shape[0], chunk_size):
        chunk = first[start : start + chunk_size]
        distances = jnp.sqrt(
            jnp.sum(
                (chunk[:, None, :] - second[None, :, :]) ** 2,
                axis=-1,
            )
        )
        total = total + jnp.sum(distances)
        count += int(distances.size)
    return total / count


def energy_distance(
    first: jax.Array, second: jax.Array, chunk_size: int = 512
) -> float:
    """Empirical energy distance between two sample sets."""
    cross = mean_pairwise_distance(first, second, chunk_size)
    first_self = mean_pairwise_distance(first, first, chunk_size)
    second_self = mean_pairwise_distance(second, second, chunk_size)
    return float(2.0 * cross - first_self - second_self)


def sample_metrics(
    samples: jax.Array | np.ndarray,
    centers: jax.Array | np.ndarray = CENTERS,
) -> dict[str, object]:
    """Cluster populations, centers, standard deviations, and width errors."""
    samples_np = np.asarray(samples, dtype=np.float64)
    centers_np = np.asarray(centers, dtype=np.float64)
    labels, _ = assign_to_nearest_center(samples_np, centers_np)
    labels = np.asarray(labels)

    clusters: dict[str, dict[str, object]] = {}
    for cluster in range(centers_np.shape[0]):
        assigned = samples_np[labels == cluster]
        population = int(assigned.shape[0])
        fraction = population / samples_np.shape[0] if len(samples_np) else 0.0
        if population:
            center = assigned.mean(axis=0)
            std = assigned.std(axis=0)
            width = float(np.sqrt((std[0] ** 2 + std[1] ** 2) / 2.0))
        else:
            center = np.full(2, np.nan)
            std = np.full(2, np.nan)
            width = float("nan")

        clusters[str(cluster)] = {
            "population": population,
            "population_fraction": float(fraction),
            "population_error": float(abs(fraction - POPULATION_TARGET)),
            "center": center.tolist(),
            "center_error": np.abs(center - centers_np[cluster]).tolist(),
            "std_x": float(std[0]),
            "std_y": float(std[1]),
            "width": width,
            "width_error": float(abs(width - WIDTH_TARGET)),
        }

    population_errors = [
        result["population_error"] for result in clusters.values()
    ]
    center_errors = [
        max(result["center_error"]) for result in clusters.values()
    ]
    width_errors = [result["width_error"] for result in clusters.values()]

    return {
        "num_samples": int(samples_np.shape[0]),
        "cluster_population_target": POPULATION_TARGET,
        "width_target": WIDTH_TARGET,
        "clusters": clusters,
        "max_population_error": float(max(population_errors)),
        "max_center_error": float(max(center_errors)),
        "max_width_error": float(max(width_errors)),
    }


__all__ = [
    "FULL_DIFFUSION_STEPS",
    "NULL_PAIRS",
    "NUM_SAMPLES",
    "POPULATION_TARGET",
    "SAMPLING_STEPS",
    "WIDTH_TARGET",
    "coarse_ddpm_step",
    "energy_distance",
    "evenly_spaced_levels",
    "make_coarse_ddpm_sampler",
    "mean_pairwise_distance",
    "sample_metrics",
]
