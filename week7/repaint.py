"""RePaint (Algorithm 6) for the frozen CIFAR-10 DDPM model."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cifar_ddpm import (
    DTYPE,
    UNet2DConfig,
    unet_apply,
)
from forward_diffusion import alpha_bars_from_betas, linear_beta_schedule

MASK_START = 6
MASK_END = 26
SEEDS = (7201, 7202, 7203, 7204)
REPAINT_LEVELS = tuple(range(1000, 19, -20)) + (0,)
IMAGES_PATH = Path(__file__).resolve().parent / "inputs" / "images.npz"


def make_repaint_mask(
    start: int = MASK_START, end: int = MASK_END
) -> np.ndarray:
    """Known-pixel mask: 1 outside rows/columns [start:end], 0 inside."""
    mask = np.ones((32, 32, 1), dtype=np.float32)
    mask[start:end, start:end, :] = 0.0
    return mask


def repaint_jumps() -> list[tuple[int, int]]:
    """Return the 50 reverse jumps 1000->980->...->20->0."""
    return list(zip(REPAINT_LEVELS[:-1], REPAINT_LEVELS[1:]))


def repaint_network_evaluations() -> int:
    """Two attempts per jump except the final jump: 99 evaluations."""
    return sum(1 if previous == 0 else 2 for _, previous in repaint_jumps())

def model_timestep(level: int) -> int:
    """Convert a 1-indexed schedule level to the model's 0-indexed timestep."""
    return int(level) - 1


def predict_clean_image(
    x_t: jax.Array,
    alpha_bar_t: jax.Array | float,
    predicted_noise: jax.Array,
) -> jax.Array:
    """Predict x0 and clip only the clean estimate to [-1,1]."""
    alpha_bar_t = jnp.asarray(alpha_bar_t, dtype=DTYPE)
    predicted_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)
    return jnp.clip(predicted_x0, -1.0, 1.0)


def jump_posterior_terms(
    alpha_bar_t: jax.Array | float,
    alpha_bar_previous: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """Return (jump variance, coefficient of predicted noise)."""
    alpha_bar_t = jnp.asarray(alpha_bar_t, dtype=DTYPE)
    alpha_bar_previous = jnp.asarray(alpha_bar_previous, dtype=DTYPE)
    jump_beta = 1.0 - alpha_bar_t / alpha_bar_previous
    jump_variance = (
        (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
        * jump_beta
    )
    noise_coefficient = jnp.sqrt(
        jnp.maximum(1.0 - alpha_bar_previous - jump_variance, 0.0)
    )
    return jump_variance, noise_coefficient


def posterior_jump(
    x_t: jax.Array,
    alpha_bar_t: jax.Array | float,
    alpha_bar_previous: jax.Array | float,
    predicted_noise: jax.Array,
    reverse_noise: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One clipped-x0 DDPM posterior reverse jump, with square roots."""
    alpha_bar_t = jnp.asarray(alpha_bar_t, dtype=DTYPE)
    alpha_bar_previous = jnp.asarray(alpha_bar_previous, dtype=DTYPE)
    predicted_x0 = predict_clean_image(
        x_t, alpha_bar_t, predicted_noise
    )
    jump_variance, noise_coefficient = jump_posterior_terms(
        alpha_bar_t, alpha_bar_previous
    )
    x_previous = (
        jnp.sqrt(alpha_bar_previous) * predicted_x0
        + noise_coefficient * predicted_noise
        + jnp.sqrt(jump_variance) * reverse_noise
    )
    return x_previous, predicted_x0


def forward_renoise_step(
    x_t: jax.Array, beta: jax.Array | float, noise: jax.Array
) -> jax.Array:
    """One forward diffusion step used by RePaint's undo_step."""
    beta = jnp.asarray(beta, dtype=DTYPE)
    return jnp.sqrt(1.0 - beta) * x_t + jnp.sqrt(beta) * noise


def known_observation(
    observation: jax.Array,
    alpha_bar_t: jax.Array | float,
    noise: jax.Array,
) -> jax.Array:
    """Forward-noised observed pixels at noise level t."""
    alpha_bar_t = jnp.asarray(alpha_bar_t, dtype=DTYPE)
    return (
        jnp.sqrt(alpha_bar_t) * observation
        + jnp.sqrt(1.0 - alpha_bar_t) * noise
    )


def merge_mask(
    known: jax.Array, unknown: jax.Array, mask: jax.Array
) -> jax.Array:
    """Merge known (mask=1) and unknown (mask=0) pixels."""
    return mask * known + (1.0 - mask) * unknown


def load_test_images(path: str | Path = IMAGES_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the four CIFAR-10 test images as NHWC float32 and their labels."""
    with np.load(path) as data:
        images = np.transpose(data["test"], (0, 2, 3, 1)).astype(np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
    if images.shape != (4, 32, 32, 3):
        raise ValueError("expected four 32x32 RGB test images")
    return images, labels


def _noise_factory(seed_keys: jax.Array, shape: tuple[int, ...]):
    """Compile independent batched noise draws from per-completion base keys."""

    @jax.jit
    def noise(counter: jax.Array) -> jax.Array:
        keys = jax.vmap(lambda key: jax.random.fold_in(key, counter))(seed_keys)
        return jax.vmap(
            lambda key: jax.random.normal(key, shape, dtype=DTYPE)
        )(keys)

    return noise


def repaint_batch(
    params: dict[str, jax.Array],
    config: UNet2DConfig,
    observations: np.ndarray,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run RePaint for a batch of (image, completion-seed) pairs.

    ``observations`` must already be shaped ``(batch, 32, 32, 3)`` with each
    source image repeated once per seed. The known region is replaced with
    forward-noised observations after every attempted jump.
    """
    observations = jnp.asarray(observations, dtype=DTYPE)
    batch = observations.shape[0]
    if batch != len(seeds):
        raise ValueError("one seed is required per batch element")
    mask = jnp.asarray(make_repaint_mask(), dtype=DTYPE)
    betas = linear_beta_schedule(1000)
    alpha_bars = alpha_bars_from_betas(betas)
    seed_keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
    # The same completion seed is used for every image, but the image index is
    # mixed into the key so each visible context gets independent noise.
    image_indices = jnp.arange(batch, dtype=jnp.int32)
    seed_keys = jax.vmap(jax.random.fold_in)(seed_keys, image_indices)
    split_keys = jax.vmap(
        lambda key: jax.random.split(key)
    )(seed_keys)
    key_initial = split_keys[:, 0]
    key_noise = split_keys[:, 1]
    x = jax.vmap(
        lambda key: jax.random.normal(key, (32, 32, 3), dtype=DTYPE)
    )(key_initial)
    noise = _noise_factory(key_noise, (32, 32, 3))
    # Seed the known border with its t=1000 forward observation before the
    # first network evaluation.
    initial_known_noise = noise(900000)
    initial_known = known_observation(
        observations, alpha_bars[1000], initial_known_noise
    )
    x = merge_mask(initial_known, x, mask)
    model = jax.jit(lambda sample, t: unet_apply(params, sample, t, config))

    evaluations = 0
    for step, (t, previous_t) in enumerate(repaint_jumps()):
        attempts = 1 if previous_t == 0 else 2
        for attempt in range(attempts):
            if attempt > 0:
                # Match RePaintScheduler.undo_step: walk forward through the
                # skipped levels with independent noise. The known pixels at
                # previous_t were already forward-noised observations, so this
                # carries the border back to level t without hidden-GT access.
                for forward_level in range(previous_t, t):
                    forward_noise = noise(
                        step * 1000 + attempt * 100 + forward_level
                    )
                    x = forward_renoise_step(
                        x, betas[forward_level], forward_noise
                    )

            predicted_noise = model(x, model_timestep(t))
            evaluations += 1

            # RePaint Algorithm 6 predicts and clips the clean image, then
            # takes the DDPM posterior reverse jump.
            reverse_noise = noise(600000 + step * 10 + attempt)
            unknown_previous, _ = posterior_jump(
                x,
                alpha_bars[t],
                alpha_bars[previous_t],
                predicted_noise,
                reverse_noise,
            )

            if previous_t > 0:
                known_noise = noise(700000 + step * 10 + attempt)
                known_at_previous = known_observation(
                    observations, alpha_bars[previous_t], known_noise
                )
                x = merge_mask(
                    known_at_previous, unknown_previous, mask
                )
            else:
                x = merge_mask(observations, unknown_previous, mask)

    return np.asarray(x, dtype=np.float32), {
        "network_evaluations": evaluations,
        "expected_network_evaluations": repaint_network_evaluations(),
        "levels": list(REPAINT_LEVELS),
        "jumps": len(repaint_jumps()),
        "seeds": list(seeds),
    }


def repaint_metrics(
    completions: np.ndarray,
    originals: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Hidden-pixel MSE, observed-pixel change, and completion variance."""
    completions = np.asarray(completions, dtype=np.float64)
    originals = np.asarray(originals, dtype=np.float64)
    if mask is None:
        mask = make_repaint_mask()
    mask = np.asarray(mask, dtype=np.float64)
    hidden = 1.0 - mask
    if completions.ndim != 5:
        raise ValueError("completions must have shape (images, seeds, 32, 32, 3)")
    if completions.shape[0] != originals.shape[0]:
        raise ValueError("completions and originals must share the image axis")

    hidden_bool = np.broadcast_to(hidden > 0.5, completions.shape[2:])
    observed_bool = ~hidden_bool
    difference_squared = (
        completions - originals[:, None, :, :, :]
    ) ** 2
    hidden_mse = np.asarray(
        [
            [
                np.mean(difference_squared[image, seed][hidden_bool])
                for seed in range(completions.shape[1])
            ]
            for image in range(completions.shape[0])
        ]
    )
    mean_completion = np.mean(completions, axis=1)
    mean_completion_mse = np.asarray(
        [
            np.mean((mean_completion[image][hidden_bool] - originals[image][hidden_bool]) ** 2)
            for image in range(completions.shape[0])
        ]
    )
    observed_change = np.max(
        np.abs(completions - originals[:, None, :, :, :])[:, :, observed_bool],
        axis=-1,
    )
    observed_change_per_completion = np.max(
        np.abs(completions - originals[:, None, :, :, :])[:, :, observed_bool],
        axis=-1,
    )
    variance = np.var(completions, axis=1)
    hidden_variance = variance[:, hidden_bool]
    hidden_std = np.sqrt(np.maximum(hidden_variance, 0.0))
    max_hidden_std = np.max(hidden_std, axis=-1)
    min_hidden_std = np.min(hidden_std, axis=-1)
    pairwise_difference = np.max(
        np.abs(
            completions[:, :, None, :, :, :]
            - completions[:, None, :, :, :, :]
        ),
        axis=(2, 3, 4, 5),
    )
    cross_image_differences = []
    for seed in range(completions.shape[1]):
        for first in range(completions.shape[0]):
            for second in range(first + 1, completions.shape[0]):
                cross_image_differences.append(
                    np.mean(
                        np.abs(
                            completions[first, seed][hidden_bool]
                            - completions[second, seed][hidden_bool]
                        )
                    )
                )

    return {
        "hidden_mse": hidden_mse.tolist(),
        "per_image_hidden_mse_max": np.max(hidden_mse, axis=1).tolist(),
        "per_image_hidden_mse_mean": np.mean(hidden_mse, axis=1).tolist(),
        "mean_completion": mean_completion,
        "mean_completion_hidden_mse": mean_completion_mse.tolist(),
        "observed_max_change": float(observed_change.max()),
        "observed_max_change_per_completion": (
            observed_change_per_completion.tolist()
        ),
        "per_image_observed_max_change": np.max(
            observed_change, axis=1
        ).tolist(),
        "max_hidden_std": np.max(max_hidden_std).item(),
        "per_image_max_hidden_std": max_hidden_std.tolist(),
        "per_image_min_hidden_std": min_hidden_std.tolist(),
        "max_pairwise_hidden_difference": float(pairwise_difference.max()),
        "cross_image_hidden_mean_abs_difference": (
            float(np.mean(cross_image_differences))
            if cross_image_differences
            else 0.0
        ),
        "variance": variance,
        "completions_in_range": bool(
            completions.min() >= -1.0 - 1e-6
            and completions.max() <= 1.0 + 1e-6
        ),
    }


__all__ = [
    "IMAGES_PATH",
    "MASK_END",
    "MASK_START",
    "REPAINT_LEVELS",
    "SEEDS",
    "load_test_images",
    "make_repaint_mask",
    "repaint_batch",
    "repaint_jumps",
    "repaint_metrics",
    "repaint_network_evaluations",
]
