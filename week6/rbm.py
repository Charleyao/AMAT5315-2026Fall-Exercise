"""Basic Restricted Boltzmann Machine for Week 6 Part 2 (VERIFY (1)).

A binary-binary RBM with 784 visible units (28x28 binarized MNIST pixels) and
128 hidden units. This module implements the model primitives only:

* binarization ``v = 1 if pixel >= 128 else 0`` and spins ``s = 2v - 1``;
* parameter initialization from ``rbm.toml``;
* the RBM energy and the visible free energy;
* hidden/visible conditional probabilities;
* binary block Gibbs sampling (hidden block, then visible block).

No contrastive-divergence training is performed here (that is a later part).

Usage::

    cd week6
    uv run python rbm.py
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import prepare_data as pd

# The learning sheet fixes float32 precision for the model.
jax.config.update("jax_enable_x64", False)

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
CONFIG_PATH = WEEK_DIR / "rbm.toml"

PIXEL_THRESHOLD = 128
VISIBLE_UNITS = 784  # 28 * 28
HIDDEN_UNITS = 128


@dataclass(frozen=True)
class RBM:
    """RBM parameters: weights ``W`` and visible/hidden biases ``a``, ``b``."""

    W: jax.Array  # (visible_units, hidden_units)
    a: jax.Array  # (visible_units,)
    b: jax.Array  # (hidden_units,)


# --- Configuration --------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the learning-sheet settings from ``rbm.toml``."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


# --- Data representation --------------------------------------------------
def binarize(images: np.ndarray, threshold: int = PIXEL_THRESHOLD) -> np.ndarray:
    """Map grayscale pixels to binary units: ``1`` if pixel >= 128 else ``0``."""
    return (np.asarray(images) >= threshold).astype(np.uint8)


def to_spins(binary: np.ndarray) -> np.ndarray:
    """Map binary pixels ``v`` to spins ``s = 2v - 1`` (values -1 or +1)."""
    return (2 * np.asarray(binary).astype(np.int8) - 1).astype(np.int8)


def load_binary_mnist(
    split: str = "train", directory: Path = MNIST_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split from the verified MNIST files as flattened binary images.

    Returns ``(images, labels)`` with ``images`` of shape ``(N, 784)`` in the
    original stored order; no shuffling or re-splitting happens here.
    """
    data = pd.load_mnist(directory)
    images = binarize(data[f"{split}_images"]).reshape(len(data[f"{split}_images"]), -1)
    return images, data[f"{split}_labels"]


def visible_bias_from_pixel_frequency(
    binary_images: np.ndarray, clip: tuple[float, float] = (0.01, 0.99)
) -> np.ndarray:
    """Visible bias ``a_i = logit(clip(p_i))`` from the training pixel frequency."""
    frequency = np.clip(
        np.asarray(binary_images).mean(axis=0, dtype=np.float64), clip[0], clip[1]
    )
    return np.log(frequency / (1.0 - frequency)).astype(np.float32)


# --- Initialization -------------------------------------------------------
def init_params(
    key: jax.Array,
    visible_units: int = VISIBLE_UNITS,
    hidden_units: int = HIDDEN_UNITS,
    weight_std: float = 0.01,
    hidden_bias: float = 0.0,
    visible_bias: np.ndarray | None = None,
) -> RBM:
    """Initialize ``W ~ Normal(0, weight_std^2)``, ``b = hidden_bias``.

    ``a`` is either supplied (logit pixel frequencies) or zero.
    """
    W = weight_std * jax.random.normal(
        key, (visible_units, hidden_units), dtype=jnp.float32
    )
    if visible_bias is None:
        a = jnp.zeros((visible_units,), dtype=jnp.float32)
    else:
        a = jnp.asarray(visible_bias, dtype=jnp.float32)
    b = jnp.full((hidden_units,), hidden_bias, dtype=jnp.float32)
    return RBM(W=W, a=a, b=b)


def init_from_config(config: dict, binary_train_images: np.ndarray) -> RBM:
    """Initialize the RBM exactly as specified in the ``[initialization]`` section."""
    init = config["initialization"]
    model = config["model"]
    visible_bias = visible_bias_from_pixel_frequency(
        binary_train_images, tuple(init["pixel_frequency_clip"])
    )
    key = jax.random.PRNGKey(init["seed"])
    return init_params(
        key,
        visible_units=model["visible_units"],
        hidden_units=model["hidden_units"],
        weight_std=init["weight_std"],
        hidden_bias=init["hidden_bias"],
        visible_bias=visible_bias,
    )


# --- Model math -----------------------------------------------------------
def energy(params: RBM, v: jax.Array, h: jax.Array) -> jax.Array:
    """RBM energy ``E(v, h) = -a·v - b·h - v^T W h``.

    Supports a single configuration (1-D ``v``/``h``) or batches (leading
    dimension) and returns the energy per configuration.
    """
    v = jnp.asarray(v, dtype=jnp.float32)
    h = jnp.asarray(h, dtype=jnp.float32)
    return -v @ params.a - h @ params.b - jnp.sum((v @ params.W) * h, axis=-1)


def free_energy(params: RBM, v: jax.Array) -> jax.Array:
    """Visible free energy ``F(v) = -a·v - sum_j softplus(b_j + (v^T W)_j)``."""
    v = jnp.asarray(v, dtype=jnp.float32)
    return -v @ params.a - jnp.sum(
        jax.nn.softplus(v @ params.W + params.b), axis=-1
    )


def hidden_prob(params: RBM, v: jax.Array) -> jax.Array:
    """``P(h_j = 1 | v) = sigmoid(b_j + (v^T W)_j)``."""
    v = jnp.asarray(v, dtype=jnp.float32)
    return jax.nn.sigmoid(v @ params.W + params.b)


def visible_prob(params: RBM, h: jax.Array) -> jax.Array:
    """``P(v_i = 1 | h) = sigmoid(a_i + (W h)_i)``."""
    h = jnp.asarray(h, dtype=jnp.float32)
    return jax.nn.sigmoid(h @ params.W.T + params.a)


def sample_bernoulli(key: jax.Array, probs: jax.Array) -> jax.Array:
    """Sample binary values with the given probabilities (float32 0/1)."""
    probs = jnp.asarray(probs, dtype=jnp.float32)
    return jax.random.bernoulli(key, probs).astype(jnp.float32)


def gibbs_step(params: RBM, v: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """One binary block-Gibbs step: resample all hidden, then all visible units."""
    key_hidden, key_visible = jax.random.split(key)
    h = sample_bernoulli(key_hidden, hidden_prob(params, v))
    v_new = sample_bernoulli(key_visible, visible_prob(params, h))
    return v_new, h


def gibbs_chain(params: RBM, v0: jax.Array, key: jax.Array, steps: int) -> jax.Array:
    """Run ``steps`` block-Gibbs transitions and return the final visible state."""
    keys = jax.random.split(key, steps)

    def body(v, step_key):
        v_new, _ = gibbs_step(params, v, step_key)
        return v_new, None

    v_final, _ = jax.lax.scan(body, jnp.asarray(v0, dtype=jnp.float32), keys)
    return v_final


def main() -> None:
    config = load_config()
    images, labels = load_binary_mnist("train")
    params = init_from_config(config, images)

    print("Week 6 Part 2 — basic RBM")
    print(f"  training images: {images.shape[0]}")
    print(f"  visible units:   {params.W.shape[0]} (28x28 binary pixels)")
    print(f"  hidden units:    {params.W.shape[1]}")
    print(f"  W shape:         {tuple(np.shape(params.W))}")
    print(f"  a shape:         {tuple(np.shape(params.a))}")
    print(f"  b shape:         {tuple(np.shape(params.b))}")
    print(f"  first labels:    {labels[:16].tolist()}")
    print(f"  binarized bits:  {images.shape[1]} per image")

    v = jnp.asarray(images[:8], dtype=jnp.float32)
    h = sample_bernoulli(jax.random.PRNGKey(0), hidden_prob(params, v))
    v_new, h_new = gibbs_step(params, v, jax.random.PRNGKey(1))
    print("\nFinite-value checks:")
    print(f"  hidden_prob finite:  {bool(jnp.isfinite(hidden_prob(params, v)).all())}")
    print(f"  visible_prob finite: {bool(jnp.isfinite(visible_prob(params, h)).all())}")
    print(f"  free_energy finite:  {bool(jnp.isfinite(free_energy(params, v)).all())}")
    print(f"  energies finite:     {bool(jnp.isfinite(energy(params, v, h)).all())}")
    print(f"  gibbs shapes:        v={v_new.shape}, h={h_new.shape}")


if __name__ == "__main__":
    main()
