"""Contrastive divergence (CD-1 and CD-10) training for Week 6 Part 2 VERIFY (2).

Both models share the same initialization (seed 6601), the same data order
(shuffles from training seed 6602) and the same settings from ``rbm.toml``;
only the Gibbs chain length differs (1 vs 10).

The CD surrogate is the epoch mean of the per-batch objective

    mean[ F(v0) - F(stop_gradient(vk)) ]

computed **before** each parameter update. It is a stochastic approximation of
the negative-log-likelihood gradient objective; it is *not* the exact NLL (the
partition function is intractable), so it is never interpreted as one.

Held-out one-step reconstruction error uses the first 64 test images and a
fixed evaluation key (seed 6606):

    h ~ Bernoulli(P(h | v0)),  v1 ~ Bernoulli(P(v | h)),
    error = mean((v0 - v1)^2).

Usage::

    cd week6
    uv run python cd_training.py
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
METRICS_PATH = ARTIFACTS_DIR / "cd_training_metrics.json"
CURVES_PATH = EVIDENCE_DIR / "cd_training_curves.png"
CHECKPOINT_PATHS = {
    "cd1": ARTIFACTS_DIR / "rbm_cd1.npz",
    "cd10": ARTIFACTS_DIR / "rbm_cd10.npz",
}

CHAIN_LENGTHS = {"cd1": 1, "cd10": 10}


# --- Data order -----------------------------------------------------------
def epoch_permutations(
    n_images: int, epochs: int, seed: int
) -> list[np.ndarray]:
    """Deterministic per-epoch shuffles shared by both models."""
    key = jax.random.PRNGKey(seed)
    permutations = []
    for _ in range(epochs):
        key, subkey = jax.random.split(key)
        permutations.append(np.asarray(jax.random.permutation(subkey, n_images)))
    return permutations


# --- CD objective and one update -----------------------------------------
def cd_surrogate(params: rbm.RBM, v0: jax.Array, vk: jax.Array) -> jax.Array:
    """Batch CD surrogate ``mean[F(v0) - F(stop_gradient(vk))]``.

    Minimizing this surrogate is equivalent to the standard CD update; the
    stop-gradient makes ``vk`` a constant in the objective.
    """
    return jnp.mean(
        rbm.free_energy(params, v0)
        - rbm.free_energy(params, jax.lax.stop_gradient(vk))
    )


@partial(jax.jit, static_argnames=("k",))
def cd_step(
    params: rbm.RBM, batch: jax.Array, key: jax.Array, learning_rate: float, k: int
):
    """One SGD step on the CD surrogate, returning ``(new_params, surrogate)``."""
    # Positive phase statistics use the data and P(h | v0).
    h0 = rbm.hidden_prob(params, batch)
    # Negative phase: k block-Gibbs steps started from the data batch.
    vk = rbm.gibbs_chain(params, batch, key, k)
    hk = rbm.hidden_prob(params, vk)

    # Surrogate is evaluated before the update.
    surrogate = cd_surrogate(params, batch, vk)

    n = batch.shape[0]
    dW = (batch.T @ h0 - vk.T @ hk) / n
    da = jnp.mean(batch - vk, axis=0)
    db = jnp.mean(h0 - hk, axis=0)

    updated = rbm.RBM(
        W=params.W + learning_rate * dW,
        a=params.a + learning_rate * da,
        b=params.b + learning_rate * db,
    )
    return updated, surrogate


@jax.jit
def one_step_reconstruction_error(
    params: rbm.RBM, v0: jax.Array, key: jax.Array
) -> jax.Array:
    """Mean squared error between inputs and a sampled one-step reconstruction."""
    key_hidden, key_visible = jax.random.split(key)
    h = rbm.sample_bernoulli(key_hidden, rbm.hidden_prob(params, v0))
    v1 = rbm.sample_bernoulli(key_visible, rbm.visible_prob(params, h))
    return jnp.mean((v0 - v1) ** 2)


# --- Training loop --------------------------------------------------------
def train_cd(
    k: int,
    config: dict,
    train_images: np.ndarray,
    test_images: np.ndarray,
    *,
    epochs: int | None = None,
    init_params: rbm.RBM | None = None,
    progress: bool = True,
) -> dict:
    """Train one CD-k model and return its parameters and recorded metrics."""
    training = config["training"]
    epochs = training["epochs"] if epochs is None else epochs
    batch_size = training["batch_size"]
    learning_rate = training["learning_rate"]

    params = (
        init_params
        if init_params is not None
        else rbm.init_from_config(config, train_images)
    )
    permutations = epoch_permutations(
        train_images.shape[0], epochs, training["seed"]
    )
    # Fixed evaluation randomness for every reconstruction measurement.
    evaluation_key = jax.random.PRNGKey(config["evaluation"]["seed"])
    base_key = jax.random.PRNGKey(training["seed"])

    initial_reconstruction = float(
        one_step_reconstruction_error(params, test_images, evaluation_key)
    )
    surrogates: list[float] = []
    reconstruction = [initial_reconstruction]

    start = time.perf_counter()
    for epoch in range(epochs):
        order = permutations[epoch]
        epoch_surrogate = 0.0
        n_batches = 0
        for batch_index, begin in enumerate(range(0, len(order), batch_size)):
            indices = order[begin:begin + batch_size]
            batch = jnp.asarray(train_images[indices], dtype=jnp.float32)
            key = jax.random.fold_in(base_key, epoch * 100_000 + batch_index)
            params, surrogate = cd_step(
                params, batch, key, learning_rate, k
            )
            epoch_surrogate += float(surrogate)
            n_batches += 1

        surrogates.append(epoch_surrogate / n_batches)
        reconstruction.append(
            float(one_step_reconstruction_error(params, test_images, evaluation_key))
        )
        if progress:
            print(
                f"    epoch {epoch + 1:2d}/{epochs}: "
                f"surrogate={surrogates[-1]:.5f}  "
                f"recon={reconstruction[-1]:.5f}"
            )

    runtime = time.perf_counter() - start
    return {
        "params": params,
        "surrogate": surrogates,
        "reconstruction_error": reconstruction,
        "runtime_seconds": runtime,
    }


# --- Metrics and figure ---------------------------------------------------
def build_metrics(config: dict, results: dict, n_test_images: int) -> dict:
    """Assemble the JSON-serialisable metrics record."""
    epochs = len(results["cd1"]["surrogate"])
    return {
        "description": (
            "CD-1 vs CD-10 on binarized MNIST. The CD surrogate "
            "mean[F(v0) - F(stop_gradient(vk))] is an approximate training "
            "objective, not the exact negative log-likelihood."
        ),
        "dataset": {"training_images": 60000, "evaluation_images": n_test_images},
        "training": config["training"],
        "evaluation": config["evaluation"],
        "chain_lengths": CHAIN_LENGTHS,
        "epochs": list(range(1, epochs + 1)),
        "initial_reconstruction_error": {
            "cd1": results["cd1"]["reconstruction_error"][0],
            "cd10": results["cd10"]["reconstruction_error"][0],
        },
        "cd1": {
            "surrogate": results["cd1"]["surrogate"],
            "reconstruction_error": results["cd1"]["reconstruction_error"],
            "runtime_seconds": results["cd1"]["runtime_seconds"],
        },
        "cd10": {
            "surrogate": results["cd10"]["surrogate"],
            "reconstruction_error": results["cd10"]["reconstruction_error"],
            "runtime_seconds": results["cd10"]["runtime_seconds"],
        },
    }


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def save_checkpoint(params: rbm.RBM, path: Path) -> Path:
    """Save frozen RBM parameters (W, a, b) to an ``.npz`` checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W=np.asarray(params.W),
        a=np.asarray(params.a),
        b=np.asarray(params.b),
    )
    return path


def load_checkpoint(path: Path) -> rbm.RBM:
    """Load RBM parameters saved by :func:`save_checkpoint`."""
    with np.load(path) as data:
        return rbm.RBM(
            W=jnp.asarray(data["W"], dtype=jnp.float32),
            a=jnp.asarray(data["a"], dtype=jnp.float32),
            b=jnp.asarray(data["b"], dtype=jnp.float32),
        )


def build_training_curves(metrics: dict):
    """Build the two-panel CD-1 vs CD-10 comparison figure."""
    epochs = metrics["epochs"]
    reconstruction_epochs = [0] + epochs

    fig, (ax_surrogate, ax_recon) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    for name, style in (("cd1", "o-"), ("cd10", "s-")):
        label = name.upper().replace("CD", "CD-")
        ax_surrogate.plot(
            epochs, metrics[name]["surrogate"], style, label=label, markersize=3
        )
        ax_recon.plot(
            reconstruction_epochs,
            metrics[name]["reconstruction_error"],
            style,
            label=label,
            markersize=3,
        )

    ax_surrogate.set_xlabel("epoch")
    ax_surrogate.set_ylabel("CD surrogate (epoch mean)")
    ax_surrogate.set_title("CD surrogate: mean[F(v0) - F(sg(vk))]")
    ax_surrogate.grid(alpha=0.3)
    ax_surrogate.legend(frameon=False)

    ax_recon.set_xlabel("epoch (0 = initialization)")
    ax_recon.set_ylabel("one-step reconstruction error")
    ax_recon.set_title("Held-out reconstruction error (64 test images)")
    ax_recon.grid(alpha=0.3)
    ax_recon.legend(frameon=False)

    fig.suptitle("CD-1 vs CD-10 on MNIST: 20 epochs, identical init and data order")
    fig.text(
        0.5, 0.005,
        "Note: the CD surrogate is a stochastic approximation of the training "
        "objective, not the exact negative log-likelihood.",
        ha="center", va="bottom", fontsize=8, style="italic", color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def save_training_curves(metrics: dict, path: Path = CURVES_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_training_curves(metrics)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Entry point ----------------------------------------------------------
def main() -> None:
    config = rbm.load_config()
    train_images, _ = rbm.load_binary_mnist("train", MNIST_DIR)
    test_images, test_labels = rbm.load_binary_mnist("test", MNIST_DIR)
    n_test = config["evaluation"]["test_images"]
    test_images = test_images[:n_test]

    # One shared initialization guarantees identical starting parameters
    # and identical held-out reconstruction at initialization.
    init_params = rbm.init_from_config(config, train_images)

    results = {}
    for name, k in CHAIN_LENGTHS.items():
        print(f"Training {name.upper()} (k={k}, {config['training']['epochs']} epochs)...")
        results[name] = train_cd(
            k, config, train_images, test_images, init_params=init_params
        )

    metrics = build_metrics(config, results, n_test)
    metrics_path = save_metrics(metrics)
    curves_path = save_training_curves(metrics)
    checkpoint_paths = {
        name: save_checkpoint(results[name]["params"], CHECKPOINT_PATHS[name])
        for name in CHAIN_LENGTHS
    }

    print("\nVERIFY (2) results")
    print(f"  evaluation images: {n_test} (first {n_test} test images)")
    for name in CHAIN_LENGTHS:
        initial = metrics["initial_reconstruction_error"][name]
        final = metrics[name]["reconstruction_error"][-1]
        runtime = metrics[name]["runtime_seconds"]
        ratio = final / initial
        print(
            f"  {name.upper()}: recon {initial:.4f} -> {final:.4f} "
            f"({100 * ratio:.1f}% of initial), runtime {runtime:.1f}s"
        )
        print(
            f"         surrogate {metrics[name]['surrogate'][0]:.4f} -> "
            f"{metrics[name]['surrogate'][-1]:.4f}"
        )

    # --- explicit verification -------------------------------------------
    for name in CHAIN_LENGTHS:
        series = (
            metrics[name]["surrogate"] + metrics[name]["reconstruction_error"]
        )
        assert all(np.isfinite(series)), f"{name}: non-finite metric"
        initial = metrics["initial_reconstruction_error"][name]
        final = metrics[name]["reconstruction_error"][-1]
        assert final < 0.5 * initial, f"{name}: final recon is not < 50% of initial"
    assert np.isclose(
        metrics["initial_reconstruction_error"]["cd1"],
        metrics["initial_reconstruction_error"]["cd10"],
    ), "CD-1 and CD-10 must start from identical parameters"
    print("\n  [ok] all metrics finite")
    print("  [ok] final reconstruction error < 50% of initial error for both models")
    print("  [ok] identical initialization for CD-1 and CD-10")
    print("  [note] the CD surrogate is an approximate objective, not the exact NLL")
    print(f"\n  metrics: {metrics_path.relative_to(WEEK_DIR)}")
    print(f"  figure:  {curves_path.relative_to(WEEK_DIR)}")
    for name, path in checkpoint_paths.items():
        print(f"  checkpoint {name}: {path.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
