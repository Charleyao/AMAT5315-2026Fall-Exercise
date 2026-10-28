"""VERIFY (3): frozen-model reconstruction and fresh generation for CD-1/CD-10.

Reuses the trained checkpoints in ``artifacts/`` (falling back to a
deterministic retrain with the exact ``rbm.toml`` settings if they are
missing). Parameters are frozen: only sampling reads them.

Two comparisons are produced:

* **Reconstruction** — one Gibbs step from held-out MNIST digits, with the
  fixed evaluation key, for CD-1 and CD-10.
* **Generation** — fresh samples started from *identical* random +/-1 spin
  configurations for both models, with *matched* sampling noise, recorded at
  steps 0, 1, 10, 100 and 1000. All 8 chains are shown; no best-sample
  selection.

Clean, recognizable digits are not required; the check is that the chains
develop connected, stroke-like ink structure rather than pixel noise.

Usage::

    cd week6
    uv run python verify3_sampling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import ndimage

import cd_training as cd
import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

RESULTS_PATH = ARTIFACTS_DIR / "verify3_sampling.json"
SNAPSHOTS_PATH = ARTIFACTS_DIR / "verify3_snapshots.npz"
RECONSTRUCTION_FIGURE = EVIDENCE_DIR / "verify3_reconstruction.png"
GENERATION_FIGURE = EVIDENCE_DIR / "verify3_generation.png"

N_HELD_OUT = 8
N_CHAINS = 8
SNAPSHOT_STEPS = [0, 1, 10, 100, 1000]
MODEL_ORDER = ("cd1", "cd10")


# --- Model checkpoints ----------------------------------------------------
def load_or_train_checkpoints(config: dict) -> tuple[dict[str, rbm.RBM], bool]:
    """Load frozen checkpoints; retrain deterministically if any is missing."""
    if all(path.is_file() for path in cd.CHECKPOINT_PATHS.values()):
        return {
            name: cd.load_checkpoint(path)
            for name, path in cd.CHECKPOINT_PATHS.items()
        }, False

    train_images, _ = rbm.load_binary_mnist("train", MNIST_DIR)
    test_images, _ = rbm.load_binary_mnist("test", MNIST_DIR)
    test_images = test_images[: config["evaluation"]["test_images"]]
    init_params = rbm.init_from_config(config, train_images)

    params = {}
    for name, k in cd.CHAIN_LENGTHS.items():
        result = cd.train_cd(
            k, config, train_images, test_images,
            init_params=init_params, progress=True,
        )
        cd.save_checkpoint(result["params"], cd.CHECKPOINT_PATHS[name])
        params[name] = result["params"]
    return params, True


# --- Random configurations and matched noise ------------------------------
def matched_randomness(steps: int, seed: int):
    """Return one initial key and a per-step key sequence shared by both models."""
    base = jax.random.PRNGKey(seed)
    initial_key, chain_key = jax.random.split(base)
    return initial_key, jax.random.split(chain_key, steps)


def initial_spins(n_chains: int, key: jax.Array, n_visible: int = rbm.VISIBLE_UNITS):
    """Random +/-1 spin configurations (shape ``(n_chains, n_visible)``)."""
    bits = jax.random.bernoulli(key, 0.5, (n_chains, n_visible))
    return jnp.where(bits, 1.0, -1.0)


def spins_to_binary(spins: np.ndarray) -> np.ndarray:
    """Map spins ``s in {-1, +1}`` to visible bits ``v = (s + 1) / 2``."""
    return ((np.asarray(spins) + 1.0) / 2.0).astype(np.float32)


# --- Frozen-model sampling ------------------------------------------------
def reconstruct(params: rbm.RBM, v0: jax.Array, key: jax.Array) -> np.ndarray:
    """One Gibbs step from data: sample ``h ~ P(h|v0)`` then ``v ~ P(v|h)``."""
    key_hidden, key_visible = jax.random.split(key)
    h = rbm.sample_bernoulli(key_hidden, rbm.hidden_prob(params, v0))
    v = rbm.sample_bernoulli(key_visible, rbm.visible_prob(params, h))
    return np.asarray(v)


@jax.jit
def _gibbs_trajectory(params: rbm.RBM, v0: jax.Array, keys: jax.Array) -> jax.Array:
    """Run one Gibbs step per key and return every intermediate visible state."""

    def body(v, step_key):
        v_new, _ = rbm.gibbs_step(params, v, step_key)
        return v_new, v_new

    _, trajectory = jax.lax.scan(body, v0, keys)
    return trajectory


def generate_snapshots(
    params: rbm.RBM, v0: jax.Array, keys: jax.Array, steps: list[int]
) -> dict[int, np.ndarray]:
    """Run the chain to ``max(steps)`` and record the visible state at each step."""
    trajectory = _gibbs_trajectory(params, v0, keys)
    snapshots = {0: np.asarray(v0, dtype=np.float32)}
    for step in steps:
        if step > 0:
            snapshots[step] = np.asarray(trajectory[step - 1], dtype=np.float32)
    return snapshots


# --- Structure metrics ----------------------------------------------------
def structure_metrics(images: np.ndarray) -> dict:
    """Ink fraction and 4-connected ink components per chain."""
    flat = np.asarray(images).reshape(-1, 28, 28) >= 0.5
    components = []
    largest_fractions = []
    for image in flat:
        labelled, count = ndimage.label(image)
        sizes = np.bincount(labelled.ravel())[1:]
        components.append(int(count))
        largest_fractions.append(float(sizes.max() / image.size) if sizes.size else 0.0)
    return {
        "chains": int(flat.shape[0]),
        "ink_fraction": float(flat.mean()),
        "mean_components": float(np.mean(components)),
        "mean_largest_fraction": float(np.mean(largest_fractions)),
        "components": components,
        "largest_fractions": largest_fractions,
    }


# --- Figures --------------------------------------------------------------
def build_reconstruction_figure(
    originals: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    labels: np.ndarray,
):
    """3 rows (input, CD-1, CD-10) x N held-out digits."""
    n = len(originals)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.35, 3 * 1.55))
    rows = [
        ("input", np.asarray(originals)),
        ("CD-1", np.asarray(reconstructions["cd1"])),
        ("CD-10", np.asarray(reconstructions["cd10"])),
    ]
    for row, (row_label, images) in enumerate(rows):
        for column in range(n):
            ax = axes[row, column]
            ax.imshow(
                images[column].reshape(28, 28),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_gid(f"row={row_label};column={column}")
            if row == 0:
                ax.set_title(f"label {int(labels[column])}", fontsize=9)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=10)

    fig.suptitle(
        "One-step reconstruction from held-out MNIST digits "
        "(fixed evaluation key, frozen parameters)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def build_generation_figure(
    snapshots: dict[str, dict[int, np.ndarray]],
    steps: list[int] = SNAPSHOT_STEPS,
    n_chains: int = N_CHAINS,
):
    """Rows = generation steps, column blocks = CD-1 and CD-10, all chains shown."""
    n_rows = len(steps)
    # 8 chains + 1 spacer + 8 chains
    grid = GridSpec(
        n_rows, 2 * n_chains + 1, wspace=0.10, hspace=0.12,
        left=0.07, right=0.99, top=0.88, bottom=0.10,
    )
    fig = plt.figure(figsize=(17.0, 8.0))
    column_blocks = {"cd1": 0, "cd10": n_chains + 1}

    for row, step in enumerate(steps):
        for name in MODEL_ORDER:
            offset = column_blocks[name]
            for chain in range(n_chains):
                ax = fig.add_subplot(grid[row, offset + chain])
                ax.imshow(
                    snapshots[name][step][chain].reshape(28, 28),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_gid(f"model={name};step={step};chain={chain}")
                if chain == 0:
                    ax.set_ylabel(f"step {step}", fontsize=9, labelpad=12)
        fig.text(0.235, 0.91, "CD-1 (k=1)", ha="center", fontsize=12)
        fig.text(0.755, 0.91, "CD-10 (k=10)", ha="center", fontsize=12)

    fig.suptitle(
        "Fresh generation from identical random +/-1 spins: "
        "CD-1 vs CD-10 over Gibbs steps",
        fontsize=13,
    )
    fig.text(
        0.5, 0.015,
        "All 8 chains shown, no best-sample selection. Initial spins and "
        "sampling noise are matched across the two models.",
        ha="center", va="bottom", fontsize=9, style="italic", color="#555555",
    )
    return fig


# --- Persistence ----------------------------------------------------------
def save_results(metrics: dict, snapshots: dict[str, dict[int, np.ndarray]]) -> tuple[Path, Path]:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    arrays = {
        f"{name}_step{step}": np.asarray(snapshots[name][step], dtype=np.uint8)
        for name in MODEL_ORDER
        for step in SNAPSHOT_STEPS
    }
    np.savez(SNAPSHOTS_PATH, **arrays)
    return RESULTS_PATH, SNAPSHOTS_PATH


def save_figures(
    originals: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    labels: np.ndarray,
    snapshots: dict[str, dict[int, np.ndarray]],
) -> tuple[Path, Path]:
    reconstruction = build_reconstruction_figure(originals, reconstructions, labels)
    reconstruction.savefig(RECONSTRUCTION_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(reconstruction)

    generation = build_generation_figure(snapshots)
    generation.savefig(GENERATION_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(generation)
    return RECONSTRUCTION_FIGURE, GENERATION_FIGURE


# --- Entry point ----------------------------------------------------------
def main() -> None:
    config = rbm.load_config()
    params, retrained = load_or_train_checkpoints(config)
    if retrained:
        print("Checkpoints were missing; retrained with the rbm.toml settings.")

    test_images, test_labels = rbm.load_binary_mnist("test", MNIST_DIR)
    originals = test_images[:N_HELD_OUT]
    labels = test_labels[:N_HELD_OUT]

    # Reconstruction: same fixed key for both frozen models.
    reconstruction_key = jax.random.PRNGKey(config["evaluation"]["seed"])
    v0 = jnp.asarray(originals, dtype=jnp.float32)
    reconstructions = {
        name: reconstruct(params[name], v0, reconstruction_key)
        for name in MODEL_ORDER
    }
    reconstruction_mse = {
        name: float(np.mean((originals - reconstructions[name]) ** 2))
        for name in MODEL_ORDER
    }

    # Generation: identical spins and matched noise for both models.
    completion = config["completion"]
    steps = [0] + list(completion["steps"])
    steps = [step for step in SNAPSHOT_STEPS if step in steps]
    max_steps = max(steps)
    initial_key, step_keys = matched_randomness(max_steps, completion["seed"])
    spins = initial_spins(completion["chains"], initial_key)
    v0_generation = jnp.asarray(spins_to_binary(spins))

    before = {name: tuple(np.asarray(x).copy() for x in params[name]) for name in MODEL_ORDER}
    snapshots = {
        name: generate_snapshots(params[name], v0_generation, step_keys, steps)
        for name in MODEL_ORDER
    }
    frozen = {
        name: all(
            np.array_equal(before[name][i], np.asarray(params[name][i]))
            for i in range(3)
        )
        for name in MODEL_ORDER
    }

    initial_identical = np.array_equal(
        snapshots["cd1"][0], snapshots["cd10"][0]
    )

    generation_metrics = {
        name: {str(step): structure_metrics(snapshots[name][step]) for step in steps}
        for name in MODEL_ORDER
    }

    metrics = {
        "description": (
            "VERIFY (3): frozen-parameter one-step reconstruction and fresh "
            "generation for CD-1 and CD-10. Generation starts from identical "
            "random +/-1 spin configurations with matched sampling noise."
        ),
        "settings": {
            "held_out_images": N_HELD_OUT,
            "chains": completion["chains"],
            "steps": steps,
            "generation_seed": completion["seed"],
            "reconstruction_seed": config["evaluation"]["seed"],
        },
        "parameters_frozen": frozen,
        "initial_configurations_identical": bool(initial_identical),
        "reconstruction": {
            "labels": labels.tolist(),
            "mse": reconstruction_mse,
        },
        "generation": generation_metrics,
    }
    results_path, snapshots_path = save_results(metrics, snapshots)
    reconstruction_path, generation_path = save_figures(
        originals, reconstructions, labels, snapshots
    )

    print("\nVERIFY (3) results")
    print("  frozen parameters:", frozen)
    print("  identical initial configurations:", bool(initial_identical))
    print("  reconstruction MSE (8 held-out digits):", {
        name: round(value, 4) for name, value in reconstruction_mse.items()
    })
    print("\n  generation structure (mean over 8 chains):")
    for name in MODEL_ORDER:
        print(f"    {name}:")
        for step in steps:
            item = generation_metrics[name][str(step)]
            print(
                f"      step {step:>4}: ink={item['ink_fraction']:.3f}  "
                f"components={item['mean_components']:6.1f}  "
                f"largest={item['mean_largest_fraction']:.3f}"
            )

    # --- explicit verification -------------------------------------------
    for name in MODEL_ORDER:
        first = generation_metrics[name]["0"]
        last = generation_metrics[name][str(max_steps)]
        assert frozen[name], f"{name}: parameters changed during sampling"
        assert last["mean_components"] < first["mean_components"], (
            f"{name}: ink did not become more connected"
        )
        assert last["mean_largest_fraction"] > first["mean_largest_fraction"], (
            f"{name}: largest component did not grow"
        )
        assert all(
            np.isfinite(value)
            for value in (generation_metrics[name][str(step)]["ink_fraction"]
                          for step in steps)
        )
    assert initial_identical, "models must start from identical spin configurations"
    assert all(item["chains"] == N_CHAINS for name in MODEL_ORDER
               for item in generation_metrics[name].values())

    print("\n  [ok] identical initial random configurations")
    print("  [ok] parameters frozen during reconstruction and generation")
    print("  [ok] ink becomes more connected / largest component grows")
    print("  [ok] 8 chains recorded per snapshot (no best-sample selection)")
    print(f"\n  results:   {results_path.relative_to(WEEK_DIR)}")
    print(f"  snapshots: {snapshots_path.relative_to(WEEK_DIR)}")
    print(f"  figure:    {reconstruction_path.relative_to(WEEK_DIR)}")
    print(f"  figure:    {generation_path.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
