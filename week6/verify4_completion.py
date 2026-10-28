"""VERIFY (4): image completion with clamped upper halves (CD-1 vs CD-10).

For the first 64 held-out MNIST test images the upper 14 rows are observed and
clamped; the lower 14 rows start from random bits and are sampled with 8
independent Gibbs chains per image. The observed rows are restored after
*every* Gibbs step.

Two conditions are compared:

* **true upper** — the image's own upper half;
* **wrong upper** — the upper half of the next test image with a different
  label (wrapping inside the 64 images), while the evaluation target remains
  the original image's true lower half.

Ink overlap on the missing lower half is the Dice coefficient
``2|A n B| / (|A| + |B|)``. Per image, overlap is averaged over its 8 chains;
the plotted error bars are standard errors across the 64 images.

Usage::

    cd week6
    uv run python verify4_completion.py
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

import rbm
import verify3_sampling as v3

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

RESULTS_PATH = ARTIFACTS_DIR / "verify4_completion.json"
SNAPSHOTS_PATH = ARTIFACTS_DIR / "verify4_snapshots.npz"
TRUE_GALLERY_PATH = EVIDENCE_DIR / "verify4_completion_true.png"
WRONG_GALLERY_PATH = EVIDENCE_DIR / "verify4_completion_wrong.png"
CURVES_PATH = EVIDENCE_DIR / "verify4_overlap_curves.png"

N_IMAGES = 64
N_CHAINS = 8
UPPER_ROWS = 14
LOWER_START = UPPER_ROWS * 28  # flattened index of the first missing pixel
STEPS = [1, 10, 100, 1000]
MODEL_ORDER = ("cd1", "cd10")
CONDITIONS = ("true", "wrong")
GALLERY_IMAGES = [0, 1, 2, 3]
GALLERY_CHAINS = [0, 1, 2, 3]


# --- Clamping mask and wrong-upper pairing --------------------------------
def upper_mask(n_visible: int = rbm.VISIBLE_UNITS) -> jax.Array:
    """1 for the observed upper 14 rows, 0 for the missing lower half."""
    mask = np.zeros(n_visible, dtype=np.float32)
    mask[:LOWER_START] = 1.0
    return jnp.asarray(mask)


def upper_half(images: np.ndarray) -> np.ndarray:
    """Keep only the upper 14 rows; the lower half is zeroed."""
    images = np.asarray(images, dtype=np.float32).copy()
    images[:, LOWER_START:] = 0.0
    return images


def wrong_upper_indices(labels: np.ndarray) -> np.ndarray:
    """For each image, the next image (wrapping) whose label differs."""
    labels = np.asarray(labels)
    n = len(labels)
    indices = []
    for i in range(n):
        for offset in range(1, n):
            j = (i + offset) % n
            if labels[j] != labels[i]:
                indices.append(j)
                break
        else:  # pragma: no cover - only if a single label exists
            raise ValueError("no image with a different label")
    return np.asarray(indices, dtype=np.int64)


# --- Clamped Gibbs sampling ----------------------------------------------
def initial_completion_state(
    observed_uppers: np.ndarray, key: jax.Array, n_chains: int = N_CHAINS
) -> jax.Array:
    """Random lower halves under each observed upper half (per chain)."""
    n_images = observed_uppers.shape[0]
    lower = jax.random.bernoulli(
        key, 0.5, (n_images * n_chains, rbm.VISIBLE_UNITS)
    ).astype(jnp.float32)
    upper = jnp.repeat(jnp.asarray(observed_uppers, jnp.float32), n_chains, axis=0)
    return jnp.where(upper_mask(), upper, lower)


@jax.jit
def _advance_clamped(
    params: rbm.RBM,
    v: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    keys: jax.Array,
) -> jax.Array:
    """Run one Gibbs step per key, restoring the observed upper half each time."""

    def body(state, step_key):
        key_hidden, key_visible = jax.random.split(step_key)
        h = rbm.sample_bernoulli(key_hidden, rbm.hidden_prob(params, state))
        v_free = rbm.sample_bernoulli(key_visible, rbm.visible_prob(params, h))
        # Clamp: observed pixels are exactly restored after every step.
        return jnp.where(mask, observed, v_free), None

    final, _ = jax.lax.scan(body, v, keys)
    return final


def completion_snapshots(
    params: rbm.RBM,
    v0: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    step_keys: jax.Array,
    steps: list[int] = STEPS,
) -> dict[int, np.ndarray]:
    """Advance to each requested step, recording the visible state."""
    snapshots = {}
    previous = 0
    state = v0
    for step in steps:
        segment = step_keys[previous:step]
        if segment.shape[0]:
            state = _advance_clamped(params, state, observed, mask, segment)
        snapshots[step] = np.asarray(state, dtype=np.float32)
        previous = step
    return snapshots


# --- Overlap metrics ------------------------------------------------------
def dice_overlap(generated: np.ndarray, target_lower: np.ndarray, n_chains: int = N_CHAINS) -> dict:
    """Dice overlap on the lower half, per chain and averaged per image."""
    generated_lower = np.asarray(generated)[:, LOWER_START:] >= 0.5
    target_lower = np.asarray(target_lower).astype(bool)
    n_images = target_lower.shape[0]
    generated_lower = generated_lower.reshape(n_images, n_chains, -1)

    intersection = np.sum(generated_lower & target_lower[:, None, :], axis=-1)
    denominator = generated_lower.sum(axis=-1) + target_lower.sum(axis=-1)[:, None]
    per_chain = np.divide(
        2.0 * intersection, denominator,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=denominator > 0,
    )
    per_image = per_chain.mean(axis=1)
    se = (
        float(per_image.std(ddof=1) / np.sqrt(n_images))
        if n_images > 1 else 0.0
    )
    return {
        "per_chain": per_chain,
        "per_image": per_image,
        "mean": float(per_image.mean()),
        "se": se,
    }


def run_condition(
    params: rbm.RBM,
    observed_uppers: np.ndarray,
    target_lower: np.ndarray,
    initial_key: jax.Array,
    step_keys: jax.Array,
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    """Run all chains for one model/condition and compute overlaps at each step."""
    v0 = initial_completion_state(observed_uppers, initial_key)
    observed = jnp.repeat(jnp.asarray(observed_uppers, jnp.float32), N_CHAINS, axis=0)
    snapshots = completion_snapshots(params, v0, observed, upper_mask(), step_keys)
    metrics = {str(step): dice_overlap(snapshots[step], target_lower) for step in STEPS}
    return snapshots, metrics


# --- Figures --------------------------------------------------------------
def _gallery_panels(
    snapshots: dict[str, dict[int, np.ndarray]],
    image_indices: list[int],
    chain_indices: list[int],
    labels: np.ndarray,
    condition: str,
):
    per_model = len(image_indices) * len(chain_indices)
    n_rows = len(STEPS)
    grid = GridSpec(
        n_rows, 2 * per_model + 1, wspace=0.06, hspace=0.06,
        left=0.045, right=0.995, top=0.83, bottom=0.11,
    )
    fig = plt.figure(figsize=(per_model * 2 * 0.60, n_rows * 0.82 + 1.9))
    offsets = {"cd1": 0, "cd10": per_model + 1}

    for row, step in enumerate(STEPS):
        for name in MODEL_ORDER:
            offset = offsets[name]
            column = 0
            for image in image_indices:
                for chain in chain_indices:
                    index = image * N_CHAINS + chain
                    ax = fig.add_subplot(grid[row, offset + column])
                    ax.imshow(
                        snapshots[name][step][index].reshape(28, 28),
                        cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                    )
                    ax.axhline(UPPER_ROWS - 0.5, color="#d62728", linewidth=0.5, alpha=0.7)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_gid(
                        f"model={name};condition={condition};step={step};"
                        f"image={image};chain={chain}"
                    )
                    if row == 0 and chain == chain_indices[0]:
                        ax.set_title(
                            f"img {image} (label {int(labels[image])})", fontsize=8
                        )
                    column += 1

        # Row label on the left of each model block.
        fig.text(0.012, 0.80 - row * (0.62 / n_rows), f"step {step}", fontsize=9, rotation=90)

    fig.text(0.26, 0.90, "CD-1 (k=1)", ha="center", fontsize=12)
    fig.text(0.74, 0.90, "CD-10 (k=10)", ha="center", fontsize=12)
    title = "true upper half" if condition == "true" else "wrong upper half"
    fig.suptitle(
        f"Image completion, {title}: upper 14 rows clamped, lower 14 sampled "
        f"({len(image_indices)} images x {len(chain_indices)} of {N_CHAINS} chains)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.015,
        "Red line separates observed (upper) from generated (lower) pixels. "
        "Observed rows are restored after every Gibbs step.",
        ha="center", va="bottom", fontsize=8, style="italic", color="#555555",
    )
    return fig


def build_true_gallery(snapshots, labels):
    return _gallery_panels(snapshots, GALLERY_IMAGES, GALLERY_CHAINS, labels, "true")


def build_wrong_gallery(snapshots, labels):
    return _gallery_panels(snapshots, GALLERY_IMAGES, GALLERY_CHAINS, labels, "wrong")


def build_overlap_figure(overlap: dict) -> plt.Figure:
    """Ink overlap vs Gibbs steps for both models and both upper halves."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    styles = {
        ("cd1", "true"): dict(color="#1f77b4", marker="o", linestyle="-", label="CD-1 true upper"),
        ("cd1", "wrong"): dict(color="#1f77b4", marker="o", linestyle="--", markerfacecolor="white", label="CD-1 wrong upper"),
        ("cd10", "true"): dict(color="#ff7f0e", marker="s", linestyle="-", label="CD-10 true upper"),
        ("cd10", "wrong"): dict(color="#ff7f0e", marker="s", linestyle="--", markerfacecolor="white", label="CD-10 wrong upper"),
    }
    for name in MODEL_ORDER:
        for condition in CONDITIONS:
            means = [overlap[name][condition][str(step)]["mean"] for step in STEPS]
            errors = [overlap[name][condition][str(step)]["se"] for step in STEPS]
            ax.errorbar(
                STEPS, means, yerr=errors, capsize=3, linewidth=1.6,
                markersize=5, **styles[(name, condition)],
            )

    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(step) for step in STEPS])
    ax.set_xlabel("Gibbs steps")
    ax.set_ylabel("ink overlap (Dice) on missing lower half")
    ax.set_title("Completion overlap: true vs wrong upper half (mean +/- SE over 64 images)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --- Persistence ----------------------------------------------------------
def save_results(
    overlap: dict,
    wrong_indices: np.ndarray,
    snapshots: dict,
    settings: dict,
) -> tuple[Path, Path]:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "VERIFY (4): clamped-upper completion on the first 64 held-out "
            "MNIST test images. Overlap is Dice on the lower half, averaged "
            "over 8 chains per image; SE across the 64 images."
        ),
        "settings": settings,
        "wrong_upper_indices": wrong_indices.tolist(),
        "overlap": {
            name: {
                condition: {
                    str(step): {
                        "mean": overlap[name][condition][str(step)]["mean"],
                        "se": overlap[name][condition][str(step)]["se"],
                        "per_image": overlap[name][condition][str(step)]["per_image"].tolist(),
                    }
                    for step in STEPS
                }
                for condition in CONDITIONS
            }
            for name in MODEL_ORDER
        },
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    arrays = {}
    for name in MODEL_ORDER:
        for condition in CONDITIONS:
            arrays[f"overlap_per_chain_{name}_{condition}"] = np.stack(
                [overlap[name][condition][str(step)]["per_chain"] for step in STEPS]
            )
            for step in STEPS:
                for image in GALLERY_IMAGES:
                    for chain in GALLERY_CHAINS:
                        arrays[f"gallery_{name}_{condition}_step{step}_img{image}_chain{chain}"] = (
                            snapshots[name][condition][step][image * N_CHAINS + chain]
                            .reshape(28, 28).astype(np.uint8)
                        )
    np.savez_compressed(SNAPSHOTS_PATH, **arrays)
    return RESULTS_PATH, SNAPSHOTS_PATH


# --- Entry point ----------------------------------------------------------
def main() -> None:
    config = rbm.load_config()
    params, retrained = v3.load_or_train_checkpoints(config)
    if retrained:
        print("Checkpoints were missing; retrained with the rbm.toml settings.")

    test_images, test_labels = rbm.load_binary_mnist("test", MNIST_DIR)
    images = test_images[:N_IMAGES]
    labels = test_labels[:N_IMAGES]
    target_lower = images[:, LOWER_START:].astype(bool)

    true_upper = upper_half(images)
    wrong_indices = wrong_upper_indices(labels)
    wrong_upper = upper_half(images[wrong_indices])

    # Shared initial randomness and matched per-step noise for every run.
    seed = config["completion"]["seed"]
    base = jax.random.PRNGKey(seed)
    initial_key, chain_key = jax.random.split(base)
    step_keys = jax.random.split(chain_key, max(STEPS))

    snapshots: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    overlap: dict[str, dict[str, dict[int, dict]]] = {}
    for name in MODEL_ORDER:
        snapshots[name], overlap[name] = {}, {}
        for condition, observed in (("true", true_upper), ("wrong", wrong_upper)):
            snaps, metrics = run_condition(
                params[name], observed, target_lower, initial_key, step_keys
            )
            snapshots[name][condition] = snaps
            overlap[name][condition] = metrics

    # --- observed pixels must never change --------------------------------
    observed_unchanged = True
    for name in MODEL_ORDER:
        for condition, observed in (("true", true_upper), ("wrong", wrong_upper)):
            reference = np.repeat(observed[:, :LOWER_START], N_CHAINS, axis=0)
            for step in STEPS:
                upper = snapshots[name][condition][step][:, :LOWER_START]
                if not np.array_equal(upper, reference):
                    observed_unchanged = False

    settings = {
        "images": N_IMAGES,
        "upper_rows_observed": UPPER_ROWS,
        "lower_rows_generated": 14,
        "chains_per_image": N_CHAINS,
        "steps": STEPS,
        "seed": seed,
        "overlap": "Dice: 2|A n B| / (|A| + |B|) on the lower 14 rows",
    }
    results_path, snapshots_path = save_results(
        overlap, wrong_indices, snapshots, settings
    )

    true_gallery = build_true_gallery({n: snapshots[n]["true"] for n in MODEL_ORDER}, labels)
    true_gallery.savefig(TRUE_GALLERY_PATH, dpi=200, bbox_inches="tight")
    plt.close(true_gallery)
    wrong_gallery = build_wrong_gallery({n: snapshots[n]["wrong"] for n in MODEL_ORDER}, labels)
    wrong_gallery.savefig(WRONG_GALLERY_PATH, dpi=200, bbox_inches="tight")
    plt.close(wrong_gallery)
    curves = build_overlap_figure(overlap)
    curves.savefig(CURVES_PATH, dpi=200, bbox_inches="tight")
    plt.close(curves)

    print("\nVERIFY (4) results: lower-half Dice overlap (mean +/- SE over 64 images)")
    for name in MODEL_ORDER:
        print(f"  {name.upper()}:")
        for condition in CONDITIONS:
            values = [
                f"{overlap[name][condition][str(step)]['mean']:.3f}"
                f"+/-{overlap[name][condition][str(step)]['se']:.3f}"
                for step in STEPS
            ]
            print(f"    {condition:<5}: " + "  ".join(values))

    print("\n  observed pixels unchanged:", observed_unchanged)

    # --- explicit verification -------------------------------------------
    assert observed_unchanged, "observed upper half was modified"
    for name in MODEL_ORDER:
        true_values = [overlap[name]["true"][str(step)]["mean"] for step in STEPS]
        wrong_values = [overlap[name]["wrong"][str(step)]["mean"] for step in STEPS]
        assert true_values[-1] > true_values[0], f"{name}: true overlap did not increase"
        assert np.argmax(true_values) == len(true_values) - 1, (
            f"{name}: true overlap is not highest at {STEPS[-1]} steps"
        )
        gap = true_values[-1] - wrong_values[-1]
        assert gap >= 0.1, f"{name}: true-vs-wrong gap {gap:.3f} < 0.1"
        print(
            f"  [ok] {name.upper()} true {true_values[-1]:.3f} vs wrong "
            f"{wrong_values[-1]:.3f} at 1000 steps (gap {gap:.3f})"
        )
    print(f"\n  results: {results_path.relative_to(WEEK_DIR)}")
    print(f"  raw:     {snapshots_path.relative_to(WEEK_DIR)}")
    print(f"  figure:  {TRUE_GALLERY_PATH.relative_to(WEEK_DIR)}")
    print(f"  figure:  {WRONG_GALLERY_PATH.relative_to(WEEK_DIR)}")
    print(f"  figure:  {CURVES_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
