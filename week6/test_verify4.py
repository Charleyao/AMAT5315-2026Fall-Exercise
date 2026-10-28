"""Tests for Week 6 Part 2 VERIFY (4): clamped-upper image completion.

Covers the clamping mask, wrong-upper pairing, frozen clamped sampling,
Dice overlap, gallery/curve layouts and the saved-artifact verification.

Run with::

    cd week6
    uv run python -m pytest -q
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

import cd_training as cd
import rbm
import verify4_completion as v4

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
CHECKPOINTS_AVAILABLE = all(path.is_file() for path in cd.CHECKPOINT_PATHS.values())

pytestmark = pytest.mark.skipif(
    not CHECKPOINTS_AVAILABLE, reason="run cd_training.py to create checkpoints"
)


@pytest.fixture(scope="module")
def config():
    return rbm.load_config()


@pytest.fixture(scope="module")
def params():
    return {name: cd.load_checkpoint(path) for name, path in cd.CHECKPOINT_PATHS.items()}


@pytest.fixture(scope="module")
def test_data():
    images, labels = rbm.load_binary_mnist("test", MNIST_DIR)
    return images[: v4.N_IMAGES], labels[: v4.N_IMAGES]


def test_wrong_upper_indices_pick_next_different_label_with_wrapping():
    labels = np.array([0, 0, 1, 1, 2])
    indices = v4.wrong_upper_indices(labels)
    np.testing.assert_array_equal(indices, [2, 2, 4, 4, 0])
    assert all(labels[i] != labels[j] for i, j in enumerate(indices))


def test_wrong_upper_indices_for_held_out_split(test_data):
    _images, labels = test_data
    indices = v4.wrong_upper_indices(labels)
    assert indices.shape == (64,)
    assert all(indices[i] != i for i in range(64))
    assert all(labels[i] != labels[indices[i]] for i in range(64))
    # "next ... wrapping": no closer index has a different label.
    for i in range(64):
        for offset in range(1, 64):
            j = (i + offset) % 64
            if labels[j] != labels[i]:
                assert indices[i] == j
                break


def test_upper_mask_and_upper_half(test_data):
    images, _labels = test_data
    mask = np.asarray(v4.upper_mask())
    assert mask.sum() == 14 * 28
    np.testing.assert_array_equal(mask[: 14 * 28], np.ones(14 * 28))
    np.testing.assert_array_equal(mask[14 * 28:], np.zeros(14 * 28))

    observed = v4.upper_half(images)
    np.testing.assert_array_equal(observed[:, : 14 * 28], images[:, : 14 * 28])
    assert observed[:, 14 * 28:].sum() == 0


def test_initial_completion_keeps_upper_and_randomizes_lower(test_data):
    images, _labels = test_data
    observed = v4.upper_half(images[:4])
    key = jax.random.PRNGKey(0)
    first = np.asarray(v4.initial_completion_state(observed, key))
    second = np.asarray(v4.initial_completion_state(observed, key))
    assert first.shape == (4 * v4.N_CHAINS, 784)
    upper = first[:, : v4.LOWER_START].reshape(4, v4.N_CHAINS, -1)
    expected = observed[:, : v4.LOWER_START][:, None, :]
    np.testing.assert_array_equal(upper, np.broadcast_to(expected, upper.shape))
    assert set(np.unique(first[:, v4.LOWER_START:]).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(first, second)


def test_clamped_sampling_never_changes_observed_pixels(test_data, params):
    images, _labels = test_data
    observed_uppers = v4.upper_half(images[:4])
    key = jax.random.PRNGKey(1)
    v0 = v4.initial_completion_state(observed_uppers, key)
    observed = jax.numpy.repeat(
        jax.numpy.asarray(observed_uppers, jax.numpy.float32), v4.N_CHAINS, axis=0
    )
    step_keys = jax.random.split(jax.random.PRNGKey(2), 5)
    snapshots = v4.completion_snapshots(
        params["cd1"], v0, observed, v4.upper_mask(), step_keys, steps=[1, 5]
    )
    reference = observed[:, : v4.LOWER_START]
    for step in (1, 5):
        np.testing.assert_array_equal(snapshots[step][:, : v4.LOWER_START], reference)
        assert set(np.unique(snapshots[step][:, v4.LOWER_START:]).tolist()) <= {0.0, 1.0}


def test_dice_overlap_values():
    target = np.zeros((1, 392), dtype=bool)
    target[0, :10] = True
    identical = np.zeros((1, 784), dtype=np.float32)
    identical[0, v4.LOWER_START: v4.LOWER_START + 10] = 1.0
    assert v4.dice_overlap(identical, target, n_chains=1)["mean"] == pytest.approx(1.0)

    empty = np.zeros((1, 784), dtype=np.float32)
    assert v4.dice_overlap(empty, target, n_chains=1)["mean"] == pytest.approx(0.0)

    half = np.zeros((1, 784), dtype=np.float32)
    half[0, v4.LOWER_START: v4.LOWER_START + 5] = 1.0
    assert v4.dice_overlap(half, target, n_chains=1)["mean"] == pytest.approx(2 * 5 / 15)


def test_overlap_figure_has_four_series_with_error_bars():
    overlap = {}
    for name in v4.MODEL_ORDER:
        overlap[name] = {}
        for condition in v4.CONDITIONS:
            overlap[name][condition] = {
                str(step): {"mean": 0.2 + 0.05 * index, "se": 0.01}
                for index, step in enumerate(v4.STEPS)
            }
    fig = v4.build_overlap_figure(overlap)
    try:
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == [
            "CD-1 true upper", "CD-1 wrong upper",
            "CD-10 true upper", "CD-10 wrong upper",
        ]
        assert len(ax.lines) + len(ax.containers) >= 4
        assert ax.get_xscale() == "log"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1", "10", "100", "1000"]
    finally:
        v4.plt.close(fig)


def test_gallery_layout_shows_steps_models_images_and_chains(test_data):
    _images, labels = test_data
    snapshots = {
        name: {
            step: np.zeros((v4.N_IMAGES * v4.N_CHAINS, 784), dtype=np.float32)
            for step in v4.STEPS
        }
        for name in v4.MODEL_ORDER
    }
    fig = v4.build_true_gallery(snapshots, labels)
    try:
        axes = [ax for ax in fig.axes if ax.images]
        expected = len(v4.STEPS) * 2 * len(v4.GALLERY_IMAGES) * len(v4.GALLERY_CHAINS)
        assert len(axes) == expected
        gids = {ax.get_gid() for ax in axes}
        for name in v4.MODEL_ORDER:
            for step in v4.STEPS:
                for image in v4.GALLERY_IMAGES:
                    for chain in v4.GALLERY_CHAINS:
                        gid = (
                            f"model={name};condition=true;step={step};"
                            f"image={image};chain={chain}"
                        )
                        assert gid in gids
        text = " ".join(item.get_text() for item in fig.texts)
        assert "Observed rows are restored after every Gibbs step" in text
    finally:
        v4.plt.close(fig)


@pytest.mark.skipif(
    not v4.RESULTS_PATH.is_file(), reason="run verify4_completion.py to create results"
)
def test_saved_results_verify_the_requirements(test_data):
    _images, labels = test_data
    metrics = json.loads(v4.RESULTS_PATH.read_text())
    assert metrics["settings"]["images"] == 64
    assert metrics["settings"]["upper_rows_observed"] == 14
    assert metrics["settings"]["chains_per_image"] == 8
    assert metrics["settings"]["steps"] == v4.STEPS
    assert metrics["settings"]["seed"] == 6607
    assert len(metrics["wrong_upper_indices"]) == 64
    assert all(
        labels[i] != labels[j] for i, j in enumerate(metrics["wrong_upper_indices"])
    )

    for name in v4.MODEL_ORDER:
        for condition in v4.CONDITIONS:
            for step in v4.STEPS:
                item = metrics["overlap"][name][condition][str(step)]
                assert len(item["per_image"]) == 64
                assert np.isfinite(item["mean"]) and np.isfinite(item["se"])
        true_values = [metrics["overlap"][name]["true"][str(s)]["mean"] for s in v4.STEPS]
        wrong_values = [metrics["overlap"][name]["wrong"][str(s)]["mean"] for s in v4.STEPS]
        # Correct-upper overlap increases with sampling steps.
        assert np.all(np.diff(true_values) > 0)
        # At 1000 steps the true upper beats the wrong upper by at least 0.1.
        assert true_values[-1] - wrong_values[-1] >= 0.1

    # Approximate sheet reference: CD-1 0.40/0.24, CD-10 0.51/0.27.
    final = {
        (name, condition): metrics["overlap"][name][condition]["1000"]["mean"]
        for name in v4.MODEL_ORDER
        for condition in v4.CONDITIONS
    }
    assert 0.33 <= final[("cd1", "true")] <= 0.44
    assert 0.17 <= final[("cd1", "wrong")] <= 0.29
    assert 0.43 <= final[("cd10", "true")] <= 0.56
    assert 0.18 <= final[("cd10", "wrong")] <= 0.31

    # Saved completions keep the observed upper half exactly.
    _images, labels = test_data
    wrong = np.asarray(metrics["wrong_upper_indices"])
    original = rbm.load_binary_mnist("test", MNIST_DIR)[0][: v4.N_IMAGES]
    with np.load(v4.SNAPSHOTS_PATH) as data:
        for condition in v4.CONDITIONS:
            source = original if condition == "true" else original[wrong]
            for name in v4.MODEL_ORDER:
                for step in v4.STEPS:
                    for image in v4.GALLERY_IMAGES:
                        for chain in v4.GALLERY_CHAINS:
                            saved = data[
                                f"gallery_{name}_{condition}_step{step}_img{image}_chain{chain}"
                            ]
                            assert saved.shape == (28, 28)
                            np.testing.assert_array_equal(
                                saved[: v4.UPPER_ROWS],
                                source[image].reshape(28, 28)[: v4.UPPER_ROWS],
                            )

    for path in (v4.TRUE_GALLERY_PATH, v4.WRONG_GALLERY_PATH, v4.CURVES_PATH):
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
