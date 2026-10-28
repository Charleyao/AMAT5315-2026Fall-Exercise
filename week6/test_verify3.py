"""Tests for Week 6 Part 2 VERIFY (3): frozen reconstruction and generation.

Checks identical initial configurations, matched randomness, frozen
parameters, all-chain recording at the requested steps, the development of
connected ink structure, and the figure/artifact layouts.

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
import verify3_sampling as v3

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
    return {
        name: cd.load_checkpoint(path)
        for name, path in cd.CHECKPOINT_PATHS.items()
    }


@pytest.fixture(scope="module")
def held_out():
    images, labels = rbm.load_binary_mnist("test", MNIST_DIR)
    return images[: v3.N_HELD_OUT], labels[: v3.N_HELD_OUT]


@pytest.fixture(scope="module")
def random_setup(config):
    initial_key, step_keys = v3.matched_randomness(1000, config["completion"]["seed"])
    spins = v3.initial_spins(config["completion"]["chains"], initial_key)
    return spins, step_keys


@pytest.fixture(scope="module")
def snapshots(params, random_setup):
    _spins, step_keys = random_setup
    v0 = jax.numpy.asarray(v3.spins_to_binary(_spins))
    return {
        name: v3.generate_snapshots(params[name], v0, step_keys, v3.SNAPSHOT_STEPS)
        for name in v3.MODEL_ORDER
    }


def test_checkpoints_have_expected_shapes(params):
    for name in v3.MODEL_ORDER:
        assert params[name].W.shape == (784, 128)
        assert params[name].a.shape == (784,)
        assert params[name].b.shape == (128,)
        assert np.isfinite(np.asarray(params[name].W)).all()


def test_matched_randomness_is_deterministic(config):
    first_init, first_keys = v3.matched_randomness(20, 6607)
    second_init, second_keys = v3.matched_randomness(20, 6607)
    np.testing.assert_array_equal(np.asarray(first_init), np.asarray(second_init))
    np.testing.assert_array_equal(np.asarray(first_keys), np.asarray(second_keys))
    assert first_keys.shape == (20, 2)


def test_initial_spins_are_strictly_plus_or_minus_one(random_setup):
    spins, _keys = random_setup
    assert spins.shape == (8, 784)
    assert set(np.unique(np.asarray(spins)).tolist()) == {-1, 1}


def test_both_models_start_from_identical_configurations(snapshots, random_setup):
    spins, _keys = random_setup
    np.testing.assert_array_equal(snapshots["cd1"][0], snapshots["cd10"][0])
    np.testing.assert_array_equal(snapshots["cd1"][0], v3.spins_to_binary(spins))


def test_snapshots_record_all_steps_and_all_chains(snapshots):
    for name in v3.MODEL_ORDER:
        assert sorted(snapshots[name]) == v3.SNAPSHOT_STEPS
        for step in v3.SNAPSHOT_STEPS:
            assert snapshots[name][step].shape == (8, 784)
            assert set(np.unique(snapshots[name][step]).tolist()) <= {0.0, 1.0}


def test_parameters_remain_frozen_during_generation(params, snapshots):
    for name in v3.MODEL_ORDER:
        loaded = cd.load_checkpoint(cd.CHECKPOINT_PATHS[name])
        for field in ("W", "a", "b"):
            np.testing.assert_array_equal(
                np.asarray(getattr(params[name], field)),
                np.asarray(getattr(loaded, field)),
            )


def test_reconstruction_uses_frozen_params_and_fixed_key(params, held_out):
    images, _labels = held_out
    v0 = jax.numpy.asarray(images, dtype=jax.numpy.float32)
    key = jax.random.PRNGKey(6606)
    before = np.asarray(params["cd1"].W).copy()
    first = v3.reconstruct(params["cd1"], v0, key)
    second = v3.reconstruct(params["cd1"], v0, key)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (v3.N_HELD_OUT, 784)
    assert set(np.unique(first).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(np.asarray(params["cd1"].W), before)


def test_generation_develops_connected_ink_structure(snapshots):
    for name in v3.MODEL_ORDER:
        first = v3.structure_metrics(snapshots[name][0])
        last = v3.structure_metrics(snapshots[name][v3.SNAPSHOT_STEPS[-1]])
        assert first["chains"] == last["chains"] == 8
        assert last["mean_components"] < first["mean_components"]
        assert last["mean_largest_fraction"] > first["mean_largest_fraction"]
        assert 0.0 <= last["ink_fraction"] <= 1.0


def test_reconstruction_figure_layout(held_out, params):
    images, labels = held_out
    v0 = jax.numpy.asarray(images, dtype=jax.numpy.float32)
    key = jax.random.PRNGKey(6606)
    recons = {
        name: v3.reconstruct(params[name], v0, key) for name in v3.MODEL_ORDER
    }
    fig = v3.build_reconstruction_figure(images, recons, labels)
    try:
        axes = [ax for ax in fig.axes if ax.images]
        assert len(axes) == 3 * v3.N_HELD_OUT
        row_labels = [ax.get_ylabel() for ax in axes if ax.get_ylabel()]
        assert sorted(row_labels) == ["CD-1", "CD-10", "input"]
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) == v3.N_HELD_OUT
    finally:
        v3.plt.close(fig)


def test_generation_figure_shows_both_models_all_chains_all_steps(snapshots):
    fig = v3.build_generation_figure(snapshots)
    try:
        axes = [ax for ax in fig.axes if ax.images]
        assert len(axes) == 2 * len(v3.SNAPSHOT_STEPS) * 8
        gids = [ax.get_gid() for ax in axes]
        for name in v3.MODEL_ORDER:
            for step in v3.SNAPSHOT_STEPS:
                for chain in range(8):
                    assert f"model={name};step={step};chain={chain}" in gids
        fig_text = " ".join(text.get_text() for text in fig.texts)
        assert "CD-1 (k=1)" in fig_text and "CD-10 (k=10)" in fig_text
        assert "no best-sample selection" in fig_text
    finally:
        v3.plt.close(fig)


@pytest.mark.skipif(
    not v3.RESULTS_PATH.is_file(), reason="run verify3_sampling.py to create results"
)
def test_saved_results_and_artifacts_verify_requirements():
    metrics = json.loads(v3.RESULTS_PATH.read_text())
    assert metrics["parameters_frozen"] == {"cd1": True, "cd10": True}
    assert metrics["initial_configurations_identical"] is True
    assert metrics["settings"]["chains"] == 8
    assert metrics["settings"]["steps"] == v3.SNAPSHOT_STEPS
    for name in v3.MODEL_ORDER:
        for step in v3.SNAPSHOT_STEPS:
            item = metrics["generation"][name][str(step)]
            assert item["chains"] == 8
            assert np.isfinite(item["ink_fraction"])
            assert np.isfinite(item["mean_components"])
        first = metrics["generation"][name]["0"]
        last = metrics["generation"][name][str(v3.SNAPSHOT_STEPS[-1])]
        assert last["mean_components"] < first["mean_components"]
        assert np.isfinite(metrics["reconstruction"]["mse"][name])

    with np.load(v3.SNAPSHOTS_PATH) as data:
        assert len(data.files) == 2 * len(v3.SNAPSHOT_STEPS)
        for name in v3.MODEL_ORDER:
            for step in v3.SNAPSHOT_STEPS:
                assert data[f"{name}_step{step}"].shape == (8, 784)

    for path in (v3.RECONSTRUCTION_FIGURE, v3.GENERATION_FIGURE):
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
