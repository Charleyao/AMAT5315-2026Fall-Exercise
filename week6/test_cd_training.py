"""Tests for Week 6 Part 2 VERIFY (2): CD-1 vs CD-10 training.

Fast tests use small subsets; the integration test validates the full
20-epoch metrics artifact produced by ``cd_training.py`` (skipped if absent).

Run with::

    cd week6
    uv run python -m pytest -q
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cd_training as cd
import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"


@pytest.fixture(scope="module")
def config():
    return rbm.load_config()


@pytest.fixture(scope="module")
def small_data():
    train, _ = rbm.load_binary_mnist("train", MNIST_DIR)
    test, _ = rbm.load_binary_mnist("test", MNIST_DIR)
    return train[:1000], test[:64]


def test_epoch_permutations_are_deterministic_and_complete():
    first = cd.epoch_permutations(100, 3, seed=6602)
    second = cd.epoch_permutations(100, 3, seed=6602)
    assert len(first) == 3
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(100))
    assert not np.array_equal(first[0], first[1])


def test_cd_surrogate_matches_definition():
    params = rbm.RBM(
        W=jnp.array([[0.2, -0.1], [0.05, 0.3], [-0.4, 0.1]]),
        a=jnp.array([0.1, -0.2, 0.3]),
        b=jnp.array([-0.1, 0.2]),
    )
    v0 = jnp.array([[1.0, 0.0, 1.0]])
    vk = jnp.array([[0.0, 1.0, 1.0]])
    expected = float(
        jnp.mean(rbm.free_energy(params, v0) - rbm.free_energy(params, vk))
    )
    assert float(cd.cd_surrogate(params, v0, vk)) == pytest.approx(expected, rel=1e-6)


def test_one_step_reconstruction_error_is_bounded_and_reproducible(config, small_data):
    _train, test = small_data
    params = rbm.init_from_config(config, _train)
    key = jax.random.PRNGKey(config["evaluation"]["seed"])
    first = float(cd.one_step_reconstruction_error(params, test, key))
    second = float(cd.one_step_reconstruction_error(params, test, key))
    assert 0.0 <= first <= 1.0
    assert first == second


def test_cd_step_changes_parameters_and_is_finite(config, small_data):
    train, _ = small_data
    params = rbm.init_from_config(config, train)
    batch = jnp.asarray(train[:100], jnp.float32)
    key = jax.random.PRNGKey(0)
    updated, surrogate = cd.cd_step(params, batch, key, 0.1, 1)
    assert updated.W.shape == params.W.shape
    assert updated.a.shape == params.a.shape
    assert updated.b.shape == params.b.shape
    assert np.isfinite(np.asarray(surrogate))
    assert not np.array_equal(np.asarray(updated.W), np.asarray(params.W))
    assert np.isfinite(np.asarray(updated.W)).all()


def test_short_training_run_records_expected_lengths(config, small_data):
    train, test = small_data
    init_params = rbm.init_from_config(config, train)
    result = cd.train_cd(
        1, config, train, test, epochs=2, init_params=init_params, progress=False
    )
    assert len(result["surrogate"]) == 2
    assert len(result["reconstruction_error"]) == 3  # initialization + 2 epochs
    assert all(np.isfinite(result["surrogate"]))
    assert all(np.isfinite(result["reconstruction_error"]))
    assert result["runtime_seconds"] > 0


def test_cd1_and_cd10_start_from_identical_parameters(config, small_data):
    train, test = small_data
    init_params = rbm.init_from_config(config, train)
    cd1 = cd.train_cd(1, config, train, test, epochs=1, init_params=init_params, progress=False)
    cd10 = cd.train_cd(10, config, train, test, epochs=1, init_params=init_params, progress=False)
    assert cd1["reconstruction_error"][0] == pytest.approx(
        cd10["reconstruction_error"][0], abs=1e-12
    )


def test_figure_has_two_panels_and_compares_both_models():
    metrics = {
        "epochs": [1, 2],
        "cd1": {
            "surrogate": [1.0, 0.5],
            "reconstruction_error": [0.17, 0.12, 0.09],
        },
        "cd10": {
            "surrogate": [1.1, 0.4],
            "reconstruction_error": [0.17, 0.11, 0.08],
        },
    }
    fig = cd.build_training_curves(metrics)
    try:
        assert len(fig.axes) == 2
        for ax in fig.axes:
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            assert labels == ["CD-1", "CD-10"]
            assert len(ax.lines) == 2
        note = " ".join(text.get_text() for text in fig.texts)
        assert "not the exact negative log-likelihood" in note
    finally:
        cd.plt.close(fig)


@pytest.mark.skipif(
    not cd.METRICS_PATH.is_file(), reason="run cd_training.py to create the metrics"
)
def test_full_metrics_and_figure_verify_the_requirements():
    metrics = json.loads(cd.METRICS_PATH.read_text())
    assert len(metrics["epochs"]) == 20
    assert metrics["chain_lengths"] == {"cd1": 1, "cd10": 10}
    assert metrics["dataset"] == {"training_images": 60000, "evaluation_images": 64}
    assert "not the exact negative log-likelihood" in metrics["description"]

    initial = metrics["initial_reconstruction_error"]
    assert initial["cd1"] == pytest.approx(initial["cd10"], abs=1e-12)
    assert 0.15 <= initial["cd1"] <= 0.19

    for name in ("cd1", "cd10"):
        surrogate = metrics[name]["surrogate"]
        reconstruction = metrics[name]["reconstruction_error"]
        assert len(surrogate) == 20
        assert len(reconstruction) == 21
        assert all(np.isfinite(surrogate))
        assert all(np.isfinite(reconstruction))
        assert reconstruction[-1] < 0.5 * reconstruction[0]
        assert 0.03 <= reconstruction[-1] <= 0.07
        assert metrics[name]["runtime_seconds"] > 0

    assert cd.CURVES_PATH.is_file()
    assert cd.CURVES_PATH.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert cd.CURVES_PATH.stat().st_size > 10_000
