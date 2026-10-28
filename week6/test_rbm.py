"""Correctness tests for the Week 6 basic RBM (Part 2, VERIFY (1)).

Covers: configuration values, MNIST binarization/spins (order preserved),
parameter initialization, energy/free-energy math against brute force, and
binary block Gibbs sampling.

Run with::

    cd week6
    uv run python -m pytest -q
"""

import gzip
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import prepare_data as pd
import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"


def raw_idx_images(path: Path) -> np.ndarray:
    """Independently parse an MNIST IDX image file (oracle for order/pixels)."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 0x00000803
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


@pytest.fixture(scope="module")
def config():
    return rbm.load_config()


@pytest.fixture(scope="module")
def binary_train():
    return rbm.load_binary_mnist("train", MNIST_DIR)


def test_config_matches_learning_sheet(config):
    assert config["training"] == {
        "images": 60000,
        "epochs": 20,
        "batch_size": 100,
        "shuffle_each_epoch": True,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "precision": "float32",
        "seed": 6602,
    }
    assert config["initialization"] == {
        "weight_std": 0.01,
        "hidden_bias": 0.0,
        "visible_bias": "logit(clipped_training_pixel_frequency)",
        "pixel_frequency_clip": [0.01, 0.99],
        "seed": 6601,
    }
    assert config["evaluation"] == {
        "test_images": 64,
        "seed": 6606,
        "reconstruction_at": ["initialization", "each_epoch"],
        "surrogate": "epoch_mean_of_batch_surrogates_before_updates",
    }
    assert config["completion"] == {
        "test_images": 64,
        "chains": 8,
        "steps": [1, 10, 100, 1000],
        "seed": 6607,
        "wrong_upper": "next_evaluation_image_with_a_different_label_wrapping_within_64",
    }
    assert config["model"] == {"visible_units": 784, "hidden_units": 128}


def test_binarization_thresholds_pixels():
    pixels = np.array([[[0, 127], [128, 255]]], dtype=np.uint8)
    binary = rbm.binarize(pixels)
    np.testing.assert_array_equal(binary, [[[0, 0], [1, 1]]])
    assert binary.dtype == np.uint8


def test_spins_are_strictly_plus_or_minus_one():
    binary = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
    spins = rbm.to_spins(binary)
    np.testing.assert_array_equal(spins, [[[-1, 1], [1, -1]]])
    assert set(np.unique(spins).tolist()) == {-1, 1}


def test_training_set_has_784_bits_per_image(binary_train):
    images, labels = binary_train
    assert images.shape == (60000, 784)
    assert labels.shape == (60000,)
    assert set(np.unique(images).tolist()) == {0, 1}


def test_original_order_is_preserved(binary_train):
    images, labels = binary_train
    raw = raw_idx_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    raw_labels = np.frombuffer(
        gzip.decompress((MNIST_DIR / "train-labels-idx1-ubyte.gz").read_bytes())[8:],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(images[:16], (raw[:16] >= 128).reshape(16, 784))
    np.testing.assert_array_equal(labels[:16], raw_labels[:16])


def test_first_training_image_pixels_match_raw_file(binary_train):
    images, labels = binary_train
    raw = raw_idx_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    np.testing.assert_array_equal(images[0].reshape(28, 28), raw[0] >= 128)
    assert labels[0] == raw_label(0)


def raw_label(index: int) -> int:
    data = gzip.decompress((MNIST_DIR / "train-labels-idx1-ubyte.gz").read_bytes())
    return int(data[8 + index])


def test_init_params_shapes_and_biases(config, binary_train):
    images, _ = binary_train
    params = rbm.init_from_config(config, images)
    assert params.W.shape == (784, 128)
    assert params.a.shape == (784,)
    assert params.b.shape == (128,)
    assert params.W.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(params.b), np.zeros(128, np.float32))

    frequencies = np.clip(images.mean(axis=0), 0.01, 0.99)
    expected_a = np.log(frequencies / (1.0 - frequencies))
    np.testing.assert_allclose(np.asarray(params.a), expected_a, rtol=1e-6, atol=1e-6)


def test_init_is_deterministic_for_seed_6601(config, binary_train):
    images, _ = binary_train
    first = rbm.init_from_config(config, images)
    second = rbm.init_from_config(config, images)
    np.testing.assert_array_equal(np.asarray(first.W), np.asarray(second.W))
    np.testing.assert_array_equal(np.asarray(first.a), np.asarray(second.a))


def test_weight_std_matches_configuration(config, binary_train):
    images, _ = binary_train
    params = rbm.init_from_config(config, images)
    assert np.asarray(params.W).std() == pytest.approx(0.01, rel=0.05)


def test_hidden_and_visible_probabilities_are_valid(config, binary_train):
    images, _ = binary_train
    params = rbm.init_from_config(config, images)
    v = jnp.asarray(images[:5], dtype=jnp.float32)
    hp = rbm.hidden_prob(params, v)
    vp = rbm.visible_prob(params, hp)
    assert hp.shape == (5, 128)
    assert vp.shape == (5, 784)
    assert np.isfinite(np.asarray(hp)).all()
    assert np.isfinite(np.asarray(vp)).all()
    assert float(hp.min()) >= 0.0 and float(hp.max()) <= 1.0
    assert float(vp.min()) >= 0.0 and float(vp.max()) <= 1.0


def small_params(seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    return rbm.RBM(
        W=jax.random.normal(k1, (3, 2)),
        a=jax.random.normal(k2, (3,)),
        b=jax.random.normal(k3, (2,)),
    )


def test_energy_matches_brute_force():
    params = small_params()
    v = jnp.array([1.0, 0.0, 1.0])
    hid = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    energies = np.asarray(rbm.energy(params, v, hid))
    W, a, b = (np.asarray(x) for x in (params.W, params.a, params.b))
    v_np = np.asarray(v)
    expected = [
        -v_np @ a - h @ b - (v_np @ W) @ h for h in np.asarray(hid)
    ]
    np.testing.assert_allclose(energies, expected, rtol=1e-6, atol=1e-6)


def test_free_energy_matches_log_sum_over_hidden_states():
    params = small_params()
    v = jnp.array([1.0, 0.0, 1.0])
    hid = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    energies = np.asarray(rbm.energy(params, v, hid))
    expected = -np.log(np.exp(-energies).sum())
    assert np.asarray(rbm.free_energy(params, v)) == pytest.approx(expected, rel=1e-6)


def test_free_energy_is_finite_for_initial_model(config, binary_train):
    images, _ = binary_train
    params = rbm.init_from_config(config, images)
    free = np.asarray(rbm.free_energy(params, jnp.asarray(images[:8], jnp.float32)))
    assert free.shape == (8,)
    assert np.isfinite(free).all()


def test_bernoulli_sampling_is_binary_and_reproducible():
    key = jax.random.PRNGKey(7)
    probs = jnp.full((100, 50), 0.5)
    first = np.asarray(rbm.sample_bernoulli(key, probs))
    second = np.asarray(rbm.sample_bernoulli(key, probs))
    assert set(np.unique(first).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(first, second)


def test_gibbs_step_returns_binary_visible_and_hidden():
    params = small_params(seed=1)
    v = jnp.full((4, 3), 0.5)
    key = jax.random.PRNGKey(11)
    v_new, h_new = rbm.gibbs_step(params, v, key)
    assert v_new.shape == (4, 3)
    assert h_new.shape == (4, 2)
    assert set(np.unique(np.asarray(v_new)).tolist()) <= {0.0, 1.0}
    assert set(np.unique(np.asarray(h_new)).tolist()) <= {0.0, 1.0}


def test_gibbs_step_is_deterministic_at_saturated_probabilities():
    # With strong positive biases and zero weights both conditionals saturate,
    # so one block-Gibbs step must return all ones.
    params = rbm.RBM(W=jnp.zeros((3, 2)), a=jnp.full((3,), 50.0), b=jnp.full((2,), 50.0))
    v_new, h_new = rbm.gibbs_step(params, jnp.zeros((1, 3)), jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(v_new), np.ones((1, 3)))
    np.testing.assert_array_equal(np.asarray(h_new), np.ones((1, 2)))


def test_gibbs_chain_keeps_shape_and_binary_values():
    params = small_params(seed=2)
    v0 = jnp.full((2, 3), 0.5)
    chain = rbm.gibbs_chain(params, v0, jax.random.PRNGKey(3), steps=10)
    assert chain.shape == (2, 3)
    assert set(np.unique(np.asarray(chain)).tolist()) <= {0.0, 1.0}
