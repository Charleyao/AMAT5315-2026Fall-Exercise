"""Tests for Week 6 Part 3 VERIFY (2): Hopfield recall capacity.

Covers the crossing estimator, batched asynchronous recall, reuse of the
VERIFY (1) pattern sets, the saved sweep (crossings, success rates, sharper
transitions) and the two-panel figure.

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

import hopfield
import hopfield_recall as recall

WEEK_DIR = Path(__file__).resolve().parent


def test_threshold_crossing_interpolates():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.8, 0.4])
    assert recall.threshold_crossing(x, y, 0.9) == pytest.approx(0.5)
    assert recall.threshold_crossing(x, y, 0.5) == pytest.approx(1.75)
    assert np.isnan(recall.threshold_crossing(x, y, 1.2))


def test_async_recall_batch_keeps_a_stored_pattern_fixed():
    pattern = hopfield.random_patterns(jax.random.PRNGKey(0), 1, 50)[0]
    J = hopfield.build_couplings(pattern[None, :])
    starts = jnp.stack([pattern, pattern, pattern])
    final, sweeps, converged = hopfield.async_recall_batch(
        J, starts, jax.random.PRNGKey(1)
    )
    for row in np.asarray(final):
        np.testing.assert_array_equal(row, np.asarray(pattern))
    assert int(sweeps) == 1
    assert bool(converged)


def test_async_recall_batch_zero_field_leaves_spins_unchanged():
    J = jnp.zeros((7, 7))
    starts = jnp.array([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]] * 2)
    final, sweeps, converged = hopfield.async_recall_batch(
        J, starts, jax.random.PRNGKey(2)
    )
    np.testing.assert_array_equal(np.asarray(final), np.asarray(starts))
    assert int(sweeps) == 1
    assert bool(converged)


def test_run_point_uses_first_min_10_patterns():
    n_spins, alpha = 100, 0.05
    key = jax.random.fold_in(jax.random.PRNGKey(recall.SEED), 0)
    point = recall.run_point(n_spins, alpha, key)
    n_patterns = int(round(alpha * n_spins))
    assert n_patterns == 5
    assert point["starts"] == min(recall.STARTS_PER_SET, n_patterns) == 5
    assert point["all_overlaps"].shape == (recall.N_SETS * 5,)
    assert point["overlap"].shape == (recall.N_SETS,)
    assert point["success"].shape == (recall.N_SETS,)
    assert np.all((point["all_overlaps"] >= -1.0) & (point["all_overlaps"] <= 1.0))
    # At alpha = 0.05 recall is essentially perfect.
    assert point["success"].mean() >= 0.95

    per_set = point["all_overlaps"].reshape(recall.N_SETS, 5)
    np.testing.assert_allclose(
        point["success"], (per_set >= recall.SUCCESS_OVERLAP).mean(axis=1)
    )
    np.testing.assert_allclose(point["overlap"], per_set.mean(axis=1))


def test_run_point_is_deterministic():
    key = jax.random.fold_in(jax.random.PRNGKey(recall.SEED), 2 * 1000 + 10)
    other_key = jax.random.fold_in(jax.random.PRNGKey(recall.SEED), 2 * 1000 + 11)
    first = recall.run_point(100, 0.15, key)
    second = recall.run_point(100, 0.15, key)
    other = recall.run_point(100, 0.16, other_key)
    np.testing.assert_array_equal(first["overlap"], second["overlap"])
    np.testing.assert_array_equal(first["success"], second["success"])
    assert not np.array_equal(first["overlap"], other["overlap"])


@pytest.mark.skipif(
    not recall.RESULTS_PATH.is_file(), reason="run hopfield_recall.py first"
)
def test_saved_recall_sweep_verifies_requirements():
    data = json.loads(recall.RESULTS_PATH.read_text())
    settings = data["settings"]
    assert settings["n_spins"] == [100, 200, 400, 800]
    assert settings["pattern_sets_per_point"] == 20
    assert settings["success_overlap_threshold"] == 0.9
    assert settings["seed"] == 5315
    assert settings["alpha_c"] == pytest.approx(0.138)
    assert len(settings["alpha"]) == 21

    crossings = {}
    widths = {}
    for n_spins in settings["n_spins"]:
        result = data["results"][str(n_spins)]
        overlap = np.asarray(result["mean_overlap"])
        success = np.asarray(result["success"])
        assert overlap.shape == (21,)
        assert success.shape == (21,)
        assert np.asarray(result["per_set_overlap"]).shape == (21, 20)
        assert np.asarray(result["per_set_success"]).shape == (21, 20)
        assert np.all(np.isfinite(overlap)) and np.all(np.isfinite(success))

        # Nearly perfect at low alpha, degraded near capacity.
        assert success[0] >= 0.95
        assert success[-1] < success[0] - 0.15
        assert overlap[-1] < overlap[0] - 0.15

        crossing = result["crossing"]
        crossings[n_spins] = crossing
        reference = result["reference_crossing"]
        assert reference == recall.REFERENCE_CROSSINGS[n_spins]
        assert abs(crossing - reference) < 0.02

        # Sharpness: width between success 0.9 and 0.5.
        widths[n_spins] = (
            recall.threshold_crossing(recall.ALPHAS, success, 0.5)
            - recall.threshold_crossing(recall.ALPHAS, success, 0.9)
        )
        assert widths[n_spins] > 0

    # Crossings decrease with N and the transition sharpens.
    ordered = [crossings[n] for n in settings["n_spins"]]
    assert all(ordered[i + 1] < ordered[i] for i in range(len(ordered) - 1))
    ordered_widths = [widths[n] for n in settings["n_spins"]]
    assert all(
        ordered_widths[i + 1] < ordered_widths[i]
        for i in range(len(ordered_widths) - 1)
    )


@pytest.mark.skipif(
    not recall.FIGURE_PATH.is_file(), reason="run hopfield_recall.py first"
)
def test_figure_has_both_panels_and_alpha_c_marker():
    assert recall.FIGURE_PATH.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(recall.RESULTS_PATH.read_text())
    fig = recall.build_figure(data)
    try:
        assert len(fig.axes) == 2
        for axis in fig.axes:
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            for n_spins in (100, 200, 400, 800):
                assert f"N = {n_spins}" in labels
            assert any("0.138" in label for label in labels)
            # A vertical line at alpha_c is present.
            assert any(
                len(line.get_xdata()) == 2
                and np.allclose(line.get_xdata(), recall.ALPHA_C)
                for line in axis.lines
            )
        assert "mean final overlap" in fig.axes[0].get_ylabel()
        assert "success probability" in fig.axes[1].get_ylabel()
    finally:
        recall.plt.close(fig)
