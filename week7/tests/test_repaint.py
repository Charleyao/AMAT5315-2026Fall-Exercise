"""Tests for Part 5 RePaint masks, schedule, and verification metrics."""

import numpy as np
import pytest

import repaint


def test_mask_hides_rows_and_columns_6_to_25():
    mask = repaint.make_repaint_mask()

    assert mask.shape == (32, 32, 1)
    assert mask[0, 0, 0] == 1.0
    assert mask[5, 15, 0] == 1.0
    assert mask[6, 6, 0] == 0.0
    assert mask[25, 25, 0] == 0.0
    assert mask[26, 26, 0] == 1.0
    assert int(mask.sum()) == 32 * 32 - 20 * 20


def test_repaint_schedule_has_50_jumps_and_99_evaluations():
    levels = repaint.REPAINT_LEVELS

    assert levels[0] == 1000
    assert levels[-2:] == (20, 0)
    assert len(levels) == 51
    assert repaint.repaint_jumps()[0] == (1000, 980)
    assert repaint.repaint_jumps()[-1] == (20, 0)
    assert len(repaint.repaint_jumps()) == 50
    assert repaint.repaint_network_evaluations() == 99


def test_test_images_load_with_expected_shape_and_labels():
    images, labels = repaint.load_test_images()

    assert images.shape == (4, 32, 32, 3)
    assert images.dtype == np.float32
    assert labels.shape == (4,)
    assert np.isfinite(images).all()


def test_repaint_metrics_detects_observed_change_and_hidden_variance():
    originals = np.zeros((2, 32, 32, 3), dtype=np.float32)
    completions = np.zeros((2, 4, 32, 32, 3), dtype=np.float32)
    mask = repaint.make_repaint_mask()
    hidden = np.broadcast_to((1.0 - mask) > 0.5, (32, 32, 3))
    for seed in range(4):
        completions[:, seed][:, hidden] = (seed + 1) * 0.1

    metrics = repaint.repaint_metrics(completions, originals, mask)

    assert metrics["observed_max_change"] == pytest.approx(0.0)
    assert metrics["max_hidden_std"] > 0.0
    assert metrics["max_pairwise_hidden_difference"] > 0.0
    assert metrics["completions_in_range"] is True
    assert len(metrics["hidden_mse"]) == 2
    assert len(metrics["hidden_mse"][0]) == 4
