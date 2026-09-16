"""Tests for the Part 4 work-normalized autocorrelation comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from compare import group_by_temperature, wolff_tau_work  # noqa: E402


def test_wolff_tau_work_converts_cluster_moves_to_sweep_work():
    """tau_work = tau_moves * <c> / L^2."""
    assert wolff_tau_work(16.0, 1024.0, 64) == pytest.approx(4.0)


def test_group_by_temperature_keeps_arrays_in_temperature_order():
    """Samples are grouped by temperature without reordering within a group."""
    temperatures = np.array([2.1, 2.0, 2.0, 2.1, 2.3])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    grouped = group_by_temperature(temperatures, values)

    assert sorted(grouped) == [2.0, 2.1, 2.3]
    np.testing.assert_allclose(grouped[2.0], [2.0, 3.0])
    np.testing.assert_allclose(grouped[2.1], [1.0, 4.0])
    np.testing.assert_allclose(grouped[2.3], [5.0])
