"""Tests for the Part 3 magnetization-trace helpers in ``scripts/trace.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from trace import first_sweeps  # noqa: E402


def test_first_sweeps_selects_the_right_temperature_in_order():
    """Only rows at ``T`` are kept and the earliest sweeps come first."""
    temperatures = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    magnetizations = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    result = first_sweeps(temperatures, magnetizations, 1.0, 2)

    np.testing.assert_allclose(result, [0.1, 0.2])


def test_first_sweeps_caps_at_the_requested_count():
    """A temperature with more sweeps than requested is truncated."""
    temperatures = np.full(10, 2.3)
    magnetizations = np.arange(10.0)

    result = first_sweeps(temperatures, magnetizations, 2.3, 4)

    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0])


def test_first_sweeps_is_empty_when_the_temperature_is_absent():
    """A temperature not present in the run yields no samples."""
    temperatures = np.array([2.0, 2.0])
    magnetizations = np.array([0.5, 0.5])

    assert first_sweeps(temperatures, magnetizations, 9.9, 5).size == 0
