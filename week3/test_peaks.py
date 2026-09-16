"""Tests for the five-point Part 2 peak fit in ``scripts/peaks.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from peaks import quadratic_peak  # noqa: E402


def test_quadratic_peak_recovers_known_vertex():
    """A noiseless five-point parabola must return its exact vertex."""
    temperature = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # chi(T) = 5 - 3 (T - 0.4)^2
    chi = 5.0 - 3.0 * (temperature - 0.4) ** 2

    peak, _, (lo, hi) = quadratic_peak(temperature, chi)

    assert peak == pytest.approx(0.4)
    assert (lo, hi) == (0, 5)


def test_quadratic_peak_centers_on_the_maximum():
    """The fit window must be the five points around the largest chi."""
    temperature = np.arange(11, dtype=float)
    chi = np.zeros(11)
    chi[7] = 10.0  # maximum at index 7
    chi[5:10] = [1.0, 3.0, 10.0, 3.0, 1.0]

    _, _, (lo, hi) = quadratic_peak(temperature, chi)

    assert (lo, hi) == (5, 10)
