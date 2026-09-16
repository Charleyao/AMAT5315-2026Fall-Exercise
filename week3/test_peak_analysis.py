"""Tests for the Part 2 susceptibility peak analysis.

These only cover the pure fitting helper in :mod:`peak_analysis`, so they run
without touching the Monte Carlo artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from peak_analysis import parabola_vertex


def test_parabola_vertex_recovers_known_vertex():
    """A noiseless parabola must return its exact vertex and height."""
    # chi(T) = 5 - 2 (T - 0.3)^2
    a, b, c = 5.0 - 2.0 * 0.3 ** 2, 4.0 * 0.3, -2.0
    t = np.array([-1.0, 0.0, 1.0])
    y = a + b * t + c * t ** 2
    sigma = np.full(3, 0.1)

    temp, temp_err, chi, chi_err, _, _ = parabola_vertex(t, y, sigma)

    assert temp == pytest.approx(0.3)
    assert chi == pytest.approx(5.0)
    # The exact points still carry the assigned measurement errors, so the
    # propagated vertex errors are finite and positive.
    assert temp_err > 0.0
    assert chi_err > 0.0


def test_parabola_vertex_errors_shrink_with_smaller_noise():
    """Tighter error bars must give a more precise vertex."""
    t = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 2.0, 1.0])
    coarse = parabola_vertex(t, y, np.full(3, 0.2))
    fine = parabola_vertex(t, y, np.full(3, 0.02))

    assert fine[1] < coarse[1]
    assert fine[3] < coarse[3]
