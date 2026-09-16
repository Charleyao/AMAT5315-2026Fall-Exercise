"""Tests for the Part 3 tau(T) helper in ``scripts/tau.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from tau import tau_vs_temperature  # noqa: E402


def _ar1(phi, size, seed):
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=size)
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for i in range(1, size):
        samples[i] = phi * samples[i - 1] + innovations[i]
    return samples


def test_tau_vs_temperature_is_sorted_and_tracks_correlation():
    """Temperatures are sorted and a correlated series has larger tau_int."""
    # A large positive offset keeps |M| from folding the Gaussian tails.
    white = 10.0 + _ar1(0.0, 100_000, seed=1)
    correlated = 10.0 + _ar1(0.8, 100_000, seed=2)

    temperatures, taus = tau_vs_temperature(
        {3.0: (white, "run"), 2.0: (correlated, "run")}
    )

    np.testing.assert_allclose(temperatures, [2.0, 3.0])
    assert 3.5 < taus[0] < 6.0  # AR(1), phi = 0.8 -> tau_int ~ 4.5
    assert taus[1] < 1.0  # white noise -> tau_int ~ 0.5
