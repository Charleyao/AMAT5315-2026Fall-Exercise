"""Tests for the RePaint completion/variance figure."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from repaint import make_repaint_mask
from repaint_plot import save_repaint_figure


def test_repaint_figure_is_written(tmp_path):
    originals = np.zeros((2, 32, 32, 3), dtype=np.float32)
    completions = np.zeros((2, 4, 32, 32, 3), dtype=np.float32)
    variance = np.zeros((2, 32, 32, 3), dtype=np.float32)
    labels = np.array([3, 8])
    output_path = tmp_path / "repaint.png"

    save_repaint_figure(
        originals,
        completions,
        variance,
        labels,
        make_repaint_mask(),
        (7201, 7202, 7203, 7204),
        str(output_path),
    )

    assert output_path.stat().st_size > 0
