"""Picture memory and four attention-style recall mechanisms for Part 4."""

from __future__ import annotations

import numpy as np

from .memory import random_unit_keys

PICTURE_SIZE = 32
M_VALUES = (64, 256, 1024, 4096)
KEY_DIM = 64
SOFTMAX_BETA = 8.0
BASE_SEED = 1730
METHODS = ("outer_product", "relu", "relu_squared", "softmax")
METHOD_LABELS = {
    "outer_product": "Outer-product",
    "relu": "ReLU",
    "relu_squared": "ReLU$^2$",
    "softmax": "Softmax ($\\beta$=8)",
}
METHOD_TEXT_LABELS = {
    "outer_product": "Outer-product",
    "relu": "ReLU",
    "relu_squared": "ReLU^2",
    "softmax": "Softmax beta=8",
}
REFERENCE_R = {
    1024: {
        "outer_product": 4.05,
        "relu": 2.84,
    },
    4096: {
        "outer_product": 7.94,
        "relu": 5.82,
        "relu_squared": 1.23,
        "softmax": 0.057,
    },
}
THEORY_R = {
    1024: {
        "outer_product": 4.0,
        "relu": 2.83,
    },
}


def _segment_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    """Distance from every grid point to a line segment."""
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    t = np.clip(
        ((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy),
        0.0,
        1.0,
    )
    return np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))


def make_character(size: int = PICTURE_SIZE) -> np.ndarray:
    """Rasterize a recognizable block letter "A" as +1 ink / -1 background.

    The learning-sheet PDF is not available in this repository, so the glyph
    could not be copied directly.  The substitution is documented in the
    README and settings artifact.
    """
    if size != PICTURE_SIZE:
        raise ValueError(
            "the documented character bitmap is defined for size=32"
        )
    yy, xx = np.mgrid[0:size, 0:size]
    left_stroke = (
        _segment_distance(xx, yy, (5.0, 27.5), (15.5, 4.0)) <= 2.2
    )
    right_stroke = (
        _segment_distance(xx, yy, (26.0, 27.5), (15.5, 4.0)) <= 2.2
    )
    crossbar = (
        (np.abs(yy - 20.0) <= 1.7) & (xx >= 10.5) & (xx <= 20.5)
    )
    ink = left_stroke | right_stroke | crossbar
    return np.where(ink, 1.0, -1.0)


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def build_memory(
    m: int,
    character: np.ndarray,
    seed: int,
    d: int = KEY_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Build M unit keys and M values; target memory is index 0."""
    if m < 1:
        raise ValueError("M must be positive")
    character_vector = np.asarray(character, dtype=np.float64).reshape(-1)
    keys = random_unit_keys(m, d, seed)
    rng = np.random.default_rng(seed + 1)
    values = np.empty((m, character_vector.size), dtype=np.float64)
    values[0] = character_vector
    if m > 1:
        values[1:] = (
            rng.integers(0, 2, size=(m - 1, character_vector.size)) * 2 - 1
        ).astype(np.float64)
    return keys, values


def attention_weights(
    similarities: np.ndarray,
    beta: float = SOFTMAX_BETA,
) -> dict[str, np.ndarray]:
    """Return the four weighting schemes applied to the key similarities."""
    similarities = np.asarray(similarities, dtype=np.float64)
    positive = np.maximum(similarities, 0.0)
    return {
        "outer_product": similarities,
        "relu": positive,
        "relu_squared": positive**2,
        "softmax": stable_softmax(beta * similarities),
    }


def recall_from_memory(
    keys: np.ndarray,
    values: np.ndarray,
    beta: float = SOFTMAX_BETA,
) -> dict[str, dict]:
    """Query with the stored character's key and measure every mechanism."""
    similarities = keys @ keys[0]
    weights_by_method = attention_weights(similarities, beta=beta)
    target_value = values[0]
    target_norm = float(np.linalg.norm(target_value))
    records = {}

    for method in METHODS:
        weights = weights_by_method[method]
        target_weight = float(weights[0])
        if target_weight <= 0.0:
            raise ValueError(f"{method} assigned non-positive target weight")

        recalled = weights @ values
        signal = target_weight * target_value
        interference = recalled - signal
        interference_norm = float(np.linalg.norm(interference))
        signal_norm = float(np.linalg.norm(signal))
        normalized = recalled / target_weight
        normalized_norm = float(np.linalg.norm(normalized))
        cosine = float(
            np.dot(normalized, target_value)
            / (normalized_norm * target_norm)
        )

        records[method] = {
            "R": interference_norm / signal_norm,
            "cosine_similarity": cosine,
            "target_weight": target_weight,
            "signal_norm": signal_norm,
            "interference_norm": interference_norm,
            "recalled_norm": float(np.linalg.norm(recalled)),
            "normalized_image": normalized.reshape(PICTURE_SIZE, PICTURE_SIZE),
            "weights": weights,
            "similarities": similarities,
        }
    return records


def run_picture_experiment(
    seed: int = BASE_SEED,
    m_values: tuple[int, ...] = M_VALUES,
    d: int = KEY_DIM,
    beta: float = SOFTMAX_BETA,
) -> dict:
    """Run all M values and all four recall mechanisms."""
    character = make_character()
    records = {}
    for m in m_values:
        # Independent memory draw for each M, with a fixed reproducible seed.
        keys, values = build_memory(m, character, seed=seed + m, d=d)
        records[m] = recall_from_memory(keys, values, beta=beta)

    return {
        "character": character,
        "character_ink_pixels": int(np.count_nonzero(character > 0)),
        "m_values": list(m_values),
        "d": d,
        "beta": beta,
        "seed": seed,
        "methods": list(METHODS),
        "records": records,
    }
