"""Associative-memory stores and capacity measurements for Week 8 Part 3.

The reported interference-to-signal ratio is an amplitude ratio,

    R = ||interference|| / ||signal|| = sqrt(energy ratio).

This is the quantity whose theoretical prediction grows like sqrt(M) at fixed
dimension; capacity is the M at which R crosses 1.
"""

from __future__ import annotations

import numpy as np

M_VALUES = np.array(
    [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    dtype=np.int64,
)
DIMENSIONS = (16, 32, 64)
MECHANISMS = (
    "outer_product",
    "least_squares",
    "relu_p1",
    "relu_p2",
    "relu_p3",
)
MECHANISM_LABELS = {
    "outer_product": "Outer product",
    "least_squares": "Least squares",
    "relu_p1": "ReLU (p=1)",
    "relu_p2": "ReLU^2 (p=2)",
    "relu_p3": "ReLU^3 (p=3)",
}
DEFAULT_DRAWS = 3
DEFAULT_QUERY_SAMPLES = 256
DEFAULT_SEED = 1730
MAX_BATCH_CELLS = 8_000_000
EXPECTED_CAPACITIES_D64 = {
    "outer_product": 67.4,
    "least_squares": 129.0,
    "relu_p1": 128.0,
    "relu_p2": 2871.0,
    "relu_p3": 38124.0,
}
EXPECTED_SLOPES = {
    "outer_product": 0.96,
    "least_squares": 0.99,
    "relu_p1": 0.95,
    "relu_p2": 1.96,
    "relu_p3": 2.79,
}


def random_unit_keys(m: int, d: int, seed: int) -> np.ndarray:
    """Generate M random unit keys in R^d (shared utility for Parts 3-4)."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((m, d))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    return keys


def random_unit_arrays(
    m: int,
    d: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate M random unit keys and M random unit values in R^d."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((m, d))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.standard_normal((m, d))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return keys, values


def measure_interference_signal(
    m: int,
    d: int,
    seed: int,
    n_queries: int = DEFAULT_QUERY_SAMPLES,
) -> dict[str, float]:
    """Measure amplitude R for all five stores on one random draw.

    Memory use is O(M d) for the keys/values plus O(M d) for each SVD or
    ReLU batch.  No M-by-M Gram or output matrix is ever constructed.
    """
    keys, values = random_unit_arrays(m, d, seed)
    query_rng = np.random.default_rng(seed + 7919)
    query_indices = query_rng.choice(
        m, size=min(n_queries, m), replace=False
    )
    query_keys = keys[query_indices]
    query_values = values[query_indices]
    n_query = len(query_indices)

    # 1. Outer-product linear store: W = sum_i v_i k_i^T.
    memory_matrix = values.T @ keys  # d x d
    recalled = query_keys @ memory_matrix.T
    signal_energy = float(np.mean(np.sum(query_values**2, axis=1)))
    outer_energy = float(np.mean(np.sum((recalled - query_values) ** 2, axis=1)))
    outer_r = float(np.sqrt(outer_energy / signal_energy))

    # 2. Least-squares linear store: W = V^T (K^T)^+.
    # For M <= d, full row rank makes the stored keys exactly interpolated.
    if m <= d:
        least_squares_r = 0.0
    else:
        left_vectors = np.linalg.svd(keys, full_matrices=False)[0]
        leverage = np.sum(left_vectors[query_indices] ** 2, axis=1)
        # Signal coefficient is leverage; interference is 1 - leverage.
        least_squares_r = float(
            np.sqrt(np.mean((1.0 - leverage) / leverage))
        )

    # 3-5. ReLU similarity stores, F(q) = sum_i v_i ReLU(k_i.q)^p.
    # Reuse each query-by-key block for all three powers.
    interference_energy = np.zeros(3, dtype=np.float64)
    batch_size = max(1, min(n_query, MAX_BATCH_CELLS // m))
    for start in range(0, n_query, batch_size):
        stop = min(start + batch_size, n_query)
        key_block = query_keys[start:stop]
        value_block = query_values[start:stop]
        similarities = key_block @ keys.T  # batch x M, not M x M
        positive = np.maximum(similarities, 0.0)
        for power_index, power in enumerate((1, 2, 3)):
            features = positive if power == 1 else positive**power
            recalled = features @ values
            interference_energy[power_index] += float(
                np.sum((recalled - value_block) ** 2)
            )
    relu_energy = interference_energy / n_query
    relu_r = np.sqrt(relu_energy / signal_energy)

    return {
        "outer_product": outer_r,
        "least_squares": least_squares_r,
        "relu_p1": float(relu_r[0]),
        "relu_p2": float(relu_r[1]),
        "relu_p3": float(relu_r[2]),
    }


def theoretical_ratios(
    m_values: np.ndarray,
    d: int,
) -> dict[str, np.ndarray]:
    """Theoretical amplitude R including finite-d sphere moments for ReLU."""
    m_values = np.asarray(m_values, dtype=np.float64)
    m_minus_one = m_values - 1.0
    return {
        "outer_product": np.sqrt(m_minus_one / d),
        "least_squares": np.sqrt(np.maximum((m_values - d) / d, 0.0)),
        "relu_p1": np.sqrt(m_minus_one / (2.0 * d)),
        "relu_p2": np.sqrt(
            3.0 * m_minus_one / (2.0 * d * (d + 2.0))
        ),
        "relu_p3": np.sqrt(
            15.0 * m_minus_one / (2.0 * d * (d + 2.0) * (d + 4.0))
        ),
    }


def capacity_from_curve(
    ratios: np.ndarray,
    m_values: np.ndarray = M_VALUES,
) -> float:
    """Interpolate the R=1 crossing on log-log axes.

    A zero below the crossing (the exact least-squares regime) is handled by
    taking the first grid point whose R reaches 1.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    m_values = np.asarray(m_values, dtype=np.float64)
    crossing = np.flatnonzero(ratios >= 1.0)
    if len(crossing) == 0:
        return float("nan")
    index = int(crossing[0])
    if index == 0:
        return float(m_values[0])
    r0 = ratios[index - 1]
    r1 = ratios[index]
    if r0 <= 0.0:
        return float(m_values[index])
    log_capacity = np.log(m_values[index - 1]) + (
        (0.0 - np.log(r0))
        * (np.log(m_values[index]) - np.log(m_values[index - 1]))
        / (np.log(r1) - np.log(r0))
    )
    return float(np.exp(log_capacity))


def run_memory_experiment(
    dimensions: tuple[int, ...] = DIMENSIONS,
    m_values: np.ndarray = M_VALUES,
    draws: int = DEFAULT_DRAWS,
    n_queries: int = DEFAULT_QUERY_SAMPLES,
    seed: int = DEFAULT_SEED,
    log=print,
) -> dict:
    """Run all dimensions, M values, and independent draws."""
    m_values = np.asarray(m_values, dtype=np.int64)
    ratios_draws = np.empty(
        (len(dimensions), len(m_values), draws, len(MECHANISMS)),
        dtype=np.float64,
    )

    for d_index, d in enumerate(dimensions):
        for m_index, m in enumerate(m_values):
            for draw in range(draws):
                draw_seed = int(
                    seed + 100003 * int(m) + 1000003 * int(d) + draw
                )
                measured = measure_interference_signal(
                    int(m), int(d), draw_seed, n_queries=n_queries
                )
                for mechanism_index, mechanism in enumerate(MECHANISMS):
                    ratios_draws[d_index, m_index, draw, mechanism_index] = (
                        measured[mechanism]
                    )
            log(
                f"[mem ] d={d:3d}, M={int(m):6d}: "
                + ", ".join(
                    f"{name}={np.median(ratios_draws[d_index, m_index, :, j]):.4g}"
                    for j, name in enumerate(MECHANISMS)
                )
            )

    ratios_median = np.median(ratios_draws, axis=2)
    capacities = {}
    for d_index, d in enumerate(dimensions):
        capacities[d] = {
            mechanism: capacity_from_curve(
                ratios_median[d_index, :, mechanism_index], m_values
            )
            for mechanism_index, mechanism in enumerate(MECHANISMS)
        }

    theoretical = {
        d: theoretical_ratios(m_values, d) for d in dimensions
    }
    return {
        "dimensions": list(dimensions),
        "m_values": [int(m) for m in m_values],
        "mechanisms": list(MECHANISMS),
        "draws": draws,
        "n_queries": n_queries,
        "seed": seed,
        "ratios_draws": ratios_draws,
        "ratios_median": ratios_median,
        "capacities": capacities,
        "theoretical": theoretical,
    }


def fit_capacity_slopes(capacities: dict[int, dict[str, float]]) -> dict[str, float]:
    """Fit log(capacity) = slope * log(d) + intercept for every mechanism."""
    dimensions = np.array(sorted(capacities), dtype=np.float64)
    slopes = {}
    for mechanism in MECHANISMS:
        values = np.array(
            [capacities[int(d)][mechanism] for d in dimensions],
            dtype=np.float64,
        )
        slopes[mechanism] = float(
            np.polyfit(np.log(dimensions), np.log(values), 1)[0]
        )
    return slopes


def local_growth_slopes(
    ratios: np.ndarray,
    m_values: np.ndarray = M_VALUES,
    lower: float = 0.15,
    upper: float = 1.0,
) -> float:
    """Fit log(R) vs log(M) in the pre-crossing regime (expected slope 0.5)."""
    ratios = np.asarray(ratios, dtype=np.float64)
    m_values = np.asarray(m_values, dtype=np.float64)
    selected = (ratios >= lower) & (ratios <= upper)
    if selected.sum() < 2:
        return float("nan")
    return float(
        np.polyfit(
            np.log(m_values[selected]), np.log(ratios[selected]), 1
        )[0]
    )
