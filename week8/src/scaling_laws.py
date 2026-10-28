"""Chinchilla scaling-law fitting and bootstrap analysis for Week 8 Part 6."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PARAMETER_NAMES = ("E", "A", "B", "alpha", "beta")
HUBER_DELTA = 0.001
N_BOOTSTRAP = 400
BOOTSTRAP_SEED = 1730
CHINCHILLA_COMPUTE = 5.76e23
CHINCHILLA_N = 70e9
CHINCHILLA_D = 1.4e12

REFERENCE_PARAMS = {
    "E": 1.817,
    "A": 478.0,
    "B": 2143.0,
    "alpha": 0.347,
    "beta": 0.367,
}

PUBLISHED_PARAMS = {
    "E": 1.69,
    "A": 406.4,
    "B": 410.7,
    "alpha": 0.34,
    "beta": 0.28,
}

# [E, log(A), log(B), alpha, beta]
BOUNDS = [
    (0.1, 5.0),
    (-10.0, 20.0),
    (-10.0, 30.0),
    (0.05, 1.5),
    (0.05, 1.5),
]


def load_chinchilla_data(path: str | Path) -> dict:
    """Load N, C, L, calculate D=C/(6N), and remove the 5 highest-loss rows."""
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    model_size = np.array([float(row["Model Size"]) for row in rows])
    compute = np.array([float(row["Training FLOP"]) for row in rows])
    loss = np.array([float(row["loss"]) for row in rows])
    data_tokens = compute / (6.0 * model_size)

    order_by_loss = np.argsort(loss)
    removed_indices = order_by_loss[-5:]
    keep_indices = np.sort(order_by_loss[:-5])

    removed = [
        {
            "model_size": float(model_size[index]),
            "training_flop": float(compute[index]),
            "data_tokens": float(data_tokens[index]),
            "loss": float(loss[index]),
        }
        for index in removed_indices
    ]
    return {
        "N": model_size[keep_indices],
        "C": compute[keep_indices],
        "D": data_tokens[keep_indices],
        "L": loss[keep_indices],
        "n_total": len(rows),
        "n_used": len(keep_indices),
        "removed": removed,
    }


def predict_loss(params, N: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Chinchilla law L = E + A/N^alpha + B/D^beta."""
    E, A, B, alpha, beta = params
    return E + A / np.asarray(N) ** alpha + B / np.asarray(D) ** beta


def unpack(vector: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert [E, logA, logB, alpha, beta] to physical parameters."""
    return (
        float(vector[0]),
        float(np.exp(vector[1])),
        float(np.exp(vector[2])),
        float(vector[3]),
        float(vector[4]),
    )


def pack(params) -> np.ndarray:
    E, A, B, alpha, beta = params
    return np.array(
        [E, np.log(A), np.log(B), alpha, beta], dtype=np.float64
    )


def huber_objective(
    vector: np.ndarray,
    N: np.ndarray,
    D: np.ndarray,
    loss: np.ndarray,
) -> float:
    """Mean Huber loss on log(L), threshold delta=0.001."""
    predicted = predict_loss(unpack(vector), N, D)
    residual = np.log(loss) - np.log(predicted)
    absolute = np.abs(residual)
    values = np.where(
        absolute <= HUBER_DELTA,
        0.5 * residual**2,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )
    return float(np.mean(values))


def grid_starts() -> list[np.ndarray]:
    """Deterministic multi-start grid for L-BFGS."""
    starts = []
    for E in (1.6, 1.8):
        for A in (200.0, 500.0, 1200.0):
            for B in (800.0, 2200.0, 6000.0):
                for alpha in (0.3, 0.4):
                    for beta in (0.3, 0.4):
                        starts.append(
                            np.array(
                                [
                                    E,
                                    np.log(A),
                                    np.log(B),
                                    alpha,
                                    beta,
                                ],
                                dtype=np.float64,
                            )
                        )
    return starts


def fit_once(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    start: np.ndarray,
):
    return minimize(
        huber_objective,
        start,
        args=(N, D, L),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={
            "maxiter": 2000,
            "ftol": 1e-15,
            "gtol": 1e-10,
            "maxls": 50,
        },
    )


def fit_scaling_law(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    starts: list[np.ndarray] | None = None,
) -> dict:
    """Fit from every grid start and keep the lowest-objective result."""
    if starts is None:
        starts = grid_starts()
    best = None
    for start in starts:
        result = fit_once(N, D, L, start)
        if best is None or result.fun < best.fun:
            best = result
    params = unpack(best.x)
    return {
        "params": dict(zip(PARAMETER_NAMES, params)),
        "vector": best.x,
        "objective": float(best.fun),
        "success": bool(best.success),
        "iterations": int(best.nit),
        "n_starts": len(starts),
    }


def bootstrap_fits(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    best_vector: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    n_starts: int = 3,
    log=None,
) -> dict:
    """Nonparametric bootstrap over the 240 runs."""
    rng = np.random.default_rng(seed)
    all_starts = grid_starts()
    starts = [
        np.asarray(best_vector, dtype=np.float64),
        all_starts[0],
        all_starts[len(all_starts) // 2],
    ][:n_starts]
    draws = np.empty((n_bootstrap, len(PARAMETER_NAMES)))
    objectives = np.empty(n_bootstrap)
    successes = 0

    for index in range(n_bootstrap):
        sample = rng.integers(0, len(N), size=len(N))
        best = None
        for start in starts:
            result = fit_once(N[sample], D[sample], L[sample], start)
            if best is None or result.fun < best.fun:
                best = result
        draws[index] = unpack(best.x)
        objectives[index] = best.fun
        successes += int(best.success)
        if log is not None and (index + 1) % 100 == 0:
            log(f"[boot] {index + 1}/{n_bootstrap} resamples")

    return {
        "draws": draws,
        "objectives": objectives,
        "n_bootstrap": n_bootstrap,
        "n_success": successes,
        "seed": seed,
        "starts_per_bootstrap": len(starts),
    }


def summarize_bootstrap(draws: np.ndarray) -> dict:
    """Standard errors and percentile 95% intervals for each parameter."""
    summary = {}
    lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
    for index, name in enumerate(PARAMETER_NAMES):
        summary[name] = {
            "mean": float(draws[:, index].mean()),
            "std_error": float(draws[:, index].std(ddof=1)),
            "ci95_lower": float(lower[index]),
            "ci95_upper": float(upper[index]),
        }
    return summary


def compute_optimal(params, compute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute-optimal N* and D* for C = 6ND."""
    _, A, B, alpha, beta = params
    compute = np.asarray(compute, dtype=np.float64)
    n_optimal = ((alpha * A) / (beta * B)) ** (1.0 / (alpha + beta)) * (
        compute / 6.0
    ) ** (beta / (alpha + beta))
    d_optimal = compute / (6.0 * n_optimal)
    return n_optimal, d_optimal


def ratio_at_compute(params, compute: float = CHINCHILLA_COMPUTE) -> dict:
    n_optimal, d_optimal = compute_optimal(params, np.array([compute]))
    return {
        "compute": float(compute),
        "N_star": float(n_optimal[0]),
        "D_star": float(d_optimal[0]),
        "D_over_N": float(d_optimal[0] / n_optimal[0]),
    }


def log_residuals(
    params,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Predicted-minus-observed log-loss residual."""
    return np.log(predict_loss(params, N, D)) - np.log(L)
