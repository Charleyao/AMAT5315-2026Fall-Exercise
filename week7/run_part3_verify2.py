"""Part 3 VERIFY (2): DDPM vs DDIM vs Flow Matching sample quality.

Run from week7/ with:

    uv run python run_part3_verify2.py

No model training is performed. Existing diffusion and Flow Matching
checkpoints are reused.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ddim import (
    exact_conditional_mean_noise,
    make_ddim_sampler,
    make_learned_noise_predictor,
)
from diffusion_model import (
    load_checkpoint as load_diffusion_checkpoint,
    load_config as load_diffusion_config,
)
from flow_matching import (
    load_checkpoint as load_flow_checkpoint,
    load_flow_config,
    make_euler_sampler,
)
from forward_diffusion import (
    alpha_bars_from_betas,
    linear_beta_schedule,
    sample_gaussian_mixture,
)
from part3_verify2 import (
    FULL_DIFFUSION_STEPS,
    NULL_PAIRS,
    NUM_SAMPLES,
    SAMPLING_STEPS,
    energy_distance,
    evenly_spaced_levels,
    make_coarse_ddpm_sampler,
    sample_metrics,
)
from part3_verify2_plot import plot_metrics_vs_steps

SEED = 2029
DDPM_NOISE_SEED = 3031
CHECKPOINT_UPDATE = 25_000

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
DIFFUSION_CHECKPOINT = (
    ARTIFACTS_DIR / "checkpoints" / f"params-{CHECKPOINT_UPDATE:06d}.npz"
)
FLOW_CHECKPOINT = (
    ARTIFACTS_DIR / "flow_checkpoints" / f"params-{CHECKPOINT_UPDATE:06d}.npz"
)


def _null_energy_distances(key: jax.Array) -> np.ndarray:
    keys = jax.random.split(key, 2 * NULL_PAIRS)
    distances = []
    for pair in range(NULL_PAIRS):
        first, _ = sample_gaussian_mixture(keys[2 * pair], NUM_SAMPLES)
        second, _ = sample_gaussian_mixture(keys[2 * pair + 1], NUM_SAMPLES)
        distances.append(energy_distance(first, second))
    return np.asarray(distances, dtype=np.float64)


def _run_record(samples: np.ndarray, target: np.ndarray) -> dict[str, object]:
    metrics = sample_metrics(samples)
    metrics["energy_distance"] = energy_distance(samples, target)
    return metrics


def main() -> int:
    if not DIFFUSION_CHECKPOINT.exists() or not FLOW_CHECKPOINT.exists():
        raise FileNotFoundError(
            "missing trained checkpoints; run Part 1/Part 2/Part 3 training first"
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    diffusion_config = load_diffusion_config()
    diffusion_params = load_diffusion_checkpoint(DIFFUSION_CHECKPOINT)
    flow_config = load_flow_config()
    flow_params = load_flow_checkpoint(FLOW_CHECKPOINT)
    betas = linear_beta_schedule()
    alpha_bars = alpha_bars_from_betas(betas)

    key = jax.random.PRNGKey(SEED)
    key_start, key_target, key_null = jax.random.split(key, 3)
    initial_noise = jax.random.normal(
        key_start, (NUM_SAMPLES, 2), dtype=jnp.float32
    )
    target_samples, _ = sample_gaussian_mixture(key_target, NUM_SAMPLES)
    target_samples = np.asarray(target_samples)
    null_distances = _null_energy_distances(key_null)
    null_q95 = float(np.quantile(null_distances, 0.95))

    print("Week 7 Part 3 VERIFY (2) -- sampling-step comparison")
    print(f"device: {jax.devices()[0]}")
    print(f"num samples per run: {NUM_SAMPLES}")
    print(f"sampling steps: {list(SAMPLING_STEPS)} + 1000 for DDPM/DDIM")
    print(f"null pairs: {NULL_PAIRS}")
    print(f"null energy 95% quantile: {null_q95:.8f}")
    print()

    learned_predictor = make_learned_noise_predictor(
        diffusion_params, diffusion_config
    )
    exact_predictor = lambda x_t, t: exact_conditional_mean_noise(
        x_t, t, alpha_bars
    )

    run_results: dict[str, dict[int, dict[str, object]]] = {
        "DDPM": {},
        "DDIM": {},
        "Flow": {},
        "DDIM_exact": {},
    }
    saved_samples: dict[str, np.ndarray] = {}
    ddpm_noise_base = jax.random.PRNGKey(DDPM_NOISE_SEED)

    for steps in SAMPLING_STEPS:
        levels = evenly_spaced_levels(steps)
        print(f"running DDPM steps={steps} ...", flush=True)
        ddpm_noise = jax.random.normal(
            jax.random.fold_in(ddpm_noise_base, steps),
            (len(levels), NUM_SAMPLES, 2),
            dtype=jnp.float32,
        )
        ddpm_sampler = make_coarse_ddpm_sampler(
            learned_predictor, alpha_bars, levels
        )
        ddpm_samples = np.asarray(ddpm_sampler(initial_noise, ddpm_noise))
        run_results["DDPM"][steps] = _run_record(
            ddpm_samples, target_samples
        )
        saved_samples[f"ddpm_{steps}"] = ddpm_samples

        print(f"running DDIM steps={steps} ...", flush=True)
        ddim_sampler = make_ddim_sampler(
            learned_predictor, alpha_bars, levels
        )
        ddim_samples = np.asarray(ddim_sampler(initial_noise)[-1])
        run_results["DDIM"][steps] = _run_record(
            ddim_samples, target_samples
        )
        saved_samples[f"ddim_{steps}"] = ddim_samples

        print(f"running Flow Matching steps={steps} ...", flush=True)
        flow_sampler = make_euler_sampler(flow_params, flow_config, steps)
        flow_samples = np.asarray(flow_sampler(initial_noise))
        run_results["Flow"][steps] = _run_record(
            flow_samples, target_samples
        )
        saved_samples[f"flow_{steps}"] = flow_samples

    for steps in (1, 2):
        print(f"running exact-predictor DDIM steps={steps} ...", flush=True)
        levels = evenly_spaced_levels(steps)
        exact_sampler = make_ddim_sampler(
            exact_predictor, alpha_bars, levels
        )
        exact_samples = np.asarray(exact_sampler(initial_noise)[-1])
        run_results["DDIM_exact"][steps] = _run_record(
            exact_samples, target_samples
        )
        saved_samples[f"ddim_exact_{steps}"] = exact_samples

    for steps in (FULL_DIFFUSION_STEPS,):
        levels = evenly_spaced_levels(steps)
        print(f"running DDPM steps={steps} ...", flush=True)
        ddpm_noise = jax.random.normal(
            jax.random.fold_in(ddpm_noise_base, steps),
            (len(levels), NUM_SAMPLES, 2),
            dtype=jnp.float32,
        )
        ddpm_sampler = make_coarse_ddpm_sampler(
            learned_predictor, alpha_bars, levels
        )
        ddpm_samples = np.asarray(ddpm_sampler(initial_noise, ddpm_noise))
        run_results["DDPM"][steps] = _run_record(
            ddpm_samples, target_samples
        )
        saved_samples[f"ddpm_{steps}"] = ddpm_samples

        print(f"running DDIM steps={steps} ...", flush=True)
        ddim_sampler = make_ddim_sampler(
            learned_predictor, alpha_bars, levels
        )
        ddim_samples = np.asarray(ddim_sampler(initial_noise)[-1])
        run_results["DDIM"][steps] = _run_record(
            ddim_samples, target_samples
        )
        saved_samples[f"ddim_{steps}"] = ddim_samples

    flow20 = run_results["Flow"][20]
    flow20_energy_pass = bool(flow20["energy_distance"] < 0.005)
    flow20_width_pass = bool(flow20["max_width_error"] < 0.03)
    overall_pass = flow20_energy_pass and flow20_width_pass

    print()
    print(
        "Method | steps | energy distance | max pop error | max width error"
    )
    for method in ("DDPM", "DDIM", "Flow", "DDIM_exact"):
        for steps in sorted(run_results[method]):
            record = run_results[method][steps]
            print(
                f"{method:9s} | {steps:4d} | "
                f"{record['energy_distance']:.6f} | "
                f"{record['max_population_error']:.6f} | "
                f"{record['max_width_error']:.6f}"
            )

    print()
    print("Exact-predictor coarse-grid comparison:")
    print("steps | DDIM learned energy | DDIM exact energy | "
          "DDIM learned width error | DDIM exact width error")
    for steps in (1, 2):
        learned = run_results["DDIM"][steps]
        exact = run_results["DDIM_exact"][steps]
        print(
            f"{steps:5d} | {learned['energy_distance']:.6f} | "
            f"{exact['energy_distance']:.6f} | "
            f"{learned['max_width_error']:.6f} | "
            f"{exact['max_width_error']:.6f}"
        )

    print()
    print(
        "Flow Matching 20-step energy distance < 0.005: "
        f"{'PASS' if flow20_energy_pass else 'FAIL'} "
        f"({flow20['energy_distance']:.6f})"
    )
    print(
        "Flow Matching 20-step max width error < 0.03: "
        f"{'PASS' if flow20_width_pass else 'FAIL'} "
        f"({flow20['max_width_error']:.6f})"
    )
    print(f"Part 3 VERIFY (2): {'PASS' if overall_pass else 'FAIL'}")

    plot_results = {
        method: {
            steps: {
                "energy_distance": record["energy_distance"],
                "max_population_error": record["max_population_error"],
                "max_width_error": record["max_width_error"],
            }
            for steps, record in method_results.items()
        }
        for method, method_results in run_results.items()
    }
    metrics_figure, _ = plot_metrics_vs_steps(
        plot_results,
        null_q95,
        str(EVIDENCE_DIR / "part3_verify2_metrics.png"),
    )
    plt.close(metrics_figure)

    results = {
        "part": "Week 7 Part 3 VERIFY (2)",
        "device": str(jax.devices()[0]),
        "num_samples": NUM_SAMPLES,
        "sampling_steps": list(SAMPLING_STEPS),
        "full_diffusion_steps": FULL_DIFFUSION_STEPS,
        "seed": SEED,
        "ddpm_noise_seed": DDPM_NOISE_SEED,
        "null_scale": {
            "num_pairs": NULL_PAIRS,
            "num_samples_per_pair": NUM_SAMPLES,
            "distances": null_distances.tolist(),
            "mean": float(null_distances.mean()),
            "q95": null_q95,
            "min": float(null_distances.min()),
            "max": float(null_distances.max()),
        },
        "configs": {
            "diffusion": asdict(diffusion_config),
            "flow": asdict(flow_config),
        },
        "checkpoints": {
            "diffusion": str(DIFFUSION_CHECKPOINT.relative_to(WEEK_DIR)),
            "flow": str(FLOW_CHECKPOINT.relative_to(WEEK_DIR)),
            "retrained": False,
        },
        "runs": {
            method: {
                str(steps): record
                for steps, record in method_results.items()
            }
            for method, method_results in run_results.items()
        },
        "verification": {
            "flow_20_energy_distance": flow20["energy_distance"],
            "flow_20_energy_below_0_005": flow20_energy_pass,
            "flow_20_max_width_error": flow20["max_width_error"],
            "flow_20_width_error_below_0_03": flow20_width_pass,
            "overall_pass": overall_pass,
        },
        "evidence": {
            "metrics_plot": "evidence/part3_verify2_metrics.png",
            "verify_log": "evidence/part3_verify2.txt",
        },
        "artifacts": {
            "results": "artifacts/part3_verify2_results.json",
            "samples": "artifacts/part3_verify2_samples.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "part3_verify2_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    save_arrays = {
        "initial_noise": np.asarray(initial_noise),
        "target_samples": target_samples,
        "null_energy_distances": null_distances,
    }
    save_arrays.update(saved_samples)
    np.savez_compressed(ARTIFACTS_DIR / "part3_verify2_samples.npz", **save_arrays)

    print("metrics plot: evidence/part3_verify2_metrics.png")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
