"""Part 2 VERIFY (1): deterministic DDIM vs stochastic DDPM.

Run from week7/ with:

    uv run python run_part2_verify1.py

No training is performed. The script loads the existing update-25000
checkpoint, runs deterministic DDIM twice, stochastic DDPM twice, and an exact
conditional-mean-predictor DDIM comparison.
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
    FULL_DDIM_LEVELS,
    exact_conditional_mean_noise,
    make_ddim_sampler,
    make_learned_noise_predictor,
)
from diffusion_model import (
    CONFIG_PATH,
    load_checkpoint,
    load_config,
    make_reverse_sampler,
    model_apply,
)
from forward_diffusion import (
    NUM_DIFFUSION_STEPS,
    alpha_bars_from_betas,
    linear_beta_schedule,
    sample_gaussian_mixture,
)
from part2_plot import (
    render_exact_predictor_comparison,
    render_sample_comparison,
    render_trajectory_comparison,
)
from part2_verify import format_part2_table, part2_metrics
from verify3 import assign_to_nearest_center

NUM_SAMPLES = 2000
START_SEED = 3
DDPM_NOISE_SEED_1 = 1000
DDPM_NOISE_SEED_2 = 1003
PREDICTOR_EVAL_SEED = 2027
CHECKPOINT_UPDATE = 25_000
TRAJECTORY_SAMPLE_INDEX = 0
SAVED_TRAJECTORY_SAMPLES = 10

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
CHECKPOINT_PATH = (
    ARTIFACTS_DIR / "checkpoints" / f"params-{CHECKPOINT_UPDATE:06d}.npz"
)


def main() -> int:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"missing trained checkpoint {CHECKPOINT_PATH}; "
            "run `uv run python run_verify2.py` first"
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    betas = linear_beta_schedule()
    alpha_bars = alpha_bars_from_betas(betas)
    params = load_checkpoint(CHECKPOINT_PATH)

    print("Week 7 Part 2 VERIFY (1) -- DDIM vs DDPM")
    print(f"device: {jax.devices()[0]}")
    print(f"checkpoint: {CHECKPOINT_PATH.relative_to(WEEK_DIR)}")
    print(f"num_samples: {NUM_SAMPLES}")
    print(f"DDIM levels: {len(FULL_DDIM_LEVELS)}")
    print(f"start seed: {START_SEED}")
    print(f"DDPM noise seeds: {DDPM_NOISE_SEED_1}, {DDPM_NOISE_SEED_2}")

    key_start, key_target = jax.random.split(jax.random.PRNGKey(START_SEED))
    x_T = jax.random.normal(
        key_start, (NUM_SAMPLES, 2), dtype=jnp.float32
    )
    target_samples, target_components = sample_gaussian_mixture(
        key_target, NUM_SAMPLES
    )

    learned_predictor = make_learned_noise_predictor(params, config)
    ddim_sampler = make_ddim_sampler(
        learned_predictor, alpha_bars, FULL_DDIM_LEVELS
    )
    print("running DDIM twice over 1000 levels ...", flush=True)
    ddim_trajectory_1 = ddim_sampler(x_T)
    ddim_trajectory_2 = ddim_sampler(x_T)
    ddim_run_difference = float(
        jnp.max(jnp.abs(ddim_trajectory_1 - ddim_trajectory_2))
    )
    ddim_samples = np.asarray(ddim_trajectory_1[-1])

    exact_predictor = lambda x_t, t: exact_conditional_mean_noise(
        x_t, t, alpha_bars
    )
    exact_ddim_sampler = make_ddim_sampler(
        exact_predictor, alpha_bars, FULL_DDIM_LEVELS
    )
    print("running exact conditional-mean DDIM comparison ...", flush=True)
    exact_trajectory = exact_ddim_sampler(x_T)
    exact_samples = np.asarray(exact_trajectory[-1])

    ddpm_sampler = make_reverse_sampler(config, betas, alpha_bars)
    ddpm_noise_1 = jax.random.normal(
        jax.random.PRNGKey(DDPM_NOISE_SEED_1),
        (NUM_DIFFUSION_STEPS, NUM_SAMPLES, 2),
        dtype=jnp.float32,
    )
    ddpm_noise_2 = jax.random.normal(
        jax.random.PRNGKey(DDPM_NOISE_SEED_2),
        (NUM_DIFFUSION_STEPS, NUM_SAMPLES, 2),
        dtype=jnp.float32,
    )
    print("running DDPM twice with different reverse noise draws ...", flush=True)
    ddpm_trajectory_1 = ddpm_sampler(params, x_T, ddpm_noise_1)
    ddpm_trajectory_2 = ddpm_sampler(params, x_T, ddpm_noise_2)
    ddpm_samples_1 = np.asarray(ddpm_trajectory_1[-1])
    ddpm_samples_2 = np.asarray(ddpm_trajectory_2[-1])

    ddim_metrics = part2_metrics(ddim_samples)
    ddpm_metrics_1 = part2_metrics(ddpm_samples_1)
    ddpm_metrics_2 = part2_metrics(ddpm_samples_2)
    exact_metrics = part2_metrics(exact_samples)

    # Compare learned and exact predictors against the true added noise on a
    # fixed evaluation batch spanning all diffusion times.
    key_x0, key_time, key_noise = jax.random.split(
        jax.random.PRNGKey(PREDICTOR_EVAL_SEED), 3
    )
    x0_eval, _ = sample_gaussian_mixture(key_x0, 20_000)
    t_eval = jax.random.randint(
        key_time, (20_000,), 1, NUM_DIFFUSION_STEPS + 1, dtype=jnp.int32
    )
    noise_eval = jax.random.normal(key_noise, x0_eval.shape, dtype=jnp.float32)
    alpha_bar_eval = alpha_bars[t_eval]
    x_t_eval = (
        jnp.sqrt(alpha_bar_eval)[:, None] * x0_eval
        + jnp.sqrt(1.0 - alpha_bar_eval)[:, None] * noise_eval
    )
    learned_eps = model_apply(params, x_t_eval, t_eval, config)
    exact_eps = exact_conditional_mean_noise(x_t_eval, t_eval, alpha_bars)
    learned_mse = float(jnp.mean((learned_eps - noise_eval) ** 2))
    exact_mse = float(jnp.mean((exact_eps - noise_eval) ** 2))
    predictor_difference = float(
        jnp.mean(jnp.abs(learned_eps - exact_eps))
    )

    ddim_fraction_pass = bool(ddim_metrics["all_population_pass"])
    ddim_width_pass = bool(ddim_metrics["all_width_pass"])
    ddpm_width_pass = bool(
        ddpm_metrics_1["all_width_pass"]
        and ddpm_metrics_2["all_width_pass"]
    )
    overall_pass = bool(
        ddim_fraction_pass
        and ddim_width_pass
        and ddpm_width_pass
        and ddim_run_difference == 0.0
    )

    print()
    print(f"Largest difference between the two DDIM runs: {ddim_run_difference:.9f}")
    print()
    print(format_part2_table(ddim_metrics, "DDIM"))
    print()
    print(format_part2_table(ddpm_metrics_1, "DDPM noise draw 1"))
    print()
    print(format_part2_table(ddpm_metrics_2, "DDPM noise draw 2"))
    print()
    print(format_part2_table(exact_metrics, "DDIM with exact conditional-mean predictor"))
    print()
    print("Predictor comparison on 20000 forward samples:")
    print(f"  learned-predictor MSE: {learned_mse:.6f}")
    print(f"  exact conditional-mean MSE: {exact_mse:.6f}")
    print(f"  mean |learned - exact|: {predictor_difference:.6f}")
    print()
    print(
        "DDIM cluster fractions within 0.015 of 0.25: "
        f"{'PASS' if ddim_fraction_pass else 'FAIL'}"
    )
    print(
        "DDIM widths within [0.29, 0.32]: "
        f"{'PASS' if ddim_width_pass else 'FAIL'}"
    )
    print(
        "DDPM widths within [0.29, 0.32] for both noise draws: "
        f"{'PASS' if ddpm_width_pass else 'FAIL'}"
    )
    print(f"Part 2 VERIFY (1): {'PASS' if overall_pass else 'FAIL'}")

    trajectory_path = EVIDENCE_DIR / "part2_trajectory_comparison.png"
    comparison_path = EVIDENCE_DIR / "part2_samples_comparison.png"
    exact_path = EVIDENCE_DIR / "part2_exact_predictor_comparison.png"

    trajectory_figure, _ = render_trajectory_comparison(
        np.asarray(ddim_trajectory_1),
        np.asarray(ddpm_trajectory_1),
        np.asarray(ddpm_trajectory_2),
        TRAJECTORY_SAMPLE_INDEX,
        str(trajectory_path),
    )
    trajectory_figure.clf()
    plt.close(trajectory_figure)
    render_sample_comparison(
        ddim_samples,
        ddpm_samples_1,
        np.asarray(target_samples),
        ddim_metrics,
        ddpm_metrics_1,
        str(comparison_path),
    )
    render_exact_predictor_comparison(
        ddim_samples,
        exact_samples,
        np.asarray(target_samples),
        exact_metrics,
        str(exact_path),
    )

    saved_indices = np.arange(
        min(SAVED_TRAJECTORY_SAMPLES, NUM_SAMPLES)
    )
    results = {
        "part": "Week 7 Part 2 VERIFY (1)",
        "algorithm": "deterministic DDIM (eta=0) vs DDPM",
        "device": str(jax.devices()[0]),
        "config_path": str(CONFIG_PATH.relative_to(WEEK_DIR)),
        "config": asdict(config),
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(WEEK_DIR)),
            "update": CHECKPOINT_UPDATE,
            "retrained": False,
        },
        "settings": {
            "num_samples": NUM_SAMPLES,
            "ddim_levels": list(FULL_DDIM_LEVELS),
            "num_ddim_levels": len(FULL_DDIM_LEVELS),
            "start_seed": START_SEED,
            "ddpm_noise_seeds": [DDPM_NOISE_SEED_1, DDPM_NOISE_SEED_2],
            "target_seed": "split(PRNGKey(3), 2)[1]",
            "predictor_eval_seed": PREDICTOR_EVAL_SEED,
            "trajectory_sample_index": TRAJECTORY_SAMPLE_INDEX,
        },
        "ddim": {
            "max_abs_difference_between_runs": ddim_run_difference,
            "metrics": ddim_metrics,
            "fraction_pass": ddim_fraction_pass,
            "width_pass": ddim_width_pass,
        },
        "ddpm": {
            "noise_draw_1": ddpm_metrics_1,
            "noise_draw_2": ddpm_metrics_2,
            "width_pass_both": ddpm_width_pass,
        },
        "exact_predictor": {
            "metrics": exact_metrics,
            "learned_mse": learned_mse,
            "exact_mse": exact_mse,
            "mean_abs_learned_minus_exact": predictor_difference,
        },
        "verification": {
            "ddim_fractions_within_0_015": ddim_fraction_pass,
            "ddim_widths_within_0_29_0_32": ddim_width_pass,
            "ddpm_widths_within_0_29_0_32": ddpm_width_pass,
            "overall_pass": overall_pass,
        },
        "evidence": {
            "trajectory_comparison": "evidence/part2_trajectory_comparison.png",
            "sample_comparison": "evidence/part2_samples_comparison.png",
            "exact_predictor_comparison": (
                "evidence/part2_exact_predictor_comparison.png"
            ),
            "verify_log": "evidence/part2_verify1.txt",
        },
        "artifacts": {
            "results": "artifacts/part2_results.json",
            "samples": "artifacts/part2_samples.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "part2_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    ddim_labels = np.asarray(assign_to_nearest_center(ddim_samples)[0])
    ddpm_labels = np.asarray(assign_to_nearest_center(ddpm_samples_1)[0])
    np.savez_compressed(
        ARTIFACTS_DIR / "part2_samples.npz",
        initial_x_T=np.asarray(x_T),
        ddim_samples=ddim_samples,
        ddpm_samples_run1=ddpm_samples_1,
        ddpm_samples_run2=ddpm_samples_2,
        exact_ddim_samples=exact_samples,
        target_samples=np.asarray(target_samples),
        target_components=np.asarray(target_components),
        ddim_labels=ddim_labels,
        ddpm_labels=ddpm_labels,
        ddim_trajectory_selected=np.asarray(ddim_trajectory_1[:, saved_indices]),
        ddpm_run1_trajectory_selected=np.asarray(
            ddpm_trajectory_1[:, saved_indices]
        ),
        ddpm_run2_trajectory_selected=np.asarray(
            ddpm_trajectory_2[:, saved_indices]
        ),
        exact_trajectory_selected=np.asarray(
            exact_trajectory[:, saved_indices]
        ),
    )

    print(f"trajectory figure: {trajectory_path.relative_to(WEEK_DIR)}")
    print(f"sample comparison: {comparison_path.relative_to(WEEK_DIR)}")
    print(f"exact predictor comparison: {exact_path.relative_to(WEEK_DIR)}")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
