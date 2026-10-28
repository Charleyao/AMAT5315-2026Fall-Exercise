"""Generate the Week 7 Part 1 forward-diffusion evidence.

Run from week7/ with:

    uv run python run_forward_diffusion.py

The script uses CPU JAX, writes a reproducible numerical summary to
artifacts/, and writes the animation, keyframe figure, and VERIFY log to
evidence/. It does not train a network or perform reverse sampling.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from forward_diffusion import (
    BETA_END,
    BETA_START,
    COMPONENT_COVARIANCE,
    COMPONENT_STD,
    CENTERS,
    NUM_COMPONENTS,
    NUM_DIFFUSION_STEPS,
    alpha_bars_from_betas,
    coordinate_stats,
    forward_trajectory,
    linear_beta_schedule,
    q_sample,
    sample_gaussian_mixture,
    verify_terminal_distribution,
)
from forward_diffusion_plot import create_forward_animation, render_keyframes

NUM_SAMPLES = 2000
SEED = 5315
FRAME_STEP = 10
KEYFRAME_STEPS = [0, 200, 400, 600, 800, 1000]
ANIMATION_FPS = 12

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"


def _stats_at(frames: jax.Array, frame_index: int) -> dict[str, float]:
    means, variances = coordinate_stats(frames[frame_index])
    return {
        "mean_x": float(means[0]),
        "mean_y": float(means[1]),
        "variance_x": float(variances[0]),
        "variance_y": float(variances[1]),
    }


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(SEED)
    key_x0, key_noise = jax.random.split(key)

    x0, components = sample_gaussian_mixture(key_x0, NUM_SAMPLES)
    noise = jax.random.normal(key_noise, x0.shape, dtype=jnp.float32)

    betas = linear_beta_schedule()
    alpha_bars = alpha_bars_from_betas(betas)

    frame_steps = list(range(0, NUM_DIFFUSION_STEPS + 1, FRAME_STEP))
    if frame_steps[-1] != NUM_DIFFUSION_STEPS:
        frame_steps.append(NUM_DIFFUSION_STEPS)

    frames = forward_trajectory(
        x0, noise, alpha_bars, jnp.asarray(frame_steps)
    )
    x_t1000 = q_sample(x0, alpha_bars[NUM_DIFFUSION_STEPS], noise)
    terminal_result = verify_terminal_distribution(x_t1000)

    keyframe_indices = [frame_steps.index(step) for step in KEYFRAME_STEPS]
    keyframe_frames = frames[jnp.asarray(keyframe_indices)]
    keyframe_stats = {
        str(step): _stats_at(frames, frame_steps.index(step))
        for step in KEYFRAME_STEPS
    }

    animation_path = EVIDENCE_DIR / "forward_diffusion.mp4"
    animation, figure, _ = create_forward_animation(
        frames, components, frame_steps, interval_ms=1000 // ANIMATION_FPS
    )
    animation.save(
        animation_path,
        writer=FFMpegWriter(fps=ANIMATION_FPS, bitrate=2400),
        dpi=120,
    )
    plt.close(figure)

    keyframes_path = EVIDENCE_DIR / "forward_diffusion_keyframes.png"
    render_keyframes(
        keyframe_frames, components, KEYFRAME_STEPS, str(keyframes_path)
    )

    results = {
        "part": "Week 7 Part 1 forward diffusion",
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "device": str(jax.devices()[0]),
        "mixture": {
            "centers": np.asarray(CENTERS).tolist(),
            "num_components": NUM_COMPONENTS,
            "weights": [1.0 / NUM_COMPONENTS] * NUM_COMPONENTS,
            "covariance": COMPONENT_COVARIANCE,
            "component_std": COMPONENT_STD,
        },
        "schedule": {
            "num_steps": NUM_DIFFUSION_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_1": float(betas[0]),
            "beta_1000": float(betas[-1]),
            "alpha_bar_1000": float(alpha_bars[-1]),
            "sqrt_alpha_bar_1000": float(jnp.sqrt(alpha_bars[-1])),
            "sqrt_one_minus_alpha_bar_1000": float(
                jnp.sqrt(1.0 - alpha_bars[-1])
            ),
        },
        "terminal_t1000": terminal_result,
        "keyframe_stats": keyframe_stats,
        "animation": {
            "path": str(animation_path.relative_to(WEEK_DIR)),
            "num_frames": len(frame_steps),
            "frame_step": FRAME_STEP,
            "fps": ANIMATION_FPS,
        },
        "verification": {
            "means_within_0_1_of_zero": terminal_result[
                "means_within_tolerance"
            ],
            "variances_within_0_1_of_one": terminal_result[
                "variances_within_tolerance"
            ],
            "passed": terminal_result["passed"],
        },
    }

    results_path = ARTIFACTS_DIR / "forward_diffusion_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    samples_path = ARTIFACTS_DIR / "forward_diffusion_samples.npz"
    np.savez_compressed(
        samples_path,
        x0=np.asarray(x0),
        components=np.asarray(components),
        noise=np.asarray(noise),
        frame_steps=np.asarray(frame_steps),
        frames=np.asarray(frames),
        x_t1000=np.asarray(x_t1000),
    )

    print("Week 7 Part 1 VERIFY (1) -- forward diffusion")
    print(f"device: {jax.devices()[0]}")
    print(f"seed: {SEED}")
    print(f"num_samples: {NUM_SAMPLES}")
    print(f"num_steps: {NUM_DIFFUSION_STEPS}")
    print(f"beta_1: {float(betas[0]):.8f}")
    print(f"beta_1000: {float(betas[-1]):.8f}")
    print(f"alpha_bar_1000: {float(alpha_bars[-1]):.8e}")
    print("t=1000 coordinate means:")
    print(f"  mean x = {terminal_result['mean_x']:+.6f}")
    print(f"  mean y = {terminal_result['mean_y']:+.6f}")
    print("t=1000 coordinate variances:")
    print(f"  variance x = {terminal_result['variance_x']:.6f}")
    print(f"  variance y = {terminal_result['variance_y']:.6f}")
    print(
        "means within 0.1 of zero: "
        f"{'PASS' if terminal_result['means_within_tolerance'] else 'FAIL'}"
    )
    print(
        "variances within 0.1 of one: "
        f"{'PASS' if terminal_result['variances_within_tolerance'] else 'FAIL'}"
    )
    print(f"VERIFY (1): {'PASS' if terminal_result['passed'] else 'FAIL'}")
    print(f"animation: {animation_path.relative_to(WEEK_DIR)}")
    print(f"keyframes: {keyframes_path.relative_to(WEEK_DIR)}")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if terminal_result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
