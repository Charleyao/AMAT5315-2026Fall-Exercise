"""Generate VERIFY (3) samples and evidence from the trained checkpoint.

Run from week7/ with:

    uv run python run_verify3.py

This script does not train. It loads the saved update-25000 checkpoint, runs
1000 DDPM reverse steps for 2000 samples, computes cluster metrics, and writes
figures/animations to evidence/ and raw arrays/results to artifacts/.
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
from matplotlib.animation import FFMpegWriter

from diffusion_model import (
    CONFIG_PATH,
    load_checkpoint,
    load_config,
    make_reverse_sampler,
)
from forward_diffusion import (
    alpha_bars_from_betas,
    linear_beta_schedule,
    sample_gaussian_mixture,
)
from verify3 import (
    NUM_SAMPLES,
    assign_to_nearest_center,
    cluster_metrics,
    format_verification_table,
)
from verify3_plot import (
    create_reverse_diffusion_animation,
    render_final_comparison,
    render_reverse_keyframes,
)

VERIFY3_SEED = 5315
CHECKPOINT_UPDATE = 25_000
TRAJECTORY_FRAME_STEP = 10
KEYFRAME_STEPS = [0, 200, 400, 600, 800, 1000]
ANIMATION_FPS = 12

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

    print("Week 7 Part 1 VERIFY (3) -- trained-model sampling")
    print(f"device: {jax.devices()[0]}")
    print(f"checkpoint: {CHECKPOINT_PATH.relative_to(WEEK_DIR)}")
    print(f"num_samples: {NUM_SAMPLES}")
    print(f"reverse_steps: {len(betas)}")
    print(f"seed: {VERIFY3_SEED}")

    key = jax.random.PRNGKey(VERIFY3_SEED)
    key_initial, key_reverse_noise, key_target = jax.random.split(key, 3)
    initial_x_T = jax.random.normal(
        key_initial, (NUM_SAMPLES, 2), dtype=jnp.float32
    )
    reverse_noise = jax.random.normal(
        key_reverse_noise,
        (1000, NUM_SAMPLES, 2),
        dtype=jnp.float32,
    )
    target_samples, target_components = sample_gaussian_mixture(
        key_target, NUM_SAMPLES
    )

    print("running 1000 DDPM reverse steps for 2000 samples ...", flush=True)
    reverse_sampler = make_reverse_sampler(config, betas, alpha_bars)
    trajectory = reverse_sampler(params, initial_x_T, reverse_noise)
    trajectory_np = np.asarray(trajectory)
    generated_samples = trajectory_np[-1]
    labels, distances = assign_to_nearest_center(generated_samples)
    labels = np.asarray(labels)
    distances = np.asarray(distances)

    if not np.isfinite(trajectory_np).all():
        raise RuntimeError("reverse trajectory contains non-finite values")

    metrics = cluster_metrics(generated_samples, labels=labels)

    print()
    print(format_verification_table(metrics))
    print()

    frame_steps = list(
        range(0, 1000 + 1, TRAJECTORY_FRAME_STEP)
    )
    animation, figure, _ = create_reverse_diffusion_animation(
        trajectory_np,
        frame_steps,
        labels=labels,
        interval_ms=1000 // ANIMATION_FPS,
    )
    animation_path = EVIDENCE_DIR / "reverse_diffusion.mp4"
    animation.save(
        animation_path,
        writer=FFMpegWriter(fps=ANIMATION_FPS, bitrate=2400),
        dpi=120,
    )
    plt.close(figure)

    render_reverse_keyframes(
        trajectory_np,
        KEYFRAME_STEPS,
        str(EVIDENCE_DIR / "reverse_diffusion_keyframes.png"),
    )
    render_final_comparison(
        generated_samples,
        np.asarray(target_samples),
        labels,
        metrics,
        str(EVIDENCE_DIR / "verify3_generated_vs_target.png"),
    )

    assigned_counts = {
        str(cluster): int(np.sum(labels == cluster))
        for cluster in range(4)
    }
    results = {
        "part": "Week 7 Part 1 VERIFY (3)",
        "device": str(jax.devices()[0]),
        "config_path": str(CONFIG_PATH.relative_to(WEEK_DIR)),
        "config": asdict(config),
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(WEEK_DIR)),
            "update": CHECKPOINT_UPDATE,
            "retrained": False,
        },
        "sampling": {
            "seed": VERIFY3_SEED,
            "num_samples": NUM_SAMPLES,
            "reverse_steps": int(len(betas)),
            "trajectory_shape": list(trajectory_np.shape),
            "trajectory_finite": bool(np.isfinite(trajectory_np).all()),
            "assigned_counts": assigned_counts,
            "assignment_distances": {
                "min": float(distances.min()),
                "mean": float(distances.mean()),
                "max": float(distances.max()),
            },
        },
        "metrics": metrics,
        "target_samples": {
            "num_samples": int(np.asarray(target_samples).shape[0]),
            "component_counts": np.bincount(
                np.asarray(target_components), minlength=4
            ).tolist(),
        },
        "evidence": {
            "reverse_animation": "evidence/reverse_diffusion.mp4",
            "reverse_keyframes": "evidence/reverse_diffusion_keyframes.png",
            "generated_vs_target": "evidence/verify3_generated_vs_target.png",
            "verify_log": "evidence/verify3.txt",
        },
        "artifacts": {
            "results": "artifacts/verify3_results.json",
            "samples": "artifacts/verify3_samples.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "verify3_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        ARTIFACTS_DIR / "verify3_samples.npz",
        generated_samples=np.asarray(generated_samples),
        target_samples=np.asarray(target_samples),
        target_components=np.asarray(target_components),
        assigned_labels=labels,
        assignment_distances=distances,
        initial_x_T=np.asarray(initial_x_T),
        reverse_trajectory=trajectory_np,
    )

    print(f"VERIFY (3): {'PASS' if metrics['passed'] else 'FAIL'}")
    print(f"animation: {animation_path.relative_to(WEEK_DIR)}")
    print("keyframes: evidence/reverse_diffusion_keyframes.png")
    print("generated vs target: evidence/verify3_generated_vs_target.png")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
