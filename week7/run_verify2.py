"""Train the Part 1 noise predictor and generate VERIFY (2) evidence.

Run from week7/ with:

    uv run python run_verify2.py

Training uses the values in diffusion.toml. The script saves parameter
checkpoints and raw arrays under artifacts/ (ignored by Git) and figures,
animations, and a numerical log under evidence/ (tracked by Git).
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
    load_config,
    make_reverse_sampler,
    save_checkpoint,
    train_checkpointed,
)
from diffusion_verify2_plot import (
    create_endpoint_evolution_animation,
    plot_training_loss,
    render_checkpoint_grid,
)
from forward_diffusion import alpha_bars_from_betas, linear_beta_schedule

NUM_TRAJECTORIES = 10
VERIFY_SEED_OFFSET = 99991
ANIMATION_FPS = 1

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def _endpoint_stats(endpoints: np.ndarray) -> dict[str, float]:
    means = endpoints.mean(axis=0)
    variances = endpoints.var(axis=0)
    return {
        "mean_x": float(means[0]),
        "mean_y": float(means[1]),
        "variance_x": float(variances[0]),
        "variance_y": float(variances[1]),
    }


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    betas = linear_beta_schedule()
    alpha_bars = alpha_bars_from_betas(betas)
    checkpoints = list(config.checkpoints)

    print("Week 7 Part 1 VERIFY (2) -- training + DDPM reverse sampling")
    print(f"device: {jax.devices()[0]}")
    print(f"config: {CONFIG_PATH.name}")
    print(f"updates: {config.updates}")
    print(f"checkpoints: {checkpoints}")
    print("training ...", flush=True)

    key = jax.random.PRNGKey(config.seed)
    params_by_checkpoint, losses = train_checkpointed(
        config,
        alpha_bars,
        key,
        chunk_updates=50,
        verbose=True,
    )

    for update, params in params_by_checkpoint.items():
        save_checkpoint(
            params, CHECKPOINT_DIR / f"params-{update:06d}.npz"
        )

    verify_key = jax.random.fold_in(
        jax.random.PRNGKey(config.seed), VERIFY_SEED_OFFSET
    )
    key_initial, key_reverse_noise = jax.random.split(verify_key)
    x_T = jax.random.normal(
        key_initial, (NUM_TRAJECTORIES, 2), dtype=jnp.float32
    )
    reverse_noise = jax.random.normal(
        key_reverse_noise,
        (1000, NUM_TRAJECTORIES, 2),
        dtype=jnp.float32,
    )

    print("generating 10 DDPM trajectories at every checkpoint ...", flush=True)
    reverse_sampler = make_reverse_sampler(config, betas, alpha_bars)
    trajectories = []
    for update in checkpoints:
        trajectory = reverse_sampler(
            params_by_checkpoint[update], x_T, reverse_noise
        )
        trajectories.append(np.asarray(trajectory))
    trajectories_array = np.stack(trajectories, axis=0)
    endpoints = trajectories_array[:, -1, :, :]

    all_finite = bool(np.isfinite(trajectories_array).all())
    if not all_finite:
        raise RuntimeError("DDPM trajectories contain non-finite values")

    losses_np = np.asarray(losses)
    checkpoint_losses: dict[str, dict[str, float | None]] = {}
    previous_checkpoint = 0
    for update in checkpoints:
        if update == 0:
            checkpoint_losses[str(update)] = {
                "mean_since_previous_checkpoint": None,
                "last_update_loss": None,
            }
        else:
            segment = losses_np[previous_checkpoint:update]
            checkpoint_losses[str(update)] = {
                "mean_since_previous_checkpoint": float(segment.mean()),
                "last_update_loss": float(losses_np[update - 1]),
            }
        previous_checkpoint = update

    final_loss = float(losses_np[-1])
    final_mean_loss = float(losses_np[-50:].mean())

    plot_training_loss(
        losses_np, checkpoints, str(EVIDENCE_DIR / "training_loss.png")
    )
    render_checkpoint_grid(
        checkpoints,
        trajectories_array,
        str(EVIDENCE_DIR / "verify2_checkpoints.png"),
    )

    animation, figure, _ = create_endpoint_evolution_animation(
        checkpoints,
        endpoints,
        losses_np,
        interval_ms=1000 // ANIMATION_FPS,
    )
    animation_path = EVIDENCE_DIR / "verify2_endpoint_evolution.mp4"
    animation.save(
        animation_path,
        writer=FFMpegWriter(fps=ANIMATION_FPS, bitrate=2400),
        dpi=130,
    )
    plt.close(figure)

    endpoint_statistics = {
        str(update): _endpoint_stats(endpoints[index])
        for index, update in enumerate(checkpoints)
    }
    results = {
        "part": "Week 7 Part 1 VERIFY (2)",
        "algorithm": "DDPM Algorithm 2, variance beta_t, no clipping, no final noise",
        "device": str(jax.devices()[0]),
        "config": asdict(config),
        "config_path": str(CONFIG_PATH.relative_to(WEEK_DIR)),
        "schedule": {
            "num_steps": int(len(betas)),
            "beta_1": float(betas[0]),
            "beta_1000": float(betas[-1]),
            "alpha_bar_1000": float(alpha_bars[-1]),
        },
        "training": {
            "losses_saved": int(len(losses_np)),
            "final_update_loss": final_loss,
            "final_50_update_mean_loss": final_mean_loss,
            "minimum_loss": float(losses_np.min()),
            "checkpoint_losses": checkpoint_losses,
        },
        "verify2": {
            "num_trajectories_per_checkpoint": NUM_TRAJECTORIES,
            "same_initial_gaussian_samples": True,
            "same_reverse_step_noise": True,
            "trajectory_shape": list(trajectories_array.shape),
            "trajectories_finite": all_finite,
            "checkpoints": checkpoints,
            "endpoint_statistics": endpoint_statistics,
        },
        "checkpoint_files": [
            str(
                (CHECKPOINT_DIR / f"params-{update:06d}.npz").relative_to(
                    WEEK_DIR
                )
            )
            for update in checkpoints
        ],
        "evidence": {
            "animation": str(animation_path.relative_to(WEEK_DIR)),
            "checkpoint_grid": "evidence/verify2_checkpoints.png",
            "training_loss": "evidence/training_loss.png",
            "verify_log": "evidence/verify2.txt",
        },
    }

    results_path = ARTIFACTS_DIR / "verify2_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        ARTIFACTS_DIR / "verify2_losses.npz",
        losses=losses_np,
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "verify2_trajectories.npz",
        trajectories=trajectories_array,
        endpoints=endpoints,
        initial_x_T=np.asarray(x_T),
        reverse_noise=np.asarray(reverse_noise),
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
    )

    print("training diagnostics:")
    for update in checkpoints:
        checkpoint_loss = checkpoint_losses[str(update)]
        if checkpoint_loss["mean_since_previous_checkpoint"] is None:
            print(f"  update={update:6d}  mean_loss=      n/a")
        else:
            print(
                f"  update={update:6d}  "
                "mean_loss="
                f"{checkpoint_loss['mean_since_previous_checkpoint']:.6f}  "
                f"last={checkpoint_loss['last_update_loss']:.6f}"
            )
    print(f"final training loss (last update): {final_loss:.6f}")
    print(f"final 50-update mean loss: {final_mean_loss:.6f}")
    print(f"trajectory array shape: {trajectories_array.shape}")
    print(f"trajectories finite: {all_finite}")
    print("VERIFY (2): COMPLETE")
    print(f"animation: {animation_path.relative_to(WEEK_DIR)}")
    print("checkpoints: evidence/verify2_checkpoints.png")
    print("training loss: evidence/training_loss.png")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
