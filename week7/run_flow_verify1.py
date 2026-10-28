"""Train the Part 3 Flow Matching velocity network and verify velocity fields.

Run from week7/ with:

    uv run python run_flow_verify1.py

The velocity network is trained from scratch using flow.toml. No Part 1 model
parameters are reused.
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

from flow_matching import (
    FLOW_CONFIG_PATH,
    VERIFY_TIMES,
    evaluate_velocity_fields,
    format_verification_table,
    load_flow_config,
    save_checkpoint,
    train_flow_checkpointed,
    velocity_apply,
)
from flow_plot import plot_flow_training_loss, plot_velocity_fields

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "flow_checkpoints"


def _checkpoint_losses(
    losses: np.ndarray, checkpoints: list[int]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    previous = 0
    for update in checkpoints:
        if update == 0:
            result[str(update)] = {
                "mean_since_previous_checkpoint": None,
                "last_update_loss": None,
            }
        else:
            segment = losses[previous:update]
            result[str(update)] = {
                "mean_since_previous_checkpoint": float(segment.mean()),
                "last_update_loss": float(losses[update - 1]),
            }
        previous = update
    return result


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_flow_config(FLOW_CONFIG_PATH)
    checkpoints = list(config.checkpoints)
    print("Week 7 Part 3 VERIFY (1) -- Flow Matching velocity fields")
    print(f"device: {jax.devices()[0]}")
    print(f"config: {FLOW_CONFIG_PATH.name}")
    print(f"updates: {config.updates}")
    print(f"seed: {config.seed}")
    print(f"checkpoints: {checkpoints}")
    print("training new velocity network from scratch ...", flush=True)

    params_by_checkpoint, losses = train_flow_checkpointed(
        config,
        jax.random.PRNGKey(config.seed),
        chunk_updates=50,
        verbose=True,
    )

    for update, params in params_by_checkpoint.items():
        save_checkpoint(
            params, CHECKPOINT_DIR / f"params-{update:06d}.npz"
        )

    final_params = params_by_checkpoint[config.updates]

    def predictor(points: jax.Array, s: float) -> jax.Array:
        s_batch = jnp.full((points.shape[0],), s, dtype=jnp.float32)
        return velocity_apply(final_params, points, s_batch, config)

    print("evaluating learned and exact velocity fields ...", flush=True)
    fields_result = evaluate_velocity_fields(
        predictor, times=VERIFY_TIMES
    )
    table = format_verification_table(fields_result)
    print()
    print(table)
    print()

    losses_np = np.asarray(losses)
    final_loss = float(losses_np[-1])
    final_mean_loss = float(losses_np[-50:].mean())
    checkpoint_losses = _checkpoint_losses(losses_np, checkpoints)

    plot_flow_training_loss(
        losses_np,
        checkpoints,
        str(EVIDENCE_DIR / "flow_training_loss.png"),
    )
    velocity_figure, _ = plot_velocity_fields(
        fields_result, str(EVIDENCE_DIR / "flow_velocity_fields.png")
    )
    plt.close(velocity_figure)

    results = {
        "part": "Week 7 Part 3 VERIFY (1)",
        "device": str(jax.devices()[0]),
        "config_path": str(FLOW_CONFIG_PATH.relative_to(WEEK_DIR)),
        "config": asdict(config),
        "training": {
            "trained_from_scratch": True,
            "losses_saved": int(len(losses_np)),
            "final_update_loss": final_loss,
            "final_50_update_mean_loss": final_mean_loss,
            "minimum_loss": float(losses_np.min()),
            "checkpoint_losses": checkpoint_losses,
        },
        "verify1": {
            "grid_extent": fields_result["grid_extent"],
            "grid_points_per_axis": fields_result["grid_points_per_axis"],
            "grid_components": fields_result["grid_components"],
            "times": fields_result["times"],
            "relative_rms": fields_result["relative_rms"],
            "threshold": 0.1,
            "pass": fields_result["pass"],
            "passed": fields_result["passed"],
            "max_relative_rms": max(fields_result["relative_rms"].values()),
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
            "velocity_fields": "evidence/flow_velocity_fields.png",
            "training_loss": "evidence/flow_training_loss.png",
            "verify_log": "evidence/flow_verify1.txt",
        },
        "artifacts": {
            "results": "artifacts/flow_results.json",
            "losses": "artifacts/flow_training_losses.npz",
            "velocity_fields": "artifacts/flow_velocity_fields.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "flow_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        ARTIFACTS_DIR / "flow_training_losses.npz",
        losses=losses_np,
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "flow_velocity_fields.npz",
        grids=np.stack(
            [
                fields_result["grids"][str(time)]
                for time in fields_result["times"]
            ]
        ),
        times=np.asarray(fields_result["times"], dtype=np.float32),
        learned_fields=np.stack(
            [
                fields_result["learned_fields"][str(time)]
                for time in fields_result["times"]
            ]
        ),
        exact_fields=np.stack(
            [
                fields_result["exact_fields"][str(time)]
                for time in fields_result["times"]
            ]
        ),
        relative_rms=np.asarray(
            [
                fields_result["relative_rms"][str(time)]
                for time in fields_result["times"]
            ],
            dtype=np.float32,
        ),
    )

    print(f"final training loss (last update): {final_loss:.6f}")
    print(f"final 50-update mean loss: {final_mean_loss:.6f}")
    print(
        "velocity field grid: "
        f"{fields_result['grid_components']} components x "
        f"{fields_result['grid_points_per_axis']}x"
        f"{fields_result['grid_points_per_axis']} local points, "
        f"extent +/-{fields_result['grid_extent']} marginal std"
    )
    print(f"Part 3 VERIFY (1): "
          f"{'PASS' if fields_result['passed'] else 'FAIL'}")
    print("velocity fields: evidence/flow_velocity_fields.png")
    print("training loss: evidence/flow_training_loss.png")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if fields_result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
