"""Part 5 VERIFY (1): RePaint completions with the frozen CIFAR-10 DDPM.

Run from week7/ with:

    uv run python run_repaint_verify1.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.dont_write_bytecode = True

import json
from pathlib import Path

import jax
import numpy as np

from cifar_ddpm import (
    CONFIG_PATH,
    WEIGHTS_PATH,
    assert_parameter_shapes,
    load_unet_config,
    load_unet_params,
)
from repaint import (
    IMAGES_PATH,
    REPAINT_LEVELS,
    SEEDS,
    load_test_images,
    make_repaint_mask,
    repaint_batch,
    repaint_metrics,
    repaint_network_evaluations,
)
from repaint_plot import save_repaint_figure

WEEK_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"


def main() -> int:
    if not IMAGES_PATH.is_file():
        raise FileNotFoundError(f"missing test images at {IMAGES_PATH}")
    if not CONFIG_PATH.is_file() or not WEIGHTS_PATH.is_file():
        raise FileNotFoundError("missing pretrained CIFAR-10 DDPM files")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_unet_config(CONFIG_PATH)
    params = load_unet_params(WEIGHTS_PATH)
    assert_parameter_shapes(params, config)
    images, labels = load_test_images(IMAGES_PATH)
    mask = make_repaint_mask()
    seeds = tuple(SEEDS) * images.shape[0]
    observations = np.repeat(images, len(SEEDS), axis=0)

    print("Week 7 Part 5 VERIFY (1) -- RePaint")
    print(f"device: {jax.devices()[0]}")
    print(f"images: {images.shape}, labels: {labels.tolist()}")
    print(f"revision: 267b167dc01f0e4e61923ea244e8b988f84deb80")
    print(f"mask: rows/cols 6:26 hidden in all RGB channels")
    print(f"levels: {REPAINT_LEVELS[0]} -> ... -> {REPAINT_LEVELS[-2]} -> 0")
    print(f"jumps: {len(REPAINT_LEVELS) - 1}, expected network evaluations: "
          f"{repaint_network_evaluations()}")
    print(f"seeds per image: {SEEDS}")
    print("running batched RePaint ...", flush=True)

    completions_flat, run_metadata = repaint_batch(
        params, config, observations, seeds
    )
    completions = completions_flat.reshape(
        images.shape[0], len(SEEDS), 32, 32, 3
    )
    metrics = repaint_metrics(completions, images, mask)

    shape_ok = completions.shape == (4, 4, 32, 32, 3)
    finite_ok = bool(np.isfinite(completions).all())
    evaluations_ok = (
        int(run_metadata["network_evaluations"])
        == repaint_network_evaluations()
        == 99
    )
    observed_ok = bool(metrics["observed_max_change"] < 1e-5)
    differ_ok = bool(
        metrics["max_hidden_std"] > 1e-4
        and metrics["max_pairwise_hidden_difference"] > 1e-4
    )
    clipping_ok = bool(metrics["completions_in_range"])
    overall_pass = bool(
        shape_ok
        and finite_ok
        and evaluations_ok
        and observed_ok
        and differ_ok
        and clipping_ok
    )

    figure_path = EVIDENCE_DIR / "cifar_repaint_completions.png"
    save_repaint_figure(
        images,
        completions,
        metrics["variance"],
        labels,
        mask,
        SEEDS,
        str(figure_path),
    )

    results = {
        "part": "Week 7 Part 5 VERIFY (1)",
        "device": str(jax.devices()[0]),
        "pretrained": {
            "source": "google/ddpm-cifar10-32",
            "revision": "267b167dc01f0e4e61923ea244e8b988f84deb80",
            "weights": str(WEIGHTS_PATH.relative_to(WEEK_DIR)),
            "retrained": False,
        },
        "data": {
            "path": str(IMAGES_PATH.relative_to(WEEK_DIR)),
            "num_images": int(images.shape[0]),
            "labels": labels.tolist(),
        },
        "algorithm": {
            "name": "RePaint Algorithm 6",
            "hidden_region": "rows 6:26, columns 6:26, all RGB",
            "levels": list(REPAINT_LEVELS),
            "jumps": len(REPAINT_LEVELS) - 1,
            "attempts_per_jump": "2 except final jump (1)",
            "network_evaluations": run_metadata["network_evaluations"],
            "expected_network_evaluations": run_metadata[
                "expected_network_evaluations"
            ],
            "forward_noise_between_attempts": True,
            "known_pixel_replacement": "forward-noised observations",
            "reverse_step": "DDPM posterior jump with clipped predicted x0",
            "model_timestep": "t-1",
            "clean_image_clip": "[-1, 1] only on each predicted clean image x0",
            "seeds_per_image": list(SEEDS),
        },
        "checks": {
            "completion_shape": list(completions.shape),
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "network_evaluations_ok": evaluations_ok,
            "hidden_mse_per_completion": metrics["hidden_mse"],
            "observed_max_change": metrics["observed_max_change"],
            "observed_max_change_per_completion": metrics[
                "observed_max_change_per_completion"
            ],
            "observed_change_below_1e_5": observed_ok,
            "max_hidden_std": metrics["max_hidden_std"],
            "per_image_max_hidden_std": metrics["per_image_max_hidden_std"],
            "max_pairwise_hidden_difference": metrics[
                "max_pairwise_hidden_difference"
            ],
            "completions_differ": differ_ok,
            "completions_in_range_minus1_plus1": clipping_ok,
            "overall_pass": overall_pass,
        },
        "evidence": {
            "completions_figure": "evidence/cifar_repaint_completions.png",
            "verify_log": "evidence/cifar_repaint.txt",
        },
        "artifacts": {
            "results": "artifacts/cifar_repaint_results.json",
            "samples": "artifacts/cifar_repaint_samples.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "cifar_repaint_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        ARTIFACTS_DIR / "cifar_repaint_samples.npz",
        originals=images,
        labels=labels,
        mask=mask,
        completions=completions,
        variance=metrics["variance"],
        hidden_mse=np.asarray(metrics["hidden_mse"]),
        observed_max_change=np.asarray(
            metrics["observed_max_change_per_completion"]
        ),
        levels=np.asarray(REPAINT_LEVELS, dtype=np.int32),
        seeds=np.asarray(SEEDS, dtype=np.int32),
    )

    print()
    print("Hidden-pixel MSE per completion:")
    print("image | seed 7201 | seed 7202 | seed 7203 | seed 7204")
    for image_index in range(completions.shape[0]):
        values = metrics["hidden_mse"][image_index]
        print(
            f"{image_index:5d} | "
            + " | ".join(f"{value:.6f}" for value in values)
        )
    print()
    print("VERIFY (1) checks")
    print(f"  completion shape {completions.shape}: "
          f"{'PASS' if shape_ok else 'FAIL'}")
    print(f"  finite completions: {'PASS' if finite_ok else 'FAIL'}")
    print(f"  network evaluations = {run_metadata['network_evaluations']} "
          f"(expected 99): {'PASS' if evaluations_ok else 'FAIL'}")
    print(
        f"  max observed-pixel change = {metrics['observed_max_change']:.3e} "
        f"(< 1e-5): {'PASS' if observed_ok else 'FAIL'}"
    )
    print(
        f"  max hidden-pixel std = {metrics['max_hidden_std']:.6f}; "
        f"max pairwise hidden difference = "
        f"{metrics['max_pairwise_hidden_difference']:.6f}: "
        f"{'PASS' if differ_ok else 'FAIL'}"
    )
    print(
        f"  completions within [-1,1]: "
        f"{'PASS' if clipping_ok else 'FAIL'}"
    )
    print(f"Part 5 VERIFY (1): {'PASS' if overall_pass else 'FAIL'}")
    print(f"figure: {figure_path.relative_to(WEEK_DIR)}")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
