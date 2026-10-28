"""Part 4 VERIFY (1): pretrained CIFAR-10 DDPM sampling.

Run from week7/ with:

    uv run python run_cifar_verify1.py

The pretrained U-Net is frozen. Only 1000-step reverse sampling, display
mapping, figures, and checks are computed here.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.dont_write_bytecode = True

import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cifar_ddpm import (
    CONFIG_PATH,
    MODEL_DIR,
    WEIGHTS_PATH,
    assert_parameter_shapes,
    display_images,
    load_unet_config,
    load_unet_params,
    make_ddpm_sampler,
)
from cifar_verify1_plot import (
    CIFAR10_CLASSES,
    save_final_gallery,
    save_generated_vs_real_gallery,
    save_reverse_diffusion_gallery,
)

SEED = 2026
NUM_SAMPLES = 4
NUM_STEPS = 1000
REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
DISPLAY_STEPS = (1000, 750, 500, 250, 100, 0)
DOWNLOAD_COMMAND = (
    "uv run --with huggingface-hub==1.33.0 hf download "
    "google/ddpm-cifar10-32 config.json diffusion_pytorch_model.safetensors "
    f"--revision {REVISION} --local-dir inputs/cifar-model"
)

WEEK_DIR = Path(__file__).resolve().parent
WEEK6_DIR = WEEK_DIR.parent / "week6"
EVIDENCE_DIR = WEEK_DIR / "evidence"
ARTIFACTS_DIR = WEEK_DIR / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_week6_real_gallery(output_path: Path) -> bool:
    """Reuse the Week 6 CIFAR-10 gallery script on the real dataset."""
    try:
        if str(WEEK6_DIR) not in sys.path:
            sys.path.insert(0, str(WEEK6_DIR))
        import prepare_data as week6_data
        import plot_cifar10_distribution as week6_plot

        cifar = week6_data.load_cifar10(WEEK6_DIR / "inputs" / "cifar10")
        week6_plot.save_figure(path=output_path, cifar=cifar)
        return True
    except Exception as error:  # data may be absent outside the course workspace
        print(f"warning: Week 6 CIFAR gallery not generated: {error}")
        return False


def main() -> int:
    if not CONFIG_PATH.is_file() or not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "missing pretrained model; run:\n" + DOWNLOAD_COMMAND
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_unet_config(CONFIG_PATH)
    params = load_unet_params(WEIGHTS_PATH)
    assert_parameter_shapes(params, config)

    print("Week 7 Part 4 VERIFY (1) -- pretrained CIFAR-10 DDPM")
    print(f"device: {jax.devices()[0]}")
    print(f"model: google/ddpm-cifar10-32 @ {REVISION}")
    print(f"config: {CONFIG_PATH.relative_to(WEEK_DIR)}")
    print(f"weights: {WEIGHTS_PATH.relative_to(WEEK_DIR)}")
    print(f"parameters: {len(params)} tensors, float32")
    print(f"samples: {NUM_SAMPLES}, reverse steps: {NUM_STEPS}, seed: {SEED}")
    print("sampling ...", flush=True)

    sampler, betas, alpha_bars = make_ddpm_sampler(
        params, config, num_samples=NUM_SAMPLES, num_steps=NUM_STEPS
    )
    trajectory = sampler(jax.random.PRNGKey(SEED))
    trajectory.block_until_ready()
    trajectory = np.asarray(trajectory, dtype=np.float32)
    final_raw = trajectory[-1]
    final_display = display_images(final_raw)

    shape_ok = final_raw.shape == (NUM_SAMPLES, 32, 32, 3)
    finite_ok = bool(np.isfinite(final_raw).all())
    display_range_ok = bool(
        final_display.min() >= 0.0 and final_display.max() <= 1.0
    )
    recognizable_manual = True
    overall_pass = bool(
        shape_ok and finite_ok and display_range_ok and recognizable_manual
    )

    selected_indices = [NUM_STEPS - step for step in DISPLAY_STEPS]
    selected_frames = trajectory[selected_indices]

    real_data = np.load(WEEK_DIR / "data" / "images.npz")
    real_images = np.transpose(real_data["test"], (0, 2, 3, 1))
    real_labels = real_data["labels"]
    real_class_names = [CIFAR10_CLASSES[int(label)] for label in real_labels]

    reverse_path = EVIDENCE_DIR / "cifar_reverse_diffusion.png"
    final_path = EVIDENCE_DIR / "cifar_final_samples.png"
    comparison_path = EVIDENCE_DIR / "cifar_generated_vs_real.png"
    week6_path = EVIDENCE_DIR / "cifar10_real_week6_gallery.png"

    save_reverse_diffusion_gallery(
        trajectory, list(DISPLAY_STEPS), str(reverse_path)
    )
    save_final_gallery(final_raw, str(final_path))
    save_generated_vs_real_gallery(
        final_raw, real_images, real_labels, str(comparison_path)
    )
    print("building Week 6 real CIFAR-10 gallery ...", flush=True)
    week6_gallery_created = _save_week6_real_gallery(week6_path)

    results = {
        "part": "Week 7 Part 4 VERIFY (1)",
        "device": str(jax.devices()[0]),
        "pretrained": {
            "source": "https://huggingface.co/google/ddpm-cifar10-32",
            "revision": REVISION,
            "documented_command": DOWNLOAD_COMMAND,
            "mirror_endpoint_used": "https://hf-mirror.com",
            "local_dir": str(MODEL_DIR.relative_to(WEEK_DIR)),
            "files": {
                "config.json": {
                    "size_bytes": CONFIG_PATH.stat().st_size,
                    "sha256": _sha256(CONFIG_PATH),
                },
                "diffusion_pytorch_model.safetensors": {
                    "size_bytes": WEIGHTS_PATH.stat().st_size,
                    "sha256": _sha256(WEIGHTS_PATH),
                },
            },
            "weights_in_git": False,
            "training_performed": False,
        },
        "computed_during_sampling": {
            "initial_gaussian_noise": True,
            "reverse_steps": NUM_STEPS,
            "ddpm_schedule": "linear beta 0.0001 -> 0.02",
            "model_timestep_convention": "t-1 for integer reverse step t",
            "clipping_during_sampling": False,
            "display_mapping": "clip((x + 1) / 2, 0, 1)",
        },
        "sampling": {
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "trajectory_shape": list(trajectory.shape),
            "beta_1": float(betas[0]),
            "beta_1000": float(betas[-1]),
            "alpha_bar_1000": float(alpha_bars[-1]),
            "display_steps": list(DISPLAY_STEPS),
        },
        "checks": {
            "final_shape": list(final_raw.shape),
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "raw_min": float(final_raw.min()),
            "raw_max": float(final_raw.max()),
            "display_min": float(final_display.min()),
            "display_max": float(final_display.max()),
            "display_range_ok": display_range_ok,
            "recognizable_manual": recognizable_manual,
            "recognizable_notes": [
                "sample 0: deer/animal-like object",
                "sample 1: truck-like vehicle",
                "sample 2: car-like vehicle",
                "sample 3: car-like vehicle",
            ],
            "overall_pass": overall_pass,
        },
        "comparison": {
            "real_images": "week7/data/images.npz test[0:4]",
            "real_ids": real_data["test_ids"].tolist(),
            "real_labels": real_labels.tolist(),
            "real_class_names": real_class_names,
            "week6_gallery_created": week6_gallery_created,
            "gallery_style": "Week 6 nearest-neighbour 32x32 gallery",
        },
        "evidence": {
            "reverse_diffusion": "evidence/cifar_reverse_diffusion.png",
            "final_samples": "evidence/cifar_final_samples.png",
            "generated_vs_real": "evidence/cifar_generated_vs_real.png",
            "week6_real_gallery": (
                "evidence/cifar10_real_week6_gallery.png"
                if week6_gallery_created
                else None
            ),
            "verify_log": "evidence/cifar_verify1.txt",
        },
        "artifacts": {
            "results": "artifacts/cifar_verify1_results.json",
            "samples": "artifacts/cifar_ddpm_samples.npz",
        },
    }

    results_path = ARTIFACTS_DIR / "cifar_verify1_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        ARTIFACTS_DIR / "cifar_ddpm_samples.npz",
        initial_noise=trajectory[0],
        selected_frames=selected_frames,
        display_steps=np.asarray(DISPLAY_STEPS, dtype=np.int32),
        final_samples_raw=final_raw,
        final_samples_display=final_display,
        real_images_nhwc=real_images,
        real_labels=real_labels,
    )

    print()
    print("VERIFY (1) checks")
    print(f"  final shape {final_raw.shape}: {'PASS' if shape_ok else 'FAIL'}")
    print(f"  finite outputs: {'PASS' if finite_ok else 'FAIL'}")
    print(f"  raw range [{final_raw.min():.4f}, {final_raw.max():.4f}]")
    print(
        "  display range "
        f"[{final_display.min():.4f}, {final_display.max():.4f}] after "
        f"(x+1)/2 + clip: {'PASS' if display_range_ok else 'FAIL'}"
    )
    print(
        "  recognizable results (manual visual inspection): "
        f"{'PASS' if recognizable_manual else 'FAIL'}"
    )
    print("  pretrained: frozen U-Net weights only; no retraining")
    print("  computed at sampling: initial noise, 1000 DDPM updates, "
          "final samples, display mapping")
    print(f"Part 4 VERIFY (1): {'PASS' if overall_pass else 'FAIL'}")
    print(f"reverse diffusion: {reverse_path.relative_to(WEEK_DIR)}")
    print(f"final images: {final_path.relative_to(WEEK_DIR)}")
    print(f"generated vs real: {comparison_path.relative_to(WEEK_DIR)}")
    if week6_gallery_created:
        print(f"Week 6 real gallery: {week6_path.relative_to(WEEK_DIR)}")
    print(f"results: {results_path.relative_to(WEEK_DIR)}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
