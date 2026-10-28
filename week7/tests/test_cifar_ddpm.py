"""Tests for the JAX CIFAR-10 DDPM U-Net and sampler."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cifar_ddpm as cd

WEEK_DIR = Path(__file__).resolve().parents[1]
CONFIG = cd.load_unet_config(WEEK_DIR / "inputs" / "cifar-model" / "config.json")
WEIGHTS_AVAILABLE = (
    WEEK_DIR / "inputs" / "cifar-model" / "diffusion_pytorch_model.safetensors"
).is_file()


def test_pinned_unet_config_matches_google_ddpm_cifar10_32():
    assert CONFIG.sample_size == 32
    assert CONFIG.in_channels == 3
    assert CONFIG.out_channels == 3
    assert CONFIG.block_out_channels == (128, 256, 256, 256)
    assert CONFIG.down_block_types == (
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    )
    assert CONFIG.up_block_types == (
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    )
    assert CONFIG.layers_per_block == 2
    assert CONFIG.norm_num_groups == 32
    assert CONFIG.norm_eps == pytest.approx(1e-6)
    assert CONFIG.attention_head_dim is None
    assert CONFIG.flip_sin_to_cos is False
    assert CONFIG.freq_shift == 1.0
    assert CONFIG.downsample_padding == 0


def test_expected_parameter_shapes_cover_the_diffusers_checkpoint():
    shapes = cd.unet_param_shapes(CONFIG)

    assert len(shapes) == 328
    assert shapes["conv_in.weight"] == (128, 3, 3, 3)
    assert shapes["time_embedding.linear_1.weight"] == (512, 128)
    assert shapes["down_blocks.1.attentions.0.query.weight"] == (256, 256)
    assert shapes["mid_block.resnets.1.conv2.weight"] == (256, 256, 3, 3)
    assert shapes["up_blocks.2.resnets.2.conv1.weight"] == (256, 384, 3, 3)
    assert shapes["conv_out.weight"] == (3, 128, 3, 3)


@pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="pretrained weights not downloaded")
def test_safetensors_match_expected_parameter_shapes():
    params = cd.load_unet_params()

    cd.assert_parameter_shapes(params, CONFIG)
    assert len(params) == 328
    assert params["conv_in.weight"].shape == (128, 3, 3, 3)
    assert bool(jnp.all(jnp.isfinite(params["conv_out.bias"])))


@pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="pretrained weights not downloaded")
def test_unet_forward_returns_finite_32x32_images():
    params = cd.load_unet_params()
    sample = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 3))

    output = jax.jit(lambda x, t: cd.unet_apply(params, x, t, CONFIG))(
        sample, 500
    )

    assert output.shape == (1, 32, 32, 3)
    assert bool(jnp.all(jnp.isfinite(output)))


@pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="pretrained weights not downloaded")
def test_two_step_sampler_returns_a_trajectory_without_shape_errors():
    params = cd.load_unet_params()
    sampler, betas, alpha_bars = cd.make_ddpm_sampler(
        params, CONFIG, num_samples=1, num_steps=2
    )

    trajectory = sampler(jax.random.PRNGKey(1))

    assert trajectory.shape == (3, 1, 32, 32, 3)
    assert betas.shape == (2,)
    assert alpha_bars.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(trajectory)))


def test_display_mapping_clips_only_after_mapping():
    raw = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    raw_copy = raw.copy()

    display = cd.display_images(raw)

    np.testing.assert_allclose(display, [0.0, 0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(raw, raw_copy)
