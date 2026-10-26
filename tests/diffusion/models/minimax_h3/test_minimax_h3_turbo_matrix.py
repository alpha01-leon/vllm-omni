# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract coverage for the LightX2V MiniMax-H3 Turbo artifact matrix."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.minimax_h3.lora import (
    _TURBO_AUDIO_SHIFT,
    _TURBO_VIDEO_SHIFT_544P,
    _TURBO_VIDEO_SHIFT_768P,
    TurboSpec,
    parse_turbo_filename,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# The eight Diffusers-layout artifacts published at lightx2v/Minimax-h3-Turbo,
# with the contract each one implies. The ComfyUI export of an artifact is a
# fused-QKV re-packing of the same weights and is not served.
PUBLISHED = [
    ("minimax_h3_fl2v_turbo_4step_v0.1.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors", "fl2v", 8, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors", "fl2v", 8, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors", "ref2v", 4, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors", "ref2v", 8, _TURBO_VIDEO_SHIFT_768P),
]


@pytest.mark.parametrize(("filename", "task", "steps", "video_shift"), PUBLISHED)
def test_every_published_artifact_is_recognised(filename: str, task: str, steps: int, video_shift: float) -> None:
    spec = parse_turbo_filename(filename)
    assert spec is not None, filename
    assert spec.filename == filename
    assert spec.task_family == task
    assert spec.denoise_steps == steps
    assert spec.video_shift == video_shift
    assert spec.audio_shift == _TURBO_AUDIO_SHIFT
    # Until the loader reads metadata, alpha is LightX2V's reference default.
    assert spec.rank == 128
    assert spec.alpha == 8.0


def test_all_eight_artifacts_are_covered() -> None:
    assert len(PUBLISHED) == 8
    assert len({name for name, *_ in PUBLISHED}) == 8


@pytest.mark.parametrize(
    "filename",
    [
        "adapter_model.safetensors",
        "minimax_h3_fl2v_turbo_v1.0_768p_bf16.safetensors",
        "minimax_h3_t2v_turbo_4step_v1.0.safetensors",
        "minimax_h3_fl2v_turbo_0step_v1.0.safetensors",
        # The ComfyUI exports are a different tensor layout, not a variant of
        # the supported one; the loader refuses them by name.
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    ],
)
def test_unrelated_names_are_rejected(filename: str) -> None:
    assert parse_turbo_filename(filename) is None


def _spec(filename: str) -> TurboSpec:
    spec = parse_turbo_filename(filename)
    assert spec is not None
    return spec


def test_sigma_points_follow_the_step_count() -> None:
    assert _spec("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors").sigma_points == 5
    assert _spec("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors").sigma_points == 9


@pytest.mark.parametrize(("filename", "task", "steps", "video_shift"), PUBLISHED)
def test_turbo_request_counts_evaluations_and_preserves_sigma_grid(filename, task, steps, video_shift):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas
    from vllm_omni.errors import OmniClientError

    spec = _spec(filename)
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.default_video_shift = video_shift
    pipeline.default_audio_shift = spec.audio_shift
    sampling = SimpleNamespace(num_inference_steps=steps, extra_args={})
    pipeline._validate_turbo_sampling(sampling, spec)
    for shift in (video_shift, spec.audio_shift):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=shift)
        # The old point-count API requested steps + 1 points on this same grid.
        base = [1.0 - i / steps for i in range(steps + 1)]
        expected = [shift * value / (1 + (shift - 1) * value) for value in base]
        assert sigmas == pytest.approx(expected, abs=1e-7)
    sampling.num_inference_steps = spec.sigma_points
    with pytest.raises(OmniClientError, match=f"num_inference_steps={steps}"):
        pipeline._validate_turbo_sampling(sampling, spec)


def test_task_family_decides_supported_tasks() -> None:
    fl2v = _spec("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")
    ref2v = _spec("minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors")
    assert fl2v.supported_tasks == frozenset({"t2va", "fl2va"})
    assert ref2v.supported_tasks == frozenset({"ref2va"})


# Sigmas ComfyUI's own KSampler printed for the FL2VA checkpoint
# (comfy.supported_models.MiniMaxH3.sampling_settings: shift 12.0, audio_shift
# 3.0) at eight euler steps. They come from ComfyUI's "simple" scheduler, which
# resamples ``ModelSamplingDiscrete.sigmas`` -- an array whose top is already
# 14.6146, not 1.0 -- so it agrees with our grid only because the H3 shift
# re-derives the base positions from the shifted sigma. If the shift formula or
# its bounds ever change, this diverges; that is the point of pinning it.
COMFYUI_FLV2A_EIGHT_STEP_SIGMAS = [
    1.0,
    0.9882,
    0.9730,
    0.9524,
    0.9231,
    0.8780,
    0.8000,
    0.6316,
    0.0,
]


@pytest.mark.parametrize("steps", [4, 8, 50])
def test_video_schedule_reproduces_comfyui_ksampler_output(steps: int) -> None:
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    sigmas = minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=12.0)
    assert len(sigmas) == steps + 1
    assert sigmas[0] == pytest.approx(1.0, abs=1e-7)
    assert sigmas[-1] == pytest.approx(0.0, abs=1e-7)
    assert sigmas == sorted(sigmas, reverse=True)
    if steps == 8:
        assert sigmas == pytest.approx(COMFYUI_FLV2A_EIGHT_STEP_SIGMAS, abs=5e-4)


def test_audio_schedule_is_the_same_grid_at_a_lower_shift() -> None:
    """H3 carries audio as ``(sigma_video / sigma_audio) * x`` at shift 3.0."""
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    video = minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=12.0)
    audio = minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=3.0)
    # Both traverse the same endpoints at different curvature. A larger shift
    # keeps the grid nearer sigma=1 for longer, so video stays strictly above
    # audio at every interior point.
    assert (video[0], video[-1]) == (audio[0], audio[-1]) == (1.0, 0.0)
    assert all(v > a for v, a in zip(video[1:-1], audio[1:-1]))


def test_explicit_base_schedule_takes_precedence_over_the_step_count() -> None:
    """Distilled checkpoints pin a non-uniform few-step grid."""
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    pinned = minimax_h3_time_shift_sigmas(num_steps=50, shift_scale=12.0, base_schedule=[1.0, 0.5, 0.0])
    assert pinned == pytest.approx([1.0, 12 * 0.5 / (1 + 11 * 0.5), 0.0], abs=1e-7)
