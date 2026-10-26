# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard the transcribed row-layout helpers in ``omni_multistep.py``.

``omni_multistep.py`` carries its own copies of ComfyUI's patchify/unpatchify
and pack/unpack_audio so the parity run needs only one interpreter. That is a
copy, and a copy can drift from its source. A silent drift would not fail the
parity test -- it would fail it *spuriously*, by reshaping rows into the wrong
order, and the resulting error looks like a quantization defect.

This pins the copies against ComfyUI's originals whenever both are importable.
Where ComfyUI is absent the test still pins the algebraic identities (that
pack/unpack and patchify/unpatchify are mutual inverses), which catches most of
the same mistakes without the dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tests.e2e.accuracy.minimax_h3.int8_comfyui_parity.omni_multistep import (
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)

# Pure tensor reshaping on CPU. Marked cpu so it runs anywhere, including on the
# machines that cannot host the full-model parity run it guards.
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _comfy_layout():
    """ComfyUI's originals, or None when it is not importable here."""
    root = os.environ.get("H3_COMFYUI_ROOT", "/data/projects/comfyui/ComfyUI")
    if not Path(root).is_dir():
        return None
    try:
        import sys

        if root not in sys.path:
            sys.path.insert(0, root)
        from comfy.ldm.minimax import model as comfy_model
    except Exception:
        # ComfyUI pulls in its own interpreter's site-packages; when those are
        # not present the import fails on an unrelated module.
        return None
    return comfy_model


@pytest.mark.parametrize("latent_t,latent_h,latent_w", [(2, 8, 14), (12, 16, 28), (7, 10, 18)])
def test_patchify_roundtrips(latent_t: int, latent_h: int, latent_w: int) -> None:
    latent = torch.randn(1, 24, latent_t, latent_h, latent_w)
    rows = patchify_video(latent, (1, 2, 2))
    assert rows.shape == (latent_t * (latent_h // 2) * (latent_w // 2), 24 * 1 * 2 * 2)
    back = unpatchify_video(rows, latent_t, latent_h // 2, latent_w // 2, c=24)
    assert torch.equal(back, latent)


@pytest.mark.parametrize("audio_t", [40, 65, 160])
def test_audio_pack_roundtrips(audio_t: int) -> None:
    latent = torch.randn(1, 32, 2, audio_t)
    rows = pack_audio(latent)
    assert rows.shape == (2 * audio_t, 32)
    back = unpack_audio(rows, ch=2)
    assert torch.equal(back, latent)


def test_matches_comfyui_video() -> None:
    comfy_model = _comfy_layout()
    if comfy_model is None:
        pytest.skip("ComfyUI not importable; inverse-property coverage only.")
    for latent_t, latent_h, latent_w in [(2, 8, 14), (12, 16, 28)]:
        latent = torch.randn(1, 24, latent_t, latent_h, latent_w)
        assert torch.equal(
            patchify_video(latent, (1, 2, 2)), comfy_model.patchify_video(latent, (1, 2, 2))
        )
        rows = patchify_video(latent, (1, 2, 2))
        assert torch.equal(
            unpatchify_video(rows, latent_t, latent_h // 2, latent_w // 2, c=24),
            comfy_model.unpatchify_video(rows, latent_t, latent_h // 2, latent_w // 2, c=24),
        )


def test_matches_comfyui_audio() -> None:
    comfy_model = _comfy_layout()
    if comfy_model is None:
        pytest.skip("ComfyUI not importable; inverse-property coverage only.")
    for audio_t in [40, 65, 160]:
        latent = torch.randn(1, 32, 2, audio_t)
        assert torch.equal(pack_audio(latent), comfy_model.pack_audio(latent))
        rows = pack_audio(latent)
        assert torch.equal(unpack_audio(rows, ch=2), comfy_model.unpack_audio(rows, ch=2))
