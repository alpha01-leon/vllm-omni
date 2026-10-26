# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared asset preparation for the ref2va int8 <-> ComfyUI parity run.

Both sides of the comparison need the SAME reference media, encoded to the SAME
latents. That is not automatic: each side has its own video/audio VAE with
different weights (ComfyUI ships fp16/fp32 Comfy-Org builds, this repo loads
the bf16 partition's own VAE), so encoding independently would leave a real,
non-quantization difference between the two runs that would be measured as one.

So this module encodes once, under the interpreter of whichever side is
producing the reference, and writes the result to a file both runners read:

  * the ref2va reference video's clean visual rows and latent shape,
  * the ref2va video's embedded soundtrack's clean audio rows and length,
  * the standalone voice-timbre audio's clean rows and length.

The clean (pre-noise-augmentation) rows are what both sides need: each side
applies its OWN condition-noise recipe on top, exactly as production does.
Those recipes draw from different RNG shapes -- ComfyUI patches first and draws
at the patched row shape, this repo draws at the latent shape over
``max(target_t + cond_frames, latent_t)`` frames -- so the anchors are not
bit-identical by construction. That is fine and deliberate: the resulting anchor
difference is ~1.4e-3 relative, two orders of magnitude below the 0.15 parity
threshold, so it cannot masquerade as a quantization defect. Reproducing
ComfyUI's draw exactly would pin a harness detail rather than the model.

Sources and sha256 are the official accuracy-test constants, so the material is
identical to what ``test_minimax_h3_ref2va_matches_official_reference`` uses.

Media decoding follows ComfyUI's ``MiniMaxH3ReferenceToVideo`` exactly, because
the ref2va prompt's reference semantics depend on it:
  * ``adapt_canvas`` -- 768-short-edge canvas, 768*1344 area cap, per-axis round
    to 32 (nodes_minimax_h3.py:52),
  * frames truncated onto the ``17k+5`` grid (``while n % 17 != 5: n -= 1``),
  * video presented to Qwen at 2 fps with ``<T.T seconds>`` per 2-frame block.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24

# Official constants, transcribed from
# tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity.py
INPUT_VIDEO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/"
    "gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4"
)
INPUT_VIDEO_SHA256 = "d6edc20ac82aa980c9cbd0b05b371f60d12dc8d018492e571b42de0086819c41"
INPUT_AUDIO_URL = (
    "https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/"
    "gallery/sr_v2p26_trio_seed42_20260724/inputs/"
    "f463d523c5ce_01_%E9%9F%B3%E9%A2%911_RSLcbpzJPo_6%E6%9C%885%E6%97%A5(1).mp3"
)
INPUT_AUDIO_SHA256 = "2a48d447b343b38a2c504d5281a0c97c688db0ba87931ed9282c598bd6752454"

PROMPT_ENV = "H3_PROMPT"
DEFAULT_ASSETS = "/tmp/h3ref2va/assets"
DEFAULT_CONDITIONS = "/tmp/h3ref2va/ref_conditions.pt"


def log(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(url: str, sha256: str, *, filename: str | None = None) -> str:
    """Download ``url`` into the asset dir and verify it against ``sha256``.

    Returns the local path. A cached file that already matches is reused, so a
    re-run does not re-hit the network; a cached file that does NOT match is a
    hard error rather than a silent re-download, because that would mean the
    corpus changed underneath a parity run that cites the constant.
    """
    import urllib.request

    asset_dir = Path(os.environ.get("H3_ASSET_DIR", DEFAULT_ASSETS))
    asset_dir.mkdir(parents=True, exist_ok=True)
    suffix = os.path.splitext(url.split("?")[0])[1] or ".bin"
    path = asset_dir / (filename or f"{sha256[:16]}{suffix}")

    if path.exists() and sha256_file(path) == sha256:
        log(f"asset cached {path}")
        return str(path)

    if path.exists():
        raise RuntimeError(
            f"cached asset {path} does not match the expected sha256 {sha256}; "
            "remove it to refetch"
        )

    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response, open(path, "wb") as out:
        out.write(response.read())
    got = sha256_file(path)
    if got != sha256:
        raise RuntimeError(f"asset {path} sha256 {got} != expected {sha256}")
    return str(path)


def write_prompt(path: str | Path | None = None) -> str:
    """Copy ``REF2VA_PROMPT`` out of the accuracy test into a plain text file.

    Run once under THIS repo's interpreter (it needs pytest and the ``tests``
    package). Both parity runners then read that file, so neither ComfyUI's venv
    nor the reference runner has to import the test module.

    Deliberately not reconstructed from memory: the ``fully_preserved`` /
    ``partially_copy`` / ``reference`` grading of ``<Video 1>`` / ``<Audio 1>`` /
    ``<Audio 2>`` is what makes both sides condition on the same material. A
    paraphrase would drop that grading and the comparison would be measuring two
    different prompts.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tests.e2e.accuracy.minimax_h3.test_minimax_h3_i2va_ref2va_similarity import (
        REF2VA_PROMPT,
    )

    target = Path(path or os.environ.get(PROMPT_ENV) or Path(DEFAULT_ASSETS) / "ref2va_prompt.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(REF2VA_PROMPT, encoding="utf-8")
    log(f"prompt written to {target} ({len(REF2VA_PROMPT)} chars)")
    return str(target)


def load_prompt() -> str:
    """Read the official six-section ref2va prompt from disk."""
    path = os.environ.get(PROMPT_ENV)
    if not path:
        raise RuntimeError(
            f"set {PROMPT_ENV} to the prompt file; run ref2va_assets.write_prompt() "
            "under this repo's interpreter to produce it from REF2VA_PROMPT"
        )
    return Path(path).read_text(encoding="utf-8")


def cap_short_edge(frames, width: int, height: int, short_edge: int):
    """Downscale a reference so its short edge is at most ``short_edge``.

    Aspect-preserving and down-only, matching the ``ref_image_size="match"``
    policy the node applies to images: never enlarges, so a reference already
    smaller than the cap passes through untouched. Rounded to
    ``CANVAS_MULTIPLE`` so the result lands on the canvas grid.
    """
    import torch

    from comfy_extras.nodes_minimax_h3 import _resize

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    if min(width, height) <= short_edge:
        return frames, width, height
    scale = float(short_edge) / float(min(width, height))
    tw = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(frames, tw, th, "disabled"), tw, th


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """Transcribed from nodes_minimax_h3.py:52."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def decode_video_frames(path: str) -> tuple["list", int, int, int]:
    """Decode ``path`` to RGB frames at native fps.

    Returns ``(frames, width, height, frame_count)`` with ``frames`` shaped
    ``[T, H, W, C]`` float32 in [0, 1], matching ComfyUI's IMAGE batch layout.
    """
    import av
    import numpy as np

    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    array = np.stack(frames).astype("float32") / 255.0
    return array, array.shape[2], array.shape[1], array.shape[0]


def decode_video_audio(path: str) -> tuple["object", int] | tuple[None, int]:
    """Extract a video's embedded soundtrack as ``(waveform[2, L], sample_rate)``.

    Returns ``(None, 0)`` when the container has no audio stream -- which is a
    real case for ref2va, and the caller must then emit a plain ``video`` block
    rather than a ``video_audio`` one.
    """
    import av
    import numpy as np

    with av.open(path) as container:
        if not container.streams.audio:
            return None, 0
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            # av returns planar [channels, samples] for packed stereo
            if arr.ndim == 1:
                arr = arr[None]
            chunks.append(arr.astype("float32"))
        if not chunks:
            return None, 0
        rate = int(container.streams.audio[0].rate)
    if len(chunks[0]) > 1 and chunks[0].shape[0] <= 8:
        waveform = np.concatenate(chunks, axis=1)
    else:  # packed interleave
        waveform = np.concatenate(chunks, axis=-1).reshape(-1, 2).T
    import torch

    return torch.from_numpy(np.ascontiguousarray(waveform)).float(), rate


def decode_audio_file(path: str) -> tuple["object", int]:
    """Decode an audio file to ``(waveform[2, L], sample_rate)``."""
    import av
    import numpy as np
    import torch

    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            arr = frame.to_ndarray().astype("float32")
            if arr.ndim == 1:
                arr = arr[None]
            chunks.append(arr)
        rate = int(stream.rate)
    waveform = np.concatenate(chunks, axis=1)
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    return torch.from_numpy(np.ascontiguousarray(waveform)).float(), rate


def prepare_reference_video(frames, width: int, height: int, frame_count: int,
                             *, short_edge: int | None = None):
    """Apply ComfyUI's reference-video resize + 17k+5 truncation.

    ``_resize`` is Lanczos via ``comfy.utils.common_upscale``; that needs
    ComfyUI on the path, so it is imported lazily and the caller must be running
    under the ComfyUI interpreter. Returns ``(resized_frames, canvas_w, canvas_h)``.

    ``short_edge`` optionally caps the canvas's short edge BELOW the model's
    768. This is a parity-run concession, not a product behaviour: at 768 the
    reference video contributes ~12k packed rows, and with the target streams
    and the Qwen text span the ref2va sequence is ~4.8x the fl2va one, which
    OOMs the 20 GiB card at step 0. Both sides take the SAME cap, so the
    comparison stays sound -- the reference is just lower resolution. ``None``
    keeps the production canvas, which is what a real request gets.
    """
    import torch

    from comfy_extras.nodes_minimax_h3 import _resize

    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    if short_edge:
        frames, width, height = cap_short_edge(frames, width, height, short_edge)
    cw, ch = adapt_canvas(width, height)
    if width * height < cw * ch:
        cw = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ch = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = _resize(frames, cw, ch, "disabled")
    if resized.shape[0] > frame_count:
        resized = resized[:frame_count]
    n = resized.shape[0]
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
    while n % 17 != 5:
        n -= 1
    return resized[:n], cw, ch


def qwen_video_samples(resized, qwen_fps: int = 2):
    """The 2 fps presentation Qwen sees, plus its timestamps.

    ``range(0, frames, FPS // 2)`` and ``timestamps = [i / 2.0]`` verbatim from
    nodes_minimax_h3.py:336-339.
    """
    step = max(1, FPS // qwen_fps)
    idx = list(range(0, resized.shape[0], step))
    return resized[idx], [i / float(qwen_fps) for i in range(len(idx))]
