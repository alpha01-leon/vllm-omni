# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""int8 ref2va DiT parity against a ComfyUI reference, over a full sampling
trajectory.

The ref2va sibling of test_int8_comfyui_parity.py. fl2va proved the int8 DiT is
faithful on a task with NO reference condition rows; ref2va is the harder case
because the packed sequence carries reference video and audio rows whose
contents, order and RoPE position assignments must match between the two
implementations. Matching latent shapes is not enough -- see compare_ref2va.py,
which asserts the layouts agree before it scores anything.

Four scripts, run in order, because they need different interpreters and two
different asset pipelines:

  0. ref2va_assets.write_prompt()  -- dumps REF2VA_PROMPT to a plain file, under
     THIS repo's interpreter (it needs pytest and the tests package; neither
     ComfyUI's venv nor the reference runner can import the accuracy test).
  1. comfyui_reference_ref2va.py   -- ComfyUI's own reference preprocessing,
     tokenizer and sampler, under ComfyUI's venv. Encodes the reference media
     once and writes the latents into the payload.
  2. omni_multistep_ref2va.py      -- this repo's minimax_h3_denoise_loop, under
     ours. CONSUMES those latents rather than re-encoding with its own VAE.
  3. compare_ref2va.py             -- scores the two and exits non-zero past
     threshold.

The two runners cannot be one process: the reference must load ComfyUI's int8
checkpoint and patcher, and this side must load the converted partition through
vLLM's quantization stack. Forcing them into one interpreter would mean picking
whichever stack loses less fidelity.

Nothing here runs in ordinary CI. It needs a converted Ref2VA int8 partition, a
Comfy-Org int8 checkpoint and the VAEs on the same 20 GiB card, and the two MUST
NOT run concurrently -- the reference alone holds ~63 GiB resident while the
DiT is loaded. TestInt8ComfyuiRef2vaParity skips unless pointed at both trees.

Two cost notes, because they shape the runtime profile:

  * The reference video encode runs on CPU at ~25 min. At the model's production
    768-short-edge canvas the reference video alone contributes ~12k packed
    rows, and the resulting sequence OOMs a 20 GiB card at step 0. Parity
    therefore runs the reference at a reduced canvas (H3_REF_SHORT_EDGE, 256 by
    default) -- the same value on both sides, so it changes the reference's
    resolution and nothing else. The encode result is cached; only the first run
    pays it.
  * The reference video encode needs the whole pipeline twice to get right; see
    the comments in comfyui_reference_ref2va.py for the three ComfyUI-specific
    traps (OOM retry, VAE layout, audio memory estimate).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.benchmark, pytest.mark.diffusion, pytest.mark.full_model]

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]

# Latent shape for the parity run. Both sides must use the same shape or there
# is nothing to compare; this one fits in ~20 GiB. The shape used for real
# output (384x672, latent_t 37) OOMs the reference at step 0.
PARITY_WIDTH = 448
PARITY_HEIGHT = 256
PARITY_LATENT_T = 12
PARITY_STEPS = 8

# Reference video canvas cap, in pixels on the short edge. 0 means the model's
# production 768, which does not fit alongside the DiT on a 20 GiB card.
PARITY_REF_SHORT_EDGE = 256

FINAL_L2REL_MAX = 0.15
PER_STEP_L2REL_MAX = 0.15
MAX_FINAL_STEP_GROWTH = 3.0


def _require(env_var: str, what: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"Set {env_var} to the {what} to run int8 ComfyUI ref2va parity.")
    return value


@pytest.fixture(scope="module")
def parity_env() -> dict[str, str]:
    """Paths for the ref2va parity scripts. All opt-in: see the module docstring."""
    partition = _require("H3_PART", "converted int8 partition (…/Ref2VA)")
    dit = _require("H3_COMFY_DIT", "Comfy-Org int8 ref2va diffusion model")
    text_encoder = _require("H3_TEXT_ENCODER_ROOT", "Qwen3-VL text encoder root")
    comfy_python = os.environ.get("H3_COMFY_PYTHON", "")
    if not comfy_python or not Path(comfy_python).exists():
        pytest.skip("Set H3_COMFY_PYTHON to ComfyUI's interpreter.")
    return {
        "H3_PART": partition,
        "H3_COMFY_DIT": dit,
        "H3_TEXT_ENCODER_ROOT": text_encoder,
        "H3_COMFY_PYTHON": comfy_python,
    }


def _run(cmd: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    log.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        pytest.fail(f"{Path(cmd[1]).name} failed (exit {result.returncode}); see {log}")


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_int8_ref2va_dit_matches_comfyui_over_full_trajectory(
    tmp_path: Path, parity_env: dict[str, str]
) -> None:
    """Both int8 ref2va DiTs must agree over the trajectory, refs included."""
    if not torch.cuda.is_available():
        pytest.skip("int8 ComfyUI ref2va parity requires CUDA.")

    ref_path = tmp_path / "ref.pt"
    omni_path = tmp_path / "omni.pt"
    prompt_path = tmp_path / "ref2va_prompt.txt"

    # The prompt has to be extracted from the accuracy test under an interpreter
    # that can import it; the runners only ever see the plain file.
    _run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); "
         "sys.path.insert(0, %r); "
         "import ref2va_assets; ref2va_assets.write_prompt(%r)" % (
             str(_THIS_DIR), str(REPO_ROOT), str(prompt_path))],
        {},
        tmp_path / "write_prompt.log",
    )

    _run(
        [parity_env["H3_COMFY_PYTHON"], "-I",
         str(_THIS_DIR / "comfyui_reference_ref2va.py")],
        {
            **parity_env,
            "OUT": str(ref_path),
            "W": str(PARITY_WIDTH),
            "H": str(PARITY_HEIGHT),
            "STEPS": str(PARITY_STEPS),
            "LATENT_T": str(PARITY_LATENT_T),
            "H3_REF_SHORT_EDGE": str(PARITY_REF_SHORT_EDGE),
            "H3_PROMPT": str(prompt_path),
        },
        tmp_path / "comfyui_reference_ref2va.log",
    )

    _run(
        [sys.executable, "-I", str(_THIS_DIR / "omni_multistep_ref2va.py")],
        {**parity_env, "H3_REF": str(ref_path), "H3_OUT": str(omni_path)},
        tmp_path / "omni_multistep_ref2va.log",
    )

    # compare_ref2va.py owns the thresholds and prints the per-step tables AND the
    # layout agreement check; run it as a subprocess so its exit status is the
    # assertion, rather than re-deriving the same metrics here and letting the
    # two drift apart.
    result = subprocess.run(
        [sys.executable, "-I", str(_THIS_DIR / "compare_ref2va.py"), str(ref_path),
         str(omni_path)],
        env={**os.environ, **parity_env},
        capture_output=True,
        text=True,
    )
    (tmp_path / "compare_ref2va.log").write_text(result.stdout + "\n" + result.stderr)
    assert result.returncode == 0, (
        f"int8 ref2va parity check failed; see {tmp_path / 'compare_ref2va.log'}"
    )
