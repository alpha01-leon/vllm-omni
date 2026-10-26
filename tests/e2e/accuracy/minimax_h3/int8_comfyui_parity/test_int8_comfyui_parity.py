# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""int8 DiT parity against a ComfyUI reference, over a full sampling trajectory.

The single-step comparison proved one DiT call is faithful. It could not say
anything about whether the residual requantization error compounds over a
sampling run -- the inputs diverge as soon as both sides free-run, which is
exactly where a small per-step error would show up. This closes that gap.

Three scripts, run in order, because they need different interpreters:

  1. comfyui_reference.py  -- ComfyUI's own KSampler, under ComfyUI's venv.
  2. omni_multistep.py     -- this repo's minimax_h3_denoise_loop, under ours.
  3. compare.py            -- scores the two and exits non-zero past threshold.

The two runners cannot be one process: the reference must load ComfyUI's
int8 checkpoint and patcher, and this side must load the converted partition
through vLLM's quantization stack. Forcing them into one interpreter would
mean picking whichever stack loses less fidelity.

Nothing here runs in ordinary CI. It needs both an int8 partition and a
Comfy-Org int8 checkpoint on the same 20 GiB card, and the two MUST NOT run
concurrently -- the reference alone holds ~63 GiB resident while the DiT is
loaded. TestInt8ComfyuiParity skips unless pointed at both trees.
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
# is nothing to compare; this one fits the reference inside 20 GiB. The shape
# used for real output (384x672, latent_t 37) OOMs the reference at step 0.
PARITY_WIDTH = 448
PARITY_HEIGHT = 256
PARITY_LATENT_T = 12
PARITY_STEPS = 8

FINAL_L2REL_MAX = 0.15
PER_STEP_L2REL_MAX = 0.15
MAX_FINAL_STEP_GROWTH = 3.0


def _require(env_var: str, what: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"Set {env_var} to the {what} to run int8 ComfyUI parity.")
    return value


@pytest.fixture(scope="module")
def parity_env() -> dict[str, str]:
    """Paths for the three scripts. All opt-in: see the module docstring."""
    partition = _require("H3_PART", "converted int8 partition (…/FL2VA)")
    dit = _require("H3_COMFY_DIT", "Comfy-Org int8 diffusion model")
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
def test_int8_dit_matches_comfyui_over_full_trajectory(
    tmp_path: Path, parity_env: dict[str, str]
) -> None:
    """Both int8 DiTs must agree over the whole 8-step trajectory, not just step 0."""
    if not torch.cuda.is_available():
        pytest.skip("int8 ComfyUI parity requires CUDA.")

    ref_path = tmp_path / "ref.pt"
    omni_path = tmp_path / "omni.pt"

    _run(
        [
            parity_env["H3_COMFY_PYTHON"], "-I", str(_THIS_DIR / "comfyui_reference.py"),
        ],
        {
            **parity_env,
            "OUT": str(ref_path),
            "W": str(PARITY_WIDTH),
            "H": str(PARITY_HEIGHT),
            "STEPS": str(PARITY_STEPS),
            "LATENT_T": str(PARITY_LATENT_T),
        },
        tmp_path / "comfyui_reference.log",
    )

    _run(
        [
            sys.executable, "-I", str(_THIS_DIR / "omni_multistep.py"),
        ],
        {**parity_env, "H3_REF": str(ref_path), "H3_OUT": str(omni_path)},
        tmp_path / "omni_multistep.log",
    )

    # compare.py owns the thresholds and prints the per-step tables; run it as
    # a subprocess so its exit status is the assertion, rather than re-deriving
    # the same metric a second time here and letting the two drift apart.
    result = subprocess.run(
        [sys.executable, "-I", str(_THIS_DIR / "compare.py"), str(ref_path), str(omni_path)],
        env={**os.environ, **parity_env},
        capture_output=True,
        text=True,
    )
    (tmp_path / "compare.log").write_text(result.stdout + "\n" + result.stderr)
    assert result.returncode == 0, f"int8 parity check failed; see {tmp_path / 'compare.log'}"
