# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ConvRot dequantization transcribed into the H3 checkpoint converter.

``convert_comfy_checkpoint.py`` used to call ``comfy_kitchen``,
``dequantize_int8_convrot_weight``. ``comfy_kitchen`` is a ComfyUI dependency and
is not in this repo's dependency tree, so that import meant the converter could
only run under a ComfyUI checkout. The function is now transcribed into the
converter; these tests pin that transcription.

Two things need pinning and they fail differently:

* The *math*. The expected checksums below were produced by ``comfy_kitchen``
  itself, at the shapes this checkpoint uses. A transcription that merely
  round-tripped against itself would pass even if the rotation were wrong, which
  is the one bug that matters here: it produces weights that load fine, quantize
  fine, and are subtly wrong.
* The *guard*. ``_build_hadamard`` only reaches powers of four. A group size of
  128 is a power of two, is easy to reach by accident (7168 = 128 * 56 looks
  groupable), and has no seed-compatible Hadamard. It must raise, not silently
  build a matrix that rotates into the wrong basis.

Storing checksums rather than the expected tensors keeps this file small and
avoids a binary fixture; fp32 output makes the hash exact, so this is bit-level
equality in all but name.

These tests do not import ComfyUI. That is the point: the converter's dependency
on it is what this file exists to keep removed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "minimax_h3" / "convert_comfy_checkpoint.py"


def _load_converter():
    """Import the tool by path; ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location("convert_comfy_checkpoint", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_converter()

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Group size 256: what the Comfy-Org H3 export declares in its ``.comfy_quant``
# sidecars (all 200 quantized modules agree).
GROUP = 256

# sha256 of the fp32 output of ``comfy_kitchen.dequantize_int8_convrot_weight``
# for int8 q drawn with ``manual_seed(0)`` and ``scale = rand(out, 1) + 0.01``.
# Captured with ComfyUI's interpreter; see the module docstring.
COMFY_KITCHEN_SHA256 = {
    (96, 256): "1b6b6410945e3852a237afe92fc0b834bd03736874e060cddb0e9b6d52d53e72",
    (64, 256): "e4d3bbcebc09cede439d28ddf5efb19095b2d9ad6f4bb3be201eb7c05a38ce01",
    (512, 7168): "21b0c292c6ce911905d355b7fd200732d7af508cdbea48abf9e5166f600e7b18",
    (256, 5376): "a5a6e354e214809302f7c0e80a84fdd6a965af3aaa0eb387bfee09b3f8cdf562",
}


def test_converter_does_not_import_comfy():
    """The whole point of the transcription. Guards against a regression.

    Matched on an import statement rather than a bare substring: the file
    explains in prose where the transcribed maths came from, and naming that
    source is worth more than the guard being simple.
    """
    tree = ast.parse(_TOOL.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    offenders = {name for name in imported if "comfy" in name.lower()}
    assert not offenders, (
        f"converter imports {offenders}; comfy_kitchen is not in this repo's "
        f"dependency tree, so the tool would stop being runnable standalone"
    )


@pytest.mark.parametrize("group", [4, 16, 64, 256, 1024])
def test_hadamard_is_orthogonal_and_normalized(group: int):
    h = converter._build_hadamard(group)
    assert h.shape == (group, group)
    # H @ H.T == I: the rotation must be exactly invertible.
    identity = h @ h.T
    assert torch.allclose(identity, torch.eye(group), atol=1e-5), f"group={group} is not orthogonal"
    # Entries are +/- 1/sqrt(group); verify the normalisation actually happened.
    assert abs(h.abs().max().item() - 1.0 / group**0.5) < 1e-6


@pytest.mark.parametrize("group", [8, 32, 128, 512])
def test_hadamard_rejects_non_fourth_powers(group: int):
    """Powers of two outside 4**k must raise rather than build a wrong basis."""
    with pytest.raises(ValueError, match="power of FOUR"):
        converter._build_hadamard(group)


@pytest.mark.parametrize("group", [0, 1, 2, 3])
def test_hadamard_rejects_too_small(group: int):
    """Below 4 there is no seed; these fail the power-of-two check first."""
    with pytest.raises(ValueError, match="power of two"):
        converter._build_hadamard(group)


def test_hadamard_rejects_non_powers_of_two():
    for size in (100, 255):
        with pytest.raises(ValueError, match="power of two"):
            converter._build_hadamard(size)


@pytest.mark.parametrize(
    "out_f,in_f",
    [
        (96, 256),     # adaLN-ish, one group
        (64, 256),
        (512, 7168),   # attn.out_proj shape from the real export
        (256, 5376),   # video_patch_proj shape from the real export
    ],
)
def test_dequantize_matches_comfy_kitchen_reference(out_f: int, in_f: int):
    """Hash the output and compare against a value captured from ``comfy_kitchen``.

    This asserts against the upstream implementation rather than against this
    file's own idea of the rotation, which is the only way to catch a rotation
    applied in the wrong place or with the wrong transpose.
    """
    torch.manual_seed(0)
    q = torch.randint(-127, 127, (out_f, in_f), dtype=torch.int8)
    scale = torch.rand(out_f, 1, dtype=torch.float32) + 0.01

    got = converter._dequantize_convrot(q, scale, GROUP)
    digest = hashlib.sha256(got.contiguous().numpy().tobytes()).hexdigest()
    expected = COMFY_KITCHEN_SHA256[(out_f, in_f)]

    assert digest == expected, (
        f"{out_f}x{in_f}: transcription diverges from comfy_kitchen. If the "
        f"rotation moved or flipped, the converted weights would load and "
        f"quantize cleanly and just be subtly wrong."
    )


def test_dequantize_rejects_indivisible_input():
    q = torch.zeros(8, 100, dtype=torch.int8)
    scale = torch.ones(8, 1)
    with pytest.raises(ValueError, match="not divisible"):
        converter._dequantize_convrot(q, scale, 256)


def test_validate_spec_rejects_power_of_two_that_is_not_a_fourth_power():
    """The spec guard must be at least as strict as the Hadamard builder.

    Previously this accepted any power of two, so a 128 group would pass
    validation and then fail (or, before the inlining, behave differently inside
    comfy_kitchen) instead of being rejected where the mistake is.
    """
    base = {"format": "int8_tensorwise", "convrot": True}

    with pytest.raises(ValueError, match="power of two"):
        converter._validate_spec({**base, "convrot_groupsize": 0}, "some.module")
    with pytest.raises(ValueError, match="power of two"):
        converter._validate_spec({**base, "convrot_groupsize": 300}, "some.module")
    with pytest.raises(ValueError, match="power of FOUR"):
        converter._validate_spec({**base, "convrot_groupsize": 128}, "some.module")

    assert converter._validate_spec({**base, "convrot_groupsize": 256}, "m") == 256
