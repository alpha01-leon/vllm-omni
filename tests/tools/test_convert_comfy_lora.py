# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MiniMax-H3 Turbo LoRA converter.

The converter's job is to undo a re-packing: Comfy-Org publishes the Turbo
adapters in ComfyUI's fused spelling, and the loader in
``vllm_omni/diffusion/models/minimax_h3/lora.py`` reads the Diffusers PEFT
spelling instead. A wrong inverse is not a crash -- it is an adapter that
loads and computes a subtly different function, so the assertions here are
about FORWARD EQUIVALENCE, not about tensors being equal.

These tests do not import ComfyUI. That is the point: the conversion rules
are read out of the source file's own metadata, and the reference for
"correct" is the loader's own arithmetic, which lives in this repo.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "minimax_h3" / "convert_comfy_lora.py"


def _load_converter():
    """Import the tool by path; ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location("convert_comfy_lora", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_converter()

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RANK = 128
HIDDEN = 5376
INNER = 7168
FF_HIDDEN = 14336

# The metadata the Comfy-Org export carries, and the rules it states.
SOURCE_METADATA = {
    "source_format": "Diffusers PEFT LoRA",
    "target_format": "ComfyUI generic LoRA",
    "qkv_fusion": "block diagonal B; concat A; alpha multiplied by 3",
    "swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]",
    "training_rank": "128",
    "training_alpha": "8.0",
    "training_scale": "0.0625",
}


def _comfy_qkv(rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor, float]:
    """A fused QKV LoRA pair in ComfyUI's layout, plus its fused alpha.

    ``lora_A`` is the row concatenation of the three projections and
    ``lora_B`` is their block-diagonal stacking, which is what the export's
    ``qkv_fusion`` metadata describes.
    """
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(3 * rank, HIDDEN, generator=generator, dtype=torch.float32)
    b = torch.zeros(3 * INNER, 3 * rank, dtype=torch.float32)
    for index in range(3):
        b[index * INNER : (index + 1) * INNER, index * rank : (index + 1) * rank] = torch.randn(
            INNER, rank, generator=generator, dtype=torch.float32
        )
    return a, b, 24.0


def _comfy_block(prefix: str, *, rank: int = RANK) -> dict[str, torch.Tensor]:
    """One ComfyUI-layout block: fused QKV, out_proj, fc1, fc2."""
    generator = torch.Generator().manual_seed(1)
    qkv_a, qkv_b, qkv_alpha = _comfy_qkv(rank)
    tensors = {
        f"{prefix}.attn.qkv_proj.lora_A.weight": qkv_a,
        f"{prefix}.attn.qkv_proj.lora_B.weight": qkv_b,
        f"{prefix}.attn.qkv_proj.alpha": torch.tensor(qkv_alpha),
        f"{prefix}.attn.out_proj.lora_A.weight": torch.randn(rank, INNER, generator=generator, dtype=torch.float32),
        f"{prefix}.attn.out_proj.lora_B.weight": torch.randn(HIDDEN, rank, generator=generator, dtype=torch.float32),
        f"{prefix}.attn.out_proj.alpha": torch.tensor(8.0),
        f"{prefix}.mlp.fc1.lora_A.weight": torch.randn(rank, HIDDEN, generator=generator, dtype=torch.float32),
        f"{prefix}.mlp.fc1.lora_B.weight": torch.randn(2 * FF_HIDDEN, rank, generator=generator, dtype=torch.float32),
        f"{prefix}.mlp.fc1.alpha": torch.tensor(8.0),
        f"{prefix}.mlp.fc2.lora_A.weight": torch.randn(rank, FF_HIDDEN, generator=generator, dtype=torch.float32),
        f"{prefix}.mlp.fc2.lora_B.weight": torch.randn(HIDDEN, rank, generator=generator, dtype=torch.float32),
        f"{prefix}.mlp.fc2.alpha": torch.tensor(8.0),
    }
    return tensors


def _write_source(path: Path, *, blocks: int = 1, metadata: dict[str, str] | None = None) -> Path:
    """Write a synthetic ComfyUI-layout Turbo LoRA."""
    tensors: dict[str, torch.Tensor] = {}
    for index in range(blocks):
        tensors.update(_comfy_block(f"diffusion_model.blocks.{index}"))
    save_file(tensors, str(path), metadata=metadata if metadata is not None else SOURCE_METADATA)
    return path


def _write_full_source(path: Path) -> Path:
    """Write a source with the COMPLETE target set the loader demands.

    The loader rejects any adapter whose raw targets are not exactly the
    published layout -- 50 transformer blocks plus 2 token-refiner blocks
    (``lora.py:122-130``) -- so a fixture that wants to exercise that check
    has to carry all of them.
    """
    tensors: dict[str, torch.Tensor] = {}
    for index in range(50):
        tensors.update(_comfy_block(f"diffusion_model.blocks.{index}"))
    for index in range(2):
        tensors.update(_comfy_block(f"diffusion_model.token_refiner.blocks.{index}"))
    save_file(tensors, str(path), metadata=SOURCE_METADATA)
    return path


def _lora_delta(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, scaling: float) -> torch.Tensor:
    """The LoRA contribution to a linear layer's output."""
    return ((x @ a.T) @ b.T) * scaling


def _read_output(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as reader:
        return {name: reader.get_tensor(name) for name in reader.keys()}


def _read_metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as reader:
        return reader.metadata() or {}


def test_split_qkv_recovers_three_independent_projections():
    """The fused pair splits into three projections of the right shape."""
    a, b, _ = _comfy_qkv()
    parts = converter._split_qkv(a, b, RANK)
    assert len(parts) == 3
    for a_part, b_part in parts:
        assert a_part.shape == (RANK, HIDDEN)
        assert b_part.shape == (INNER, RANK)


def test_split_qkv_is_forward_equivalent_to_the_fused_pair():
    """Splitting the fused QKV reproduces the fused forward.

    ComfyUI infers the rank from ``lora_A``'s own row count
    (``comfy/weight_adapter/lora.py``: ``rank = down.shape[0]``), so the fused
    file's scaling is ``alpha / (3 * rank)`` -- which is exactly the
    per-projection scaling the split file uses. Both sides therefore have to
    agree, and the split has to reproduce the fused output row for row.
    """
    a, b, fused_alpha = _comfy_qkv()
    parts = converter._split_qkv(a, b, RANK)
    x = torch.randn(16, HIDDEN, dtype=torch.float32)

    fused_scaling = fused_alpha / a.shape[0]
    fused = _lora_delta(a, b, x, fused_scaling)

    per_projection_scaling = 8.0 / RANK
    split = torch.cat(
        [_lora_delta(a_part, b_part, x, per_projection_scaling) for a_part, b_part in parts],
        dim=1,
    )
    # The two sides sum the same 5376-term dot product in a different order, so
    # they agree to fp32 accumulation noise rather than bit for bit.
    torch.testing.assert_close(fused, split, rtol=1e-4, atol=1e-4)


def test_split_qkv_rejects_a_non_block_diagonal_source():
    """A source whose B is not block diagonal is rejected, not silently split.

    Splitting such a matrix would drop the off-diagonal mass, and the adapter
    would then compute a different function while looking perfectly plausible.
    """
    a, b, _ = _comfy_qkv()
    b[0, RANK] = 1.0
    with pytest.raises(ValueError, match="not block diagonal"):
        converter._split_qkv(a, b, RANK)


def test_split_qkv_rejects_a_wrong_rank():
    a, b, _ = _comfy_qkv()
    with pytest.raises(ValueError, match="expected rank"):
        converter._split_qkv(a, b, RANK + 1)


def test_swap_ffn_halves_produces_the_diffusers_order():
    """The FFN output is the Diffusers ``[value;gate]`` order.

    The loader swaps the halves back before handing the tensor to the model's
    fused gate/up projection, so the file has to carry the Diffusers order.
    """
    b = torch.arange(2 * FF_HIDDEN, dtype=torch.float32).unsqueeze(1).expand(2 * FF_HIDDEN, RANK)
    swapped = converter._swap_ffn_halves(b)
    assert swapped.shape == b.shape
    # The loader's own transform takes value, gate = chunk(2) and rebuilds
    # (gate, value); applying it to our output must recover the input.
    value, gate = swapped.chunk(2, dim=0)
    torch.testing.assert_close(torch.cat((gate, value), dim=0), b)


def test_swap_ffn_halves_rejects_an_odd_row_count():
    with pytest.raises(ValueError, match="not an even split"):
        converter._swap_ffn_halves(torch.zeros(7, RANK))


def test_convert_writes_the_diffusers_layout(tmp_path):
    """The converted file carries the names and shapes the loader expects."""
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    dst = tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    stats = converter.convert_lora(src, dst)
    assert stats["pairs"] == 6
    assert stats["split_qkv"] == 3

    out = _read_output(dst)
    assert "transformer_blocks.0.attn.to_q.lora_A.default.weight" in out
    assert "transformer_blocks.0.attn.to_k.lora_B.default.weight" in out
    assert "transformer_blocks.0.attn.to_v.lora_A.default.weight" in out
    assert "transformer_blocks.0.attn.to_out.0.lora_A.default.weight" in out
    assert "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight" in out
    assert "transformer_blocks.0.ff.net.2.lora_B.default.weight" in out
    assert out["transformer_blocks.0.attn.to_q.lora_A.default.weight"].shape == (RANK, HIDDEN)
    assert out["transformer_blocks.0.attn.to_q.lora_B.default.weight"].shape == (INNER, RANK)
    assert out["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"].shape == (2 * FF_HIDDEN, RANK)
    # No ComfyUI-style names may survive.
    assert not any("diffusion_model" in name for name in out)
    assert not any(name.endswith(".alpha") for name in out)


def test_convert_maps_the_token_refiner_namespace(tmp_path):
    """``token_refiner.blocks`` becomes ``token_refiner.refiner_blocks``."""
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    tensors = _read_output(src)
    tensors.update(_comfy_block("diffusion_model.token_refiner.blocks.0"))
    save_file(tensors, str(src), metadata=SOURCE_METADATA)

    dst = tmp_path / "out.safetensors"
    converter.convert_lora(src, dst)
    out = _read_output(dst)
    assert "token_refiner.refiner_blocks.0.attn.to_q.lora_A.default.weight" in out
    assert "transformer_blocks.0.attn.to_q.lora_A.default.weight" in out


def test_convert_output_name_drops_the_comfyui_infix(tmp_path):
    """The default output name is the source name without ``_comfyui_``."""
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    converter.convert_lora(src)
    # The converter also drops the ``_bf16`` suffix, because the loader's name
    # regex treats it as an optional decoration rather than part of the id.
    assert (tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors").exists()


def test_converted_output_is_parseable_by_the_loader(tmp_path):
    """The converted file satisfies the loader's own name and metadata contract."""
    from vllm_omni.diffusion.models.minimax_h3.lora import parse_turbo_filename

    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    dst = tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    converter.convert_lora(src, dst)

    spec = parse_turbo_filename(dst.name)
    assert spec is not None
    assert spec.task_family == "fl2v"
    assert spec.denoise_steps == 8

    metadata = _read_metadata(dst)
    assert metadata["key_format"] == "minimax-h3-diffusers"
    assert float(metadata["alpha"]) == 8.0


def test_converted_output_passes_the_loader_validation(tmp_path):
    """``_validate_and_convert_tensors`` accepts the converted file.

    That function is the loader's gate: it checks the target set, the shapes
    and the FFN half order, and rejects anything it cannot consume.
    """
    from vllm_omni.diffusion.models.minimax_h3.lora import _validate_and_convert_tensors

    # The loader's gate checks the WHOLE target set (50 transformer blocks plus
    # 2 token-refiner blocks), so the fixture has to carry all of them.
    src = _write_full_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    dst = tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    converter.convert_lora(src, dst)

    from safetensors import safe_open

    with safe_open(str(dst), framework="pt") as reader:
        tensors = _validate_and_convert_tensors(reader)
    # 52 blocks x 6 targets x 2 tensors.
    assert len(tensors) == 52 * 6 * 2


def test_converted_file_is_forward_equivalent_to_the_source(tmp_path):
    """The converted adapter computes the same function as the source.

    This is the assertion that matters: the two files have different tensor
    layouts, so equality of tensors is not the goal -- equality of the
    LoRA contribution to a linear layer's output is.
    """
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    dst = tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    converter.convert_lora(src, dst)

    source_tensors = _read_output(src)
    out = _read_output(dst)
    x = torch.randn(16, HIDDEN, dtype=torch.float32)
    per_projection_scaling = 8.0 / RANK

    # --- QKV: the source is fused, the output is three projections
    a = source_tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
    b = source_tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    fused_alpha = float(source_tensors["diffusion_model.blocks.0.attn.qkv_proj.alpha"].item())
    fused = _lora_delta(a, b, x, fused_alpha / a.shape[0])
    split = torch.cat(
        [
            _lora_delta(
                out[f"transformer_blocks.0.attn.{proj}.lora_A.default.weight"],
                out[f"transformer_blocks.0.attn.{proj}.lora_B.default.weight"],
                x,
                per_projection_scaling,
            )
            for proj in ("to_q", "to_k", "to_v")
        ],
        dim=1,
    )
    torch.testing.assert_close(fused, split, rtol=1e-4, atol=1e-4)

    # --- FFN fc1: the source is [gate;value], the output is [value;gate]
    fc1_a = source_tensors["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]
    fc1_b = source_tensors["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
    src_fc1 = _lora_delta(fc1_a, fc1_b, x, per_projection_scaling)
    out_a = out["transformer_blocks.0.ff.net.0.proj.lora_A.default.weight"]
    out_b = out["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"]
    # The loader swaps the halves back before use.
    value, gate = out_b.chunk(2, dim=0)
    dst_fc1 = _lora_delta(out_a, torch.cat((gate, value), dim=0), x, per_projection_scaling)
    # The two sides sum the same long dot product in a different order.
    torch.testing.assert_close(src_fc1, dst_fc1, rtol=1e-4, atol=1e-4)


def test_convert_rejects_a_source_without_the_expected_metadata(tmp_path):
    """A source that does not declare the fusion rules is rejected.

    Silently assuming the rules would produce a wrong adapter that still
    loads, which is the failure mode this script exists to prevent.
    """
    src = _write_source(
        tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        metadata={"source_format": "Diffusers PEFT LoRA"},
    )
    # The rules are checked in a fixed order, so the FIRST one this file is
    # missing is the one reported -- here "target_format".
    with pytest.raises(ValueError, match="target_format"):
        converter.convert_lora(src, tmp_path / "out.safetensors")


def test_convert_rejects_a_source_without_the_fusion_rules(tmp_path):
    """A source that does not declare the QKV fusion rule is rejected.

    Silently assuming the rules would produce a wrong adapter that still
    loads, which is the failure mode this script exists to prevent.
    """
    metadata = {k: v for k, v in SOURCE_METADATA.items() if k != "qkv_fusion"}
    src = _write_source(
        tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="qkv_fusion"):
        converter.convert_lora(src, tmp_path / "out.safetensors")


def test_convert_rejects_a_source_with_a_different_fusion_rule(tmp_path):
    metadata = dict(SOURCE_METADATA)
    metadata["swi_glu_mapping"] = "something else entirely"
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", metadata=metadata)
    with pytest.raises(ValueError, match="swi_glu_mapping"):
        converter.convert_lora(src, tmp_path / "out.safetensors")


def test_convert_rejects_a_fused_alpha_that_contradicts_the_unfused_one(tmp_path):
    """A QKV alpha that is not three times the unfused alpha is rejected."""
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    tensors = _read_output(src)
    tensors["diffusion_model.blocks.0.attn.qkv_proj.alpha"] = torch.tensor(11.0)
    save_file(tensors, str(src), metadata=SOURCE_METADATA)
    with pytest.raises(ValueError, match="fused qkv alpha"):
        converter.convert_lora(src, tmp_path / "out.safetensors")


def test_convert_rejects_a_source_without_an_alpha(tmp_path):
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    tensors = _read_output(src)
    for name in list(tensors):
        if name.endswith(".alpha"):
            del tensors[name]
    save_file(tensors, str(src), metadata=SOURCE_METADATA)
    with pytest.raises(ValueError, match="no alpha"):
        converter.convert_lora(src, tmp_path / "out.safetensors")


def test_convert_rejects_a_rank_that_contradicts_the_metadata(tmp_path):
    src = _write_source(tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    # The TENSOR check fires first: the fused lora_A has 3*128 rows, so a
    # per-projection rank of 64 does not divide it.
    with pytest.raises(ValueError, match="rank"):
        converter.convert_lora(src, tmp_path / "out.safetensors", rank=64)


def test_convert_rejects_a_rank_the_metadata_contradicts(tmp_path):
    """A metadata training_rank that disagrees with the tensors is rejected."""
    metadata = dict(SOURCE_METADATA)
    metadata["training_rank"] = "64"
    src = _write_source(
        tmp_path / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="training_rank"):
        converter.convert_lora(src, tmp_path / "out.safetensors", rank=128)


def test_tool_does_not_import_comfyui():
    """The converter must not depend on ComfyUI.

    The sibling DiT converter keeps the same rule, and it matters: the
    conversion rules are read from the artifact's own metadata, so the
    script runs anywhere torch and safetensors are installed.
    """
    import ast

    tree = ast.parse(_TOOL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "comfy" not in alias.name.lower(), f"the tool must not import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            assert "comfy" not in module, f"the tool must not import from {node.module}"
