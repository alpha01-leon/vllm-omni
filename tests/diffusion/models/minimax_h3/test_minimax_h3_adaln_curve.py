# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Curve-form AdaLN tests: the ``adaln_t_table`` export shape.

Comfy-Org H3 exports replace the ``time_embedder`` MLP with a precomputed
``[rows, 8]`` lookup curve. The native path cannot express it (the two spans are
different 8-dimensional subspaces), so the model selects the curve form from the
arch config.
"""

import math
from types import SimpleNamespace

import pytest
import torch


def _od_config(*, adaln_table_dim: int = 0):
    arch = {
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 4,
        "adaln_out_features": 18 * 8,
        "final_adaln_out_features": 2 * 8,
        "rope_inv_freq_len": 2,
    }
    if adaln_table_dim:
        arch["adaln_table_dim"] = adaln_table_dim
    return SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def _reference_lookup(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Verbatim port of ``comfy/ldm/minimax/model.py``'s curve branch."""
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def test_lookup_matches_comfyui_reference():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnTable

    table = torch.randn(1025, 8)
    module = MiniMaxH3AdalnTable(8)
    t = torch.linspace(0.0, 1.0, 1026)[:-1].clamp(0.0, 1.0)

    assert torch.equal(module(t, table), _reference_lookup(table, t))


def test_lookup_endpoints_are_exact():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnTable

    table = torch.randn(1025, 8)
    module = MiniMaxH3AdalnTable(8)

    assert torch.equal(module(torch.tensor([0.0]), table)[0], table[0])
    assert torch.equal(module(torch.tensor([1.0]), table)[0], table[-1])


def test_lookup_clamps_out_of_range_t():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnTable

    table = torch.randn(1025, 8)
    module = MiniMaxH3AdalnTable(8)

    below = module(torch.tensor([-5.0]), table)
    above = module(torch.tensor([5.0]), table)
    assert torch.equal(below[0], table[0])
    assert torch.equal(above[0], table[-1])


def test_lookup_rejects_degenerate_table():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnTable

    module = MiniMaxH3AdalnTable(8)
    with pytest.raises(ValueError, match="at least 2 rows"):
        module(torch.tensor([0.5]), torch.randn(1, 8))


def test_lookup_interpolates_linearly():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnTable

    table = torch.randn(1025, 8)
    module = MiniMaxH3AdalnTable(8)

    # Exactly halfway between grid rows 512 and 513 is their mean. The table
    # spans t in [0, 1] over 1024 intervals, so the midpoint of that interval
    # is (512 + 0.5) / 1024.
    midpoint = 512.5 / 1024.0
    got = module(torch.tensor([midpoint]), table)[0]
    assert torch.allclose(got, (table[512] + table[513]) / 2, atol=1e-6)


def test_arch_config_defaults_to_native_form():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig

    assert MiniMaxH3DiTArchConfig.from_mapping({}).adaln_table_dim == 0
    assert MiniMaxH3DiTArchConfig.from_mapping({"adaln_table_dim": 8}).adaln_table_dim == 8


def test_arch_config_rejects_negative_width():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig

    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3DiTArchConfig.from_mapping({"adaln_table_dim": -1})


def test_adaln_proj_skips_silu_in_curve_form(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _FakeLinear(torch.nn.Module):
        def __init__(
            self,
            *args,
            input_size=None,
            output_size=None,
            params_dtype=None,
            bias=True,
            quant_config=None,
            prefix="",
            **kwargs,
        ):
            num_heads = kwargs.pop("total_num_heads", None)
            num_kv_heads = kwargs.pop("total_num_kv_heads", None)
            # Callers pass sizes positionally (e.g. AdalnProj) or by keyword.
            if input_size is None and len(args) >= 1:
                input_size = args[0]
            if output_size is None and len(args) >= 2:
                output_size = args[1]
            del args, kwargs
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.in_features = input_size
            self.out_features = output_size
            self.prefix = prefix
            self.quant_config = quant_config
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            else:
                self.register_parameter("bias", None)
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads

    class _FakeAttention(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    curve_arch = h3.MiniMaxH3DiTArchConfig(hidden_size=8, adaln_table_dim=8)
    curve = h3.MiniMaxH3AdalnProj(
        curve_arch,
        18 * 8,
        None,
        expand_ratio=6,
        modality_num=3,
        prefix="blocks.0",
    )
    assert curve.apply_silu is False
    assert curve.linear.in_features == 8

    # The native form keeps the wide 2688-dim embedder output.
    native_arch = h3.MiniMaxH3DiTArchConfig(hidden_size=8)
    native = h3.MiniMaxH3AdalnProj(
        native_arch,
        18 * 8,
        None,
        expand_ratio=6,
        modality_num=3,
        prefix="blocks.0",
    )
    assert native.apply_silu is True
    assert native.linear.in_features == native_arch.time_embed_dim == 2688


def test_dit_registers_curve_buffer_under_checkpoint_key(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _FakeLinear(torch.nn.Module):
        def __init__(
            self,
            *args,
            input_size=None,
            output_size=None,
            params_dtype=None,
            bias=True,
            quant_config=None,
            prefix="",
            **kwargs,
        ):
            num_heads = kwargs.pop("total_num_heads", None)
            num_kv_heads = kwargs.pop("total_num_kv_heads", None)
            # Callers pass sizes positionally (e.g. AdalnProj) or by keyword.
            if input_size is None and len(args) >= 1:
                input_size = args[0]
            if output_size is None and len(args) >= 2:
                output_size = args[1]
            del args, kwargs
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.in_features = input_size
            self.out_features = output_size
            self.prefix = prefix
            self.quant_config = quant_config
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            else:
                self.register_parameter("bias", None)
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads

    class _FakeAttention(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    model = h3.MiniMaxH3DiTModel(_od_config(adaln_table_dim=8))

    # The checkpoint key is ``adaln_t_table``. A buffer owned by the submodule
    # would surface as ``adaln_t_table.adaln_t_table`` and never be loaded.
    assert "adaln_t_table" in dict(model.named_buffers())
    assert model.adaln_t_table is not None
    assert model.time_embedder is None


def test_dit_builds_time_embedder_for_native_form(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _FakeLinear(torch.nn.Module):
        def __init__(
            self,
            *args,
            input_size=None,
            output_size=None,
            params_dtype=None,
            bias=True,
            quant_config=None,
            prefix="",
            **kwargs,
        ):
            num_heads = kwargs.pop("total_num_heads", None)
            num_kv_heads = kwargs.pop("total_num_kv_heads", None)
            # Callers pass sizes positionally (e.g. AdalnProj) or by keyword.
            if input_size is None and len(args) >= 1:
                input_size = args[0]
            if output_size is None and len(args) >= 2:
                output_size = args[1]
            del args, kwargs
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.in_features = input_size
            self.out_features = output_size
            self.prefix = prefix
            self.quant_config = quant_config
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            else:
                self.register_parameter("bias", None)
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads

    class _FakeAttention(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    model = h3.MiniMaxH3DiTModel(_od_config())
    assert model.time_embedder is not None
    assert model.adaln_table is None
    assert "adaln_t_table" not in dict(model.named_buffers())


def test_curve_table_is_fp32_whitelisted():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MINIMAX_H3_FP32_BUFFER_NAMES

    assert "adaln_t_table" in MINIMAX_H3_FP32_BUFFER_NAMES


def test_adaln_cache_disabled_for_curve_form(monkeypatch):
    """The prebuilt cache replays ``silu(embedder(t))`` and has no curve input."""
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _FakeLinear(torch.nn.Module):
        def __init__(
            self,
            *args,
            input_size=None,
            output_size=None,
            params_dtype=None,
            bias=True,
            quant_config=None,
            prefix="",
            **kwargs,
        ):
            num_heads = kwargs.pop("total_num_heads", None)
            num_kv_heads = kwargs.pop("total_num_kv_heads", None)
            # Callers pass sizes positionally (e.g. AdalnProj) or by keyword.
            if input_size is None and len(args) >= 1:
                input_size = args[0]
            if output_size is None and len(args) >= 2:
                output_size = args[1]
            del args, kwargs
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.in_features = input_size
            self.out_features = output_size
            self.prefix = prefix
            self.quant_config = quant_config
            self.weight = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(1, dtype=params_dtype))
            else:
                self.register_parameter("bias", None)
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads

    class _FakeAttention(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    model = h3.MiniMaxH3DiTModel(_od_config(adaln_table_dim=8))
    assert model.adaln_cache.max_bytes == 0