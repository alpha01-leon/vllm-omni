# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert a Comfy-Org MiniMax-H3 DiT export into a vllm-omni FL2VA partition.

The ComfyUI export is fully quantized, which is the only form that fits a 20 GiB
card. Nothing about it is directly loadable: the int8 weights sit in a rotated
(ConvRot) basis, the fused QKV uses ComfyUI's grouped layout, and AdaLN ships as
a precomputed curve rather than a ``time_embedder`` MLP.

This script undoes all three offline so the runtime keeps using the existing
serialized-int8 path (see ``vllm_omni/quantization/int8_config.py``) with no new
quantization scheme and no ConvRot kernel.

Run it with ComfyUI's interpreter -- ``comfy_kitchen`` is a ComfyUI dependency and
is not part of the vllm-omni dependency tree:

    /data/projects/comfyui/ComfyUI/venv/bin/python convert_comfy_checkpoint.py

The text encoder is deliberately not converted: the NVFP4 weights only exist to
shrink 63 GiB to 15 GiB on disk, and 100 of their 350 layers carry ModelOpt AWQ
``pre_quant_scale`` vectors that ``__metadata__`` does not advertise. Point the
converted partition at the BF16 source text encoder instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ComfyUI's fused QKV splits one projection into [3, H, D] blocks, while
# vllm-omni's loader wants interleaved [H, 3, D]. Candidate A in the design doc;
# the loader's own ``_reorder_grouped_qkv_to_qkv`` implements candidate B, so we
# must do this first or the loader will permute our rows a second time.
#
# Whether a tensor is quantized at all is decided by the presence of its
# ``.comfy_quant`` sidecar, never by module names: ``token_refiner`` reuses the
# same names (``.attn.qkv_proj``, ``.mlp.fc1``, ...) as the DiT blocks but stays
# BF16.

def unquantized_layers(config: dict[str, Any]) -> list[str]:
    """Fully-qualified prefixes the checkpoint leaves unquantized.

    ``adaln_proj`` matters most: the Comfy-Org export ships those weights as
    plain F16, so routing them through the serialized-int8 path allocates a
    ``weight_scale`` nothing ever fills and drives the projections with
    uninitialized scales.

    ``token_refiner`` is included for the same reason under a different
    spelling: the converter emits it as BF16 pass-through, but the model builds
    it with the DiT's quant config, so its projections would otherwise be
    quantized from unquantized checkpoints.

    Names are expanded rather than left as substrings because the quant config's
    ``is_layer_skipped`` call uses the default exact-match mode.
    """
    layers = ["video_patch_proj", "audio_patch_proj", "condition_proj"]
    for block in range(int(config["num_layers"])):
        layers.append(f"blocks.{block}.adaln_proj.linear")
    layers.append("final_layer.adaln_proj.linear")
    for block in range(int(config["token_refiner_num_layers"])):
        for proj in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            layers.append(f"token_refiner.blocks.{block}.{proj}")
    return layers


def _json_key(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True)


def _load_quant_spec(reader: Any, name: str) -> dict[str, Any]:
    """Decode one ``.comfy_quant`` sidecar blob."""
    return json.loads(bytes(reader.get_tensor(f"{name}.comfy_quant")).decode())


def _validate_spec(spec: dict[str, Any], name: str) -> int:
    """Check the export matches the single scheme this converter implements."""
    if spec.get("format") != "int8_tensorwise":
        raise ValueError(f"{name}: unsupported quant format {spec.get('format')!r}")
    if not spec.get("convrot"):
        raise ValueError(f"{name}: expected a convrot (rotated basis) export")
    group = int(spec["convrot_groupsize"])
    if group & (group - 1) or group < 4:
        raise ValueError(f"{name}: convrot group size {group} is not a power of four")
    return group


def _dequantize_convrot(q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """Undo ConvRot: dequantize per-row, then rotate back to the original basis."""
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight

    return dequantize_int8_convrot_weight(q.float(), scale.float(), group)


def _requantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8, matching ``[N, 1]`` fp32 scale layout.

    Per-channel (not per-group) is what vllm-omni's serialized-int8 loader
    allocates, and it is what ``int8_tensorwise`` already meant in the export:
    the convrot ``group_size`` sizes the Hadamard rotation, not the quantization.
    """
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _reorder_qkv(
    w: torch.Tensor,
    scale: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ComfyUI [3, H, D] -> interleaved [H, 3, D] rows, with scales kept in step.

    ``scale`` may be None for BF16 passthrough weights, which have no scales.
    """
    if w.shape[0] != 3 * num_heads * head_dim:
        raise ValueError(f"qkv rows {w.shape[0]} != 3*{num_heads}*{head_dim}")
    out_cols = w.shape[1]
    # [3, H, D, K] -> [H, 3, D, K] -> [3*H*D, K]
    weight = w.reshape(3, num_heads, head_dim, out_cols).permute(1, 0, 2, 3).reshape(3 * num_heads * head_dim, out_cols)
    if scale is None:
        return weight.contiguous(), None
    # Scale rows follow the same permutation; each head contributed one q, one k
    # and one v row, so split per head and re-emit in q/k/v order.
    per_head = scale.reshape(num_heads, 3 * head_dim)
    parts = per_head.split([head_dim, head_dim, head_dim], dim=1)
    reordered = torch.cat([p.reshape(-1) for p in parts]).reshape(-1, 1)
    return weight.contiguous(), reordered.contiguous()


def convert_dit(
    src: Path,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Convert one Comfy-Org DiT safetensors file into vllm-omni layout."""
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])

    tensors: dict[str, torch.Tensor] = {}
    stats = {"quantized": 0, "reordered": 0, "reordered_bf16": 0, "passthrough": 0}

    with safe_open(str(src), framework="pt") as reader:
        keys = list(reader.keys())
        # The presence of a `.comfy_quant` sidecar is the only reliable signal
        # that a tensor is quantized: `token_refiner` uses the same module names
        # (`.attn.qkv_proj`, `.mlp.fc1`, ...) as the DiT blocks but ships BF16.
        quantized_modules = {k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")}
        has_table = "adaln_t_table" in keys
        if has_table:
            table = reader.get_tensor("adaln_t_table")
            # Curve-form AdaLN: no time_embedder at all, and every adaln_proj
            # consumes these columns directly (no SiLU).
            tensors["adaln_t_table"] = table.to(torch.float32).contiguous()
            width = int(table.shape[1])

        for name in keys:
            if name.endswith(".comfy_quant") or name == "adaln_t_table":
                continue
            module = name
            for suffix in (".weight", ".bias", ".weight_scale"):
                if module.endswith(suffix):
                    module = module[: -len(suffix)]
                    break

            if module in quantized_modules:
                # `keys` also yields the `.weight_scale` entry, so count the
                # layer once rather than once per tensor.
                if not name.endswith(".weight_scale"):
                    stats["quantized"] += 1
                spec = _load_quant_spec(reader, module)
                group = _validate_spec(spec, module)
                w = _dequantize_convrot(reader.get_tensor(f"{module}.weight"), reader.get_tensor(f"{module}.weight_scale"), group)
                if module.endswith(".attn.qkv_proj"):
                    w, scale = _reorder_qkv(w, reader.get_tensor(f"{module}.weight_scale").float(), num_heads, head_dim)
                    if not name.endswith(".weight_scale"):
                        stats["reordered"] += 1
                else:
                    scale = reader.get_tensor(f"{module}.weight_scale").float()
                q, s = _requantize_int8(w)
                tensors[f"{module}.weight"] = q
                tensors[f"{module}.weight_scale"] = s
                continue

            tensor = reader.get_tensor(name)
            # The BF16 source uses vllm's interleaved [H, 3, D] QKV layout while
            # ComfyUI stores every fused QKV as [3, H, D]. The quantized DiT
            # blocks are reordered above, but token_refiner is BF16 and reaches
            # here as passthrough, so it needs the same permutation.
            if name.endswith(".attn.qkv_proj.weight"):
                tensor = _reorder_qkv(tensor.float(), None, num_heads, head_dim)[0].to(tensor.dtype)
                stats["reordered"] += 1
            tensors[name] = tensor.contiguous()
            stats["passthrough"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    return stats


def write_config(config: dict[str, Any], has_table: bool, table_width: int, out: Path) -> None:
    """Emit the DiT config declaring curve AdaLN and serialized int8."""
    new = dict(config)
    new["_class_name"] = "MiniMaxH3DiTModel"
    if has_table:
        # Consumed by MiniMaxH3DiTArchConfig.from_mapping; selecting the curve
        # form is what makes the model build MiniMaxH3AdalnTable instead of the
        # time_embedder, and sizes every adaln_proj input to the table width.
        new["adaln_table_dim"] = table_width
        new.pop("time_embed_dim", None)
    new["quantization_config"] = {
        "quant_method": "int8",
        # Must be explicit. Auto-detection routes through
        # ``build_quant_config(method, **kwargs)``, which splats the dict into
        # ``DiffusionInt8Config(**kw)`` and never runs ``from_config`` -- so
        # without this flag ``is_checkpoint_int8_serialized`` stays False, the
        # loader picks Int8OnlineLinearMethod, and every ``weight_scale`` is
        # dropped as an unknown tensor.
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "dynamic",
        # Only the 200 fused DiT projections are serialized-int8; AdaLN and
        # patch projections keep their checkpoint dtypes.
        "ignored_layers": unquantized_layers(config),
    }
    out.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")


def link_shared_components(src_root: Path, out_root: Path) -> None:
    """Point the partition at the BF16 source for everything not converted.

    Only the DiT is converted. The text encoder keeps its BF16 weights because
    the NVFP4 export carries ModelOpt AWQ ``pre_quant_scale`` vectors that
    ``__metadata__`` does not advertise (see the module docstring), and the VAEs
    are unquantized in both trees.
    """
    for name in ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae"):
        src = src_root / name
        if not src.exists():
            continue
        dst = out_root / name
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(src, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Comfy-Org diffusion_models/*.safetensors")
    parser.add_argument("--bf16-root", type=Path, required=True, help="BF16 source partition, for config + text_encoder")
    parser.add_argument("--out", type=Path, required=True, help="destination partition directory")
    parser.add_argument("--dry-run", action="store_true", help="inspect without writing")
    args = parser.parse_args()

    source_config = json.loads((args.bf16_root / "transformer" / "config.json").read_text(encoding="utf-8"))

    if args.dry_run:
        with safe_open(str(args.src), framework="pt") as reader:
            keys = list(reader.keys())
            quant = [k for k in keys if k.endswith(".comfy_quant")]
            specs = {_json_key(_load_quant_spec(reader, k.removesuffix(".comfy_quant"))) for k in quant}
            print(f"keys={len(keys)} quantized={len(quant)}")
            print(f"distinct specs={specs}")
            print(f"curve table: {'adaln_t_table' in keys}")
        return 0

    print(f"converting {args.src} -> {args.out}")
    # The pipeline resolves ``<model_root>/FL2VA`` for a partition directory, so
    # --out is the model root and the partition lives one level below it.
    partition = args.out / args.bf16_root.name
    stats = convert_dit(args.src, partition / "transformer", source_config)
    print(f"  {stats}")

    table_path = partition / "transformer" / "diffusion_pytorch_model.safetensors"
    with safe_open(str(table_path), framework="pt") as reader:
        keys = list(reader.keys())
        width = int(reader.get_slice("adaln_t_table").get_shape()[1]) if "adaln_t_table" in keys else 0
    write_config(source_config, bool(width), width, partition / "transformer" / "config.json")
    link_shared_components(args.bf16_root, partition)

    shutil.copy2(args.bf16_root / "model_index.json", partition / "model_index.json")
    print(f"wrote {partition}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
