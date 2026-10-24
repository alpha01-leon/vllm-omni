# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert a Comfy-Org MiniMax-H3 Turbo LoRA export into the Diffusers PEFT
layout the Turbo loader expects.

Comfy-Org publishes the Turbo adapters in ComfyUI's generic LoRA spelling:
the attention QKV projection is one fused ``qkv_proj`` tensor, the FFN uses
``fc1``/``fc2``, and the blocks live under ``diffusion_model.blocks``. The
loader in ``vllm_omni/diffusion/models/minimax_h3/lora.py`` expects the
Diffusers PEFT spelling instead: ``attn.to_q``/``to_k``/``to_v``,
``ff.net.0.proj``/``ff.net.2``, under ``transformer_blocks``, with
``.lora_A.default.weight``/``.lora_B.default.weight`` suffixes. It also
refuses the file NAME that Comfy-Org uses (``..._comfyui_bf16.safetensors``),
because the name carries the sampler contract -- steps, resolution, alpha --
and the ``_comfyui_`` infix would sit between that contract and the suffix
the regex allows.

So the export is not loadable as published, and this script undoes the
re-packing. The source file's own metadata states the fusion rules it
applied, and this script reads and checks them rather than assuming them:

``qkv_fusion``
    ``"block diagonal B; concat A; alpha multiplied by 3"`` -- the fused
    ``lora_B`` is block diagonal across the three projections and the fused
    ``lora_A`` is their row concatenation, and the fused alpha is three times
    the per-projection alpha.

``swi_glu_mapping``
    ``"Diffusers [value;gate] -> ComfyUI [gate;value]"`` -- the fused FFN
    projection stores gate first. The Diffusers side wants value first, and
    the loader swaps the halves back, so this script emits ``[value;gate]``.

The alpha needs a word of its own. The loader takes ONE alpha for the whole
adapter, from the file's metadata, and derives ``scaling = alpha / rank``
(``vllm/lora/lora_weights.py:33``). The fused file's per-tensor alpha is
24.0 = 3 x 8.0, which is the *fused* value; a single global alpha of 24.0
would scale every projection three times too hard. This script therefore
writes the per-projection alpha (8.0) as the global one, which makes the
fused 24.0 split into three consistent 8.0s -- and lands exactly on the
``training_scale`` of 0.0625 that the source file declares.

What this script deliberately does NOT do:

* It does not relax the loader's filename regex. The name encodes the
  sampler's contract, so the fix belongs here rather than in the loader.
* It does not touch the runtime. LoRA is applied in output space on top of
  whatever ``quant_method.apply`` returned, so an int8 partition needs no
  runtime change for an adapter to bind.
* It does not validate the adapter against a reference implementation. It
  proves the re-packing is faithful -- the converted adapter computes the
  same function as the fused one -- and nothing more. Whether the adapter is
  any good is a separate question, and it is the base checkpoint's question.

Run with any interpreter that has torch and safetensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The metadata keys Comfy-Org writes on a Turbo export. They are the
# contract the export was built against; without them this script would be
# guessing at a layout it cannot verify.
_SOURCE_FORMAT_KEY = "source_format"
_TARGET_FORMAT_KEY = "target_format"
_QKV_FUSION_KEY = "qkv_fusion"
_SWI_GLU_MAPPING_KEY = "swi_glu_mapping"

_EXPECTED_SOURCE_FORMAT = "Diffusers PEFT LoRA"
_EXPECTED_TARGET_FORMAT = "ComfyUI generic LoRA"
_EXPECTED_QKV_FUSION = "block diagonal B; concat A; alpha multiplied by 3"
_EXPECTED_SWI_GLU_MAPPING = "Diffusers [value;gate] -> ComfyUI [gate;value]"

# The loader reads this key and rejects any other value
# (vllm_omni/diffusion/models/minimax_h3/lora.py:262-266). Declaring it
# removes the "this file may be a different format" ambiguity entirely.
_OUTPUT_KEY_FORMAT = "minimax-h3-diffusers"
_OUTPUT_ALPHA_KEY = "alpha"

# ComfyUI's block prefixes and the loader's Diffusers-side ones.
_COMFY_BLOCK_PREFIX = "diffusion_model.blocks."
_COMFY_REFINER_PREFIX = "diffusion_model.token_refiner.blocks."
_DIFFUSERS_BLOCK_PREFIX = "transformer_blocks."
_DIFFUSERS_REFINER_PREFIX = "token_refiner.refiner_blocks."

# The ComfyUI-side fused target names.
_COMFY_QKV_TARGET = "attn.qkv_proj"
_COMFY_OUT_TARGET = "attn.out_proj"
_COMFY_FC1_TARGET = "mlp.fc1"
_COMFY_FC2_TARGET = "mlp.fc2"

# The Diffusers-side targets the fused ones expand to.
_QKV_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v")
_OUT_SUFFIX = "attn.to_out.0"
_FC1_SUFFIX = "ff.net.0.proj"
_FC2_SUFFIX = "ff.net.2"

# How many projections the fused QKV packs. Also the divisor for the fused
# alpha.
_QKV_PROJECTIONS = 3

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"
_ALPHA_SUFFIX = ".alpha"
_OUTPUT_A_SUFFIX = ".lora_A.default.weight"
_OUTPUT_B_SUFFIX = ".lora_B.default.weight"

# The loader's filename regex rejects the ComfyUI infix
# (vllm_omni/diffusion/models/minimax_h3/lora.py:38-41), and the infix
# sits between the version and the ``_bf16`` suffix.
_COMFY_INFIX = "_comfyui_"


def _read_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as reader:
        return reader.metadata() or {}


def _check_source_metadata(metadata: dict[str, str], path: Path) -> None:
    """Refuse a file whose declared fusion rules are not the ones we invert.

    Every rule this script applies comes from the source file's metadata. If
    a future export declares different rules -- a different fusion, a
    different half order -- silently applying today's rules would produce a
    plausible-looking but wrong adapter. Failing loudly is the only safe
    response.
    """
    for key, expected in (
        (_SOURCE_FORMAT_KEY, _EXPECTED_SOURCE_FORMAT),
        (_TARGET_FORMAT_KEY, _EXPECTED_TARGET_FORMAT),
        (_QKV_FUSION_KEY, _EXPECTED_QKV_FUSION),
        (_SWI_GLU_MAPPING_KEY, _EXPECTED_SWI_GLU_MAPPING),
    ):
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"{path.name}: metadata {key}={actual!r}, expected {expected!r}; "
                f"this script only inverts the published Comfy-Org Turbo layout"
            )


def _split_qkv(
    a: torch.Tensor,
    b: torch.Tensor,
    rank: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split a fused QKV LoRA into its three projections.

    ``a`` is the row concatenation of the three ``lora_A`` matrices and ``b``
    is the block-diagonal stacking of the three ``lora_B`` matrices, so each
    projection takes its own row band of ``a`` and its own block of ``b``.

    The block-diagonal claim is checked rather than assumed: if the source
    file's ``b`` had off-diagonal mass, the split below would silently drop
    it, and the adapter would compute a different function than the fused
    one while looking perfectly plausible.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"fused qkv lora_A/lora_B must be matrices, got {tuple(a.shape)}/{tuple(b.shape)}")
    if a.shape[0] % _QKV_PROJECTIONS != 0:
        raise ValueError(f"fused qkv lora_A has {a.shape[0]} rows, not divisible by {_QKV_PROJECTIONS}")
    if b.shape[1] % _QKV_PROJECTIONS != 0:
        raise ValueError(f"fused qkv lora_B has {b.shape[1]} columns, not divisible by {_QKV_PROJECTIONS}")
    per_rank = a.shape[0] // _QKV_PROJECTIONS
    out_rows = b.shape[0] // _QKV_PROJECTIONS
    if per_rank != rank:
        raise ValueError(f"fused qkv lora_A has {per_rank} rows per projection, expected rank {rank}")
    if b.shape[1] != per_rank * _QKV_PROJECTIONS:
        raise ValueError(f"fused qkv lora_B has {b.shape[1]} columns, expected {per_rank * _QKV_PROJECTIONS}")

    off_diagonal = 0
    for index in range(_QKV_PROJECTIONS):
        block = b[index * out_rows : (index + 1) * out_rows]
        for other in range(_QKV_PROJECTIONS):
            if other == index:
                continue
            off_diagonal += int((block[:, other * per_rank : (other + 1) * per_rank] != 0).sum())
    if off_diagonal:
        raise ValueError(
            f"fused qkv lora_B is not block diagonal: {off_diagonal} non-zero entries outside the "
            f"three diagonal blocks; the source file does not match its declared qkv_fusion"
        )

    parts = []
    for index in range(_QKV_PROJECTIONS):
        a_part = a[index * per_rank : (index + 1) * per_rank]
        b_part = b[index * out_rows : (index + 1) * out_rows, index * per_rank : (index + 1) * per_rank]
        parts.append((a_part.contiguous(), b_part.contiguous()))
    return parts


def _swap_ffn_halves(b: torch.Tensor) -> torch.Tensor:
    """Turn ComfyUI's ``[gate;value]`` row order into Diffusers' ``[value;gate]``.

    The loader swaps the halves back (``vllm_omni/diffusion/models/minimax_h3/lora.py:208``)
    before handing the tensor to the model's fused gate/up projection, so the
    file has to carry the Diffusers order.
    """
    if b.ndim != 2:
        raise ValueError(f"fused ff lora_B must be a matrix, got {tuple(b.shape)}")
    if b.shape[0] % 2 != 0:
        raise ValueError(f"fused ff lora_B has {b.shape[0]} rows, which is not an even split")
    value, gate = b.chunk(2, dim=0)
    return torch.cat((gate, value), dim=0).contiguous()


def _parse_comfy_name(name: str) -> tuple[str, str, str] | None:
    """Split a ComfyUI tensor name into (block prefix, target, side).

    ``block prefix`` is the Diffusers-side prefix including the block index
    (``transformer_blocks.7.``), ``target`` is the ComfyUI-side fused target
    (``attn.qkv_proj``), and ``side`` is ``"a"`` or ``"b"``. Returns ``None``
    for a name this script does not handle.
    """
    for comfy_prefix, diffusers_prefix in (
        (_COMFY_REFINER_PREFIX, _DIFFUSERS_REFINER_PREFIX),
        (_COMFY_BLOCK_PREFIX, _DIFFUSERS_BLOCK_PREFIX),
    ):
        if not name.startswith(comfy_prefix):
            continue
        rest = name[len(comfy_prefix) :]
        parts = rest.split(".")
        if len(parts) < 4:
            return None
        block_index = parts[0]
        target = ".".join(parts[1:-2])
        tail = parts[-2:]
        if tail == ["lora_A", "weight"]:
            side = "a"
        elif tail == ["lora_B", "weight"]:
            side = "b"
        else:
            return None
        return f"{diffusers_prefix}{block_index}.", target, side
    return None


def _output_name(src: Path, dst: Path | None) -> Path:
    """The destination path, with the ComfyUI infix removed from the name.

    The loader matches the file name against a regex that fixes the
    ``_<version>`` and ``_bf16`` parts of the contract, so the ``_comfyui_``
    infix has to come out for the name to be recognised at all.
    """
    if dst is not None:
        return dst
    if _COMFY_INFIX not in src.stem:
        raise ValueError(
            f"{src.name}: expected a {_COMFY_INFIX!r} infix to strip; pass --dst explicitly to name the output yourself"
        )
    # Strip ONLY the infix word, keeping the separators around it:
    # "..._v1.0_comfyui_bf16" -> "..._v1.0_bf16". Replacing the whole
    # "_comfyui_" token would eat the separator and glue "v1.0" to "bf16".
    return src.with_name(src.stem.replace(_COMFY_INFIX, "_") + src.suffix)


def convert_lora(
    src: Path,
    dst: Path | None = None,
    *,
    rank: int = 128,
) -> dict[str, int]:
    """Convert one Comfy-Org Turbo LoRA export into the Diffusers PEFT layout.

    ``dst`` defaults to the source path with the ComfyUI infix removed.
    Returns a statistics dict with the same shape as the sibling DiT
    converter's: every key is present from the start, so the printed line
    always has the same fields.
    """
    stats = {"blocks": 0, "pairs": 0, "split_qkv": 0, "swapped_ffn": 0, "passthrough": 0}
    metadata = _read_metadata(src)
    _check_source_metadata(metadata, src)

    tensors: dict[str, torch.Tensor] = {}
    alphas: dict[str, set[float]] = {}
    blocks: set[str] = set()

    with safe_open(str(src), framework="pt") as reader:
        for name in reader.keys():
            parsed = _parse_comfy_name(name)
            if parsed is None:
                if name.endswith(_ALPHA_SUFFIX):
                    # An alpha tensor names its target the same way the weight
                    # tensors do, minus the side suffix.
                    stem = name[: -len(_ALPHA_SUFFIX)]
                    alpha_target = _parse_comfy_name(stem + _A_SUFFIX)
                    if alpha_target is None:
                        raise ValueError(f"{src.name}: unexpected tensor {name!r}")
                    alphas.setdefault(alpha_target[1], set()).add(float(reader.get_tensor(name).item()))
                    continue
                raise ValueError(f"{src.name}: unexpected tensor {name!r}")
            prefix, target, side = parsed
            tensor = reader.get_tensor(name)

            if target == _COMFY_QKV_TARGET:
                if side == "b":
                    # Already emitted together with its lora_A above.
                    continue
                # The fused pair is split in one place: read the sibling lora_B
                # here, so the split sees both halves at once and the
                # block-diagonal check has the whole matrix to look at.
                b_name = name[: -len(_A_SUFFIX)] + _B_SUFFIX
                parts = _split_qkv(tensor, reader.get_tensor(b_name), rank)
                for suffix, (a_part, b_part) in zip(_QKV_SUFFIXES, parts, strict=True):
                    tensors[f"{prefix}{suffix}{_OUTPUT_A_SUFFIX}"] = a_part
                    tensors[f"{prefix}{suffix}{_OUTPUT_B_SUFFIX}"] = b_part
                    stats["split_qkv"] += 1
                blocks.add(prefix)
                continue

            if target == _COMFY_FC1_TARGET:
                if side == "b":
                    tensors[f"{prefix}{_FC1_SUFFIX}{_OUTPUT_B_SUFFIX}"] = _swap_ffn_halves(tensor)
                    stats["swapped_ffn"] += 1
                else:
                    tensors[f"{prefix}{_FC1_SUFFIX}{_OUTPUT_A_SUFFIX}"] = tensor
                blocks.add(prefix)
                continue

            if target == _COMFY_OUT_TARGET:
                suffix = _OUT_SUFFIX
            elif target == _COMFY_FC2_TARGET:
                suffix = _FC2_SUFFIX
            else:
                raise ValueError(f"{src.name}: unsupported target {target!r} in {name!r}")

            out_suffix = _OUTPUT_A_SUFFIX if side == "a" else _OUTPUT_B_SUFFIX
            tensors[f"{prefix}{suffix}{out_suffix}"] = tensor
            stats["passthrough"] += 1
            blocks.add(prefix)

    stats["blocks"] = len(blocks)
    stats["pairs"] = len(tensors) // 2

    out_path = _output_name(src, dst)
    _write_output(out_path, tensors, metadata, alphas, rank)
    return stats


def _write_output(
    dst: Path,
    tensors: dict[str, torch.Tensor],
    source_metadata: dict[str, str],
    alphas: dict[str, set[float]],
    rank: int,
) -> None:
    """Write the converted adapter with the metadata the loader reads.

    The loader takes ONE alpha for the whole adapter and derives
    ``scaling = alpha / rank``.

    Only the QKV target's alpha was multiplied by the fusion factor; every
    other target kept the per-projection alpha. So the global alpha is taken
    from the UNFUSED targets, and the fused one is checked against it rather
    than being used -- dividing the fused value by three would be right only
    if the file had no other targets.
    """
    if not tensors:
        raise ValueError("the converted adapter is empty")
    # The alpha of the UNFUSED targets is the adapter's real alpha; the fused
    # QKV alpha is a multiple of it and only serves as a cross-check.
    unfused_alphas = sorted(
        {
            value
            for target, values in alphas.items()
            if target != _COMFY_QKV_TARGET
            for value in values
        }
    )
    fused_alphas = sorted(
        {value for target, values in alphas.items() if target == _COMFY_QKV_TARGET for value in values}
    )
    if not unfused_alphas:
        raise ValueError("the source adapter declares no alpha for an unfused target")
    if len(unfused_alphas) > 1:
        raise ValueError(f"the source adapter declares differing unfused alphas {unfused_alphas}; expected one value")
    alpha = unfused_alphas[0]
    if not (alpha > 0):
        raise ValueError(f"the converted adapter would carry a non-positive alpha {alpha}")
    if fused_alphas:
        expected_fused = alpha * _QKV_PROJECTIONS
        for value in fused_alphas:
            if abs(value - expected_fused) > 1e-6 * max(1.0, abs(expected_fused)):
                raise ValueError(
                    f"the source adapter's fused qkv alpha is {value}, but {expected_fused} was expected "
                    f"from the unfused alpha {alpha} and {_QKV_PROJECTIONS} fused projections"
                )

    metadata: dict[str, Any] = {
        "format": "pt",
        # The loader reads the KEY "key_format" and compares its VALUE against
        # "minimax-h3-diffusers" (lora.py:262-266). Writing the constant as the
        # key instead would leave the format undeclared.
        "key_format": _OUTPUT_KEY_FORMAT,
        _OUTPUT_ALPHA_KEY: str(alpha),
    }
    # Keep the provenance the source file carried, minus the keys that
    # described the SOURCE layout (which no longer describes this file).
    for key in ("training_rank", "training_scale", "base_model", "qkv_fusion", "swi_glu_mapping"):
        if key in source_metadata:
            metadata[key] = source_metadata[key]
    if "training_rank" in metadata and int(metadata["training_rank"]) != rank:
        raise ValueError(
            f"the source adapter declares training_rank={metadata['training_rank']}, but rank {rank} was assumed"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst), metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Comfy-Org Turbo LoRA .safetensors")
    parser.add_argument("--dst", type=Path, default=None, help="output path (default: src with _comfyui_ removed)")
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank (default: 128)")
    parser.add_argument("--dry-run", action="store_true", help="inspect without writing")
    args = parser.parse_args()

    if not args.src.is_file():
        print(f"no such file: {args.src}", file=sys.stderr)
        return 1

    if args.dry_run:
        metadata = _read_metadata(args.src)
        _check_source_metadata(metadata, args.src)
        with safe_open(str(args.src), framework="pt") as reader:
            keys = list(reader.keys())
        print(f"keys={len(keys)} metadata={json.dumps(metadata, sort_keys=True)}")
        print(f"would write {args.dst or _output_name(args.src, None)}")
        return 0

    print(f"converting {args.src} -> {args.dst or _output_name(args.src, None)}")
    stats = convert_lora(args.src, args.dst, rank=args.rank)
    print(f"  {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
