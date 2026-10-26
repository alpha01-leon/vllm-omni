# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vllm-omni side of the ref2va parity run: the FULL denoise loop on the
converted int8 partition, tapping the DiT velocity at every step.

The ref2va sibling of omni_multistep.py. Driven by comfyui_reference_ref2va.py's
sigma schedule, its initial noise and -- crucially -- its reference condition
rows, so both trajectories start from an identical state AND condition on
identical material. Each side then free-runs its own: that is deliberate,
because feeding a shared x every step would resynchronise the two
implementations and hide exactly the effect under test.

The reference latents are CONSUMED, not recomputed. The two trees ship
different VAE builds (Comfy-Org fp16/fp32 here, the bf16 partition's own VAE
on the omni side); re-encoding locally would fold a VAE difference into a
number this test reports as quantization error. The reference runner encodes
once and this side reads those latents.

Two ref2va-specific details, both of which the fl2va runner never had to face:

1. The packed layout carries reference rows. This side builds its layout with
   ``minimax_h3_packed_sequence_ref2va_blocks`` and dumps it alongside the
   taps, so compare.py can assert the two sides agree on the reference rows,
   their order and their RoPE position assignments -- not merely on shapes.

2. The condition anchors are noised by each side's OWN production recipe. The
   recipes draw from different RNG shapes (ComfyUI patches then draws at the
   row shape; this repo draws at the latent shape over
   ``max(target_t + cond_frames, latent_t)``), so anchors are not bit-equal by
   construction. Measured difference is ~1.4e-3 relative, two orders of
   magnitude below the 0.15 parity threshold, so each side keeps its own recipe
   instead of pinning a harness detail.

Audio runs on this repo's shift=3 grid rather than ComfyUI's carried video
schedule, as in fl2va, and the script asserts the two parameterisations agree on
the audio row timestep so a schedule change fails loudly instead of showing up
as a fake divergence.

Run with this repo's interpreter:

    python omni_multistep_ref2va.py
"""
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("H3_OMNI_ROOT", "/data/projects/vllm/vllm-omni"))

PART = os.environ.get("H3_PART", "/data/models/MiniMax-H3/Ref2VA_comfyint8/Ref2VA")
CKPT = f"{PART}/transformer"
REF = os.environ.get("H3_REF", "/tmp/h3ref2va/ref.pt")
OUT = os.environ.get("H3_OUT", "/tmp/h3ref2va/omni.pt")

_CFG_CTX = None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------
# Row layout, transcribed from ComfyUI's comfy/ldm/minimax/model.py -- identical
# to the fl2va runner, which already proved these against the layout parity
# unit test (test_layout_parity.py).
#
# Pure tensor reshapes, but importing them would drag in ComfyUI's whole
# package and therefore its interpreter's site-packages. Transcribing keeps
# both sides runnable under one interpreter each and makes the layout contract
# this test depends on visible in one place.
# ---------------------------------------------------------------------------


def patchify_video(latent, patch_size=(1, 2, 2)):
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    """[B*t*h*w, C*pt*ph*pw] -> [B, C, t*pt, h*ph, w*pw]."""
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    """[B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)."""
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    """[ch*T, 32] -> [1, 32, ch, T], inverse of :func:`pack_audio`."""
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def init_dist():
    """Single-rank distributed/TP env + current VllmConfig.

    vLLM's parallel linears read world size, TP size and a global config that
    only the engine normally installs.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29514")

    global _CFG_CTX
    if _CFG_CTX is None:
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CFG_CTX = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device)))
        _CFG_CTX.__enter__()

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)


def build_model():
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    cfg = json.load(open(f"{CKPT}/config.json"))
    qcfg = cfg.pop("quantization_config")
    log(f"quant: method={qcfg['quant_method']} serialized={qcfg['is_checkpoint_int8_serialized']} "
        f"ignored={len(qcfg['ignored_layers'])}")

    quant_config = DiffusionInt8Config(
        is_checkpoint_int8_serialized=bool(qcfg["is_checkpoint_int8_serialized"]),
        activation_scheme=qcfg.get("activation_scheme", "dynamic"),
        ignored_layers=qcfg.get("ignored_layers") or [],
    )
    od = OmniDiffusionConfig(
        tf_model_config=TransformerConfig.from_dict({**cfg, "_class_name": "MiniMaxH3DiTModel"})
    )
    od.cache_config = {"minimax_h3_adaln_cache": False}
    od.quantization_config = quant_config

    init_dist()
    log("building model...")
    model = MiniMaxH3DiTModel(od, quant_config=quant_config, diffusers_weights=False)
    log(f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    from safetensors import safe_open

    t0 = time.time()

    def it():
        for f in sorted(glob.glob(f"{CKPT}/*.safetensors")):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    yield k, h.get_tensor(k)

    loaded = model.load_weights(it())
    log(f"loaded {len(loaded)} tensors in {time.time()-t0:.1f}s")
    model.post_load_weights()

    # The engine runs this before eval(): it rewrites quantized layers into the
    # kernel's layout (the CUTLASS int8 kernel needs its weight transposed).
    for m in model.modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(m)
    log("process_weights_after_loading done")
    return model.eval()


def build_ref_blocks(conditions):
    """The reference block descriptors for the ref2va packed layout.

    Same shapes and order as the reference's ``minimax_refs`` list, in the
    schema ``minimax_h3_packed_sequence_ref2va_blocks`` parses. The video block
    carries its soundtrack (``video_audio``) because this material has one; the
    standalone voice-timbre clip is a separate ``audio`` block.
    """
    blocks = []
    video_shape = conditions["video_shape"]
    rt = int(conditions["audio_t"])
    blocks.append({
        "kind": "video_audio" if rt else "video",
        "ref_audio_t": rt,
        "latent_t": int(video_shape[0]),
        "latent_h": int(video_shape[1]),
        "latent_w": int(video_shape[2]),
    })
    blocks.append({"kind": "audio", "ref_audio_t": int(conditions["standalone_t"])})
    return blocks


def main():
    ref = torch.load(REF, weights_only=False)
    meta = ref["meta"]
    sigmas_v = [float(s) for s in ref["sigmas"]]
    conditions = ref["ref_conditions"]
    log(f"ref meta: { {k: v for k, v in meta.items() if k != 'prompt'} }")
    log(f"ref conditions: video_shape={conditions['video_shape']} "
        f"audio_t={conditions['audio_t']} standalone_t={conditions['standalone_t']}")

    # Video sigma grid from the reference. Audio on this repo's own shift=3
    # grid -- what the server uses -- asserted equal to ComfyUI's internal
    # derivation (t_a = 1 - time_shift_sigma(sigma_v, 12, 3)).
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    audio_shift = float(os.environ.get("H3_AUDIO_SHIFT", "3.0"))
    sigmas_a = minimax_h3_time_shift_sigmas(num_steps=len(sigmas_v) - 1, shift_scale=audio_shift)

    def _tss(s, sv, sa):
        base = s / (sv + s * (1.0 - sv))
        return sa * base / (1.0 + (sa - 1.0) * base)

    expected_a = [_tss(s, 12.0, audio_shift) for s in sigmas_v]
    drift = max(abs(x - y) for x, y in zip(sigmas_a, expected_a))
    log(f"ref  video sigmas: {['%.6f' % s for s in sigmas_v]}")
    log(f"omni audio sigmas: {['%.6f' % s for s in sigmas_a]}")
    log(f"comfy eff. audio:  {['%.6f' % s for s in expected_a]}")
    if drift > 1e-4:
        raise SystemExit(
            f"audio sigma grid mismatch: omni={sigmas_a} vs comfy-derived={expected_a} "
            f"(max drift {drift:.2e}). Refusing to compare trajectories under "
            f"different schedules -- that would be measured as a DiT error."
        )
    log(f"audio schedule agrees with ComfyUI's internal derivation (max drift {drift:.2e})")

    model = build_model()
    device = torch.device("cuda")
    blocks = list(model.blocks)
    block_params = {id(p) for b in blocks for p in b.parameters()}
    for p_ in model.parameters():
        if id(p_) not in block_params:
            p_.data = p_.data.to(device)
    for _, b_ in model.named_buffers():
        b_.data = b_.data.to(device)
    log(f"non-block state resident, vram={torch.cuda.memory_allocated()/2**30:.2f} GiB")

    from vllm_omni.diffusion.offloader.layerwise_backend import _install_layerwise_hook_group
    from vllm_omni.platforms import current_omni_platform

    stream = current_omni_platform.Stream()
    hooks = _install_layerwise_hook_group(blocks, device, stream, pin_memory=False)
    hooks[0].prefetch_layer(non_blocking=False)
    log(f"offload ready, vram={torch.cuda.memory_allocated()/2**30:.2f} GiB")

    noise_v = ref["noise_video"].to(device)
    noise_a = ref["noise_audio"].to(device)
    text_states = ref["text_states"].to(device)
    text_tags = ref["tags"]
    if text_tags is None:
        raise SystemExit("reference has no token tags; the text span would carry tag 1 everywhere")
    latent_t, lat_h, lat_w = noise_v.shape[2], noise_v.shape[3], noise_v.shape[4]
    audio_t = noise_a.shape[-1]

    from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        MINIMAX_H3_IMGVID_COND_TIMESTEP,
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    ref_blocks = build_ref_blocks(conditions)
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=int(text_states.shape[1]),
        latent_t=latent_t,
        latent_h=lat_h,
        latent_w=lat_w,
        audio_t=audio_t,
        audio_channel=2,
        ref_blocks=ref_blocks,
    )
    # The text span's tags come from the reference, exactly as the pipeline does
    # (pipeline_minimax_h3.py:1897-1898): the packed default is tag 1 everywhere
    # except that the encoder's vision-pad positions carry tag 0.
    tags = packed["token_tags"].clone()
    tags[packed["text_pos"]] = text_tags.view(-1).to(tags.dtype)

    branch = MiniMaxH3DenoiseBranch(
        packed=packed, text_embeddings=text_states[0], token_tags=tags, device=device)
    branch.prepare_rope_table(model)
    log(f"branch: seq_len={branch.seq_len} used_len={branch.used_len} "
        f"img_pos={branch.img_pos.shape[0]} audio_pos={branch.audio_pos.shape[0]} "
        f"update={int(branch.update_mask.sum())}/{branch.update_mask.numel()} "
        f"audio_update={int(branch.audio_update_mask.sum())}/{branch.audio_update_mask.numel()}")

    # ---- condition anchors: the reference's CLEAN rows, noised by THIS side's
    # production recipe. Patchify first so the clean rows are in the packed
    # layout the loop expects.
    video_rows = patchify_video(noise_v.to(torch.float32), (1, 2, 2))
    audio_rows = pack_audio(noise_a.to(torch.float32))
    log(f"target rows: video={tuple(video_rows.shape)} audio={tuple(audio_rows.shape)}")

    clean_visual = patchify_video(
        conditions["video_latent"].to(torch.float32), (1, 2, 2))
    visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_visual,
        condition_shapes=[conditions["video_shape"]],
        target_latent_t=latent_t,
        imgvid_cond_num_frames=1,
        seed=meta["seed"],
        noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    # audio anchors, in layout order: the video's soundtrack first, then the
    # standalone clip (encoder_processing.py:816-834 emits them the same way).
    audio_lengths = [int(conditions["audio_t"]), int(conditions["standalone_t"])]
    clean_audio = [pack_audio(conditions[k].to(torch.float32))
                   for k in ("audio_latent", "standalone_latent")
                   if conditions[k] is not None]
    clean_audio = clean_audio[:len(audio_lengths)]
    audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
        torch.cat(clean_audio, dim=0),
        condition_audio_t=audio_lengths,
        seed=meta["seed"],
        noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    log(f"anchors: visual={tuple(visual_anchor.shape)} audio={tuple(audio_anchor.shape)} "
        f"expected_cond_rows={int((~branch.update_mask).sum())} "
        f"expected_audio_ref_rows={int((~branch.audio_update_mask).sum())}")

    taps = []
    raw = []

    def tap_hook(mod, args, kwargs, out):
        # Tap the DiT INPUT as well as its output: that is what makes the
        # free-running trajectory comparable, because both sides are compared
        # at the x they each actually denoised from rather than at an
        # off-by-one from where the hook sits.
        #
        # Only the TARGET rows are unpacked. The model returns every image/audio
        # row and x/audio_x carry every packed row, reference rows included;
        # unpatchifying those into target geometry would produce a tensor of the
        # wrong shape and a comparison that means nothing.
        v_rows, a_rows = out[0], out[1]
        x = kwargs["x"]
        audio_x = kwargs["audio_x"]
        vm = branch.update_mask_dev
        am = branch.audio_update_mask_dev
        raw.append({
            "v_video": unpatchify_video(
                v_rows[vm].detach().float().cpu(), latent_t, lat_h // 2, lat_w // 2, c=24),
            "v_audio": unpack_audio(a_rows[am].detach().float().cpu(), ch=2),
            "x_video": unpatchify_video(
                x[0][branch.img_pos_dev][vm].detach().float().cpu(),
                latent_t, lat_h // 2, lat_w // 2, c=24),
            "x_audio": unpack_audio(
                audio_x[0][branch.audio_pos_dev][am].detach().float().cpu(), ch=2),
            "unique_timesteps": kwargs["unique_timesteps"].detach().float().cpu(),
        })
        return out

    model.register_forward_hook(tap_hook, with_kwargs=True)

    # The loop's initial rows must span EVERY image/audio row of the layout,
    # reference rows included -- not just the target's. The reference rows'
    # values are the noise-augmented anchors, exactly as the production path
    # assembles them (pipeline_minimax_h3.py:1919-1958): place the target noise
    # in the update-masked rows and the anchors in the rest.
    full_video = torch.zeros(branch.img_pos.shape[0], 96, dtype=torch.float32, device=device)
    full_video[branch.update_mask_dev] = video_rows.to(device)
    full_audio = torch.zeros(branch.audio_pos.shape[0], 32, dtype=torch.float32, device=device)
    full_audio[branch.audio_update_mask_dev] = audio_rows.to(device)
    log(f"full rows: video={tuple(full_video.shape)} audio={tuple(full_audio.shape)}")

    log("running denoise loop...")
    t0 = time.time()
    final_v, final_a = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=full_video,
        initial_audio_rows=full_audio,
        keyframe_cond_rows=visual_anchor,
        audio_ref_rows=audio_anchor,
        sigmas_video=sigmas_v,
        sigmas_audio=sigmas_a,
        device=device,
        sampler="euler",
    )
    log(f"loop done in {time.time()-t0:.1f}s, {len(raw)} taps")

    # Select the TARGET rows before unpacking, exactly as the pipeline does
    # (pipeline_minimax_h3.py:2011-2040): the loop returns every row, reference
    # rows included, and unpatchifying the whole thing would silently reshape
    # reference rows into target geometry.
    target_v = unpatchify_video(
        final_v[branch.update_mask_dev].detach().float().cpu(),
        latent_t, lat_h // 2, lat_w // 2, c=24)
    target_a = unpack_audio(
        final_a[branch.audio_update_mask_dev].detach().float().cpu(), ch=2)
    torch.save({
        "sigmas": ref["sigmas"],
        "sigmas_audio": torch.tensor(sigmas_a, dtype=torch.float32),
        "raw": raw,
        "final": [target_v, target_a],
        # Dumped so compare.py can assert the reference rows, their order and
        # their RoPE positions match the reference's -- not just the shapes.
        "layout": {
            "seq_len": int(packed["seq_len"]),
            "used_len": int(packed["cu_seqlens"][1]),
            "position_ids": packed["img_position_ids"].cpu(),
            "img_pos": packed["img_pos"].cpu(),
            "update_mask": packed["update_mask"].cpu(),
            "audio_pos": packed["audio_pos"].cpu(),
            "audio_update_mask": packed["audio_update_mask"].cpu(),
            # The EFFECTIVE tags: packed defaults with the reference's
            # text-span tags written in, which is what the branch actually ran.
            # Saving packed["token_tags"] instead would ship a text span marked
            # all-1 and make compare.py's tag check compare the wrong span.
            "token_tags": tags.cpu(),
        },
        "ref_blocks": ref_blocks,
        # The CLEAN reference rows this side was given, plus the noised anchors
        # it actually built, so compare.py can report how far the two sides'
        # condition recipes (which draw from different RNG shapes) diverge.
        "anchors": {
            "clean_video": clean_visual,
            "video": visual_anchor,
            "clean_audio": torch.cat(clean_audio, dim=0),
            "audio": audio_anchor,
        },
        "meta": meta,
    }, OUT)
    log(f"saved {OUT} ({os.path.getsize(OUT)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
