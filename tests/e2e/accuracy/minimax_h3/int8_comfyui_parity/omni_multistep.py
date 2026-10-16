# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vllm-omni side of the multi-step comparison: run the FULL denoise loop on the
converted int8 partition, tapping the DiT velocity at every step.

Driven by comfyui_reference.py's sigma schedule and the same initial noise, so
both trajectories start from an identical state. Each side then free-runs its
own: that is deliberate, because feeding a shared x every step would
resynchronise the two implementations and hide exactly the effect under test
(does the residual requantization error compound along the trajectory?).

This goes through minimax_h3_denoise_loop -- the function the server actually
calls -- rather than a hand-rolled loop, so the code under test is the
production path.

Audio runs on this repo's own shift=3 grid rather than ComfyUI's carried video
schedule. That is the correct default, verified by measurement rather than by
argument: the final audio latent lands at L2rel 0.068 on the shift=3 grid vs
0.516 on the video grid, and its rms matches the reference's (0.3181 vs 0.3175).
The script asserts the two parameterisations agree on the audio row timestep,
so a schedule change here fails loudly instead of showing up as a fake
divergence.

Run with this repo's interpreter:

    python omni_multistep.py
"""
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("H3_OMNI_ROOT", "/data/projects/vllm/vllm-omni"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


PART = os.environ.get("H3_PART", "/data/models/MiniMax-H3/FL2VA_confyint8/FL2VA")
CKPT = f"{PART}/transformer"
REF = os.environ.get("H3_REF", "/tmp/h3ref/ref_ms_448x256_s8.pt")
OUT = os.environ.get("H3_OUT", "/tmp/h3ref/omni_ms_448x256_s8.pt")

_CFG_CTX = None


# ---------------------------------------------------------------------------
# Row layout, transcribed from ComfyUI's comfy/ldm/minimax/model.py.
#
# These four are pure tensor reshapes, but importing them would drag in ComfyUI's
# whole package (and therefore its interpreter's site-packages, including modules
# this repo does not have). Transcribing them keeps both sides of the comparison
# runnable under a single interpreter each, and makes the layout contract this
# test depends on visible in one place.
#
# parity against ComfyUI is asserted by the unit test, so a drift here fails
# loudly instead of silently reshaping rows into the wrong order.
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
    os.environ.setdefault("MASTER_PORT", "29513")

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
    from vllm_omni.quantization.int8_config import DiffusionInt8Config
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

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


def main():
    ref = torch.load(REF, weights_only=False)
    meta = ref["meta"]
    log(f"ref meta: {meta}")
    sigmas_v = [float(s) for s in ref["sigmas"]]

    # Take the VIDEO grid from the reference. For AUDIO, take omni's own
    # shift=3 grid -- which is what the server uses -- and assert it equals what
    # ComfyUI's DiT actually ran the audio rows at.
    #
    # ComfyUI carries both streams on one video-sigma schedule and converts
    # inside the model: t_a = 1 - time_shift_sigma(sigma_v, 12, 3)
    # (comfy/ldm/minimax/model.py:630). Feeding omni the video grid for audio
    # would run its audio rows at the wrong timestep -- that is a harness bug,
    # not a model defect, and it looks exactly like a broken audio path.
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    audio_shift = float(os.environ.get("H3_AUDIO_SHIFT", "3.0"))
    sigmas_a = minimax_h3_time_shift_sigmas(num_steps=len(sigmas_v) - 1, shift_scale=audio_shift)

    def _tss(s, sv, sa):
        base = s / (sv + s * (1.0 - sv))
        return sa * base / (1.0 + (sa - 1.0) * base)

    expected_a = [_tss(s, 12.0, audio_shift) for s in sigmas_v]
    drift = max(abs(x - y) for x, y in zip(sigmas_a, expected_a))
    log(f"ref  video sigmas: {['%.6f' % s for s in sigmas_v]}")
    log(f"omni video sigmas: {['%.6f' % s for s in sigmas_v]}")
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
    latent_t, lat_h, lat_w = noise_v.shape[2], noise_v.shape[3], noise_v.shape[4]
    audio_t = noise_a.shape[-1]

    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )

    packed = minimax_h3_packed_sequence(
        text_len=text_states.shape[1], latent_t=latent_t, latent_h=lat_h, latent_w=lat_w,
        audio_t=audio_t, audio_channel=2, include_keyframe_cond=False)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed, text_embeddings=text_states[0],
        token_tags=packed["token_tags"], device=device)
    branch.prepare_rope_table(model)
    log(f"branch: seq_len={branch.seq_len} used_len={branch.used_len}")

    video_rows = patchify_video(noise_v.to(torch.float32), (1, 2, 2))
    audio_rows = pack_audio(noise_a.to(torch.float32))
    log(f"rows: video={tuple(video_rows.shape)} audio={tuple(audio_rows.shape)}")

    taps = []

    def on_step(step, v_rows, a_rows):
        # Same latent-space conversion as the velocity taps, so these compare
        # against the reference's per-step model input directly.
        taps.append({
            "video": unpatchify_video(
                v_rows.detach().float().cpu(), latent_t, lat_h // 2, lat_w // 2, c=24),
            "audio": unpack_audio(a_rows.detach().float().cpu(), ch=2),
        })

    # Tap the raw velocity the loop feeds the solver. Hooking the model keeps
    # this out of the loop body, so the loop under test is unmodified.
    #
    # Convert to the latent space ComfyUI reports its velocity in: the DiT works
    # in packed rows, and unpatchify/unpack are the exact inverses of the
    # patchify/pack above, so this is a layout change and not a computation.
    raw = []

    def tap_hook(mod, args, kwargs, out):
        v_rows, a_rows = out[0], out[1]
        # Tap the DiT INPUT too, not just the output. That is what makes the
        # free-running trajectory comparable: both sides are compared at the
        # x they each actually denoised from, so a drop here is real drift
        # rather than an off-by-one from where the hook was placed.
        x = kwargs["x"]
        audio_x = kwargs["audio_x"]
        raw.append({
            "v_video": unpatchify_video(
                v_rows.detach().float().cpu(), latent_t, lat_h // 2, lat_w // 2, c=24),
            "v_audio": unpack_audio(a_rows.detach().float().cpu(), ch=2),
            "x_video": unpatchify_video(
                x[0][branch.img_pos_dev].detach().float().cpu(),
                latent_t, lat_h // 2, lat_w // 2, c=24),
            "x_audio": unpack_audio(
                audio_x[0][branch.audio_pos_dev].detach().float().cpu(), ch=2),
            "unique_timesteps": kwargs["unique_timesteps"].detach().float().cpu(),
        })
        return out

    model.register_forward_hook(tap_hook, with_kwargs=True)

    log("running denoise loop...")
    t0 = time.time()
    final_v, final_a = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video_rows,
        initial_audio_rows=audio_rows,
        keyframe_cond_rows=None,
        audio_ref_rows=None,
        sigmas_video=sigmas_v,
        sigmas_audio=sigmas_a,
        device=device,
        on_step=on_step,
        sampler="euler",
    )
    log(f"loop done in {time.time()-t0:.1f}s, {len(taps)} steps, {len(raw)} taps")

    out_v = unpatchify_video(final_v, latent_t, lat_h // 2, lat_w // 2, c=24)
    out_a = unpack_audio(final_a.float().cpu(), ch=2)
    torch.save({
        "sigmas": ref["sigmas"],
        "sigmas_audio": torch.tensor(sigmas_a, dtype=torch.float32),
        "taps": taps,
        "raw": raw,
        "final": [out_v, out_a],
        "meta": meta,
    }, OUT)
    log(f"saved {OUT} ({os.path.getsize(OUT)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
