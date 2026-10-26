# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ComfyUI reference: full N-step ref2va H3 sampling, dumping the packed layout,
the MODEL INPUT and the raw VELOCITY at every step.

The ref2va sibling of comfyui_reference.py. Everything ComfyUI owns is reached
through its public path -- load_diffusion_model -> ModelPatcher -> CFGGuider ->
its own ModelSamplingAV -> MiniMaxH3Model.forward -- so the schedule, the audio
carry and the CFG arithmetic are upstream's, not ours.

Three things make this more than a copy of the fl2va runner:

1. Reference preprocessing goes through ComfyUI's OWN helpers
   (``_resize``, ``_encode_ref_audio``, ``adapt_canvas``) and the tokenizer
   goes through ``tokenize_with_weights(..., minimax_ref_items=...)``, so the
   reference canvas, the 17k+5 frame grid and the 2 fps Qwen presentation are
   upstream's. Re-deriving them is exactly how a harness ends up comparing two
   different resize policies and calling the difference a model bug.

2. Reference condition rows ride in the packed sequence, so a matching latent
   shape does NOT make the two sides comparable. The layout is captured with
   the taps and compare.py asserts the ref rows, their order and their RoPE
   position assignments agree. Same shapes, different condition = a harness
   difference, not a quantization difference. Verified for this pair: the two
   layouts agree to 7e-15 (float64 summation order only).

3. The VAEs are ComfyUI's own Comfy-Org builds, which differ from the bf16
   partition's VAEs in this repo. The reference latents are written to the
   payload so the omni runner consumes THE SAME latents; otherwise a VAE
   difference would be measured as a quantization difference.

Conventions the consumer must undo -- same as fl2va, see compare_ref2va.py:
  * forward returns -v, so a faithful match is cos ~= -1.0;
  * audio is CARRIED on the video sigma schedule, not its own.

Resolution: 448x256 / latent_t=12, the shape the fl2va parity uses, because it
fits in ~20 GiB. The real-output shape (384x672, latent_t 37) OOMs the reference
at step 0. Both sides MUST use the same shape to compare.

Run with the ComfyUI interpreter, not this repo's:

    /path/to/ComfyUI/venv/bin/python comfyui_reference_ref2va.py
"""
import gc
import glob
import os
import sys
import time
from pathlib import Path

import torch

COMFYUI_ROOT = os.environ.get("H3_COMFYUI_ROOT", "/data/projects/comfyui/ComfyUI")
sys.path.insert(0, COMFYUI_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy.conds  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.text_encoders.hunyuan_video  # noqa: E402
import comfy.text_encoders.minimax as cte  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.minimax import model as h3model  # noqa: E402

import ref2va_assets as assets  # noqa: E402
from comfy_extras import nodes_minimax_h3 as nmh3  # noqa: E402

COMFY = os.environ.get("H3_COMFY_ROOT", "/data/models/Comfy-Org/MiniMax-H3")
DIT = os.environ.get(
    "H3_COMFY_DIT",
    f"{COMFY}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)
TE = os.environ.get("H3_TEXT_ENCODER_ROOT", "/data/models/MiniMax-H3/Ref2VA")
# ComfyUI's own text encoder for H3, in ComfyUI's native layout. This is what a
# real ComfyUI workflow loads, and it is the DEFAULT for that reason.
#
# Do not point this at the bf16 partition's text_encoder directory: ComfyUI's
# detect_te_model (comfy/sd.py:1677-1679) identifies H3 by
# "visual.deepstack_merger_list.0.norm.weight" + "model.layers.49...", and the
# partition's TE stores "model.visual.*", so detection falls through to
# TEModel.QWEN3VL_8B -- the wrong model, silently. The Comfy-Org file stores
# "visual.*" and detects as QWEN3VL_32B correctly.
#
# The two checkpoints are not interchangeable numerically: they differ by
# L2rel 0.047 in the resulting text states (the Comfy-Org one is NVFP4-quantized,
# the partition's is bf16). Since BOTH sides of the parity comparison consume
# the SAME text_states -- this runner writes them into the payload and the omni
# runner reads them back -- the choice does not affect what the comparison
# measures. It only decides which conditioning the reference is conditioned on.
COMFY_TE = os.environ.get(
    "H3_COMFY_TEXT_ENCODER",
    f"{COMFY}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
)
VIDEO_VAE = os.environ.get("H3_VIDEO_VAE", f"{COMFY}/vae/minimax_h3_video_vae_fp16.safetensors")
AUDIO_VAE = os.environ.get("H3_AUDIO_VAE", f"{COMFY}/vae/minimax_h3_audio_vae_fp32.safetensors")

OUT = os.environ.get("OUT", "/tmp/h3ref2va/ref.pt")
W = int(os.environ.get("W", "448"))
H = int(os.environ.get("H", "256"))
STEPS = int(os.environ.get("STEPS", "8"))
LATENT_T = int(os.environ.get("LATENT_T", "0"))
AUDIO_T = int(os.environ.get("AUDIO_T", "0"))
SEED = int(os.environ.get("SEED", "1234"))
# Reference-video canvas cap. 0 = the model's production 768 short edge.
#
# At 768 the reference video alone contributes ~12k packed rows, making the
# ref2va sequence ~4.8x the fl2va one -- which OOMs a 20 GiB card at step 0,
# not because of the DiT's weights but because attention activations scale with
# sequence length. Parity therefore runs the reference at a reduced canvas.
# Both sides take the SAME value (the omni side reads it out of this payload's
# meta), so this changes the reference's resolution and nothing else.
REF_SHORT_EDGE = int(os.environ.get("H3_REF_SHORT_EDGE", "256"))
DTYPE = torch.bfloat16


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_cuda():
    gc.collect()
    mm.unload_all_models()
    torch.cuda.empty_cache()


def load_vae_wrapper(path):
    """Load a MiniMax H3 VAE as a comfy.sd.VAE, i.e. exactly what a node gets.

    The wrapper matters: it owns the [0,1] -> [-1,1] conversion, the latent
    frame padding for non-17k+5 inputs, and the detection branch that picks
    the model class. Calling the raw module would skip all three.
    """
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    log(f"VAE {os.path.basename(path)}: latent_dim={vae.latent_dim} "
        f"channels={vae.latent_channels} dtype={vae.vae_dtype}")
    return vae


def install_taps(patcher):
    """Per-step taps on the raw DiT velocity plus its input, and the layout.

    Hooked at MiniMaxH3Model.forward (the wrapper) so the tap sees exactly what
    the sampler's x0 conversion consumes -- the same tensors vllm-omni's
    model() returns.
    """
    dm = patcher.model.diffusion_model
    taps = {"__step__": []}

    def fwd_hook(mod, args, kwargs, out):
        payload = kwargs.get("minimax_payload") or {}
        layout = payload.get("layout")
        taps["__step__"].append({
            "x_in": [None if s is None else s.detach().float().cpu() for s in args[0]],
            "timestep": args[1].detach().float().cpu(),
            "v": [None if s is None else s.detach().float().cpu() for s in out],
            "layout": None if layout is None else {
                "seq_len": layout.seq_len,
                "position_ids": layout.position_ids.detach().cpu(),
                "img_pos": layout.img_pos.detach().cpu(),
                "img_update": layout.img_update.detach().cpu(),
                "audio_pos": layout.audio_pos.detach().cpu(),
                "audio_update": layout.audio_update.detach().cpu(),
                "segments": list(layout.segments),
            },
            "payload": {
                k: payload.get(k)
                for k in ("seed", "visual_cond_noise_aug", "audio_cond_noise_aug")
            },
        })
        return out

    dm.register_forward_hook(fwd_hook, with_kwargs=True)
    return taps


def cached_video_encode(cache_dir, cache_key, compute):
    """Memoize the reference video encode to ``cache_dir``.

    The encode takes ~30 min on CPU at the 1344x768 reference canvas, which is
    the cost of doing this correctly on a 20 GiB card rather than a one-off
    mistake. Re-running the harness -- after a bug fix on a LATER stage, say --
    should not pay it again, and a cached latent is bit-identical to a fresh
    one because the encoder is deterministic. The key includes the input shape
    and the VAE path so a different material or a different VAE cannot silently
    reuse a stale file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"video_latent_{cache_key}.pt"
    if path.exists():
        log(f"video latent cache hit {path}")
        return torch.load(path, weights_only=False)
    out = compute()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(out, tmp)
    os.replace(tmp, path)
    log(f"video latent cached to {path}")
    return out


def encode_video_vae(vae, frames, label):
    """Encode a [T,H,W,C] reference batch, falling back to CPU on OOM.

    The reference canvas is 1344x768 (adapt_canvas). At that size the encoder's
    activations plus the 5.2 GiB fp16 VAE exceed a 20 GiB card -- both the
    untiled path and ComfyUI's own internal tiling OOM at step 0 of the first
    downsample block. A real workflow would get a bigger card or a smaller
    reference; this parity run has neither, so the encode runs on CPU.

    Two details make the CPU path actually work, and both are ComfyUI-specific:

      * ``MiniMaxH3VideoVAE.encode`` is called directly rather than through the
        ``comfy.sd.VAE`` wrapper. The wrapper pins pixels to ``vae.device``,
        which is CUDA, while the model-level ``tiled_encode`` slices tiles out
        of the tensor it is handed -- so a CPU model fed CUDA tiles dies on a
        device mismatch inside ``slow_conv3d_forward``. Calling the model with
        an explicit CPU device sidesteps that. The wrapper's only other jobs
        here are the [0,1] -> [-1,1] conversion and latent padding, both of
        which are reproduced below.
      * ``frames`` are padded to a whole 17-frame clip first. The frame count
        reaching a non-17k+5 input gets clipped inside ``encode_temporal``,
        which silently drops frames; the reference here is already on the grid,
        but padding makes the invariant explicit instead of incidental.

    This is a device substitution, not a fidelity compromise: the encoder has
    no randomness (posterior mean, no sampling), so the latents are a
    deterministic function of the weights and the pixels. Both sides consume
    the SAME latents from this call regardless of where it ran. It is slow
    (minutes), which is why it happens before the DiT is loaded.
    """
    device = mm.get_torch_device()
    try:
        out = vae.encode(frames)
        log(f"{label}: encoded {tuple(frames.shape)} -> {tuple(out.shape)} on {device}")
        return out
    except torch.OutOfMemoryError:
        log(f"{label}: {tuple(frames.shape)} OOM on {device}, falling back to CPU")
        free_cuda()

    vae.patcher.unpatch_model(torch.device("cpu"), unpatch_weights=True)
    vae.patcher.model.to("cpu")
    mm.soft_empty_cache()

    # Reproduce comfy.sd.VAE.encode's preprocessing exactly (sd.py:1362-1387),
    # which is TWO transposes, not one: [T,H,W,C] -> movedim(-1,1) -> [T,C,H,W]
    # -> movedim(1,0).unsqueeze(0) -> [1,C,T,H,W]. Doing only the first leaves a
    # 4-D [C,T,H,W] and _normalize_pixels' [1,3,1,1,1] buffer then fails to
    # broadcast. Pixels go to [-1,1] because _normalize_pixels maps that onto
    # ImageNet norm.
    #
    # Frame padding is deliberately NOT done here: encode_temporal already pads
    # each clip to clip_length internally (vae.py:550-552), and prepare_-
    # reference_video has the count on the 17k+5 grid anyway.
    model = vae.first_stage_model
    samples = frames[..., :3].movedim(-1, 1).movedim(1, 0).unsqueeze(0).cpu()
    samples = (samples * 2.0 - 1.0).to(vae.vae_dtype)
    # fp16 conv on CPU is emulated and roughly an order of magnitude slower
    # than fp32, which matters when this encode is the run's long pole. The
    # weights are cast to match, otherwise the conv sees fp16 weights against
    # an fp32 input.
    vae.patcher.model.to(torch.float32)
    samples = samples.to(torch.float32)
    assert samples.ndim == 5 and samples.shape[1] == 3, f"bad layout {tuple(samples.shape)}"

    t0 = time.time()
    # no_grad matters a lot here: ComfyUI's wrapper normally runs under the
    # sampler's inference_mode, but calling the model directly leaves autograd
    # live, which keeps every activation alive and turns a minutes-long encode
    # into an hours-long one. The encode has no gradients to compute.
    with torch.no_grad():
        out = model.encode(samples, device=torch.device("cpu"))
    log(f"{label}: encoded {tuple(frames.shape)} -> {tuple(out.shape)} on CPU "
        f"in {time.time()-t0:.0f}s")
    return out


def configure_cpu_threads():
    """Give the CPU fallback encode real parallelism.

    torch defaults to one thread per core divided against an inter-op pool,
    which leaves this single-threaded conv chain at ~110% CPU. The encode is
    minutes of pure conv, so this is the difference between a tolerable run and
    an intolerable one. Must be set before any tensor work.
    """
    import os

    threads = int(os.environ.get("H3_CPU_THREADS", "0")) or min(16, os.cpu_count() or 8)
    torch.set_num_threads(threads)
    log(f"cpu threads: {threads}")


def encode_audio_vae(vae, waveform, sample_rate, label):
    """Encode one reference waveform to [1,32,2,T], at 32 kHz.

    Calls the model directly rather than ``comfy.sd.VAE.encode`` because that
    wrapper's ``memory_used_encode`` lambda indexes ``shape[2]`` as a sample
    count, which only holds for the 1-D waveform layout -- it raises IndexError
    on the [B,2,L] tensor this node passes. ``_encode_ref_audio`` (the node's
    own helper) only touches ``audio_vae.encode``, so calling the model here is
    the same call the node makes, minus the broken memory estimate.

    The 32 kHz resample is the node's (``getattr(audio_vae, "audio_sample_rate",
    32000)``), reproduced verbatim.
    """
    import torchaudio

    with torch.no_grad():
        vae_sr = getattr(vae, "audio_sample_rate", 32000)
        wave = waveform.float()
        if wave.ndim == 1:
            wave = wave[None]
        if int(sample_rate) != int(vae_sr):
            wave = torchaudio.functional.resample(wave, int(sample_rate), int(vae_sr))
        # audio_vae.encode unpacks waveform as (B, S, length) and processes the S
        # stereo rows independently. The node's `waveform[:1].movedim(1, -1)` is
        # a no-op for the [1, 2, L] it receives here, so pass that layout
        # straight through rather than reshaping it away.
        if wave.ndim == 2:
            wave = wave.unsqueeze(0)
        z = vae.encode(wave[:1])
    log(f"{label}: encoded {tuple(waveform.shape)}@{sample_rate} -> {tuple(z.shape)} "
        f"(rt={z.shape[-1]})")
    return z, int(z.shape[-1])


def main():
    torch.manual_seed(SEED)
    configure_cpu_threads()
    # ComfyUI reports what it actually loaded vs offloaded at INFO (the
    # "loaded partially ... MB loaded ... lowvram patches" line). Without this
    # the only signal is an OOM much later, which does not distinguish "offloaded
    # but the sequence is too big" from "never offloaded at all".
    import logging

    # ComfyUI's offload accounting is a plain logging.info on the root logger
    # ("loaded partially; ... MB loaded, ... lowvram patches"), emitted at load
    # time. It is the only way to tell "offloaded but the sequence is too big"
    # apart from "never offloaded at all", and the OOM that follows either way
    # looks identical. ComfyUI's own import path configures the root logger, so
    # basicConfig alone is a no-op here -- set the level directly.
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(logging.INFO)
        for handler in root.handlers:
            handler.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="[comfy] %(message)s")

    # ComfyUI decides vram_state at IMPORT time from its CLI args
    # (model_management.py:573-591), so assigning mm.vram_state afterwards does
    # not survive -- the sampler's per-step load_models_gpu() sees ~19.7 GB
    # "free" and pulls the entire 19.5 GiB DiT back onto the card, which OOMs
    # immediately. Setting args.lowvram makes LOW_VRAM the state ComfyUI itself
    # installs, so every later load honours it.
    if hasattr(mm, "lowvram_available"):
        mm.lowvram_available = True
    import comfy.cli_args as comfy_args

    comfy_args.args.lowvram = True
    mm.vram_state = mm.VRAMState.LOW_VRAM
    log(f"vram_state forced to {mm.vram_state} (args.lowvram=True)")
    mm.vram_state = mm.VRAMState.LOW_VRAM
    # Reserve headroom for activations the DiT's own loaders do not account for
    # (the packed ref2va sequence plus its attention buffers). extra_reserved_memory()
    # reads this module constant, so it has to be the real name -- the fl2va
    # runner sets a non-existent `EXTRA_RESERVED` attribute, which silently does
    # nothing.
    mm.EXTRA_RESERVED_VRAM = int(os.environ.get("H3_RESERVE_VRAM_MB", "3500")) * 1024 * 1024
    log(f"reserving {mm.EXTRA_RESERVED_VRAM / 2**20:.0f} MiB of VRAM")

    prompt = assets.load_prompt()
    log(f"prompt: {len(prompt)} chars, first line {prompt.splitlines()[0]!r}")

    video_path = assets.fetch_asset(
        assets.INPUT_VIDEO_URL, assets.INPUT_VIDEO_SHA256, filename="input_video.mp4")
    audio_path = assets.fetch_asset(
        assets.INPUT_AUDIO_URL, assets.INPUT_AUDIO_SHA256, filename="input_audio.mp3")

    # ---- target shape, on the 17k+5 grid the ComfyUI length widget produces
    latent_t = LATENT_T or 12
    frames_n = 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 and (latent_t - 2) % 5 == 0 else 5
    audio_t = AUDIO_T or int(round(frames_n / 24.0 * 40.0))
    log(f"target: W={W} H={H} latent_t={latent_t} frames={frames_n} audio_t={audio_t}")

    # ---- reference video: ComfyUI's resize + 17k+5 truncation
    log("decoding reference video")
    raw_frames, vw, vh, vcount = assets.decode_video_frames(video_path)
    log(f"reference video {vcount} frames @ {vw}x{vh}")
    resized, cw, ch = assets.prepare_reference_video(
        raw_frames, vw, vh, frames_n,
        short_edge=REF_SHORT_EDGE or None)
    log(f"reference canvas cap: short_edge={REF_SHORT_EDGE or 'model default 768'}")
    ref_latent_t = (resized.shape[0] - 2) // 5 + 1
    log(f"reference video -> {resized.shape[0]} frames ({cw}x{ch}), latent_t={ref_latent_t}")

    qwen_frames, qwen_ts = assets.qwen_video_samples(resized)
    log(f"qwen sees {qwen_frames.shape[0]} frames at 2 fps, ts[:4]={qwen_ts[:4]}")

    # ---- reference audio: the video's own soundtrack + the standalone mp3
    log("decoding reference audio")
    video_wave, video_rate = assets.decode_video_audio(video_path)
    audio_wave, audio_rate = assets.decode_audio_file(audio_path)
    log(f"video soundtrack: {'none' if video_wave is None else tuple(video_wave.shape)} @{video_rate}")
    log(f"standalone audio: {tuple(audio_wave.shape)} @{audio_rate}")

    # ---- VAE encode the references (ComfyUI's VAEs; the omni runner reuses
    #      these exact latents rather than re-encoding with its own VAE)
    log("loading video VAE")
    video_vae = load_vae_wrapper(VIDEO_VAE)
    cache_key = (f"{assets.sha256_file(VIDEO_VAE)[:12]}_t{resized.shape[0]}"
                 f"_w{cw}_h{ch}")
    ref_video_latent = cached_video_encode(
        os.environ.get("H3_REF_CACHE", "/tmp/h3ref2va/cache"),
        cache_key,
        lambda: encode_video_vae(video_vae, resized, "ref video"),
    )
    log(f"ref video latent {tuple(ref_video_latent.shape)}")

    log("loading audio VAE")
    audio_vae = load_vae_wrapper(AUDIO_VAE)
    if video_wave is not None:
        video_audio_latent, ref_audio_t = encode_audio_vae(
            audio_vae.first_stage_model, video_wave, video_rate, "video soundtrack")
    else:
        video_audio_latent, ref_audio_t = None, 0
    standalone_latent, standalone_t = encode_audio_vae(
        audio_vae.first_stage_model, audio_wave, audio_rate, "standalone audio")
    log(f"ref audio latents: soundtrack rt={ref_audio_t}, standalone rt={standalone_t}")

    del video_vae, audio_vae
    free_cuda()

    # ---- build the reference blocks exactly as MiniMaxH3ReferenceToVideo does
    ref_blocks = [{
        "kind": "video_audio" if ref_audio_t else "video",
        "latent_t": int(ref_video_latent.shape[2]),
        "latent_h": int(ch // 16),
        "latent_w": int(cw // 16),
        "ref_audio_t": int(ref_audio_t),
        "latent": ref_video_latent,
        "audio_latent": video_audio_latent,
    }, {
        "kind": "audio",
        "ref_audio_t": int(standalone_t),
        "audio_latent": standalone_latent,
    }]
    log(f"ref blocks: " + "; ".join(
        f"{b['kind']}(t={b.get('latent_t')},rt={b['ref_audio_t']},"
        f"{b.get('latent_h')}x{b.get('latent_w')})" for b in ref_blocks))

    # ---- text encoder
    log("loading text encoder")
    tokenizer = cte.MiniMaxH3Tokenizer()
    te_sd = {}
    for f in sorted(glob.glob(f"{TE}/text_encoder/*.safetensors")):
        te_sd.update(comfy.utils.load_torch_file(f, device=torch.device("cpu")))
    if not te_sd and os.path.exists(COMFY_TE):
        # ComfyUI's own H3 text encoder. Preferred over the partition's: see the
        # COMFY_TE comment above for why the partition's layout is not loadable
        # by ComfyUI's detection path.
        te_sd = comfy.utils.load_torch_file(COMFY_TE, device=torch.device("cpu"))
        log(f"text encoder: {COMFY_TE}")
    detect = comfy.text_encoders.hunyuan_video.llama_detect(
        comfy.utils.state_dict_prefix_replace(te_sd, {"model.language_model.": "model."}), "")
    log(f"llama_detect -> {detect}")
    clip_model = cte.te(**detect)(device=mm.intermediate_device(), dtype=DTYPE)

    # The vision tower needs its prefix fixed up when the checkpoint stores it
    # as `model.visual.*`. llama_detect's rename only covers the language model
    # (`model.language_model.` -> `model.`), and load_sd calls
    # load_state_dict(strict=False), so a mismatch is SILENT: every
    # vision-tower weight stays at its uninitialized all-zero value while the
    # language model loads fine.
    #
    # ComfyUI's own H3 text encoder already stores `visual.*` and needs no
    # rename -- it is the default. The bf16 partition's text_encoder does NOT
    # (it stores `model.visual.*`), so the rename is applied only when those
    # keys are actually present.
    #
    # This is invisible for text-only conditioning, which is why the fl2va
    # parity run never hit it -- but ref2va feeds Qwen video vision blocks, and
    # an all-zero patch_embed plus all-zero blocks makes the text states NaN.
    # Symptom to recognise: `text_states mean=nan` right before the DiT loads.
    vision_keys = {k: te_sd.pop(k) for k in list(te_sd) if k.startswith("model.visual.")}
    if vision_keys:
        te_sd.update({f"visual.{k[len('model.visual.'):]}": v for k, v in vision_keys.items()})
    clip_model.load_sd(te_sd)
    log(f"TE loaded (renamed {len(vision_keys)} vision-tower keys)")
    uninitialized = [n for n, p in clip_model.named_parameters() if p.abs().max() == 0]
    if uninitialized:
        raise RuntimeError(
            f"{len(uninitialized)} text-encoder parameters are still all-zero after "
            f"load_sd, e.g. {uninitialized[:3]}; the vision tower or language model did "
            f"not load and any vision conditioning will be NaN")

    # ref_items order IS the presentation contract: the soundtrack's <Audio j>
    # label is emitted immediately BEFORE its <Video k>
    # (nodes_minimax_h3.py:330-341), and the standalone audio last.
    ref_items = []
    if video_audio_latent is not None:
        ref_items.append({"type": "audio"})
    ref_items.append({"type": "video", "data": qwen_frames, "timestamps": qwen_ts})
    ref_items.append({"type": "audio"})

    tokens = tokenizer.tokenize_with_weights(prompt, minimax_ref_items=ref_items)
    # SD1ClipModel.encode_token_weights does the `token_weight_pairs[clip_name]`
    # indexing itself, so hand it the tokenizer's whole dict rather than the
    # already-indexed list -- indexing twice raises TypeError.
    out = clip_model.encode_token_weights(tokens)
    text_states = out[0]
    extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
    tags = extra.get("minimax_token_tags")
    log(f"text_states {tuple(text_states.shape)} mean={text_states.float().mean():.4f} "
        f"std={text_states.float().std():.4f}")
    if not torch.isfinite(text_states).all():
        raise RuntimeError(
            "text_states contain non-finite values; the Qwen conditioning did not load "
            "or the vision blocks produced NaN. Check the text_encoder directory and "
            "the vision-tower prefix fixup above.")
    if tags is not None:
        log(f"token tags ({int((tags == 0).sum())} video / {int((tags == 1).sum())} text)")
    else:
        log("token tags: None")

    del clip_model
    free_cuda()

    # ---- DiT
    log("loading DiT " + DIT)
    patcher = comfy.sd.load_diffusion_model(DIT)
    if patcher is None:
        raise RuntimeError("ComfyUI could not detect the MiniMax-H3 diffusion model")
    log(f"DiT patcher={type(patcher.model).__name__} size={patcher.model_size()/2**30:.2f} GiB")

    noise = comfy.sample.prepare_noise(
        comfy.nested_tensor.NestedTensor([
            torch.zeros([1, 24, latent_t, H // 16, W // 16]),
            torch.zeros([1, 32, 2, audio_t]),
        ]),
        SEED,
    )
    nv, na = noise.unbind()
    log(f"noise streams={[tuple(t.shape) for t in noise.unbind()]}")

    # The DiT is 19.5 GiB and the card is 19.6 GiB, so every block must be
    # offloaded to CPU and streamed back one at a time -- there is no budget
    # that fits weights AND attention activations simultaneously. lowvram_model_memory
    # is the ceiling ComfyUI compares each module against; 2 GiB leaves the rest
    # for activations over the packed ref2va sequence.
    #
    # NOTE: do NOT detach() before this. The fl2va runner does detach + partially_load,
    # which reports "kept 0.00 GiB" and leaves the model fully resident, which OOMs at
    # step 0 for any sequence longer than fl2va's.
    patcher.load(lowvram_model_memory=int(os.environ.get("H3_LOWVRAM_MB", "2000")) * 1024 * 1024)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"pre-sampling vram={torch.cuda.memory_allocated()/2**30:.2f} GiB "
        f"free={torch.cuda.mem_get_info()[0]/2**30:.2f} GiB, "
        f"model_lowvram={patcher.model.model_lowvram}")

    patcher.model_options.setdefault("transformer_options", {})
    taps = install_taps(patcher)

    # model_sampling lives on the patcher's model: pull the schedule off the same
    # object the sampler will use, never re-derive it from source.
    model_sampling = patcher.get_model_object("model_sampling")
    log(f"shift={getattr(model_sampling, 'shift', None)} "
        f"audio_shift={getattr(model_sampling, 'audio_shift', None)} "
        f"audio_scale={model_sampling.audio_scale}")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", STEPS).to(mm.get_torch_device())
    log(f"sigmas={['%.6f' % v for v in sigmas.tolist()]}")
    sampler = comfy.samplers.sampler_object("euler")

    cond = [[text_states, {
        "minimax_token_tags": tags,
        "minimax_refs": ref_blocks,
        "minimax_visual_cond_noise_aug": h3model.VISUAL_COND_TIMESTEP,
        "minimax_audio_cond_noise_aug": h3model.AUDIO_COND_TIMESTEP,
    }]]
    negative = [[torch.zeros_like(text_states), {}]]

    log("sampling...")
    samples = comfy.samplers.sample(
        patcher, noise, cond, negative, 1.0, mm.get_torch_device(), sampler, sigmas,
        patcher.model_options, latent_image=noise, disable_pbar=False, seed=SEED)
    log(f"sampled, {len(taps['__step__'])} steps tapped")
    for i, rec in enumerate(taps["__step__"]):
        vv, va = rec["v"]
        log(f"  step {i} t={rec['timestep'].flatten()[0].item():.4f} "
            f"video rms={vv.pow(2).mean().sqrt():.4f} audio rms={va.pow(2).mean().sqrt():.4f}")

    payload = {
        "sigmas": sigmas.float().cpu(),
        "steps": taps["__step__"],
        "noise_video": nv.cpu(),
        "noise_audio": na.cpu(),
        "text_states": text_states.cpu(),
        "tags": None if tags is None else tags.cpu(),
        "final": [s.detach().float().cpu() for s in samples.unbind()],
        # Consumed by the omni runner instead of re-encoding: the two trees ship
        # different VAE builds, and a VAE difference must not be scored as a
        # quantization difference.
        "ref_conditions": {
            "video_latent": ref_video_latent.detach().float().cpu(),
            "video_shape": (int(ref_video_latent.shape[2]), int(ch // 16), int(cw // 16)),
            "audio_latent": None if video_audio_latent is None
                            else video_audio_latent.detach().float().cpu(),
            "audio_t": int(ref_audio_t),
            "standalone_latent": standalone_latent.detach().float().cpu(),
            "standalone_t": int(standalone_t),
        },
        "meta": {
            "W": W, "H": H, "STEPS": STEPS, "latent_t": latent_t, "audio_t": audio_t,
            "frames": frames_n, "seed": SEED, "prompt": prompt,
            "ref_frames": int(resized.shape[0]), "ref_canvas": (int(cw), int(ch)),
        },
    }
    torch.save(payload, OUT)
    log(f"saved {len(taps['__step__'])} steps to {OUT} ({os.path.getsize(OUT)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
