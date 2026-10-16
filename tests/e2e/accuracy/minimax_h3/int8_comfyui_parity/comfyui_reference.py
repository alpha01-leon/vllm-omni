# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ComfyUI reference: full N-step H3 sampling, dumping the MODEL INPUT and the
raw VELOCITY at every step.

Why per-step taps and not just the final latent: the single-step comparison
already showed cos 0.9993 on one DiT call, which says nothing about whether the
error compounds over a trajectory. Both sides here free-run their own
trajectory from the same noise, so the per-step state error IS the propagation
measurement.

Everything goes through ComfyUI's public path (load_diffusion_model ->
ModelPatcher -> CFGGuider -> its own ModelSamplingAV), so the schedule, the
audio carry scale and the CFG arithmetic are all upstream's, not ours.

Two conventions that the consumer must undo -- both look like defects if missed:

  * This model's forward returns -v, so a faithful match is cos ~= -1.0.
  * Audio is CARRIED on the video sigma schedule, not its own. The pack holds
    audio as (sigma_a / sigma_v) * x_audio, and the wrapper's audio output has
    an additive carry term. See compare.py for the inverse, which compare.py
    applies before scoring audio.

Resolution: 256x448 with latent_t=12 fits in ~20 GiB and runs 8 steps in ~16 s
of sampling. The 384x672 / latent_t=37 shape OOMs at step 0, so parity runs at
the smaller shape on purpose -- both sides must use the SAME shape to compare.

Run with the ComfyUI interpreter, not this repo's:

    /path/to/ComfyUI/venv/bin/python comfyui_reference.py
"""
import gc
import glob
import json
import os
import sys
import time

import torch

# Must be importable before the comfy.* imports below. This runs under the
# ComfyUI virtualenv, which is not this repo's interpreter.
COMFYUI_ROOT = os.environ.get("H3_COMFYUI_ROOT", "/data/projects/comfyui/ComfyUI")
sys.path.insert(0, COMFYUI_ROOT)

import comfy.conds
import comfy.model_management as mm
import comfy.model_detection
import comfy.nested_tensor
import comfy.ops
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.supported_models
import comfy.text_encoders.hunyuan_video
import comfy.text_encoders.minimax as cte
import comfy.utils
import comfy.ldm.minimax.model as h3model

COMFY = os.environ.get("H3_COMFY_ROOT", "/data/models/Comfy-Org/MiniMax-H3")
DIT = os.environ.get(
    "H3_COMFY_DIT",
    f"{COMFY}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
TE = os.environ.get("H3_TEXT_ENCODER_ROOT", "/data/models/MiniMax-H3/FL2VA")

OUT = os.environ.get("OUT", "/tmp/h3ref/ref_multistep.pt")
W = int(os.environ.get("W", "448"))
H = int(os.environ.get("H", "256"))
STEPS = int(os.environ.get("STEPS", "8"))
LATENT_T = int(os.environ.get("LATENT_T", "0"))
AUDIO_T = int(os.environ.get("AUDIO_T", "0"))
SEED = 1234
# Taps are big: the packed sequence at this shape is ~4k rows x 5120, so the
# step-0 block dump alone is hundreds of MB.
DTYPE = torch.bfloat16


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def install_taps(patcher):
    """Per-step taps on the raw DiT velocity, plus its input, at every block.

    Hooked at MiniMaxH3Model.forward (the wrapper) rather than _forward so the
    tap sees exactly the tensors the sampler's x0 conversion consumes -- the
    same ones vllm-omni's model() returns.
    """
    dm = patcher.model.diffusion_model
    taps = {"__step__": []}

    def fwd_hook(mod, args, kwargs, out):
        step = len(taps["__step__"])
        taps["__step__"].append({
            "x_in": [None if s is None else s.detach().float().cpu() for s in args[0]],
            "timestep": args[1].detach().float().cpu(),
            "v": [None if s is None else s.detach().float().cpu() for s in out],
        })
        return out

    dm.register_forward_hook(fwd_hook, with_kwargs=True)

    # Per-block hidden states, step 0 only: enough to localise a divergence to a
    # layer without storing 8x the memory. Step 0 is the same comparison the
    # earlier single-step run already did in depth; what is new here is the
    # per-step velocity trace, not a repeat of the layer hunt.
    blocks = {}

    def blk_hook(i):
        def h(m, inp, out):
            if len(taps["__step__"]) == 1:
                blocks[i] = out.detach().float().cpu()
        return h

    for i, b in enumerate(dm.blocks):
        b.register_forward_hook(blk_hook(i))
    taps["blocks"] = blocks
    return taps


def main():
    torch.manual_seed(SEED)
    mm.vram_state = mm.VRAMState.LOW_VRAM
    mm.EXTRA_RESERVED = 512 * 1024 * 1024

    log(f"loading DiT {DIT}")
    patcher = comfy.sd.load_diffusion_model(DIT)
    if patcher is None:
        raise RuntimeError("ComfyUI could not detect the MiniMax-H3 diffusion model")
    log(f"DiT patcher={type(patcher.model).__name__} "
        f"size={patcher.model_size()/2**30:.2f} GiB")
    patcher.load(lowvram_model_memory=2000 * 1024 * 1024)

    # ---- text encoder
    log("loading text encoder")
    tokenizer = cte.MiniMaxH3Tokenizer()
    te_sd = {}
    for f in sorted(glob.glob(f"{TE}/text_encoder/*.safetensors")):
        te_sd.update(comfy.utils.load_torch_file(f, device=torch.device("cpu")))
    detect = comfy.text_encoders.hunyuan_video.llama_detect(
        comfy.utils.state_dict_prefix_replace(te_sd, {"model.language_model.": "model."}), "")
    log(f"llama_detect -> {detect}")
    clip_model = cte.te(**detect)(device=mm.intermediate_device(), dtype=DTYPE)
    clip_model.load_sd(te_sd)
    log("TE loaded")

    prompt = os.environ.get(
        "PROMPT",
        "a calm ocean wave rolling toward a sandy beach at sunset, gentle foam, wide cinematic shot.")
    out = clip_model.encode_token_weights(tokenizer.tokenize_with_weights(prompt))
    text_states = out[0]
    extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
    tags = extra.get("minimax_token_tags")
    log(f"text_states {tuple(text_states.shape)} mean={text_states.float().mean():.4f} "
        f"std={text_states.float().std():.4f}")
    log(f"token tags: {None if tags is None else tags.view(-1).tolist()}")

    cond = [[text_states, {"minimax_token_tags": tags}]]
    negative = [[torch.zeros_like(text_states), {}]]

    # ---- latent shape, derived with vllm-omni's own planner relation so the
    # two sides are forced to the same shape: latent_t = 5n+2 frames,
    # frames = 17n+5, audio_t = round(frames/24 * 40).
    latent_t = LATENT_T or 12
    frames = 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 and (latent_t - 2) % 5 == 0 else 5
    audio_t = AUDIO_T or int(round(frames / 24.0 * 40.0))
    video = torch.zeros([1, 24, latent_t, H // 16, W // 16])
    audio = torch.zeros([1, 32, 2, audio_t])
    noise = comfy.sample.prepare_noise(comfy.nested_tensor.NestedTensor([video, audio]), SEED)
    nv, na = noise.unbind()
    log(f"latent_t={latent_t} frames={frames} audio_t={audio_t} "
        f"streams={[tuple(t.shape) for t in noise.unbind()]}")

    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    patcher.detach()
    gc.collect()
    torch.cuda.empty_cache()
    patcher.model.model_lowvram = True
    kept = patcher.partially_load(mm.get_torch_device())
    log(f"partially_load kept {kept/2**30:.2f} GiB of {patcher.model_size()/2**30:.2f} GiB")
    gc.collect()
    torch.cuda.empty_cache()
    log(f"pre-sampling vram={torch.cuda.memory_allocated()/2**30:.2f} GiB "
        f"free={torch.cuda.mem_get_info()[0]/2**30:.2f} GiB")

    patcher.model_options.setdefault("transformer_options", {})
    taps = install_taps(patcher)

    # model_sampling lives on the patcher's model, so pull the schedule off the
    # same object the sampler will use -- never re-derive it from source.
    model_sampling = patcher.get_model_object("model_sampling")
    log(f"shift={getattr(model_sampling, 'shift', None)} "
        f"audio_shift={getattr(model_sampling, 'audio_shift', None)} "
        f"audio_scale={model_sampling.audio_scale}")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", STEPS)
    sigmas = sigmas.to(mm.get_torch_device())
    log(f"sigmas={['%.6f' % v for v in sigmas.tolist()]}")

    sampler = comfy.samplers.sampler_object("euler")

    log("sampling...")
    samples = comfy.samplers.sample(
        patcher, noise, cond, negative, 1.0, mm.get_torch_device(), sampler, sigmas,
        patcher.model_options, latent_image=noise, disable_pbar=False, seed=SEED)
    log(f"sampled, {len(taps['__step__'])} steps tapped")
    log(f"sigmas={['%.6f' % v for v in sigmas.tolist()]}")
    for rec in taps["__step__"]:
        vv, va = rec["v"]
        log(f"  step t={rec['timestep'].flatten()[0].item():.4f} "
            f"video rms={vv.pow(2).mean().sqrt():.4f} audio rms={va.pow(2).mean().sqrt():.4f}")

    payload = {
        "sigmas": sigmas.float().cpu(),
        "steps": taps["__step__"],
        "blocks": taps.get("blocks", {}),
        "noise_video": nv.cpu(),
        "noise_audio": na.cpu(),
        "text_states": text_states.cpu(),
        "tags": None if tags is None else tags.cpu(),
        "final": [s.detach().float().cpu() for s in samples.unbind()],
        "meta": {"W": W, "H": H, "STEPS": STEPS, "latent_t": latent_t, "audio_t": audio_t,
                 "seed": SEED, "prompt": prompt},
    }
    torch.save(payload, OUT)
    log(f"saved {len(taps['__step__'])} steps to {OUT} "
        f"({os.path.getsize(OUT)/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
