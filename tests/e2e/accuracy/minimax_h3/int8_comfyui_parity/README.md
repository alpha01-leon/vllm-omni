# MiniMax-H3 int8 DiT ↔ ComfyUI parity

Does the converted int8 DiT still match ComfyUI's int8 reference **across a whole
sampling trajectory**, not just on a single forward pass?

The single-step comparison (cos 0.9993 on one DiT call) says nothing about
compounding. As soon as both implementations free-run, their states drift, and a
small per-step error either gets re-absorbed each step or integrates into a
different picture. This closes that gap.

## Result (measured 2026-10-16)

256×448, `latent_t=12`, 8 steps, seed 1234, the Comfy-Org int8 checkpoint.

| | value |
|---|---|
| per-step DiT error, step 0 (identical state both sides) | video 3.4 %, audio 3.0 % |
| trajectory state error | 0.0017 → 0.037 video, 0.0016 → 0.036 audio |
| per-step growth factor | 1.17 1.28 1.63 1.82 1.45 1.71 2.04 — bounded, sublinear |
| **final latent L2rel** | **video 0.077, audio 0.068** |
| final latent rms | omni 1.1371 vs ref 1.1379; audio 0.3181 vs ref 0.3175 |

Per-step fidelity is ~flat, so the requantization error is a **fixed per-step
residual, not a growing one**. The trajectory stays on the reference's rails.

## Running it

Three steps, because the two runners need different interpreters.

```bash
export H3_PART=/data/models/MiniMax-H3/FL2VA_comfyint8/FL2VA   # converted partition
export H3_COMFY_DIT=/data/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export H3_TEXT_ENCODER_ROOT=/data/models/MiniMax-H3/FL2VA
export H3_COMFY_PYTHON=/data/projects/comfyui/ComfyUI/venv/bin/python

W=448 H=256 STEPS=8 LATENT_T=12 OUT=/tmp/ref.pt $H3_COMPY_PYTHON -I comfyui_reference.py
python -I omni_multistep.py            # H3_REF / H3_OUT override the paths
python -I compare.py /tmp/ref.pt /tmp/omni.pt   # exit code is the assertion
```

Or through pytest, which drives all three:

```bash
pytest tests/e2e/accuracy/minimax_h3/int8_comfyui_parity/test_int8_comfyui_parity.py
```

It skips unless every one of those env vars points at a real tree.

## Constraints that are easy to trip over

**The two sides must not run concurrently.** The reference holds ~63 GiB
resident while its DiT is loaded; running both at once gets the pair OOM-killed
silently — the symptom is a script vanishing with exit 137, not an error.

**Both sides must use the same shape.** The shape used for real output
(384×672, `latent_t` 37) OOMs the reference at step 0. 256×448 fits and runs 8
steps in ~16 s of sampling, so that is what parity uses — a deliberate trade of
resolution for comparability.

**Thresholds, not equality.** The two int8 DiTs derive from the same source
weights but not the same bytes: omni re-quantizes ConvRot per output channel and
reorders QKV, while ComfyUI keeps the Comfy-Org layout. bf16 accumulation order
differs too, so bit-exactness is not on the table and asserting it would fail
for the wrong reason.

## Three conventions that masquerade as defects

Each of these produced a convincing-looking failure before being pinned down.
All three are handled in `compare.py`, with the derivation in its docstring.

1. **ComfyUI returns −v.** Its `MiniMaxH3Model.forward` ends in
   `return [-video_out, -audio_out]`. A faithful match is cos ≈ **−1.0**, not
   +1.0.

2. **Audio is carried on the video sigma schedule.** ComfyUI packs both streams
   onto one grid: it carries audio as `(sigma_a/sigma_v) · x_audio`
   (`model.py:561-600`) and sets the audio row timestep to
   `1 - time_shift_sigma(sigma_v, 12, 3)` (`model.py:630`). omni instead runs
   each stream on its own grid (shift 12 / shift 3) with no carry. Both are
   valid; comparing them needs a conversion, and skipping it makes audio read
   cos 0.78 — indistinguishable from a broken audio path.

3. **omni's shift=3 audio grid is the correct one.** Measured, not argued: the
   final audio latent lands at L2rel 0.068 on the shift=3 grid vs 0.516 on the
   video grid, and its rms matches the reference's (0.3181 vs 0.3175).
   `omni_multistep.py` asserts the two parameterisations agree on the audio row
   timestep, so a schedule change fails loudly instead of showing up as a fake
   divergence.

## Files

| file | role |
|---|---|
| `comfyui_reference.py` | ComfyUI's own KSampler under ComfyUI's venv; taps every step |
| `omni_multistep.py` | this repo's `minimax_h3_denoise_loop`; taps every step |
| `compare.py` | scores both, prints the tables, exits non-zero past threshold |
| `test_layout_parity.py` | pins the transcribed row-layout helpers against ComfyUI |
| `test_int8_comfyui_parity.py` | pytest wrapper driving all three |

`omni_multistep.py` carries transcribed copies of ComfyUI's
patchify/unpatchify and pack/unpack_audio rather than importing them, so the
parity run needs only one interpreter. `test_layout_parity.py` exists to catch
that copy drifting: a silent drift would not fail parity, it would fail it
*spuriously* with a quantization-looking error.

Tensor products under `/tmp` run to ~1.6 GiB (the step-0 block dump is most of
it) — keep them out of version control.
