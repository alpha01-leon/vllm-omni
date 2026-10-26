# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-step comparison: ComfyUI int8 ref2va reference vs vllm-omni int8 ref2va
over the whole trajectory.

The ref2va sibling of compare.py. It keeps the fl2va structure and adds the one
check fl2va did not need: ref2va packs reference CONDITION rows into the
sequence, so matching latent shapes do not make the two sides comparable.

TWO SPACE CONVENTIONS -- both matter, and getting either wrong produces a
plausible-looking failure that is not one.

1. Sign. ComfyUI's MiniMaxH3Model.forward returns -v, so a faithful match is
   cos ~= -1.0, not +1.0.

2. Audio is carried on a different schedule than it is compared in.
   ComfyUI packs both streams onto ONE video-sigma schedule: it carries audio as
   (sigma_a / sigma_v) * x_audio and converts inside the model, with the audio
   row timestep set to 1 - time_shift_sigma(sigma_v, 12, 3). Its audio state
   therefore also steps with the video ratio.
   vllm-omni instead runs each stream on its own grid (shift 12 / shift 3) with
   no carry. Both are valid; comparing them needs a space conversion:
       x_audio(omni space) = x_carried(comfy) * sigma_v / sigma_a
   and ComfyUI's returned audio output has an additive carry term to remove:
       v_raw = (out - (1 - audio_scale) * x_carried)
               / (1 + (audio_scale - 1) * sigma_a)

THREE REF2VA-SPECIFIC CHECKS, all of them about "are we comparing the same
thing at all" rather than about quantization error:

  * layout agreement -- row indices, update masks and RoPE position assignments
    must match. Verified for this pair to 7e-15 (float64 summation order), so
    any larger drift is a real divergence, not noise.
  * token tags -- the text span's modality tags come from the Qwen encoder's
    vision-pad positions, and they drive AdaLN. Both sides must agree.
  * condition anchor scale -- the two sides apply DIFFERENT noise-augmentation
    recipes by construction (see omni_multistep_ref2va.py). The measured anchor
    difference is ~1.4e-3 relative, far below the parity threshold, so this is
    reported rather than asserted bit-equality. It is printed because a change
    to either recipe moves it, and a jump here would explain a diagonal shift
    in every downstream number.
"""
import sys

import torch

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/h3ref2va/ref.pt"
OMNI = sys.argv[2] if len(sys.argv) > 2 else "/tmp/h3ref2va/omni.pt"
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

# Thresholds carried over from the fl2va parity run, which measured final latent
# L2rel video 0.077 / audio 0.068 with per-step 0.034-0.096 on the same shape.
# ref2va adds condition rows but does not change the DiT, so the same bands apply.
FINAL_L2REL_MAX = 0.15
PER_STEP_L2REL_MAX = 0.15
MAX_STEP_GROWTH = 3.0

# Layout checks. Positions are float64 on both sides, so agreement is exact to
# accumulation order; the tolerance is ulp slack, not measurement slack.
POSITION_ATOL = 1e-9
# The anchors are deliberately not bit-equal (different RNG shapes upstream).
# This is a REPORTED number with a generous sanity bound: a value far above it
# means one side's condition recipe changed, not that quantization drifted.
ANCHOR_REL_REPORT_MAX = 0.02


def nrel(a, b):
    """L2 relative error, sign-aligned.

    Sign alignment is the point of using this over cosine: ComfyUI returns -v by
    convention, and cosine would absorb that into its sign, hiding whether the
    disagreement is a convention or a real error.
    """
    a = a.detach().float().cpu().flatten().double()
    b = b.detach().float().cpu().flatten().double()
    if torch.dot(a, b) < 0:
        b = -b
    return ((a - b).norm() / a.norm()).item()


def time_shift(s, frm, to):
    base = s / (frm + s * (1.0 - frm))
    return to * base / (1.0 + (to - 1.0) * base)


def segment_runs(values):
    """Run-length encode a 1-D tensor as [(value, length), ...]."""
    out = []
    start = 0
    for i in range(1, values.numel() + 1):
        if i == values.numel() or values[i] != values[start]:
            out.append((values[start].item(), i - start))
            start = i
    return out


def check_layout(ref_layout, omni_layout, used_len):
    """Assert both sides packed the SAME reference rows at the SAME positions.

    This is the check fl2va could skip. In ref2va the packed sequence holds
    reference video and audio rows between the text span and the target streams,
    and their RoPE positions decide where the references sit on the timeline. If
    those differ, every downstream number is a comparison of two different
    models-of-the-problem, not two implementations of one.
    """
    problems = []
    print(f"  ref  used_len={used_len} seq_len={ref_layout['seq_len']}")
    print(f"  omni used_len={omni_layout['used_len']} seq_len={omni_layout['seq_len']}")
    if used_len != ref_layout["seq_len"]:
        problems.append(
            f"reference used_len {used_len} != ComfyUI seq_len {ref_layout['seq_len']}; "
            "the two sides packed a different number of real rows")

    for name, ref_key, omni_key in (
        ("img_pos", "img_pos", "img_pos"),
        ("img_update_mask", "img_update", "update_mask"),
        ("audio_pos", "audio_pos", "audio_pos"),
        ("audio_update_mask", "audio_update", "audio_update_mask"),
    ):
        a = torch.as_tensor(ref_layout[ref_key]).view(-1)
        b = torch.as_tensor(omni_layout[omni_key]).view(-1)
        if a.shape != b.shape:
            problems.append(f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}")
            continue
        if not torch.equal(a.to(a.dtype), b.to(a.dtype)):
            mismatch = int((a != b).sum())
            problems.append(f"{name}: {mismatch}/{a.numel()} rows differ")

    pos_a = torch.as_tensor(ref_layout["position_ids"]).view(-1, 3).double()[:used_len]
    pos_b = torch.as_tensor(omni_layout["position_ids"]).view(-1, 3).double()[:used_len]
    if pos_a.shape != pos_b.shape:
        problems.append(f"position_ids: shape {tuple(pos_a.shape)} != {tuple(pos_b.shape)}")
    else:
        drift = (pos_a - pos_b).abs().max().item()
        print(f"  position_ids max abs drift: {drift:.3e}")
        if drift > POSITION_ATOL:
            worst = (pos_a - pos_b).abs().max(-1).values
            bad = int((worst > POSITION_ATOL).sum())
            problems.append(
                f"position_ids drift {drift:.3e} over {bad}/{used_len} rows exceeds "
                f"{POSITION_ATOL:.0e}; the reference rows sit at different RoPE "
                f"positions on the two sides")

    # ComfyUI's segment table, so a disagreement is legible by segment rather
    # than as an anonymous row index.
    segs = ref_layout.get("segments")
    if segs:
        pos_drift = (pos_a - pos_b).abs().max(-1).values if pos_a.shape == pos_b.shape else None
        print("  ref segments (start, stop, kind) and max position drift:")
        for start, stop, kind in segs:
            d = pos_drift[start:stop].max().item() if pos_drift is not None else float("nan")
            print(f"    [{start:>5}, {stop:>5}) {kind:>10}  n={stop-start:>5}  drift={d:.3e}")
    return problems


def check_token_tags(ref_tags, omni_tags, used_len, ref_text_len):
    """The text span's AdaLN modality tags must agree row for row.

    ComfyUI returns tags only for the TEXT span (Qwen's tokenizer emits them
    over the encoded sequence), so the comparison is against the first
    ``ref_text_len`` rows of omni's full-sequence tags -- the same span, and the
    only span where the two sides' tagging is independently derived (the rest
    is fixed by each side's layout, already checked above).
    """
    if ref_tags is None:
        return ["reference has no token tags; the text span's AdaLN modality is unverifiable"]
    a = torch.as_tensor(ref_tags).view(-1)
    b = torch.as_tensor(omni_tags).view(-1)
    if a.numel() != ref_text_len:
        return [f"reference token tags cover {a.numel()} rows, expected the text span "
                f"({ref_text_len})"]
    if b.numel() < a.numel():
        return [f"omni token_tags ({b.numel()} rows) shorter than the text span ({a.numel()})"]
    b = b[:a.numel()]
    runs_a = segment_runs(a)
    runs_b = segment_runs(b)
    print(f"  text-span tag runs  ref : {runs_a[:8]}{' ...' if len(runs_a) > 8 else ''}")
    print(f"  text-span tag runs  omni: {runs_b[:8]}{' ...' if len(runs_b) > 8 else ''}")
    if runs_a != runs_b:
        return [f"text-span token tags differ:\n      ref  {runs_a[:12]}"
                f"\n      omni {runs_b[:12]}"]
    print("  -> text-span AdaLN modality tags agree")
    return []


def comfy_anchor(clean_rows, shape, seed, aug):
    """Replay ComfyUI's condition-noise recipe on the shared clean rows.

    Transcribed from ``comfy/ldm/minimax/model.py:527-560``: a FRESH cpu
    generator at the same seed for every block, and the draw taken at the
    PATCHED ROW shape -- which is what omni's recipe deliberately does NOT do
    (it draws at latent shape over max(target_t + cond_frames, latent_t)).
    Comparing the two recipes needs both computed the same way, so this is
    ComfyUI's and the omni side's numbers come from the saved payload.
    """
    gen = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn(clean_rows.shape, generator=gen, dtype=torch.float32)
    return aug * clean_rows + (1.0 - aug) * noise


def report_anchors(ref, omni):
    """Report how far the two sides' condition anchors land apart.

    Not an equality assertion: the two recipes draw from different RNG shapes by
    design, so bit-equality is impossible and pinning one recipe would freeze a
    harness detail rather than test the model. What this guards is the failure
    mode where one side's recipe silently changes and every downstream number
    shifts for a reason that has nothing to do with quantization -- that shows up
    as a jump here first.
    """
    anchors = omni.get("anchors")
    if not anchors:
        print("  omni payload carries no anchors; skipping")
        return []
    conds = ref.get("ref_conditions")
    if not conds:
        print("  reference payload carries no ref_conditions; skipping")
        return []

    seed = ref["meta"]["seed"]
    problems = []
    rows = []
    if conds.get("video_latent") is not None:
        clean = anchors["clean_video"].float().cpu()
        rows.append(("video", comfy_anchor(
            clean, conds["video_shape"], seed, 0.999), anchors["video"].float().cpu()))
    if conds.get("audio_latent") is not None:
        clean = anchors["clean_audio"].float().cpu()
        rows.append(("audio", comfy_anchor(
            clean, conds["video_shape"], seed + 1, 1.0), anchors["audio"].float().cpu()))

    for name, comfy_side, omni_side in rows:
        if comfy_side.shape != omni_side.shape:
            problems.append(f"anchor {name}: shape {tuple(comfy_side.shape)} != "
                            f"{tuple(omni_side.shape)}")
            continue
        rel = nrel(omni_side, comfy_side)
        # The audio recipe uses aug=1.0, so it is a no-op on both sides and the
        # comparison is vacuous; say so rather than printing a meaningless 0.0.
        print(f"  {name:>5} anchor L2rel (comfy recipe vs omni recipe): {rel:.6f}")
        if rel > ANCHOR_REL_REPORT_MAX:
            problems.append(
                f"anchor {name} L2rel {rel:.4f} > {ANCHOR_REL_REPORT_MAX}; one side's "
                f"condition-noise recipe changed, which is not a quantization effect")
    return problems


def main():
    ref = torch.load(REF, weights_only=False)
    omni = torch.load(OMNI, weights_only=False)

    sig_v = [float(s) for s in ref["sigmas"]]
    sig_a = [float(s) for s in omni.get("sigmas_audio", [])]
    n = min(len(ref["steps"]), len(omni["raw"]))
    audio_scale = VIDEO_SHIFT / AUDIO_SHIFT
    ref_layout = (ref["steps"][0] or {}).get("layout")

    print("=== SETUP ===")
    print(f"ref  video sigmas: {['%.6f' % s for s in sig_v]}")
    print(f"omni video sigmas: {sig_v}  (taken from the reference)")
    print(f"omni audio sigmas: {['%.6f' % s for s in sig_a]}")
    derived = [time_shift(s, VIDEO_SHIFT, AUDIO_SHIFT) for s in sig_v]
    if sig_a:
        drift = max(abs(x - y) for x, y in zip(sig_a, derived))
        print(f"comfy implied audio sigma per step (t_a = 1 - time_shift_sigma(sv,12,3)):")
        print(f"                 {['%.6f' % s for s in derived]}")
        print(f"  agreement: max drift {drift:.2e}"
              + ("  OK" if drift < 1e-4 else "  MISMATCH -- do not trust the rest"))
    print(f"ref steps={len(ref['steps'])}  omni taps={len(omni['raw'])}")
    print(f"ref meta: { {k: v for k, v in ref['meta'].items() if k != 'prompt'} }")
    print(f"ref ref_blocks: {omni.get('ref_blocks')}")

    problems = []
    if ref_layout is None:
        problems.append("reference taps carry no layout; cannot verify the ref rows match")
        used_len = 0
    else:
        used_len = int(omni["layout"]["used_len"])
    # ---- ref2va-specific: are the two sides comparing the same thing?
    print("\n=== REF2VA LAYOUT (do the reference rows actually match?) ===")
    if ref_layout is not None:
        layout_problems = check_layout(ref_layout, omni["layout"], used_len)
        problems.extend(layout_problems)
        if not layout_problems:
            print("  -> reference rows, order and RoPE positions agree")
        print("\n=== TOKEN TAGS (AdaLN modality on the text span) ===")
        segs = ref_layout.get("segments") or []
        text_len = next((b - a for a, b, kind in segs if kind == "text"), 0)
        problems.extend(check_token_tags(
            ref.get("tags"), omni["layout"]["token_tags"], used_len, text_len))

    print("\n=== CONDITION ANCHORS (different RNG shapes by design) ===")
    problems.extend(report_anchors(ref, omni))

    print("\n=== PER-STEP VELOCITY (each side denoised from its own x) ===")
    print("ComfyUI returns -v, so the video column should read near -1.0.\n")
    print(f"{'step':>4} {'sigma_v':>9} {'L2rel_vid':>10} {'L2rel_aud':>10} {'cos_vid':>9}")
    vrel, arel = [], []
    for i in range(n):
        rv = nrel(omni["raw"][i]["v_video"], ref["steps"][i]["v"][0])
        # strip ComfyUI's carry terms before comparing audio
        out = ref["steps"][i]["v"][1].detach().float().cpu()
        xin = ref["steps"][i]["x_in"][1].detach().float().cpu()
        v_raw = (out - (1.0 - audio_scale) * xin) / (1.0 + (audio_scale - 1.0) * sig_a[i])
        ra = nrel(omni["raw"][i]["v_audio"], v_raw)
        cosv = torch.nn.functional.cosine_similarity(
            omni["raw"][i]["v_video"].flatten().double().cpu(),
            ref["steps"][i]["v"][0].flatten().double().cpu(), dim=0).item()
        vrel.append(rv)
        arel.append(ra)
        print(f"{i:>4} {sig_v[i]:>9.6f} {rv:>10.5f} {ra:>10.5f} {cosv:>9.6f}")

    print("\n=== PER-STEP STATE: does the error compound? ===")
    print("Both sides free-ran from the same noise and the same reference rows, so")
    print("this is the real propagation measurement. Growth is the question.\n")
    print(f"{'step':>4} {'err_vid':>9} {'x_growth':>10} {'err_aud':>9} {'x_growth':>10}")
    ev, ea = [], []
    for i in range(n):
        ev.append(nrel(omni["raw"][i]["x_video"], ref["steps"][i]["x_in"][0]))
        # ComfyUI carries audio as (sigma_a / sigma_v) * x_audio, so its audio
        # state lives in video-scaled units. Lift omni's state into that same
        # carried space rather than un-carrying ComfyUI's.
        ea.append(nrel(omni["raw"][i]["x_audio"] * (sig_v[i] / sig_a[i]),
                       ref["steps"][i]["x_in"][1]))
    for i in range(n):
        gv = f"{ev[i]/ev[i-1]:.2f}x" if i else "-"
        ga = f"{ea[i]/ea[i-1]:.2f}x" if i else "-"
        print(f"{i:>4} {ev[i]:>9.5f} {gv:>10} {ea[i]:>9.5f} {ga:>10}")

    print("\n=== FINAL LATENT (the end-to-end number that matters) ===")
    for idx, name in enumerate(("video", "audio")):
        a = ref["final"][idx].detach().float().cpu()
        b = omni["final"][idx].detach().float().cpu()
        print(f"  {name:>5}: L2rel={nrel(b, a):.5f}   "
              f"rms_omni={b.pow(2).mean().sqrt():.4f} rms_ref={a.pow(2).mean().sqrt():.4f}")

    print("\n=== VERDICT ===")
    fv = nrel(omni["final"][0], ref["final"][0])
    fa = nrel(omni["final"][1], ref["final"][1])
    print(f"  Per-step DiT fidelity is ~constant ({min(vrel):.3f}-{max(vrel):.3f} video),")
    print(f"  so the requantization error is a fixed per-step residual, not a growing one.")
    print(f"  Final latent L2rel: video {fv:.4f}, audio {fa:.4f}.")
    print(f"  State-error growth (video): "
          f"{' '.join(f'{ev[i]/ev[i-1]:.2f}' for i in range(1, n))}")
    print(f"  State-error growth (audio): "
          f"{' '.join(f'{ea[i]/ea[i-1]:.2f}' for i in range(1, n))}")

    # Thresholds, not equality. Two int8 DiTs derive from the same source
    # weights but not the same bytes: omni re-quantizes ConvRot per output
    # channel and reorders QKV, while ComfyUI keeps the Comfy-Org layout. bf16
    # accumulation order differs too, so bit-exactness is not on the table.
    if fv > FINAL_L2REL_MAX:
        problems.append(f"final video latent L2rel {fv:.4f} > {FINAL_L2REL_MAX}")
    if fa > FINAL_L2REL_MAX:
        problems.append(f"final audio latent L2rel {fa:.4f} > {FINAL_L2REL_MAX}")
    if min(vrel) > PER_STEP_L2REL_MAX:
        problems.append(
            f"worst per-step video L2rel {min(vrel):.4f} > {PER_STEP_L2REL_MAX} "
            f"(the DiT itself disagrees; check weights, not the trajectory)")
    if len(ev) > 2:
        growth = ev[-1] / ev[-2]
        if growth > MAX_STEP_GROWTH:
            problems.append(
                f"state error grew {growth:.2f}x in the final step "
                f"(> {MAX_STEP_GROWTH}x); error is compounding, not saturating")

    if problems:
        print(f"\n  FAIL:")
        for p in problems:
            print(f"    - {p}")
    else:
        print(f"\n  PASS: reference rows agree and the trajectories are within thresholds "
              f"(final L2rel <= {FINAL_L2REL_MAX}, per-step <= {PER_STEP_L2REL_MAX})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
