# InternVL3-8B BNB4 diversity pilot — 2026-09-23

## Motivation

MiniCPM-V-4.5 INT4 showed real complementarity but weak standalone accuracy:
- fixed first-100 pilot: 89/100 vs FT-8B 94/100;
- untouched confirmation 907: 787/907 vs FT-8B 858/907;
- MiniCPM rescued 17 of 49 FT-8B confirmation errors, but regressed 88 FT-correct rows.

The next experiment therefore tests a stronger alternate VLM family rather than further post-hoc routing of MiniCPM.

## Model and loading

- model family: InternVL3-8B
- local checkpoint: full OpenGVLab/InternVL3-8B
- local inference only; no external inference API
- BNB NF4 4-bit at load time
- BF16 compute
- InternVL native dynamic 448px tiling
- max_num=12
- native model.chat generation
- deterministic generation: do_sample=false, num_beams=1
- final output requested as one lowercase a/b/c/d

The official AWQ checkpoint was downloaded separately, but its config declares AWQ 4-bit. The existing baseline environment already failed on another AWQ checkpoint because Transformers required AutoAWQ. To avoid destabilizing the working Qwen environment with an archived/extra quantization dependency, this experiment uses the full checkpoint plus the already-working bitsandbytes 4-bit path instead.

## Stage 0 — smoke 8

Pass if:
- no OOM/runtime failure;
- parse >=7/8;
- outputs are stable a/b/c/d;
- VRAM is safe enough to continue.

## Stage 1 — fixed first 100 VAL-A rows

Reference:
- FT-8B direct = 94/100.

Advance to full VAL-A if:
- standalone >=92/100 and at least 3 FT-error rescues; or
- standalone >=94/100 regardless of rescue count.

Otherwise stop unless a clearly pre-specified paired-complementarity criterion is met.

## Stage 2 — full VAL-A

Only after Stage 1 passes. Evaluate:
- standalone accuracy;
- FT-8B rescue/regression counts;
- disagreement rate;
- category diagnostics.

No Public-LB submission before this validation decision.
