# Step3-VL-10B experimental BNB4 diversity pilot — 2026-09-23

## Motivation

Two alternate VLMs were tested after the current Qwen-heavy ensemble:
- MiniCPM-V-4.5 INT4: weak standalone, confirmed complementarity but poor routing precision.
- InternVL3-8B BNB4: first-100 gate passed, but confirmation 907 fell to 87.21% with 22 rescues and 89 regressions, so the branch was stopped.

Step3-VL-10B is the final broad model-family hop before returning to targeted reasoning/specialist methods.

## Runtime constraint

The official Step3-VL-10B Transformers path documents BF16 inference. The local RTX 5060 Ti has 16 GB VRAM, so this experiment uses an explicitly experimental bitsandbytes NF4 4-bit load.

A load-only check succeeded before this protocol:
- class: Step3VL10BForCausalLM
- peak allocated VRAM after load: 7.48 GB

A successful load does not prove that 4-bit generation quality matches the officially supported BF16 path.

## Inference

- local checkpoint: downloads/models/Step3-VL-10B
- AutoProcessor + official chat template
- model: AutoModelForCausalLM with official key_mapping
- quantization: BNB NF4 4-bit, double quant, BF16 compute
- deterministic generation: do_sample=false
- thinking mode retained; the model is not treated as a guaranteed non-thinking model
- max_new_tokens=256
- final answer requested as FINAL: a/b/c/d
- parser first reads the post-</think> segment and prefers an explicit FINAL marker

## Stage 0 — smoke 8

Pass if:
- no OOM/runtime failure;
- parsed >=7/8;
- no systematic 256-token truncation;
- peak VRAM remains safely under the 16 GB device limit.

Smoke accuracy is diagnostic only.

## Stage 1 — fixed first 100 VAL-A rows

Reference FT-8B direct accuracy: 94/100.

Advance only if either:
1. Step3 >=94/100, or
2. Step3 >=92/100 AND rescues at least 4 FT-8B errors.

Otherwise stop the Step3 branch.

## Stage 2

Only after Stage 1 passes:
- run the untouched remaining 907 confirmation rows;
- inspect standalone accuracy, rescue/regression, disagreement, and parse/truncation behavior;
- do not use Public LB to tune a router.
