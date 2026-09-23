# MiniCPM-V-4.5-AWQ Diversity Pilot — 2026-09-23

## Goal

Add a model with a different vision stack to the existing Qwen-heavy pool and measure both standalone accuracy and paired complementarity.

This branch is not a replacement decision based only on standalone score. A weaker but complementary model may still be useful downstream.

## Model

- local model: `openbmb/MiniCPM-V-4_5-AWQ`
- downloaded weights are local; inference is local GPU inference, not an external inference API
- native MiniCPM `model.chat` path
- `enable_thinking=False`
- `sampling=False`
- prompt asks for one lowercase choice a/b/c/d
- deterministic parser records raw response and parse failures

Official model-card usage documents `AutoModel.from_pretrained(..., trust_remote_code=True)`, `AutoTokenizer`, and `model.chat` for image chat.

## Stages

### Stage 0 — load/format smoke
Run 8–32 VAL-A rows.
Pass only if:
- model loads without OOM/dependency failure;
- parse rate >= 95%;
- no repeated runtime errors.

Accuracy is diagnostic only at this tiny size.

### Stage 1 — matched 100-row pilot
Use the same first 100 VAL-A rows already used by the FT-8B direct/option-verification comparison.

Reference on these 100 rows:
- FT-8B direct scoring: 94/100.

Decision:
- MiniCPM >=94/100: advance to full VAL-A.
- 92–93: run paired comparison; advance only if complementarity is substantial enough to justify the cost.
- <=91: stop unless paired error rescue is unexpectedly strong.

### Stage 2 — full VAL-A
Only after Stage 1.
Evaluate standalone accuracy plus paired rescue/regression against FT-8B direct.

Do not submit MiniCPM or ensemble it on Public LB before the full validation decision.
