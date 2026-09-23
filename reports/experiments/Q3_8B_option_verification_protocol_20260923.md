# Qwen3-VL-8B Option Verification Scoring Protocol — 2026-09-23

## Hypothesis

The existing direct four-choice scorer asks the model to bind answer content to the displayed label a/b/c/d in one pass.

A different inference formulation may reduce choice-label binding errors:

1. present one candidate answer text at a time;
2. ask whether that candidate is correct;
3. score binary correctness as logit(a=correct) - logit(b=incorrect);
4. repeat for all four candidates;
5. choose the candidate with the largest correctness margin.

This changes the **decision formulation**, not model weights, resolution, or training.

## Locked configuration

- model: Qwen3-VL-8B-Instruct
- adapter: Q3_8B_FT_R32_LR2E4_E1
- quantization: 4-bit NF4 + double quant
- compute: BF16
- image profile: min_pixels=200704, max_pixels=501760 (640 profile)
- evaluation set: fixed VAL-A 1,007
- candidate verification labels:
  - a = candidate is correct
  - b = candidate is incorrect
- candidate score: logit(a) - logit(b)
- final answer: argmax candidate score
- no generation chain-of-thought
- no Public-LB tuning

Reference:
- FT-8B direct P0: 952/1007 = 94.5382%

## Cost

Four verification forwards per VQA row. Therefore full VAL-A is expected to cost roughly four times one direct-scoring pass unless later batching is separately validated.

## Predefined decision gate

After the full 1,007-row VAL-A run:

- **Strong pass:** >= 958/1007 AND net gain >= +6 versus direct P0.
  - eligible for paired/category analysis and a later bounded test run.
- **Weak signal:** 954–957.
  - analyze paired rescues/regressions only; do not immediately run full test.
- **Reject:** <= 953.
  - stop this branch.

Even on a strong pass, compare paired rescues/regressions with the existing direct scorer before authorizing the ~4x-cost test inference.

## Outputs

The script saves:
- predictions.csv
- choice_probabilities.csv
- verification_scores.csv
- run.json

The exported p_a..p_d values are softmax-normalized binary verification margins. They are **candidate scores**, not calibrated probabilities of correctness.
