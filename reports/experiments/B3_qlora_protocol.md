# B3 — QLoRA Experiment Protocol

Date: 2026-09-21

## Updated reference after router re-check

On the 400-row QLoRA holdout:
- B0 direct/standard: **361/400 = 90.25%**
- price-routed system: **360/400 = 90.00%**

The price router therefore does not remain the deployment reference.

**Current reference = B0 direct + standard.**

## Why use inner dev?

The 400-row holdout has not been used for QLoRA hyperparameter selection and is frozen now.

Create:
- inner train ≈ 3,294
- inner dev ≈ 300

from the 3,594-row qlora_train, with zero normalized-template and image-path overlap.

All QLoRA development decisions are made on inner dev.

Only after a configuration is chosen:
1. retrain it on all 3,594 qlora_train rows
2. evaluate once on the final 400 holdout

## Conservative first configuration

Because B0 is already ~90%, the first goal is to avoid catastrophic forgetting.

Locked config:
- base: Qwen2.5-VL-3B-Instruct
- 4-bit NF4 + double quant
- BF16 compute
- standard dynamic resolution
- LoRA target: q_proj, k_proj, v_proj, o_proj
- r=8
- alpha=16
- dropout=0.05
- lr=5e-5
- epochs=1
- micro batch=1
- gradient accumulation=16
- cosine schedule
- warmup=5%
- answer-token-only loss
- no MLP adapters in the first run

## Stage order

### B3-P2 smoke
Train on 32 rows only.

Pass if:
- no OOM
- no masking assertion failure
- loss finite
- adapter saves successfully
- peak VRAM < 15.5 GB

### B3-P3 inner-dev run
Train on full inner train for 1 epoch, evaluate on inner dev.

Practical gate relative to inner-dev B0:
- **net +4 or better with <=4 regressions** → promote config
- net +1~3 → weak signal; inspect categories before one allowed revision
- net <=0 → reject first config

### B3-P4 final
Once config is selected:
- retrain from base on all 3,594 qlora_train rows
- evaluate exactly once on 400 final holdout

Final adoption:
- must beat B0 361/400
- **365/400 or better** is a clear practical win (+1.0 pp)
- 362–364 is marginal and should be weighed against complexity
- <=361 reject


## Smoke-test correction after first run

The first 32-row smoke completed without OOM:
- peak VRAM: **5.81 GB**
- mean microbatch loss: **0.863**
- adapter files saved

However PyTorch emitted:
- `None of the inputs have requires_grad=True. Gradients will be None`

Because the run used re-entrant gradient checkpointing, this warning makes gradient flow ambiguous enough that the smoke cannot be accepted as final proof of a valid training path.

The trainer is corrected to:
- use **non-reentrant gradient checkpointing**
- assert a non-zero LoRA gradient norm on the first microbatch
- assert at least one trainable parameter receives non-zero gradient
- compute and assert non-zero LoRA parameter change after training
- abort immediately on non-finite loss

The smoke gate is unchanged, with two explicit additions:
- non-zero first gradient norm
- non-zero adapter parameter L2 change


---

## Post-protocol result note — 2026-09-22

This section records results obtained after the protocol above was written. It does **not** rewrite the preregistered gate.

### B3-P3 inner-dev outcome

Using the locked first configuration trained on the 3,294-row inner train:
- B0: **271/300 = 90.33%**
- QLoRA: **275/300 = 91.67%**
- wrong→right (rescue): **10**
- right→wrong (regression): **6**
- net gain: **+4**
- McNemar exact p: **0.4544982910**

The preregistered B3-P3 promotion gate required **net +4 or better AND <=4 regressions**. Therefore this run **did not pass the preregistered promotion gate**.

### One-time frozen-holdout confirmation of the same adapter

After the inner-dev result, the same 3,294-row-trained adapter
`B3P3_r8_lr5e5_epoch1` was evaluated once on the frozen 400-row QLoRA holdout without further QLoRA tuning.

Important protocol distinction:
- this was **not** the originally specified B3-P4 retraining on all 3,594 qlora_train rows;
- no 3,594-row retrain was performed before this holdout evaluation;
- the holdout had previously been used for fixed B0/router comparison, but not for QLoRA hyperparameter selection.

Paired result:
- B0: **361/400 = 90.25%**
- B3P3 QLoRA: **369/400 = 92.25%**
- delta: **+2.00 percentage points**
- both correct: **356**
- B0 only / regression: **5**
- QLoRA only / rescue: **13**
- both wrong: **26**
- net gain: **+8**
- prediction disagreement rate: **5.5%**
- McNemar exact p: **0.0962524414**

Category deltas on this holdout included:
- scene_text: **90.37% → 92.66%**, net +5
- other: **92.38% → 96.19%**, net +4
- price: unchanged
- spatial: unchanged
- menu: net -1

Interpretation:
- the direction of improvement observed on inner dev reproduced on the frozen holdout and exceeded the protocol's practical 365/400 threshold;
- however, because the original inner-dev promotion gate failed and the prescribed 3,594-row retraining step was not performed, this result must be described as a **one-time confirmation of the existing B3P3 adapter**, not as a clean execution of the original B3-P4 path;
- p=0.096 does not justify a p<0.05 statistical-significance claim.

### Test inference artifact status

The same B3P3 adapter completed inference on all **6,714** test rows with direct prompt + standard resolution.
Reported run:
- rows: **6,714 / 6,714**
- elapsed: **4,161.28 s (~69.35 min)**
- peak allocated VRAM: **7.39 GB**
- parse/test-gold metrics: unavailable because test has no answer labels

Artifacts produced locally:
- `output/baseline/B3P3_test_QLoRA_probs_v2/predictions.csv`
- `output/baseline/B3P3_test_QLoRA_probs_v2/choice_probabilities.csv`

The probability file stores `id,p_a,p_b,p_c,p_d`, defined as first-generation-step logits normalized over choices a/b/c/d. These are candidate scores, not calibrated probabilities of correctness.

Public leaderboard score remains **unconfirmed** in this report.
