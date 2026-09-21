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
