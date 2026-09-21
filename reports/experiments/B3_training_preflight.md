# B3 — Training-side Adaptation Preflight

Date: 2026-09-21

## Why move to training now?

B2-D failed its pre-registered tiling gate:
- 4/35 perception errors rescued
- 2/35 correct controls broken
- target-localization rescued 0/3

The corrected implementation produced the same aggregate result.

Inference-side branches have now been tested enough to justify moving to training-side adaptation.

## Data discipline

Never train on:
- random validation
- grouped validation
- fresh price/phone audit

A new QLoRA holdout is created **before training** from the untouched pool.

The split builder keeps connected rows together when they share:
- normalized question template, OR
- exact image path

Target:
- safe pool expected: ~3,994 rows
- fresh QLoRA holdout: ~400 rows
- QLoRA train: ~3,594 rows

## Stage B3-P0 — environment preflight

Before writing/running training:
1. check peft
2. check bitsandbytes
3. check accelerate
4. test actual 4-bit model load on RTX 5060 Ti
5. record peak VRAM

If 4-bit loading fails on the Windows environment, do **not** fight the tooling blindly. Fall back to BF16 LoRA feasibility testing.

## Stage B3-P1 — baseline on new holdout

Run current reference inference on the new QLoRA holdout before any training.

This provides a fresh benchmark for the training branch.

## Training principles

The future trainer must:
- use all available image resolution behavior, not fixed 384x384
- train only assistant answer token(s)
- mask system/user/image/prompt tokens with -100
- use low-rank adapters only
- start with 1 epoch
- keep the new holdout untouched during optimization

No full training starts until P0/P1 pass.
