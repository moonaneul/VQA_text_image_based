# B1C Fresh Audit Plan

Date: 2026-09-21

## Audit split creation

Existing validation IDs were excluded before sampling.

- train rows: **6,714**
- unique IDs excluded by random/grouped validation: **2,420**
- untouched remaining pool: **4,294**
- available price rows after exclusion: **570**
- available phone rows after exclusion: **142**
- audit sample: **300**
  - price: **200**
  - phone: **100**
- seed: **20260921**

Audit file: `splits/audit_price_phone.csv`

## Why this audit exists

Random validation has been used repeatedly for ablation/tuning.
Grouped validation was then used to confirm B1C and revealed category heterogeneity.

The audit set is intended as a fresh confirmation set for the refined hypothesis:

> B1C (high resolution + binding-aware prompt) is useful for **price** questions, while phone should remain on B0 unless independently supported.

## Locked evaluation protocol

Run exactly two configurations on the same 300 rows:

1. **Audit-B0**
   - direct
   - standard
   - no quantization
   - seed 20260921

2. **Audit-B1C**
   - binding_aware
   - high
   - no quantization
   - seed 20260921

No prompt/resolution changes are allowed after observing Audit-B0.

## Locked decision rule

Primary decision is on **price (n=200)**.

| Price net gain | Decision |
|---:|---|
| **+4 or more**, with <=2 regressions | keep price-only routing candidate |
| +2 to +3 | weak replication; do not lock yet |
| <= +1 | close routing branch |
| negative | reject |

Phone is secondary:
- report independently
- because grouped phone regressed, keep B0 unless the fresh audit shows a clear independent positive signal

## Audit protection

Once evaluated, these 300 audit IDs must be excluded from any future QLoRA/fine-tuning data.

This audit should not be repeatedly reused for prompt search.
