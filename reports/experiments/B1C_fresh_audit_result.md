# B1C Fresh Audit Result — Price-only Routing Confirmed

Date: 2026-09-21

## 한눈에 보기

Fresh untouched audit:
- price: 200
- phone: 100
- total: 300

| Category | B0 | B1C | Wrong→Right | Right→Wrong | Net |
|---|---:|---:|---:|---:|---:|
| **price** | 180/200 = 90.0% | **184/200 = 92.0%** | **5** | **1** | **+4** |
| phone | 89/100 = 89.0% | 87/100 = 87.0% | 1 | 3 | **-2** |
| all 300 | 269/300 = 89.67% | 271/300 = 90.33% | 6 | 4 | +2 |

> **Decision:** the pre-registered price gate was met exactly: **net +4 with only 1 regression**. Price-only routing is therefore locked as the current candidate. Phone stays on B0.

## Pre-registered gate check

Price gate:
- net +4 or more
- regressions <= 2

Observed:
- net = **+4**
- regressions = **1**

Result: **PASS**

Phone was secondary and had already shown instability on grouped validation. Fresh audit again regressed:
- net = **-2**

Result: **B0 retained for phone**

## Evidence across splits

### Price

| Split | B0 → B1C net |
|---|---:|
| Random | **+2** |
| Grouped | **+4** |
| Fresh audit | **+4** |

Direction is positive in all three evaluations.

The random and grouped splits are not fully independent because their validation IDs overlap, so they should not be treated as three independent statistical replications. The fresh audit is the key confirmation because it excluded every ID used by both existing validation splits.

### Phone

| Split | B0 → B1C net |
|---|---:|
| Random | +2 |
| Grouped | **-1** |
| Fresh audit | **-2** |

Phone does not replicate and is not routed to B1C.

## Locked router

```text
question category
      |
      +-- price --> high resolution + binding-aware prompt
      |
      +-- all other categories --> standard resolution + direct prompt
```

This is intentionally narrower than the original price+phone hypothesis.

## Hybrid scores if price-only routing is applied

### Random full validation
- B0: 1216/1341 = **90.68%**
- Price-only hybrid: 1218/1341 = **90.83%**
- delta: **+0.15 pp**

### Grouped full validation
- B0: 1232/1343 = **91.73%**
- Price-only hybrid: 1236/1343 = **92.03%**
- delta: **+0.30 pp**

### Fresh audit
Use B1C on price and B0 on phone:
- price B1C: 184/200
- phone B0: 89/100
- hybrid: **273/300 = 91.00%**
- B0: 269/300 = 89.67%
- delta: **+1.33 pp on the price/phone-enriched audit**

## Statistical interpretation

Fresh-audit price McNemar:
- wins = 5
- losses = 1
- exact p = **0.21875**

This is not statistically significant at 0.05.

So the claim is not "proven improvement." The correct claim is:

> The price-only route passed a pre-registered practical gate on a fresh untouched audit and showed the same positive direction on both prior validation views.

## Expected leaderboard-scale effect

Price is about 13% of the dataset.

Observed full-validation effects:
- Random: +0.15 pp
- Grouped: +0.30 pp

The audit price delta was +2.0 pp, which would imply roughly +0.27 pp overall if the test distribution and effect transfer similarly.

Therefore a reasonable local expectation is approximately **+0.15 to +0.30 pp overall**, not a guarantee.

## What is now frozen

- price router rule
- binding-aware prompt text
- high resolution preset
- direct/standard fallback
- audit 300 IDs

The audit IDs must never enter QLoRA/fine-tuning data.

## Next modeling branch

Do not continue tuning this router on the audit.

Next improvement work should target remaining failure families on a separate development pool:
1. layout-aware grounding for residual binding/localization
2. scene-text OCR/exact-string failures
3. QLoRA only after inference-side grounding experiments plateau
