# B0 Zero-shot Baseline Result

Date: 2026-09-21

## 1. Experiment identity

- Run family: **B0**
- Stage: zero-shot baseline
- Purpose: establish the reference point for every later model improvement
- Previous best: none
- Improvement vs previous model: **N/A — this is the first full baseline**

## 2. Modeling setup

- Model: **Qwen2.5-VL-3B-Instruct**
- Model class: 3B multimodal instruction-tuned VLM
- Competition-data training: none
- Prompt: **direct**
- Resolution: **standard**
  - min_pixels: 200,704
  - max_pixels: 802,816 (~0.8MP)
- Quantization: none
- Decision method: free generation, then parse to a/b/c/d
- Seed: 20260921
- GPU: NVIDIA GeForce RTX 5060 Ti
- PyTorch: 2.11.0+cu128

## 3. Strategy

EDA showed that scene-text, price, phone, and menu questions account for 68.7% of train and 67.7% of test.

The initial strategy was deliberately conservative:

> Before fine-tuning, measure how far a strong pretrained VLM can go with a minimal prompt and near-native image information, then use the remaining errors to decide whether the true bottleneck is OCR recognition, target localization, reasoning/binding, or answer selection.

This avoids spending GPU time on LoRA before knowing what needs to be fixed.

### Fixed variables

- model
- direct prompt
- standard resolution
- no quantization
- deterministic generation
- same seed

### Validation views

- Random stratified validation: leaderboard-like primary view
- Grouped validation: template/image-generalization guardrail

## 4. Results

| Metric | Random validation | Grouped validation |
|---|---:|---:|
| Samples | 1,341 | 1,343 |
| Accuracy | **90.68%** | **91.73%** |
| Correct | 1,216 | 1,232 |
| Errors | 125 | 111 |
| Unparsed | 0 | 0 |
| OCR-heavy accuracy | 89.91% | 91.55% |
| Runtime | 792.3s | 792.3s |
| Throughput | 1.69 rows/s | 1.70 rows/s |

Grouped minus random accuracy: **+1.06 percentage points**.

Because B0 is zero-shot and does not train on the competition train split, this difference is not evidence of template memorization. At this stage it mainly indicates that the grouped validation sample is slightly easier for this pretrained model.

## 5. Category results

| Category | Random | n | Grouped | n |
|---|---:|---:|---:|---:|
| scene_text | 91.17% | 668 | 92.23% | 669 |
| price | 87.78% | 180 | 91.21% | 182 |
| phone | 83.72% | 43 | 87.80% | 41 |
| menu | 83.87% | 31 | 83.87% | 31 |
| spatial | 84.21% | 57 | 88.89% | 54 |
| other | 93.39% | 348 | 92.35% | 353 |
| count | 100.00% | 10 | 100.00% | 9 |
| color | 100.00% | 4 | 100.00% | 4 |

The 100% count/color values are based on very small sample sizes and should not be treated as stable estimates.

## 6. Error concentration

Random validation errors by category:

- scene_text: 59
- price: 22
- phone: 7
- menu: 5
- spatial: 9
- other: 23
- count/color: 0

OCR-heavy categories contain **93 of 125 errors (74.4%)**, while representing **922 of 1,341 samples (68.8%)**.

This supports the original hypothesis that text-centric questions remain the main improvement target even though the overall zero-shot baseline is already strong.

## 7. What this experiment tells us

1. **Qwen2.5-VL-3B-Instruct is already a strong base model** for this competition.
2. The current bottleneck is not output parsing: there were **0 unparsed answers**.
3. OCR-heavy questions remain the largest improvement pool.
4. Scene-text contributes the largest absolute number of errors because it is the dominant category.
5. Price, phone, and menu have lower accuracy than the overall baseline, so an OCR-aware intervention is justified.
6. Fine-tuning is not yet justified because we have not isolated whether the errors are caused by prompt behavior, image resolution, OCR recognition, or reasoning.
7. Random/grouped comparison becomes much more important after competition-data fine-tuning begins.

## 8. Decision

**Keep B0 as the reference baseline.**

Do not change multiple components yet.

The next experiment should isolate prompt behavior only.

## 9. Next experiment: A1 prompt ablation

Change only:

```
direct -> ocr_deliberate
```

Keep fixed:

- Qwen2.5-VL-3B-Instruct
- standard ~0.8MP resolution
- no quantization
- free generation
- same seed
- same validation split

Primary signals:

- overall accuracy delta
- OCR-heavy accuracy delta
- scene_text / price / phone / menu deltas
- B0 wrong -> A1 right
- B0 right -> A1 wrong
- net sample gain

If A1 improves random validation meaningfully, confirm it on grouped validation before moving to resolution experiments.

## 10. Portfolio takeaway

This baseline demonstrates:

- dataset-driven model selection rather than starting with arbitrary fine-tuning
- dual validation design for leaderboard-like performance and generalization
- controlled ablation planning
- category-level error analysis
- compute-aware experimentation on a single 16GB GPU
- a clear baseline from which every later improvement can be attributed to a specific intervention
