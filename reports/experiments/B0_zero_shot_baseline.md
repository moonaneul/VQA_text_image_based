# B0 Zero-shot Baseline Result

Date: 2026-09-21

## Configuration

- Model: Qwen2.5-VL-3B-Instruct
- Prompt: direct
- Resolution: standard
  - min_pixels: 200,704
  - max_pixels: 802,816 (~0.8MP)
- Quantization: none
- Decision: free generation
- Seed: 20260921
- GPU: NVIDIA GeForce RTX 5060 Ti
- PyTorch: 2.11.0+cu128

## Results

| Metric | Random validation | Grouped validation |
|---|---:|---:|
| Samples | 1,341 | 1,343 |
| Accuracy | 90.68% | 91.73% |
| Correct | 1,216 | 1,232 |
| Errors | 125 | 111 |
| Unparsed | 0 | 0 |
| OCR-heavy accuracy | 89.91% | 91.55% |
| Runtime | 792.3s | 792.3s |
| Throughput | 1.69 rows/s | 1.70 rows/s |

Grouped minus random accuracy: **+1.06 percentage points**.

Because B0 is zero-shot and does not train on the project train split, this gap should not be interpreted as template memorization. At this stage it mainly indicates that the grouped validation sample is slightly easier for this model despite having stricter template/image grouping.

## Category results

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

## Error concentration

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

## Interpretation

1. Qwen2.5-VL-3B-Instruct is a strong baseline for this dataset.
2. Output parsing is not currently a bottleneck: 0 unparsed responses.
3. The largest absolute error pool is scene_text because it is the dominant category.
4. Price, phone, and menu have lower accuracy than the overall baseline and are high-value targets for OCR-aware improvements.
5. The random/grouped difference is not evidence of overfitting at B0 because the model has not been fine-tuned on the competition train split.
6. Once fine-tuning starts, the same random/grouped comparison becomes important for detecting template-dependent gains.

## Next experiment

**A1: prompt-only ablation**

Keep model, resolution, quantization, generation settings, seed, and validation split fixed.

Change only:

```
direct -> ocr_deliberate
```

Run A1 on random validation first. Compare B0 vs A1 sample-by-sample.

Primary signals:

- overall accuracy delta
- OCR-heavy accuracy delta
- scene_text / price / phone / menu deltas
- B0 wrong -> A1 right
- B0 right -> A1 wrong

If A1 produces a meaningful gain, confirm the same configuration on grouped validation before moving to resolution experiments.
