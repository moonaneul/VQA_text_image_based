# B3-P1 — Fresh QLoRA Holdout Baseline

Date: 2026-09-21

## Split / environment

- original train: **6,714**
- IDs excluded from prior random/grouped/audit sets: **2,720**
- safe pool: **3,994**
- QLoRA train: **3,594**
- fresh QLoRA holdout: **400**
- normalized-template overlap train↔holdout: **0**
- image-path overlap train↔holdout: **0**

Environment:
- GPU: NVIDIA GeForce RTX 5060 Ti
- BF16: supported
- PEFT: installed
- bitsandbytes: installed
- accelerate: installed
- 4-bit model load: **success**
- 4-bit load peak VRAM: **3.50 GB**

## B0 on fresh holdout

Configuration:
- Qwen2.5-VL-3B-Instruct
- direct prompt
- standard resolution
- no quantization
- greedy generation

Result:
- **361 / 400 = 90.25%**
- parse failures: 0

By category:

| Category | Accuracy | Count |
|---|---:|---:|
| scene_text | **90.37%** | 218 |
| price | **92.31%** | 39 |
| other | **92.38%** | 105 |
| spatial | **81.25%** | 16 |
| menu | **83.33%** | 12 |
| count | **100%** | 7 |
| phone | 33.33% | 3 |

Phone n=3 is too small for a meaningful category conclusion.

## Interpretation

The fresh holdout baseline (**90.25%**) is close to the earlier random B0 (**90.68%**), which is reassuring for the split quality.

This holdout is now the training branch's locked final evaluation set. It must not be used for hyperparameter search.

Before training, evaluate the already-locked price-only router on the same 400 rows. That establishes the true current-best inference baseline that QLoRA must beat.

## Next

1. Extract 39 price rows.
2. Run high + binding-aware on price only.
3. Merge those predictions into the B0 400-row predictions.
4. Compare B0 vs routed-current-best.
5. Freeze that score.
6. Only then train one pre-registered QLoRA configuration.
