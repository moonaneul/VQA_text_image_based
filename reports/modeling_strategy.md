# Modeling Strategy

## 1. Problem framing

This challenge is a Korean 4-choice VQA task, but the current EDA indicates that the dominant difficulty is not generic object recognition.

- train: 6,714 samples
- dev: 2,683 samples
- test: 6,714 samples
- scene-text + price + phone + menu:
  - train: 68.7%
  - test: 67.7%
- median image scale is roughly 720x960 (~0.69 MP)
- about 72% of images are portrait-oriented

Observed failure-prone cases include small storefront text, overexposed night signs, phone numbers, price boards, multi-column menus, dates, product names, and exact-string comparisons.

**Working hypothesis:** the main bottleneck is a mixture of OCR recognition, target localization, text-to-entity binding, and multiple-choice decision stability.

Therefore the project prioritizes controlled inference-side experiments before fine-tuning.

---

## 2. Evaluation design

Two validation views are maintained.

### Random stratified validation
Primary score for leaderboard-like distribution.

- 1,341 samples
- stratified by answer position and question type
- 95 normalized question-template overlaps remain between train and validation
- no exact-image overlap

### Grouped validation
Generalization guardrail.

- 1,343 samples
- normalized question-template groups and exact-image SHA groups are kept intact
- zero template overlap between train and validation
- zero exact-image overlap

The two scores are not collapsed into one number.

```
generalization_gap = random_accuracy - grouped_accuracy
```

Interpretation:

- Random ↑ / Grouped ↑: preferred
- Random ↑ / Grouped ≈: may exploit leaderboard-like template structure
- Random ↑ / Grouped ↓: possible shortcut learning or overfitting
- Random ≈ / Grouped ↑: generalization improvement, leaderboard effect uncertain
- both ↓: reject

---

## 3. Sequential ablation roadmap

Only one factor is changed at a time.

| Stage | Fixed | Changed | Main question |
|---|---|---|---|
| B0 | Qwen2.5-VL-3B, 0.8MP, generation | none | What is the reproducible zero-shot baseline? |
| A1 | model, 0.8MP, generation | prompt | Does OCR-aware prompting help? |
| A2 | model, winning prompt, generation | resolution 0.8→1.0MP | Is small-text visibility a bottleneck? |
| A3 | model, winning prompt, generation | resolution 0.8→0.4MP | How sensitive is performance to image detail? |
| A4 | best prompt/resolution | generation→constrained choice scoring | Is output/choice instability a bottleneck? |
| A5 | best VLM setup | no OCR→external OCR | Does explicit text extraction add signal? |
| A6 | best inference setup | zero-shot→QLoRA | Is task-specific adaptation still necessary? |

A full Cartesian search is intentionally avoided. If two factors appear to interact, a small 2x2 follow-up experiment is used.

---

## 4. Core metrics

Every run should record:

- overall accuracy
- scene_text / price / phone / menu / count / color / spatial / other accuracy
- OCR-heavy aggregate accuracy
- random accuracy
- grouped accuracy
- random-grouped gap
- parse failure rate
- mean latency
- total runtime
- peak VRAM if available

Per-sample prediction logs should include:

```
id
question_type
answer
prediction
correct
raw_output
score_a
score_b
score_c
score_d
confidence_margin
```

The score columns are optional until constrained scoring is implemented.

---

## 5. Error taxonomy

Errors are separated before choosing the next modeling intervention.

| Error type | Meaning | Likely next action |
|---|---|---|
| OCR recognition | correct region, wrong characters | higher resolution / OCR |
| target localization | model reads the wrong sign or region | crop / region proposal / bbox-aware OCR |
| number binding | numbers are read but attached to wrong item | bbox-aware reasoning |
| exact-string confusion | near-match string chosen | candidate comparison / scoring |
| question understanding | text is visible but intent is misunderstood | prompt / fine-tuning |
| visual reasoning | count/color/spatial error unrelated to text | base VLM reasoning |
| choice-position bias | answer changes with option order | constrained scoring / permutation diagnostic |
| output parsing | answer intent is correct but parsing fails | constrained output |
| ambiguity / label issue | sample itself is uncertain | manual review / flag |

A diagnostic subset of roughly 150-200 errors should be manually annotated before expensive changes.

---

## 6. External OCR entry criteria

External OCR is not added automatically.

Start OCR experiments only if all of the following are approximately true:

1. OCR-heavy categories account for a large share of remaining errors.
2. Manual review shows at least ~30-40% of sampled errors are genuine text-recognition failures.
3. Increasing resolution from 0.8MP to 1.0MP gives limited improvement.

Test OCR in increasing complexity:

1. plain OCR transcript
2. OCR transcript + bounding boxes
3. selective OCR only for OCR-heavy question types
4. selective top-k OCR lines relevant to the question/options

---

## 7. Fine-tuning entry criteria

Do not start QLoRA simply because zero-shot is imperfect.

Start a pilot only after:

- prompt/resolution/scoring improvements have mostly saturated
- a meaningful portion of remaining errors occur even when the necessary text is visible
- OCR does not resolve the dominant remaining failure modes
- the error analysis suggests learnable task-specific reasoning/binding mistakes

Pilot success should be judged by validation accuracy, not training loss.

Provisional continuation criteria versus the best non-fine-tuned baseline:

- Random: +1.0 to +1.5 pp or more
- Grouped: non-decreasing, preferably +0.5 pp or more

If Random improves while Grouped drops materially, treat the result as possible memorization rather than success.

---

## 8. QLoRA data-collator requirement

The historical notebook behavior

```python
labels = input_ids.clone()
```

must not be reused.

Loss should be applied only to the assistant answer span.

Conceptually:

```
system/user/image/prompt tokens -> -100
assistant answer token(s)       -> supervised
padding                         -> -100
```

Multimodal processor outputs required by Qwen must be preserved, including image tensors/grid metadata.

---

## 9. RTX 5060 Ti 16GB practical scope

Reasonable:

- Qwen2.5-VL-3B inference
- 0.8-1.0MP batch-1 experiments
- constrained choice scoring
- selective permutation diagnostics
- external OCR preprocessing
- 4-bit QLoRA with gradient checkpointing and accumulation

Low priority:

- full-parameter fine-tuning
- exhaustive prompt x resolution x OCR x permutation grid search
- large multi-model ensembles before error diversity is established

---

## 10. Portfolio narrative

The project is documented as an evidence-driven modeling process rather than a single final score.

1. inspect the complete dataset and identify the dominant OCR-heavy structure
2. design leakage-aware random and grouped validation
3. define bottleneck hypotheses
4. run controlled ablations
5. classify errors into recognition/localization/binding/reasoning failures
6. adopt only components whose contribution is measured
7. move to QLoRA only when inference-side fixes are no longer sufficient

This makes the project explainable, reproducible, and suitable for ML/AI portfolio review.
