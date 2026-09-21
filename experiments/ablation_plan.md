# Ablation Plan

## Principle

Change one factor at a time, keep per-sample predictions, and compare the same validation samples pairwise.

## Run matrix

| Run ID | Prompt | Resolution | Decision | OCR | Purpose |
|---|---|---:|---|---|---|
| B0 | direct | 0.8MP | generation | off | reproducible baseline |
| A1 | ocr_deliberate | 0.8MP | generation | off | prompt effect |
| A2 | winner | 1.0MP | generation | off | high-resolution effect |
| A3 | winner | 0.4MP | generation | off | low-resolution sensitivity |
| A4 | winner | best | constrained label scoring | off | choice-decision stability |
| A5 | winner | best | best | on | OCR contribution |
| A6 | best | best | best | best | QLoRA contribution |

Do not run all combinations up front.

## Decision rules

### Prompt
Promote the new prompt if:
- overall accuracy improves meaningfully, and
- OCR-heavy categories improve without a large regression elsewhere.

### Resolution
Promote 1.0MP if:
- overall improves by roughly >=0.7-1.0 pp, or
- OCR-heavy improves by roughly >=1.5-2.0 pp,
- with acceptable runtime/VRAM cost.

These are working thresholds, not statistical guarantees.

### Choice scoring
Prefer constrained a/b/c/d scoring if it:
- removes parse failures,
- is at least as accurate as free generation,
- improves prediction stability or confidence calibration.

### Permutation diagnostic
Use a stratified subset first.
Escalate only when answer-order sensitivity is material.

Useful warning signals:
- semantic consistency <90%, or
- accuracy changes by >=1.5 pp across option orders.

### External OCR
Escalate only after manual error analysis confirms recognition failures are a major bottleneck.

### QLoRA
Start only after inference-side ablations mostly plateau and remaining errors appear learnable rather than visibility-limited.

---

## Pairwise comparison

For every pair of runs, record:

```
old_wrong_new_right
old_right_new_wrong
both_right
both_wrong
net_gain
```

This is more informative than accuracy alone.

When enough samples are available, use paired bootstrap confidence intervals or McNemar's test to avoid over-interpreting tiny differences.

---

## W&B run schema

Suggested config fields:

```
model_name
prompt_version
resolution_name
min_pixels
max_pixels
decision_method
ocr_enabled
quantization
seed
split_name
git_commit
```

Suggested summary metrics:

```
accuracy
ocr_heavy_accuracy
scene_text_accuracy
price_accuracy
phone_accuracy
menu_accuracy
count_accuracy
color_accuracy
spatial_accuracy
other_accuracy
parse_failure_rate
mean_latency
total_runtime
peak_vram
generalization_gap
```

Suggested artifacts:

- per-sample prediction CSV
- metrics JSON
- run config JSON
- optional error-analysis CSV
- optional confusion/change table between runs

---

## Experiment discipline

- No leaderboard-driven prompt tweaking.
- No test-label inference from answer-position frequencies.
- No use of dev answer1-5 as pseudo-labels until their reliability is validated on a manually labeled subset.
- Do not call training loss a model-selection metric.
- Do not change prompt and resolution in the same comparison unless specifically testing interaction.
