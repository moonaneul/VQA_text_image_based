# B2-D — Scene-text Multi-view Grounding Diagnostic

Date: 2026-09-21

## Why B2 now?

The price-only router is frozen after passing the fresh audit.

The largest remaining error category is scene_text.

B0 random scene_text:
- 668 samples
- 59 errors

Manual root-cause breakdown inside scene_text:

| Cause | Errors |
|---|---:|
| OCR recognition | 21 |
| exact-string confusion | 11 |
| target localization | 3 |
| question understanding | 11 |
| visual/spatial reasoning | 6 |
| number/text binding | 5 |
| ambiguity/label | 2 |

The first three form a **perception/grounding family: 35/59 = 59.3%**.

A1 OCR-aware prompt and A2 global high-resolution did not improve scene_text. Therefore the next visual experiment must be structurally different from "tell the model to read better" or "slightly raise max_pixels."

## Diagnostic design

Build a 70-row diagnostic:
- all **35 scene_text perception/grounding errors**
- **35 matched scene_text samples B0 already gets right**

This is intentionally not an unbiased score estimate. It is a mechanism diagnostic.

## B2-D method

For each image create 5 views:
1. full image
2. top-left 60% crop
3. top-right 60% crop
4. bottom-left 60% crop
5. bottom-right 60% crop

The corner crops overlap in the center.

For each view:
- same direct prompt
- standard resolution
- compute logits for single-token a/b/c/d labels

Primary aggregation is locked before evaluation:

```text
combined(choice)
  = full-image log P(choice)
  + max over crops log P(choice)
```

This preserves global context while allowing one enlarged crop to contribute strong local evidence.

## Diagnostic gate

Primary metric:
- rescue among 35 known perception errors
- damage among 35 correct controls

Decision:

| Result | Next |
|---|---|
| **>=8 rescued and <=3 controls broken** | run full scene_text validation |
| 5–7 rescued and <=3 broken | promising; inspect changed cases first |
| <5 rescued or >3 controls broken | stop tiling branch |

The threshold is fixed before the run.

## Why not QLoRA yet?

Because 35 scene_text errors are specifically perception/localization related and may be recoverable without training.

If B2-D fails, that is evidence to stop spending inference complexity on this branch and move toward reasoning-focused QLoRA.
