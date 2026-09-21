# B2-D Result — Tiling Branch Closed

Date: 2026-09-21

## Result

Diagnostic composition:
- 35 known scene-text perception/grounding errors
- 35 baseline-correct controls

Corrected B2-D result:
- rescued errors: **4 / 35**
- broken controls: **2 / 35**
- net diagnostic gain: **+2**
- decision: **STOP_TILING_BRANCH**

By failure type:

| Failure | Count | Rescued | Rescue rate |
|---|---:|---:|---:|
| OCR recognition | 21 | 3 | 14.3% |
| exact-string confusion | 11 | 1 | 9.1% |
| target localization | 3 | 0 | 0.0% |

Pre-registered gate:
- full scene-text run requires >=8 rescues and <=3 broken controls
- promising inspection requires 5–7 rescues and <=3 broken controls

Observed rescue count **4** is below both gates.

## Implementation correction check

The first run normalized scores only inside a/b/c/d. The corrected run used full-vocabulary log probabilities as planned.

The corrected run produced the **same aggregate outcome: 4 rescues / 2 regressions**.

Therefore the negative decision is not caused by the scoring normalization bug.

## Decision

Do not:
- run tiled inference over all 668 scene-text validation rows
- tune crop fraction on this diagnostic
- add more tile layouts
- spend inference latency on this branch

The branch has insufficient upside relative to complexity and ~5-view inference cost.

## Strategic implication

Inference-side experiments have now tested:
- OCR-aware prompting
- global higher max-resolution
- constrained-output diagnostics
- binding-aware prompting
- selective high+binding routing
- explicit multi-view tiling

Only the **price-only high+binding router** survived independent audit.

The remaining dominant failure mass is reasoning/association:
- number/text binding
- question understanding
- visual/spatial reasoning

Next branch: **training-side adaptation**, but only after a clean split and memory/environment preflight.
