# TEAM-E / Team Handoff Reconciliation — 2026-09-22

## Purpose

Record the gap between the team-level handoff state and the latest TEAM-E/B3P3 local results without modifying the team handoff's historical claims.

## TEAM-E state confirmed locally

- B3P3 inner-dev: B0 271/300 (90.33%) → QLoRA 275/300 (91.67%)
- Inner-dev paired changes: rescue 10, regression 6, net +4
- The preregistered inner-dev promotion gate (net >=4 AND regression <=4) was not passed.
- Same B3P3 adapter was later evaluated once on the frozen 400-row holdout:
  - B0 361/400 (90.25%)
  - B3P3 369/400 (92.25%)
  - rescue 13, regression 5, net +8
  - both correct 356, both wrong 26
  - McNemar exact p=0.0962524414
- B3P3 test inference completed for all 6,714 rows.
- Local artifacts:
  - output/baseline/B3P3_test_QLoRA_probs_v2/predictions.csv
  - output/baseline/B3P3_test_QLoRA_probs_v2/choice_probabilities.csv
  - output/B3P3_submission.csv
- Public LB for B3P3 remains unconfirmed in this record.

## Team-artifact availability conflict on the current machine

The team handoff names the following artifacts/paths:
- predictions/EXP-001T_test_scoring.csv
- predictions/EXP-003T_8B_P0P1_PERM.csv
- C:/Users/SSAFY/Downloads/submission_probs.csv (TEAM-C)

On the current project machine/session, direct existence checks returned **False** for all three paths.

Therefore:
- do not claim those team probability files are locally available here;
- do not re-run expensive 4B/8B/P1 inference just to recreate them;
- D-PERM vs B3P3 complementarity analysis is blocked until the saved team probability artifacts are transferred or their actual workspace path is provided.

## Next low-compute step

1. Preserve current D-PERM 0.95055 as the team baseline.
2. Obtain/transfer the saved team probability artifacts from the workspace described by TEAM_AI_HANDOFF.
3. Until then, B3P3 standalone submission can be evaluated using the already-created output/B3P3_submission.csv without any new GPU inference.
4. Once team probabilities are available, run CPU-only complementarity analysis before considering any new ensemble submission.

No new training or inference is authorized by this note.


## Latest team artifact update — 2026-09-22 11:52 KST report

A teammate supplied three probability artifacts and a newer Public LB state after the original TEAM_AI_HANDOFF snapshot:

- TEAM-C 896 P0: `TEAM-C896_P0_probs.csv`, Fine-tuned Qwen3-VL-4B, image token 896, n_perm=1, team-reported Public LB **0.94518**
- Local Qwen3-VL-4B zero-shot: `EXP-001T_test_scoring.csv`, team-reported Public LB **0.92701**
- Local Qwen3-VL-8B P0+P1 permutation: `EXP-003T_8B_P0P1_PERM.csv`, team-reported Public LB **0.94250**

The teammate additionally reports the latest team best as:
- **TEAM-C896 80% + Permutation-8B 20%**
- Public LB **0.95144**

This is newer than the original handoff snapshot's D-PERM 0.95055. Treat it as a **newer TEAM_REPORTED state** until the team Decision Log / registry is synchronized; do not silently rewrite the historical handoff snapshot.

### Uploaded artifact integrity checks

The three supplied CSVs were inspected directly:
- each has **6,714 rows**
- all three have identical ID sets
- duplicate IDs: **0**
- no NaN/non-finite candidate scores after normalization
- four-choice score sums normalize cleanly to 1

Pairwise argmax disagreement on test:
- TEAM-C896 vs Permutation-8B: **390 / 6,714 = 5.81%**
- TEAM-C896 vs Local-4B: **357 / 6,714 = 5.32%**
- Permutation-8B vs Local-4B: **429 / 6,714 = 6.39%**

Three-model agreement pattern:
- all three same: **6,146**
- TEAM-C896 = Permutation-8B only: **178**
- TEAM-C896 = Local-4B only: **211**
- Permutation-8B = Local-4B only: **139**
- all three different: **40**

Using weighted log probabilities with weights TEAM-C896=0.80 and Permutation-8B=0.20, the reconstructed 80/20 ensemble changes **130** test argmax predictions relative to TEAM-C896 and **270** relative to Permutation-8B. Test gold is unavailable, so these changes are diversity diagnostics only.

### Remaining blocker for B3P3 complementarity

The B3P3 probability artifact is still required in the analysis environment:
`output/baseline/B3P3_test_QLoRA_probs_v2/choice_probabilities.csv`

Once supplied, perform the planned CPU-only ID-joined analysis against the current 80/20 team ensemble before proposing any new GPU work.
