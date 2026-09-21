# B3-P1b — Price Router Fails on Fresh QLoRA Holdout

Date: 2026-09-21

## Result

Fresh QLoRA holdout: 400 rows.

B0:
- **361/400 = 90.25%**

Price-only routed candidate:
- **360/400 = 90.00%**
- net **-1**
- prediction changes: 2
- wrong→right: 0
- right→wrong: 1

Price subset:
- B0: **36/39 = 92.31%**
- high + binding-aware: **35/39 = 89.74%**
- net **-1**

## Decision

The price-only router is **not robust enough to remain the deployment/default candidate**.

It previously showed positive gains on random/grouped and on the first fresh audit, but this second fresh evaluation moved in the opposite direction.

Therefore:
- current best reference returns to **B0 direct + standard**
- price router is retained only as an exploratory ablation result
- do not use it as the default test-time routing rule
- do not tune the router on this holdout

## Training-evaluation discipline

The 400-row QLoRA holdout has now been used to compare fixed inference systems, but it has **not** been used for training hyperparameter selection.

From this point:
- freeze the 400-row holdout
- create an inner dev split from qlora_train
- select QLoRA configuration only on inner dev
- retrain the selected configuration on all 3,594 qlora_train rows
- evaluate the final adapter on the 400-row holdout exactly once
