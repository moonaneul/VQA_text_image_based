# Step3-VL-10B BNB4 pilot result — 2026-09-23

## Setup
- checkpoint: stepfun-ai/Step3-VL-10B
- runtime: experimental bitsandbytes NF4 4-bit, BF16 compute
- fixed first 100 rows of VAL-A
- reference: FT Qwen3-VL-8B direct = 94/100

## Result
- Step3-VL-10B: 91/100 = 91.0%
- FT Qwen3-VL-8B: 94/100 = 94.0%
- rescue (FT wrong, Step3 right): 6
- regression (FT right, Step3 wrong): 9
- net: -3
- both right: 85
- both wrong: 0
- disagreement rate: 15%
- McNemar exact p = 0.60723876953125

By category:
- scene_text: tie, 47/51 each; rescue 4 / regression 4
- price: Step3 +1 net on n=7
- spatial: Step3 -2 net on n=5
- menu: Step3 -1 net on n=2
- other: Step3 -1 net on n=28

## Gate decision
The preregistered continuation rule was:
1. accuracy >=94/100, or
2. accuracy >=92/100 and rescue >=4.

Step3 scored 91/100, so it fails the gate despite 6 rescues.

## Decision
Stop the Step3 branch. Do not run confirmation 907 or test inference.

The broad alternate-model exploration is closed:
- MiniCPM-V-4.5 INT4: weak standalone; complementarity confirmed, routing precision insufficient.
- InternVL3-8B BNB4: first-100 gate passed but confirmation 907 failed strongly.
- Step3-VL-10B BNB4: first-100 gate failed.

Preserve the current best submission:
- Public LB: 0.95263
- file: output/ENSEMBLE_TEAM722_PERM1805_B3P3_0475_FT8B_05.csv
- weights: TEAM-C896 0.7220 / Perm8B 0.1805 / B3P3 0.0475 / FT8B640P0 0.0500
