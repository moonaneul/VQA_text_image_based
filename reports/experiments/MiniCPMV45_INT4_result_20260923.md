# MiniCPM-V-4.5 INT4 diversity result — 2026-09-23

## Fixed pilot
First 100 VAL-A:
- FT-8B direct: 94/100
- MiniCPM-V-4.5 INT4: 89/100
- rescue 4, regression 9, disagreement 14%

## Confirmation 907
- FT-8B direct: 858/907 = 94.5976%
- MiniCPM-V-4.5 INT4: 787/907 = 86.7696%
- rescue: 17
- regression: 88
- both right: 770
- both wrong: 32
- disagreement: 12.6792%

The predefined complementarity check (>=17 rescues among the 49 FT errors) was met exactly, but MiniCPM is much weaker standalone.

## Post-hoc FT-confidence analysis on the 907
Among disagreement cases:
- regression FT confidence median: 0.981528
- rescue FT confidence median: 0.658458
- regression FT margin median: 0.972533
- rescue FT margin median: 0.492921

This shows strong separation in distribution, but simple low-margin routing was not strong enough:
- q=0.1 margin gate: -0.110 pp
- q=0.2 margin gate: +0.221 pp (8 rescue, 6 regression; net +2)
- q=0.3: +0.000 pp
- q=0.4: -0.662 pp
- q=0.5: -1.985 pp

Because these thresholds were inspected post-hoc on the same confirmation set, the +0.221 pp result is diagnostic only and is not a clean adoption result.

## Decision
Do not run MiniCPM on test and do not submit a MiniCPM-routed ensemble from this analysis. Preserve it as evidence that a different model can rescue a meaningful fraction of FT-8B errors, but move to a stronger alternate VLM family.
