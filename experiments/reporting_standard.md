# Experiment Reporting Standard

Every meaningful modeling experiment in this repository is recorded as an engineering decision, not only as a score.

The documentation should be readable by someone who did not participate in the project.

## Reader-first rule

Every experiment report should answer these five questions within the first screen:

1. **What was changed?**
2. **Why was it changed?**
3. **What model/pipeline was used?**
4. **How much did performance change?**
5. **What do we do next?**

Prefer short tables, diagrams, and compact callouts over long paragraphs.

## Recommended report order

### 0. One-screen summary
Start with a compact summary table:

| Item | Value |
|---|---|
| Model | ... |
| Single changed variable | ... |
| Random accuracy | ... |
| Grouped accuracy | ... |
| Delta vs previous best | ... |
| Decision | Adopt / Reject / Confirm |
| Next experiment | ... |

Then add one sentence:

> **Conclusion:** what this experiment taught us and why the next step follows.

### 1. Experiment identity
Record:
- run ID
- date
- stage
- parent baseline / previous best
- whether this is a new best

### 2. Modeling setup
Record:
- model/checkpoint
- model size/class
- whether competition data was used for training
- prompt version
- image-resolution policy
- quantization
- decision method
- seed
- hardware
- validation split

### 3. Strategy and hypothesis
State one explicit hypothesis.

Also record:
- what stays fixed
- what changes
- why this experiment is worth running
- what result would make us stop pursuing the idea

### 4. Results
Always report:
- overall accuracy
- OCR-heavy accuracy
- category accuracies
- correct/error counts
- parse failures
- runtime / throughput
- random and grouped results when available

### 5. Improvement vs previous best
Record:
- accuracy delta in percentage points
- OCR-heavy delta
- category deltas
- wrong -> right count
- right -> wrong count
- net sample gain
- whether the gain survives grouped validation

For the first baseline, explicitly write that improvement is not applicable.

### 6. Error interpretation
Explain what the result says about the likely bottleneck:
- OCR recognition
- target localization
- number/text binding
- question understanding
- spatial reasoning
- answer-position/decision instability
- parsing

Do not credit a component unless the ablation isolates it.

### 7. Decision
Choose one:
- adopt
- reject
- keep as diagnostic only
- confirm on grouped validation
- escalate to next stage

State the reason in one or two sentences.

### 8. Next experiment
Pick exactly one primary next experiment.

Record:
- hypothesis
- one changed variable
- fixed variables
- success criterion
- stop condition

### 9. Portfolio takeaway
Add 2-4 bullets describing what the experiment demonstrates professionally.

Examples:
- leakage-aware validation design
- controlled ablation rather than random tuning
- OCR-vs-reasoning bottleneck analysis
- compute-aware experimentation under a 16GB GPU constraint

## Visualization rule

Whenever the data supports it, include at least one visualization that helps answer a decision question.

Preferred visuals:

- **Model-evolution flowchart**: what changed between stages
- **Accuracy comparison bar chart**: random vs grouped
- **Category accuracy chart**: where the model is weak
- **Error composition chart**: where remaining errors are concentrated
- **Delta chart**: which categories improved or regressed
- **Wrong→Right / Right→Wrong summary**: whether a change is genuinely helpful

Avoid decorative charts. A visualization should exist only if it makes a modeling decision easier to understand.

## Writing style rule

- Use Korean or plain English consistently within a section.
- Prefer short sentences.
- Explain ML terms the first time they appear.
- Put the conclusion before the implementation detail.
- Use percentage points (pp) when describing accuracy changes.
- Do not call a tiny change an improvement without paired evidence.
- Clearly label hypotheses, measured facts, and interpretations.

## Model-evolution rule

The chronological model summary in `reports/model_evolution.md` must be updated after every important run.

The summary should answer:

1. What model/pipeline was used?
2. What single strategy changed?
3. What score changed?
4. Why was the change adopted or rejected?
5. What did the experiment teach us?

The top-level dashboard in `reports/README.md` should also be updated when a new best or major strategic decision is reached.
