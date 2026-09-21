# Experiment Reporting Standard

Every meaningful modeling experiment in this repository is recorded as an engineering decision, not only as a score.

## Required sections

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

State the reason.

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

## Model-evolution rule

The chronological model summary in `reports/model_evolution.md` must be updated after every important run.

The summary should answer:

1. What model/pipeline was used?
2. What single strategy changed?
3. What score changed?
4. Why was the change adopted or rejected?
5. What did the experiment teach us?
