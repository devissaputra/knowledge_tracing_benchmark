# Model and Benchmark Card

## Purpose
This research bundle compares transparent and recurrent next-response probability models on learner-disjoint ASSISTments 2009 sequences.

It is designed for methodological inspection, reproducible benchmarking, and AI in Education research discussion. It is not a deployment-ready learner decision system.

## Compared models

### Global prior
A single correctness probability estimated from training interactions only.

### Skill prior
A training-only correctness rate for each observed skill key, with the global prior as fallback.

### Fixed BKT
Bayesian Knowledge Tracing with explicit fixed parameters. State is maintained independently for each learner-skill key and predictions are emitted before the current response is observed.

### Validation-tuned BKT
The same BKT state update with parameters selected on validation learners from the frozen grid in `src/run_experiment.py`.

### PFA-style logistic model
Logistic regression using skill identity and only prior learner-skill successes and failures. Current-response information is excluded from the features.

### Compact GRU knowledge tracer
A recurrent sequence model using previous skill-response tokens and the queried skill. Unknown training skills and the first interaction use the documented prior fallback. Full experiments run the prespecified seeds 13, 42, and 73.

This GRU is a compact benchmark implementation, not a claim of exact reproduction of a canonical DKT codebase.

## Data boundary
The study uses the pinned Atomi serialization of ASSISTments 2009 documented in `DATA.md`. The runner verifies aligned binary response sequences and rejects duplicate learner rows before splitting.

## Model-selection boundary
Training learners fit priors, PFA, and GRU parameters. Validation learners select BKT parameters and GRU stopping epoch. Final test learners are not used for model selection.

## Reported evidence
The bundle reports discrimination, probability quality, calibration, first-seen-versus-repeated skill slices, learner-block uncertainty, repeated-seed GRU stability, and skill-level error diagnostics.

The authoritative machine-readable output is `results/metrics.json`.

## Intended use
- research benchmarking;
- reproducibility review;
- learner-modeling method comparison;
- portfolio evidence for AI in Education research capability.

## Not intended for
- high-stakes learner placement;
- permanent ability or intelligence labeling;
- admissions or disciplinary decisions;
- automated intervention without separate validity evidence and educator oversight;
- claims that prediction accuracy demonstrates learning effectiveness.

## Important limitations
The dataset is historical and observational. Skill identifiers can be noisy or composite. Several models omit item difficulty and broader context. A knowledge-tracing probability is a model-dependent predictive state, not a direct measurement of mastery or ability. Results from one dataset and protocol do not establish general validity.
