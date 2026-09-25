# Model Card — BKT Research Baseline

## Purpose
Transparent next-response probability baseline for the ASSISTments 2009 research bundle.

## Model
Bayesian Knowledge Tracing with explicit fixed parameters. State is maintained separately for each observed learner-skill key and reset for each learner.

## Comparator models
Training-only global correctness rate and training-only per-skill correctness prior.

## Intended use
Research benchmarking and methodological inspection.

## Not intended for
High-stakes learner placement, permanent ability labeling, disciplinary action, or automated intervention without educator review.

## Important limitations
Fixed parameters are not fitted to the dataset; item difficulty and multi-skill structure are simplified; mastery is latent and not directly observed.
