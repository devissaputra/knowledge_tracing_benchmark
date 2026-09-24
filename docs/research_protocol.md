# Research protocol

## Project

Knowledge Tracing Benchmark

## Questions

1. How do BKT, DKT, and attention-based approaches compare under learner-aware splits?
2. How sensitive are results to sequence truncation and cold-start learners?
3. Are gains preserved after calibration and subgroup analysis?

## Baseline methods

- Bayesian Knowledge Tracing
- binary response traces
- explicit BKT parameters
- mastery probability updates
- benchmark ready interface

## Evidence to collect

Start from the current transparent baseline and record every transformation needed to produce a mastery probability after each observed binary response. Keep a clear boundary between synthetic demonstration data and any future empirical dataset.

## Validation

Use learner disjoint splits for generalization to new learners and temporal evaluation for future responses. Compare BKT with stronger models only after all methods share the same data preprocessing and split logic.

## What counts as a useful result

The next build should add one recurrent and one attention based implementation behind the same learner disjoint evaluation interface. Comparisons should include calibration and cold start behavior, not only predictive ranking.

## Threats to validity

Skill tagging errors, short sequences, changing item difficulty, multiple latent skills, and leakage across learner histories can distort knowledge tracing results.
