# Research Bundle Evidence Contract

## Identity

**Area:** AI in Education  
**Study:** multi-model knowledge tracing on learner-disjoint ASSISTments 2009 sequences  
**Retrieval artifact:** pinned Atomi/ASSISTments2009 sequence representation

## Required evidence

A valid full run must record or enforce:

1. mirror repository, exact revision, parquet path, and SHA-256;
2. learner and interaction counts;
3. one unique serialized learner row per `user_id`, with duplicate learner rows rejected;
4. learner-disjoint train/validation/test counts and split seed;
5. train-only global and skill priors;
6. fixed and validation-tuned BKT parameters;
7. PFA features using only prior learner-skill outcomes;
8. GRU skill vocabulary, seeds, validation histories, and fallback rules;
9. pre-response prediction timing;
10. ROC-AUC, average precision, Brier, log loss, and ECE-10;
11. first-seen skill for learner versus repeated-skill slices;
12. learner-block bootstrap Brier differences versus the skill prior;
13. repeated-seed GRU summary;
14. high-error skill diagnostics with minimum support;
15. software environment and generated calibration figure.

## Authoritative evidence

The single machine-readable empirical source is `results/metrics.json`.

Generated derivatives are:
- `results/summary.md`;
- `paper/results.md`;
- `results/figures/calibration.png`.

A second independent status JSON is intentionally not maintained because duplicate result manifests can become contradictory.

## Statistical boundary

Interactions within one learner are dependent. Uncertainty therefore resamples learners as blocks instead of pretending individual responses are independent observations.

Repeated neural seeds measure optimization sensitivity under the same dataset and split; they are not independent replications across populations.

## Terminology boundary

`first_seen_skill_for_learner` means the first occurrence of that observed skill key in a particular learner's sequence. It must not be described as a skill unseen in training.

## Non-claims

The bundle does not claim latent knowledge is directly observed, prediction quality implies learning benefit, the GRU is a canonical DKT reproduction, one dataset generalizes to all learners, or any model should make high-stakes learner decisions.
