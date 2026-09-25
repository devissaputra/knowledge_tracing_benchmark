# Research Bundle Evidence Contract

## Identity

**Area:** AI in Education  
**Study:** multi-model knowledge tracing on learner-disjoint ASSISTments 2009 sequences  
**Retrieval artifact:** Atomi/ASSISTments2009 pinned revision

## Required evidence

A valid full run records:

1. mirror repository, exact revision, parquet path and SHA-256
2. learner and interaction counts
3. learner-disjoint train/validation/test counts and split seed
4. train-only global and skill priors
5. fixed and validation-tuned BKT parameters
6. PFA feature definition using only prior learner-skill outcomes
7. GRU skill vocabulary, seeds, validation histories and fallback rules
8. pre-response prediction timing
9. ROC-AUC, average precision, Brier, log loss and ECE-10
10. first-seen versus repeated-skill slices
11. learner-block bootstrap Brier differences versus skill prior
12. high-error skill diagnostics with minimum support
13. software environment and generated calibration figure

## Statistical boundary

Interactions within one learner are dependent. The uncertainty analysis therefore resamples learners as blocks instead of pretending individual responses are independent observations.

## Non-claims

The bundle does not claim latent knowledge is directly observed, prediction quality implies learning benefit, the GRU is a canonical DKT reproduction, one dataset generalizes to all learners, or any model should make high-stakes learner decisions.
