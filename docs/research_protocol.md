# Research Protocol

## Study objective
Evaluate next-response probability models on real ASSISTments 2009 learner sequences under a learner-disjoint protocol, while separating transparent baselines from higher-capacity sequence models.

## Research questions
1. Do stateful learner models improve held-out next-response probability quality over global and skill priors?
2. Does validation-tuned Bayesian Knowledge Tracing improve over a fixed BKT parameterization?
3. Does a PFA-style logistic model using only prior learner-skill successes and failures improve probability quality?
4. Does a compact GRU knowledge tracer add useful sequence information beyond transparent baselines?
5. Do conclusions differ between a learner's first observed encounter with a skill and repeated encounters?
6. Are Brier-score differences robust when uncertainty is resampled at the learner level?
7. How stable is the GRU result across the prespecified random seeds?

## Data and study unit
The scientific source is ASSISTments 2009. The executable pipeline retrieves the pinned Atomi/ASSISTments2009 sequence mirror described in `DATA.md`.

One serialized row must correspond to one unique `user_id`. The runner rejects duplicate learner rows before partitioning. This makes the learner-disjoint claim an executable invariant rather than a documentation-only assumption.

## Split and model-selection boundary
Learners are deterministically shuffled with seed 42 and split:
- 70% training;
- 15% validation;
- 15% final test.

No learner may cross partitions.

Training learners are used to estimate global and skill priors, fit PFA, construct the GRU skill vocabulary, and train GRU parameters. Validation learners are used to select BKT parameters and GRU stopping epoch. Final test learners are evaluated only after model selection.

## Prediction timing
For every stateful model, the probability for the current interaction is emitted before the current response is observed. Historical responses may inform future predictions, but the response being predicted may not leak into its own probability.

## Models
1. Global training correctness rate.
2. Training-skill prior with global fallback.
3. Fixed-parameter BKT.
4. Validation-tuned BKT over the frozen grid in `src/run_experiment.py`.
5. PFA-style logistic regression with skill identity plus prior learner-skill success and failure counts.
6. Compact GRU knowledge tracer using response-conditioned skill tokens and a queried skill embedding.

The GRU is a compact recurrent benchmark. It is not presented as a canonical reproduction of every Deep Knowledge Tracing implementation.

## Evaluation
Primary held-out metrics:
- ROC-AUC;
- average precision;
- Brier score;
- log loss;
- ECE-10 calibration error.

The evaluation additionally reports:
- first-seen skill for learner versus repeated-skill slices;
- learner-block bootstrap Brier differences relative to the skill-prior baseline;
- high-error skills with minimum support;
- repeated-seed GRU stability across seeds 13, 42, and 73.

The phrase **first-seen skill for learner** means the learner has not previously encountered that observed skill key in the sequence. It does not mean the skill is absent from the training partition.

## Statistical boundary
Interactions from the same learner are dependent. Uncertainty therefore resamples learners as blocks rather than treating interactions as independent observations.

Repeated GRU runs quantify random-seed sensitivity. They are not independent dataset replications.

## Threats to validity
ASSISTments 2009 is a historical tutoring dataset. Skill tags can be noisy or composite, opportunity order reflects platform and curriculum decisions, and item difficulty and other contextual variables are omitted from several models. Predictive metrics do not measure instructional benefit, causal learning gain, fairness, or validity for consequential learner decisions.

## Authoritative outputs
`results/metrics.json` is the machine-readable empirical record. `results/summary.md`, `paper/results.md`, and `results/figures/calibration.png` are generated from the same experiment runner and should not be hand-edited with independent numerical claims.
