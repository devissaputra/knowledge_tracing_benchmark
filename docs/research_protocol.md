# Research Protocol

## Question
Does transparent fixed-parameter BKT improve held-out next-response probability quality over non-stateful training priors on real ASSISTments 2009 learner sequences?

## Split
Learners are shuffled with seed 42 and split 70/15/15. The validation partition is reserved for future parameter/model selection; the current fixed-parameter BKT does not tune on it.

## Prediction timing
For a learner-skill state, the model first emits `P(correct)`, then observes the binary response, then updates mastery. This prevents current-response leakage.

## Baselines
1. global training correctness;
2. per-skill training correctness with global fallback;
3. fixed-parameter BKT.

## Primary metrics
- ROC-AUC, when both classes are present;
- Brier score;
- log loss.

## Secondary slices
- cold-start skill interactions: first time a learner encounters the observed skill key;
- repeated-skill interactions.

## Threats to validity
Fixed BKT parameters may be misspecified. Skill tags may not define independent knowledge components. Item difficulty is omitted. Learner histories are observational traces from one platform/ecosystem. Predictive metrics do not measure learning benefit.
