# Calculation guide

## Question and evidence

How much does learner history improve next-response prediction?

ASSISTments 2009 representation: 4,148 learners, 274,331 interactions.

**Status:** RECORDED EXTERNAL-DATA STUDY | full experiment not rerun in this review.

## Design

Learner-disjoint train/validation/test; priors, BKT, PFA and GRU; predict before updating with the current answer.

## Calculation and interpretation

`BKT prediction = mastery×(1-slip) + (1-mastery)×guess.`

After a response, Bayes updating and the learning transition produce the next mastery state. Brier and log loss assess probabilities; learner-block bootstrap respects within-learner dependence. Model mastery is not directly measured knowledge.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| skill_prior | 0.20965724966908303 | Brier ↓ | `test_metrics.skill_prior.all.brier` |
| bkt_fixed | 0.21870446416122305 | Brier ↓ | `test_metrics.bkt_fixed.all.brier` |
| bkt_validation_tuned | 0.20237712440747457 | Brier ↓ | `test_metrics.bkt_validation_tuned.all.brier` |
| pfa_logistic | 0.19760582856803 | Brier ↓ | `test_metrics.pfa_logistic.all.brier` |
| gru_kt_seed_42 | 0.1799516881075459 | Brier ↓ | `test_metrics.gru_kt_seed_42.all.brier` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

On 623 held-out learners, the seed-42 GRU achieves ROC-AUC 0.7470 and Brier score 0.1800, compared with 0.6982 and 0.1976 for PFA. The three GRU seeds average ROC-AUC 0.7471 with sample SD 0.0008. These results support better response prediction under this historical benchmark; they do not establish instructional benefit or justify high-stakes interpretations of a learner’s knowledge.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. Stored empirical results were inspected, not independently reproduced from raw data.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`predict_correct_probability`](src/knowledge_tracing_benchmark/core.py#L23) | Inspect the explicit implementation and its callers. |
| [`update_mastery`](src/knowledge_tracing_benchmark/core.py#L30) | Inspect the explicit implementation and its callers. |
| [`trace`](src/knowledge_tracing_benchmark/core.py#L44) | Inspect the explicit implementation and its callers. |
| [`predictive_trace`](src/knowledge_tracing_benchmark/core.py#L56) | Inspect the explicit implementation and its callers. |
| [`aligned`](src/run_experiment.py#L45) | Inspect the explicit implementation and its callers. |
| [`validate_unique_learner_rows`](src/run_experiment.py#L55) | Require exactly one serialized row per learner before any partitioning. |
| [`load_real_rows`](src/run_experiment.py#L72) | Inspect the explicit implementation and its callers. |
| [`split_rows`](src/run_experiment.py#L106) | Inspect the explicit implementation and its callers. |
| [`train_priors`](src/run_experiment.py#L116) | Inspect the explicit implementation and its callers. |
| [`expected_calibration_error`](src/run_experiment.py#L133) | Inspect the explicit implementation and its callers. |
| [`metric_bundle`](src/run_experiment.py#L146) | Inspect the explicit implementation and its callers. |
| [`prediction_frame`](src/run_experiment.py#L161) | Inspect the explicit implementation and its callers. |
| [`prior_predictions`](src/run_experiment.py#L171) | Inspect the explicit implementation and its callers. |
| [`bkt_predictions`](src/run_experiment.py#L191) | Inspect the explicit implementation and its callers. |
| [`tune_bkt`](src/run_experiment.py#L209) | Inspect the explicit implementation and its callers. |
| [`pfa_features`](src/run_experiment.py#L225) | Inspect the explicit implementation and its callers. |
| [`fit_pfa`](src/run_experiment.py#L243) | Inspect the explicit implementation and its callers. |
| [`predict_pfa`](src/run_experiment.py#L255) | Inspect the explicit implementation and its callers. |
| [`collate_kt`](src/run_experiment.py#L288) | Inspect the explicit implementation and its callers. |
| [`skill_vocabulary`](src/run_experiment.py#L313) | Inspect the explicit implementation and its callers. |
| [`sequence_loss`](src/run_experiment.py#L318) | Inspect the explicit implementation and its callers. |
| [`train_gru`](src/run_experiment.py#L337) | Inspect the explicit implementation and its callers. |
| [`predict_gru`](src/run_experiment.py#L366) | Inspect the explicit implementation and its callers. |
| [`sliced_metrics`](src/run_experiment.py#L394) | Inspect the explicit implementation and its callers. |
| [`summarize_gru_seed_runs`](src/run_experiment.py#L405) | Summarize full-test GRU metrics across prespecified random seeds. |
| [`learner_block_bootstrap_brier_delta`](src/run_experiment.py#L421) | Inspect the explicit implementation and its callers. |
| [`skill_error_analysis`](src/run_experiment.py#L443) | Inspect the explicit implementation and its callers. |
| [`write_calibration_figure`](src/run_experiment.py#L452) | Inspect the explicit implementation and its callers. |
| [`write_summary`](src/run_experiment.py#L478) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L514) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L598) | Inspect the explicit implementation and its callers. |
| [`forward`](src/run_experiment.py#L307) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

After a response, Bayes updating and the learning transition produce the next mastery state. Brier and log loss assess probabilities; learner-block bootstrap respects within-learner dependence. Model mastery is not directly measured knowledge. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
