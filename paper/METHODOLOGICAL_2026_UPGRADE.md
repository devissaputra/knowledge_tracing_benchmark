> **ARCHIVED DRAFT NOTICE (26 October 2026):** This file preserves the design-stage manuscript with PENDING fields. Its outcomes are now executed and documented in [METHODOLOGICAL_2026_EXECUTED_RESULTS.md](METHODOLOGICAL_2026_EXECUTED_RESULTS.md) and the complete [main_methodological_2026.tex](main_methodological_2026.tex). Do not submit this archived draft as the final paper.\n\n# Toward a History-Budget Sensitivity Audit for Knowledge Tracing: Source Provenance, Information-Restricted Baselines, and Cross-Year Evidence

**26 October 2026 — research draft, not a completed new experiment or a submission-ready replacement.** The numerical results of the newly frozen Stage-3 history-budget study must be inserted only after independent runner success and author QA. Earlier verified numerical results below are historical evidence and were already used for earlier preprint drafts.

**Author:** Devis Wawan Saputra (authorship and contribution disclosure require personal approval)

## Abstract (conditional working version)

Knowledge tracing benchmarks often compare model families whose access to historical response information and whose underlying data transformations differ. Rather than treating historical model rankings as architecture effects, this study audits three separate sources of benchmark sensitivity: event-level provenance, the amount of prior learner history made available to predictors, and representation of that allowed history. Previous documented re-executions on a pinned third-party ASSISTments 2009 mirror and an independently sourced corrected 2009 release reveal nonidentical source representations: 817 of 4,148 shared learners have differing skill–correctness sequences, and the mirror omits 8,774 labeled events relative to the corrected official source. An earlier fresh, nonoverlapping ASSISTments 2015 test of 750 learners found an interaction-weighted Brier difference (extended GRU minus enriched PFA) of −0.004681 (95% learner-cluster percentile interval [−0.006073, −0.003346]), conditional on separately retrained models. We now prospectively freeze a third, disjoint 2015 learner cohort to estimate predictive sensitivity to history budgets of one versus 20 preceding interactions, comparing window-based count logistic regression, an ordered-feature logistic baseline, and a compact window-GRU on identical response targets. **Results from this latest experiment are pending execution/verification and are not yet asserted here.** The study is a test of benchmark interpretation, not a novel model architecture, causal educational intervention, or generalization guarantee.

## 1. Introduction and specific research contribution

Claims that sequential architectures outperform interpretable baselines are sensitive to the availability and representation of input data. Reproducible benchmarking therefore requires (a) identifying the exact event source; (b) fixing the observable learner history permitted at prediction time; (c) matching target events, learner splits, preprocessing scope, and uncertainty estimands; and (d) distinguishing computational/predictive results from causal pedagogical value.

**Research questions:**

- **RQ1 — provenance:** Can the common transformed ASSISTments 2009 mirror stand in for the source-corrected official 2009 Skill Builder data without changing its labeled event population? Earlier checksum-verified direct audits show **no**.
- **RQ2 — history budget:** Holding cohort, splits, test targets and model capacity fixed *within model family*, how does restricting the previous observable response stream to K=1 versus K=20 alter test Brier score and discrimination?
- **RQ3 — order representation:** At the same history budget and on exactly the same target responses, what is the descriptive predictive gap between a logistic model with summary counts, a logistic model augmented with derivable order proxies, and a small recurrent history encoder?
- **RQ4 — continuity across years:** Do earlier corrected-2009 and fresh 2015 direction-specific results survive careful qualification when placed alongside a new 2015 window study? This is comparison of the published scopes, **not** frozen-weight transfer nor a controlled cross-year experimental effect.

The study's proposed incremental contribution is an **explicit two-axis source/history-window audit**, not the first proposal of input-matched knowledge tracing.

## 2. Literature positioning (necessary non-overclaiming)

- Polydoros (2026), *Cross-Cohort Knowledge-Tracing Benchmark*, Zenodo v2.0.1 (https://doi.org/10.5281/zenodo.22131530), compares PFA, BKT, DKT and SAKT on corrected ASSISTments 2009 and 2015. Hence neither cross-year comparison nor student-level cluster uncertainty is new here.
- Padi et al. (2026), *Stable and Faithful Explanations for Knowledge Tracing*, arXiv:2609.28502, explicitly studies information-matched baselines and duplicated KC records. Hence information parity and detecting leakage are not new constructs here.
- Badran and Preisach (2026), *Enhancing Knowledge Tracing through Leakage-Free and Recency-Aware Embeddings*, SN Computer Science 7:774, DOI 10.1007/s42979-026-05344-3, addresses multi-KC leakage and recency. This paper does **not** claim first detection of KC leakage, nor does it introduce their recency encoding.
- Gervet et al. (2020), *When Is Deep Learning the Best Approach to Knowledge Tracing?*, Journal of Educational Data Mining 12(3), demonstrates model-dependent comparisons on KT benchmarks.
- Piech et al. (2015), *Deep Knowledge Tracing*, NeurIPS 28, introduces neural sequential models and is background, not a method newly invented here.
- Eedi NeurIPS 2020 challenge (Wang et al., 2020, arXiv:2007.12061) publishes a larger, newer-source benchmark under CC BY-NC-ND 4.0; no experiment on Eedi is claimed in the current work.

**What remains distinct and testable:** An explicitly pre-specified new learner subset with same-target restriction to previous 1 versus 20 response events and three representations, interpreted alongside a concrete **different-source 2009 count/sequence discrepancy**. Novelty is bounded; review by a subject specialist is necessary.

## 3. Sources and prior verified findings

### 3.1 Corrected 2009 vs transformed mirror

Pinned third-party mirror SHA-256: `2053b7ea63f1d5ea953834bf82ed89543e1f11bae1e3e7535c2f9d7f49d4dda7`. Corrected official 2009 source SHA-256: `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`. The mirror includes 4,148 learner sequences and 274,331 labeled events; corrected official source includes 4,163 learners and 283,105 labeled events; 817/4,148 common learners differ in at least the count or sequence of skill–response pairs. These facts **reject exact source parity** and explain why old mirror-based test metrics cannot be silently labeled "corrected official 2009" results.

### 3.2 Earlier predictive reanalyses

Corrected official 2009 exploratory reanalysis: enriched PFA AUC 0.69508/Brier 0.196755 versus extended GRU AUC 0.73243/Brier 0.185074; 625 held-out learners/48,842 responses, with GRU minus PFA paired pooled Brier −0.011681 (2,000 learner-cluster bootstrap 95% interval [−0.013718, −0.009862]). This is **exploratory**, and its split/representation is not held fixed against the mirror benchmark; no model ranking change is causally attributed to source correction.

2015 independent-year Stage 2: exclude 5,000 previously tested learners before selecting 5,000 new learners. Enriched PFA AUC 0.681286/Brier 0.186954 and GRU AUC 0.699683/Brier 0.182273 on 750 test learners/26,697 events, Brier delta −0.004681 [−0.006073, −0.003346]. Stage-1 result was discarded as confirmatory evidence after baseline numerical nonconvergence. Different-year models were retrained, not transported unchanged.

## 4. New window-budget experiment — frozen before fresh data evaluation

Complete operational specification and decision gates are in [METHODOLOGICAL_2026_PROTOCOL.md](../METHODOLOGICAL_2026_PROTOCOL.md); executable implementation is [scripts/matched_history_budget_2026.py](../scripts/matched_history_budget_2026.py).

Official ASSISTments 2015 CSV SHA-256 `75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0`. Exclude all Stage-1 and Stage-2 learners (10,000 IDs), sample a new 4,000 from the remainder with seed 20261026, and split by learner with the same seed 2,800/600/600 for train/validation/test. Predictions for every model and history budget concern exactly the event targets at t>=1; the current response is never an input.

- **K=1:** the single immediately preceding observed (skill, response) pair and current skill ID.
- **K=20:** at most 20 immediately preceding observed (skill, response) pairs and current skill ID.
- **Window-count logistic (PFA+):** current skill, previous same-skill success/failure counts and all-skill success/failure counts within the K-window.
- **Ordered-feature logistic:** same features, plus previous response outcome, most recent same-skill response sign and lag since that same-skill observation (all derivable from the same window).
- **Window-GRU:** a compact recurrent model reading only K preceding event tokens, current skill as prediction query, with train-only fitting and validation-only early stopping.

All models share the **observable input stream**, learner populations, and test events; architectures, feature engineering and parameter counts differ. Numeric classifiers must converge and GRU training must complete under identical frozen hyperparameters across budgets.

### Predeclared descriptive primary table — do not fill until real results exist

| History budget | Model | Test Brier | Test ROC-AUC | Test log loss | Learners / test events |
|---|---|---:|---:|---:|---|
| K=1 | Window-count PFA+ | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |
| K=1 | Ordered-feature logistic | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |
| K=1 | Window-GRU | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |
| K=20 | Window-count PFA+ | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |
| K=20 | Ordered-feature logistic | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |
| K=20 | Window-GRU | **PENDING** | **PENDING** | **PENDING** | 600 / PENDING |

Matched comparison: GRU Brier minus ordered-feature logistic Brier at each K, paired learner-cluster bootstrap (2,000 repetitions); positive/negative differences interpreted descriptively, and compared across budgets. All repeated choices are labeled exploratory with no family-wise multiplicity guarantee.

## 5. Validity threats and interpretation bounds

- New sample comes from the *same historical platform/year* as Stage 1 and 2; no modern-school deployment or external ecological validity follows.
- Split-seed and model-hyperparameter sensitivity are not completely assessed by a single stage-3 split and GRU seed.
- Ordered-feature logistic transforms historical signals explicitly, whereas recurrent models learn their own representations; equal input *availability* is not equal expressive capacity or optimization.
- Sequence labeling is based on recorded `log_id`, not on proof of clinical/educational learning states. Learner-based partitions prevent train–test ID overlap but do not rule out all upstream duplicated-question artifacts.
- Correction of 2009 data is a provenance result. A direct *paired* source-effect estimate would require aligned exact learner-event IDs and targets, unavailable from the transformed mirror.
- Bootstrap CIs are conditional on analyzed learners/model fits and not prospective causal intervals; model selection, multiple comparisons and the reviewed historical corpus limit confirmation language.
- Eedi and attention-model benchmarks are promising additional contexts but have **not** been executed and must not appear as study results.
- Disclosure of substantial generative AI involvement in coding/manuscript drafting is required by author policy; author must independently check data, code, citations and final interpretation.

## 6. Conditional scientific conclusion

**Until Stage-3 completes:** Earlier corrected-source and independent-year historical evidence shows reproducible predictive advantages of the tested GRU over the specific summary PFA baseline, but cannot attribute them to recurrence itself. The newly specified history-budget experiment is not claimed as executed, favorable, significant or publication-ready yet.

**After execution:** The conclusion must be driven by observed changes and confidence intervals; an absence of predicted differences, reversal, or numerical failure is a reportable result rather than grounds for secretly retuning and replacing the study.

## Links and reproducibility

- [Official ASSISTments 2009 and corrected file](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010).
- [Official ASSISTments 2015](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data).
- [Prior corrected source reanalysis](../results/corrected_2009_sensitivity.json).
- [Prior independent-year 2015 result](../results/confirmatory_assist2015_stage2.json).
- New Stage-3 results: [planned fixed output path](../results/matched_history_budget_2026.json), only valid when source file is actually created by the successful real-data pipeline.
