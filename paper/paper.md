# Knowledge-Tracing Reproducibility and Information-Budget Auditing

**Current research status (22 October 2026):** The final author-review manuscript is [main_integrated_20261022.tex](main_integrated_20261022.tex), together with the three figures in the separately packaged arXiv source archive. This repository text is a project overview, **not** a second competing manuscript.

## Empirical findings

- **ASSISTments 2009 transformed mirror, learner-disjoint original test:** 4,148 learners, 274,331 interactions in the full representation, 623 test learners, 41,228 test responses. Standard PFA ROC-AUC 0.6982 and Brier 0.1976; original seed-42 GRU AUC 0.7470 and Brier 0.1800. Full original workflow reproduced on the *same pinned mirror*, GitHub Actions [39655888620](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39655888620).
- **Exploratory 2009 fixed-test robustness:** Stronger PFA with prior across-skill counts: AUC 0.7135, Brier 0.1929. Extended seed-42 GRU: AUC 0.7491, Brier 0.1791. GRU-minus-enriched-PFA pooled Brier difference -0.013874, learner-cluster 95% interval [-0.017231, -0.011049]. All these changes were evaluated after the original test was inspected and therefore are explicitly **exploratory**.
- **Independent-year ASSISTments 2015, preliminary test:** A first 5,000-learner sampling protocol produced nominal favorable differences but the enriched PFA optimizer failed to converge. [Run 39754046492](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39754046492) is an **inconclusive diagnostic pilot**, not valid confirmation.
- **New, nonoverlapping 2015 cohort, convergence-corrected protocol:** All 5,000 pilot learners were excluded; second sample 5,000 different learners (3,500 train, 750 validation, 750 test). On 26,697 fresh held-out responses, enriched PFA achieved AUC 0.68129, Brier 0.18695; extended GRU AUC 0.69968, Brier 0.18227. Interaction-weighted paired Brier delta -0.0046813, 95% learner-cluster bootstrap interval [-0.0060730,-0.0033457], 2,000 draws. All numerical/data gates and PFA convergence passed. Evidence: [run 39754854426](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39754854426), [final JSON](../results/confirmatory_assist2015_stage2.json). **This supports only a narrow directional predictive replication on the tested 2015 sample.**

## Manuscript and audit sources

- [Canonical updated LaTeX manuscript](main.tex)
- [Manuscript build and review instructions](BUILD_AND_AUTHOR_REVIEW.md) · [Final numerical/source-package QA results](FINAL_QA_RESULTS_20261022.json)
- [Source-parity and novelty assessment](PROVENANCE_NOVELTY_AUDIT_20261022.md)
- [External cohort replication details](external_replication_assist2015.md)
- [First 2015 protocol](../CONFIRMATORY_PROTOCOL_2015.md) and [corrective-stage protocol](../CONFIRMATORY_STAGE2_PROTOCOL_2015.md)
- [Exploratory code PR #1](https://github.com/devissaputra/knowledge_tracing_benchmark/pull/1) and [independent-cohort code PR #2](https://github.com/devissaputra/knowledge_tracing_benchmark/pull/2)

## Non-negotiable interpretation bounds

The transformed Atomi 2009 parquet lacks the raw problem/event identifiers and independently verified chronology required to establish exact parity with the **corrected official 2009--2010 source**. Learner-disjoint splitting alone does not remove any within-sequence contamination caused upstream. The 2015 new-cohort result does **not** validate 2009 transformation equivalence.

The GRU has ordered cross-skill history and greater capacity, while PFA uses summary history counts. The observed gap **must not** be attributed solely to recurrent architecture. The 2015 improvement does not establish intervention effects, fairness, educational gains, or universal transportability.

The arXiv source package compiles cleanly and automated checks pass, but **author verification, novelty judgment and final arXiv upload are pending**. This is a working preprint candidate, not a peer-reviewed study.
