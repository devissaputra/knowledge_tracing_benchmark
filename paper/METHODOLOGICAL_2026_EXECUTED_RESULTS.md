# Executed 2026 matched-history-budget results — 26 October 2026

**Scientific outcome:** An interpretable order-aware logistic regression scored better Brier loss than the compact GRU on the same next-response targets under both fixed observable history budgets. The earlier GRU-over-PFA result does not generalize to every classical baseline under matched input stream availability.

The official 2015 source SHA-256 was verified; stage 3 excluded the full 5,000-learner stage-1 sample and full 5,000-learner stage-2 sample. Of 19,840 source-eligible students, 9,840 were eligible for further sampling and 4,000 were selected for stage 3 (138,129 source events). Learner splits: 2,800 train / 600 validation / 600 test. **582** test learners produced **20,592** eligible targets at t>=1. The other 18 test learners had no post-first response and were properly excluded from next-response metrics.

## Actual same-target results

| Previous history K | Model | Brier (lower) | ROC-AUC (higher) | Log loss | ECE-10 |
|---|---|---:|---:|---:|---:|
| 1 | Count-based PFA+ | 0.177882 | 0.684440 | 0.534589 | 0.009320 |
| 1 | Order-aware logistic | **0.177882** | 0.684436 | 0.534589 | 0.009321 |
| 1 | GRU | 0.180003 | 0.670909 | 0.540658 | 0.005781 |
| 20 | Count-based PFA+ | 0.176674 | 0.691058 | 0.531590 | 0.006126 |
| 20 | Order-aware logistic | **0.173885** | 0.705044 | 0.524095 | 0.007251 |
| 20 | GRU | 0.175140 | 0.699361 | 0.527153 | 0.011658 |

Paired learner-cluster bootstrap, 2,000 draws: **GRU minus order-aware Brier K=1 +0.002121** (95% interval [0.001198,0.003013]), **K=20 +0.001255** ([0.000075,0.002426]). GRU-minus-count at K=20 is **−0.001533**, interval [−0.002761, −0.000279], demonstrating the difference that comparator choice makes.

Improvements in Brier from K=1 to K=20: count-based 0.001208, order-aware 0.003997, GRU 0.004863. Interpret intervals descriptively: fixed data split/model fits, one GRU seed, no multiple-comparison correction.

## Audited execution status

- New protocol committed **before evaluating stage-3 cohort**: [frozen protocol](../METHODOLOGICAL_2026_PROTOCOL.md)
- [Python implementation](../scripts/matched_history_budget_2026.py)
- [Real-data, checksum-verified and learner-disjoint aggregate JSON](../results/matched_history_budget_2026.json)
- [Successful GitHub Actions run 40310733288](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/40310733288).
- [Final 13-page author-review manuscript](main_methodological_2026.tex).
- [41 internal manuscript and numerical QA checks](METHODOLOGICAL_2026_FINAL_QA.json).
- [Figure generator](generate_figures.py).
- User-facing final PDF, minimal arXiv ZIP, and complete evidence package are preserved in ChatGPT Library folder `/Paper1_Methodological_2026`.

### Transparency on prior release assertion
The initial run 40310088864 completed all model training, both history horizons and aggregate output. Its release gate nevertheless failed on a mistaken assertion requiring 600 **evaluated** learners, confusing split members with learners having t>=1 targets. The corrected check permits 582 observed evaluable test learners and verifies exactly equal event counts across models, without changing any data, model, tuning, or observed result. The final run succeeds.

### Contribution boundaries
The study is **not** the first knowledge-tracing information-parity comparison, not the first source correction study and not a model architecture innovation. Polydoros (2026, DOI 10.5281/zenodo.22131530), Padi et al. (2026, arXiv:2609.28502), and Badran & Preisach (2026, DOI 10.1007/s42979-026-05344-3) are material prior work. The incremental contribution is a frozen, newly nonoverlapping 2015 cohort with explicitly constrained K=1 versus K=20 history budgets and order-feature comparator, interpreted alongside independently verified 2009 source mismatch.

There is no basis for causal effect on educational outcomes, current schooling deployment, universal superiority of logistic models, or cross-year frozen-weight transfer. Next strengthening steps would be independent contemporary datasets, multiple splits/seeds and attention baselines; **none are claimed completed**.

**Status:** Audited working preprint for named human author's scientific approval, not externally peer reviewed and not submitted to arXiv.
