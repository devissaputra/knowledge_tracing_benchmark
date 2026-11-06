# Paper 1: final scientific evidence, source-provenance and publication audit (22 October 2026)

## Outcome of remaining feasible investigations

### Official corrected ASSISTments 2009 direct source parity audit — completed

- Official corrected dataset, downloaded from maintainer-linked file: SHA-256 `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`, 346,860 total corrected student-problem rows, 283,105 usable labeled records, 4,163 learners.
- Previously analyzed pinned Atomi mirror: SHA-256 `2053b7ea63f1d5ea953834bf82ed89543e1f11bae1e3e7535c2f9d7f49d4dda7`, 274,331 interactions, 4,148 learners.
- Shared 4,148 learners; official corrected data have 15 learners absent from the mirror. Same normalized ordered (skill, response) sequence for 3,331; discrepancy for 817. Full exact data equivalence is **refuted by data**, not merely unverified.
- Analysis with code and aggregate results: https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39773920168 ; source script `scripts/audit_corrected_2009.py` on the PR branch.
- **Caveat**: the raw corrected source includes event identifiers; the mirror does not. The audit cannot reconstruct which exact upstream preprocessing events were omitted/altered or prove a complete one-to-one original-event crosswalk. Sorting by `order_id` is a declared convention; physical clock time is not independently reconstructed.

### Corrected official 2009 exploratory model refit — completed

- Corrected source frozen checksum; 4,163 learners, 283,105 skill-labeled records after excluding 63,755 records without a usable skill/outcome. Learner-disjoint fixed split 2,914 train, 624 validation, 625 test; test has 48,842 interactions.
- Enriched PFA: ROC-AUC **0.69508**, Brier **0.19675**, log loss **0.58086**; optimizer converged (930 iterations), no warning.
- Extended GRU: ROC-AUC **0.73243**, Brier **0.18507**, log loss **0.55124**; best validation epoch 13.
- Interaction-pooled Brier contrast GRU minus PFA: **−0.0116811**, 95% learner-cluster percentile interval **[−0.0137180, −0.0098625]**, 2,000 draws.
- Source protocol and code fixed before this run; executed evidence https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39774414969 . **Exploratory**, not a new untouched 2009 confirmation, and does not isolate recurrence from feature/parameter budgets.

### New independent-year 2015 replication — completed in prior work

- Stage 1 pilot had PFA solver nonconvergence; it must not be counted as confirmation.
- Stage 2 excludes *all* first-stage sampled learners. 5,000 disjoint learners, 3,500 train / 750 validation / 750 test, 26,697 held-out events.
- Converged enriched PFA: Brier **0.18695**, AUC **0.68129**. GRU: Brier **0.18227**, AUC **0.69968**. Paired interaction-pooled difference **−0.0046813**, 95% interval **[−0.0060730, −0.0033457]**; passes prespecified *stage-2* diagnostic and directional criterion.
- GitHub Actions execution https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/39754854426 . Results are distinct-year, same platform family; not cross-institutional transportability, strict information matching, or a preregistered trial.

## Novelty and inference decision

The paper contributes auditable reproduction, direct corrected-source parity falsification, explicitly documented failure/correction of the 2015 pilot, comparison of distinct bootstrap estimands, exploratory official corrected-source refit, and a separate-year directional predictive replication. It does **not** contribute a new model architecture or prove superiority of recurrent networks under equal information/parameter budgets.

Important relevant prior work exists: Georgios Polydoros's 2026 corrected-source cross-cohort package https://doi.org/10.5281/zenodo.22131530 and Padi et al.'s 2026 leakage/information matching paper https://arxiv.org/abs/2609.28502 . This bounds novelty and should not be hidden. We cannot infer novelty acceptance by an academic editor or certify originality from automated similarity checks.

## Documentation and release checks

- Updated scientific manuscript distinguishes the mirror from official corrected source, and corrected versus independent-year experiments.
- Includes all numeric source and bootstrap limits, data access links and hashes.
- Source arXiv ZIP includes only `main.tex` and three used PNG figures; no auxiliary files, data records, output PDFs or logs.
- Source compiled afresh twice, cross-references and bibliography resolved. Generated PDF remains selectable text and visually inspected.
- Author must personally read, verify claims and give approval; arXiv account access, any required endorsement, license and final submission need the account holder. arXiv moderation is not independent academic peer review. External human reviewer feedback is recommended for stronger credibility but not a prerequisite for posting a preprint.

**Overall scientific status:** technically reproducible, carefully qualified **working preprint candidate**. Explicitly not externally peer-reviewed, not certified novel, and not yet submitted to arXiv. The corrected 2009 results strengthen robustness but source processing assumptions and unequal model information remain substantive limitations.