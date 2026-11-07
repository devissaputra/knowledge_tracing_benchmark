# Manuscript and GitHub evidence reconciliation (7 November 2026)

This is an audit note, not external peer review or a claim of completed end-to-end independent reproduction.

## Manuscript and canonical evidence

The 6 November 2026 PDF, *When Knowledge-Tracing Rankings Depend on Available History: A Source-Correction and Matched-Information Audit on ASSISTments 2009–2015*, is the author's current PDF supplied for review. Its independent checks cite [run 40724144930](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/40724144930) for the 2015 matched-history computation, [run 40724726591](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/40724726591) for the corrected 2009 source audit, and [run 40731376499](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/40731376499) for corrected-2009 model estimates. GitHub reports completed-successful jobs in all three. Those earlier independent checks strengthen computational provenance, although this review did not itself rerun data and models. The repository contains the earlier LaTeX manuscript at `paper/main_methodological_2026.tex`. The 6 November PDF has not been established as byte-identical or line-by-line synchronized with that LaTeX version. Do not identify the earlier LaTeX file as the exact source of that later PDF without completing reconciliation.

Canonical executed stage-3 evidence is in:
- `METHODOLOGICAL_2026_PROTOCOL.md`
- `scripts/matched_history_budget_2026.py`
- `results/matched_history_budget_2026.json`
- `paper/METHODOLOGICAL_2026_EXECUTED_RESULTS.md`

At K=20, the recorded Brier scores are 0.176674 (count), 0.173885 (order-aware logistic), and 0.175140 (GRU), on 20,592 common target responses from 582 evaluable test learners. The learner-cluster bootstrap interval for GRU minus order-aware logistic is [0.000075, 0.002426]. The reported stage-3 numbers agree with the 6 November PDF at display precision. This check does **not** establish that every other table, figure, reference or supplement in the PDF is synchronized.

## The 6 November GitHub Actions failure was post-computation

[Workflow run 40310733288](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/runs/40310733288) finished smoke checks, downloaded the declared 2015 source, computed both history budgets, passed numerical assertions and uploaded an aggregate artifact. Its overall status was **failure** because the final `git push` to `research/kt-methodological-2026-history-budget` was rejected as non-fast-forward after a local commit. Previous README/notes calling this a "successful workflow run" without qualification were misleading.

The workflow has been changed on this review branch to retain uploaded artifacts but not auto-commit or push them to a shared research branch. This avoids the known post-computation push conflict. It is a workflow reliability correction, **not** a new reproduction run. The corrected workflow requires a fresh successful run before its CI status may be described as passing end-to-end.

## Still outstanding

1. Reconstruct the exact 6 November paper source, figure files and reference database in a versioned folder; confirm PDF and source match.
2. Compare all scientific tables, figures, citations and claims against pinned JSON and independently inspect source hashes and scripts.
3. Run the revised CI workflow once and confirm all steps succeed.
4. Keep author approval and scholarly attribution separate from computational QA.

No dataset, model result, interval or scientific interpretation has been changed by this reconciliation note.
