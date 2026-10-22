# Canonical Paper 1 — build and author review

The **actual** updated manuscript is `paper/main.tex` on the research branch `research/kt-fresh-cohort-20261022`. The previous `paper/main_integrated_20261022.tex` is synchronized for revision history. The distribution in ChatGPT Library `/Paper1_Final_20261022/Paper1_Actual_Manuscript_Source_20261022.zip` includes four required PNG figures and is directly compilable.

## Build from repository root

```bash
python paper/generate_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figure generator uses the actual archived GitHub Actions JSON outputs; it does not fabricate any research data. It needs `matplotlib` and `numpy`. When working from the downloadable source ZIP, figures are already provided, so `latexmk -pdf main.tex` suffices.

## What changed in the manuscript

1. Preserves the frozen 2009 transformed-mirror benchmark and the exploratory enhanced-PFA/GRU analyses.
2. Documents the initial ASSISTments 2015 test (25,089 held-out responses) with actual model scores and its failed numerical convergence gate. The pilot is **not** counted as confirmatory evidence.
3. Integrates the independent, learner-disjoint 2015 corrective-stage test (750 held-out learners; 26,697 responses), numerical point estimates, bootstrap interval and figure.
4. States explicitly that the bootstrap resampled learners **without refitting the training models**. This captures held-out sample uncertainty conditional on fitted models, not optimization instability or between-cohort variability.
5. Preserves caveats: 2009 mirror not proven equivalent to corrected official data; model information/capacity unmatched; no causal or instructional-effect claims.

## Record of local QA

- 10-page A4, selectable text, all pages checked by rendering.
- Four figures included; no missing references or LaTeX errors; no overfull LaTeX boxes.
- PDF successfully rebuilt from source in a clean temporary directory.
- 19/19 consistency/build checks passed in the accompanying `MANUSCRIPT_QA.json`.

## Final gates not delegated

The author must review the science, verify the provenance and calculations personally, approve contribution/originality and disclosure statements, and choose the arXiv category and submit. The branch must not be silently treated as peer-reviewed or as evidence of data parity with corrected ASSISTments 2009-2010.