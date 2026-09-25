# Reproducibility Protocol

## Environment

Use Python 3.11 and install the checked-in dependency list:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Frozen run

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python src/run_experiment.py
```

The runner downloads the pinned ASSISTments mirror revision, records the parquet SHA-256, applies the learner-disjoint split, tunes BKT on validation learners only, fits PFA and GRU models using training data, and evaluates only on final test learners.

## Frozen comparisons

The study reports:

1. global correctness prior;
2. skill prior with global fallback;
3. fixed-parameter BKT;
4. validation-tuned BKT;
5. PFA-style logistic regression;
6. compact GRU knowledge tracer.

The GRU is repeated at the seeds declared in `src/run_experiment.py`. Brier-score uncertainty is resampled by learner so within-learner interactions remain together.

## Generated evidence

A successful full run writes machine-readable metrics, a human-readable summary, paper results, and figures under `results/` and `paper/`. Numerical claims in the manuscript should come from those generated artifacts rather than being copied by hand.

## Reproduction boundary

Re-running at the pinned mirror revision should reproduce the data bytes. Exact floating-point results can still vary with dependency, BLAS, operating-system, and hardware differences; environment versions are therefore recorded in generated metrics.
