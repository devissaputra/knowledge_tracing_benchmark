# Reproducibility Protocol

## Environment

The repository keeps two dependency files for different purposes:

- `requirements.txt` provides bounded top-level ranges for normal development and CI;
- `requirements-repro.txt` pins the exact top-level versions used by the successful empirical GitHub Actions environment recorded on 2026-09-25.

For a reproduction run, use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt
pip install pytest==9.1.1
```

For ordinary development against compatible newer patch/minor releases, use `requirements.txt`.

## Frozen run

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python src/run_experiment.py
```

The runner downloads the pinned ASSISTments mirror revision, records the parquet SHA-256, verifies one unique learner row per `user_id`, applies the learner-disjoint split, tunes BKT on validation learners only, fits PFA and GRU models using training data, and evaluates only on final test learners.

## Frozen comparisons

The study reports:

1. global correctness prior;
2. skill prior with global fallback;
3. fixed-parameter BKT;
4. validation-tuned BKT;
5. PFA-style logistic regression;
6. compact GRU knowledge tracer.

The full GRU experiment uses seeds 13, 42, and 73. Individual runs remain in `results/metrics.json`, and the generated summary reports repeated-seed stability. Brier-score uncertainty is resampled by learner so within-learner interactions remain together.

## Generated evidence

A successful full run writes:

- `results/metrics.json`;
- `results/summary.md`;
- `results/figures/calibration.png`;
- `paper/results.md`.

These files are generated from the same runner. Numerical claims in the manuscript should be taken from those artifacts rather than copied independently.

## Reproduction boundary

The dataset bytes are pinned by upstream Git revision and recorded SHA-256. Exact floating-point results can still vary with operating system, CPU/GPU implementation, BLAS libraries, and transitive dependencies. The generated metrics therefore record key runtime package versions, and `requirements-repro.txt` records the successful top-level package set.

The exact historical GitHub Actions environment remains the strongest reproduction reference because a Python lock file cannot fully freeze system libraries or hardware behavior.
