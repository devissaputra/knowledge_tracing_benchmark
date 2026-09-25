# Knowledge Tracing Benchmark on ASSISTments 2009

[![CI](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/empirical.yml)

**Research Bundle · AI in Education · learner modeling and knowledge tracing**

This repository is a learner-disjoint empirical benchmark on real ASSISTments 2009 sequences. It compares non-stateful priors, probabilistic knowledge tracing, performance-factor features, and a compact recurrent knowledge tracer rather than presenting fixed BKT as the whole benchmark.

## Research questions

1. Do stateful learner models improve held-out next-response probability quality over global and skill priors?
2. Does validation-tuned BKT improve over a fixed parameterization?
3. Does a PFA-style logistic model using prior learner-skill successes and failures improve probability quality?
4. Does a compact GRU knowledge tracer add useful sequence information beyond transparent baselines?
5. Are conclusions different for first-seen skills versus repeated skill encounters?
6. Are Brier-score differences robust when uncertainty is resampled at the learner rather than interaction level?

## Real dataset and provenance

The executable adapter uses the public Atomi/ASSISTments2009 sequence mirror pinned to revision c72a664a9693547fb206652ed2ce18e62d320c7d. The mirror currently contains 4,148 learner rows with aligned sequence fields including user_id, skill_ids and grades.

The mirror is a retrieval representation, not the scientific authority for the original ASSISTments collection. The repository records the exact mirror revision and SHA-256 of the downloaded parquet file and cites the original ASSISTments lineage separately.

## Learner-disjoint protocol

- 70% learners: train
- 15% learners: validation
- 15% learners: final test
- fixed split seed: 42
- no learner crosses partitions
- all next-response probabilities are emitted before observing the current response

## Models

1. global training correctness rate
2. training skill prior with global fallback
3. fixed-parameter BKT
4. validation-tuned BKT over a frozen parameter grid
5. PFA-style logistic regression using skill identity and prior learner-skill success/failure counts
6. compact GRU knowledge tracer with train-skill vocabulary and first-interaction/unseen-skill prior fallback

The GRU is a compact recurrent benchmark, not a claim of reproducing every canonical DKT implementation.

## Evaluation

The final learner-disjoint test set reports ROC-AUC, average precision, Brier score, log loss and ECE-10. Metrics are also sliced into first-seen and repeated-skill interactions. The primary uncertainty analysis resamples test learners as blocks and reports Brier-score differences against the skill-prior baseline.

The bundle also identifies high-Brier skills with at least 100 test interactions as an error-analysis diagnostic.

## Run

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=src pytest -q
    PYTHONPATH=src python src/run_experiment.py

The legacy command python scripts/run_research.py routes to the same current protocol.

## Interpretation boundary

Knowledge-tracing probabilities are model states and predictive summaries, not direct measurements of knowledge. Better held-out prediction does not establish better teaching, justify ability labels, or validate automated educational decisions. Skill tags, item difficulty, opportunity order and missing context can all change the interpretation.

## Professor review path

README.md → DATA.md → src/run_experiment.py → src/knowledge_tracing_benchmark/core.py → results/summary.md → results/metrics.json → ETHICS.md → paper/paper.md.
