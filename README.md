# Knowledge Tracing Benchmark — Research Bundle

[![CI](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/knowledge_tracing_benchmark/actions/workflows/ci.yml)

**Research Bundle · AI in Education · learner modeling and knowledge tracing**

This repository is now an empirical knowledge-tracing bundle built around **real ASSISTments 2009 learner sequences**. The previous toy sequences remain only as unit-test/smoke-test fixtures; they are not research evidence.

## Empirical question

> On learner-disjoint ASSISTments 2009 holdout data, does a transparent fixed-parameter Bayesian Knowledge Tracing (BKT) baseline improve next-response probability quality over global-rate and skill-prior baselines?

A second analysis separates **cold-start skill encounters** from later encounters to expose where a stateful learner model has evidence to update.

## Real dataset

Default research source: `Atomi/ASSISTments2009`, a sequence-formatted public mirror of ASSISTments 2009.

The dataset contains one row per learner with aligned sequences including:

- `user_id`
- `skill_ids`
- `skill_names`
- `grades` (binary correctness)
- `attempt_counts`
- `answer_types`

The Hugging Face viewer exposes about 4.15k learner rows. The original data lineage is ASSISTments. See [docs/dataset_card.md](docs/dataset_card.md) for provenance and the distinction between the original source and the convenience mirror.

## Frozen protocol

1. Load the real learner-sequence dataset.
2. Validate that `skill_ids` and `grades` are aligned.
3. Split **learners**, not interactions: 70% train, 15% validation, 15% test, seed 42.
4. Estimate two non-stateful baselines from train learners only:
   - global correctness rate;
   - per-skill correctness prior with global fallback.
5. Evaluate fixed-parameter BKT on each test learner independently.
6. For every test interaction, predict correctness **before** observing that response.
7. Report ROC-AUC, Brier score and log loss.
8. Report the same metrics for first-seen skill interactions and repeated skill interactions when estimable.
9. Record dataset size, split counts and BKT parameters.

No learner in the test set contributes to training baselines.

## Implemented models

### Global-rate baseline
One probability estimated from training interactions.

### Skill-prior baseline
Training-only correctness rate per skill, with global fallback for unseen test skills.

### Bayesian Knowledge Tracing
Transparent BKT with explicit:
- initial mastery;
- learning probability;
- guess probability;
- slip probability.

The current research bundle deliberately does **not** claim DKT or attention models are implemented. Those are future benchmark extensions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_research.py
```

Offline tests and the small smoke demo:

```bash
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

## Research Bundle contents

- real learner data;
- learner-disjoint evaluation;
- explicit next-response prediction timing;
- transparent baselines;
- cold-start analysis;
- dataset card;
- research protocol;
- ethics and risks;
- machine-readable results;
- CI and tests;
- paper-ready research brief.

See [RESEARCH_BUNDLE.md](RESEARCH_BUNDLE.md).

## Responsible interpretation

Knowledge-tracing probabilities are model states, not direct measurements of human knowledge. Skill tags can be incomplete, multi-skill identifiers are treated as observed composite keys in this baseline, and correctness can reflect guessing, slips, item difficulty and context. The system must not be used to label learners as permanently capable or incapable.
