# Research Bundle Evidence Contract

## Identity
**Area:** AI in Education  
**Study:** transparent knowledge tracing on real learner sequences  
**Dataset:** ASSISTments 2009 via Atomi/ASSISTments2009

## Required empirical evidence
A valid run records:
1. dataset identifier and source;
2. learner count and interaction count;
3. learner-disjoint train/validation/test membership counts;
4. training-only global and skill priors;
5. BKT parameter values;
6. pre-response predictions only;
7. ROC-AUC, Brier score and log loss for each baseline;
8. cold-start versus repeated-skill metrics where defined;
9. random seed and software versions.

## Non-claims
The bundle does not claim latent mastery is directly observed, that fixed BKT parameters are optimal, that one dataset represents all learners, or that prediction quality establishes pedagogical benefit.

## Professor review path
`README.md` → `docs/dataset_card.md` → `docs/research_protocol.md` → `src/knowledge_tracing_benchmark/core.py` → `scripts/run_research.py` → `tests/` → generated `results/research_metrics.json`.
