# Related Work and Benchmark Positioning

This repository is an original compact benchmark implementation. It does not claim to reproduce the full training procedures or reported results of the papers below. The literature is used to position the model families and evaluation choices.

## Bayesian Knowledge Tracing
Corbett and Anderson introduced knowledge tracing as a probabilistic model of changing learner state during skill acquisition.

- Corbett, A. T., & Anderson, J. R. (1995). *Knowledge tracing: Modeling the acquisition of procedural knowledge*. User Modeling and User-Adapted Interaction, 4, 253–278. DOI: https://doi.org/10.1007/BF01099821

This bundle includes both a fixed BKT baseline and a validation-tuned BKT variant so the comparison is not restricted to one arbitrary parameterization.

## Performance Factors Analysis
Pavlik, Cen, and Koedinger proposed Performance Factors Analysis as a logistic alternative that models prior successes and failures.

- Pavlik, P. I., Cen, H., & Koedinger, K. R. (2009). *Performance Factors Analysis — A New Alternative to Knowledge Tracing*. AIED 2009, 531–538. DOI: https://doi.org/10.3233/978-1-60750-028-5-531

The repository's PFA-style comparator uses skill identity plus prior learner-skill success and failure counts. It is intentionally described as PFA-style rather than as an exact reproduction of every specification in the paper.

## Recurrent knowledge tracing
Piech et al. demonstrated recurrent neural networks for knowledge tracing.

- Piech, C., et al. (2015). *Deep Knowledge Tracing*. NeurIPS 2015. https://arxiv.org/abs/1506.05908

The compact GRU in this repository tests whether recurrent sequence information adds predictive value beyond transparent baselines. It is not labeled a canonical DKT reproduction.

## Attention-based knowledge tracing
Attention models are important later comparators but are not implemented in the current frozen benchmark.

- Pandey, S., & Karypis, G. (2019). *A Self-Attentive Model for Knowledge Tracing*. https://arxiv.org/abs/1907.06837
- Ghosh, A., Heffernan, N., & Lan, A. S. (2020). *Context-Aware Attentive Knowledge Tracing*. KDD 2020. https://arxiv.org/abs/2007.12324

## Comparative evidence
Gervet et al. compared deep, logistic, and Markov learner-performance models across multiple real datasets and emphasized that the strongest model family depends on dataset conditions and feature design.

- Gervet, T., Koedinger, K., Schneider, J., & Mitchell, T. (2020). *When is Deep Learning the Best Approach to Knowledge Tracing?* Journal of Educational Data Mining, 12(3), 31–54. https://doi.org/10.5281/zenodo.4143614

That comparative perspective motivates this repository's use of multiple transparent baselines, calibration-aware metrics, learner-disjoint evaluation, repeated neural seeds, and explicit limitations rather than a single-model demonstration.

## Scope boundary
The present research question is deliberately narrower than a state-of-the-art leaderboard claim. It asks what can be learned from a reproducible comparison of priors, BKT, PFA-style features, and a compact recurrent model on one pinned ASSISTments 2009 representation. Attention-based models and additional datasets are appropriate extensions, not unimplemented claims.
