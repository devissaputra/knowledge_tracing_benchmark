# Ethics, Validity, and Misuse Risks

This document expands the concise boundary in `../ETHICS.md` for researchers reviewing the empirical design.

## What the model output means

Each model estimates the probability of the next observed binary response under this dataset and protocol.

That probability is not a direct measurement of intelligence, motivation, effort, disability, future potential, or stable mastery. Even the BKT latent state is a model-dependent construct rather than an observed psychological variable.

## Historical-data risk

ASSISTments 2009 is a historical tutoring dataset. Its traces can encode:

- curriculum order;
- platform item-selection policies;
- uneven opportunities to practice;
- incomplete or composite skill tags;
- omitted item difficulty and context;
- population- and period-specific behavior.

A model can therefore predict responses well while relying on regularities that do not transport to another course, platform, learner population, or year.

## Leakage and evaluation safeguards

This bundle uses several safeguards:

- duplicate `user_id` rows are rejected before partitioning;
- learners are disjoint across train, validation, and final test partitions;
- stateful predictions are emitted before the current response is observed;
- transparent priors, BKT, and PFA-style models are compared alongside the GRU;
- validation learners are separated from final test learners;
- uncertainty is resampled at the learner level;
- first-seen and repeated-skill interactions are reported separately;
- recurrent results are repeated across prespecified random seeds;
- skill-level errors are inspected rather than relying on one aggregate score.

These safeguards reduce specific methodological risks. They do not prove external validity, fairness, or educational benefit.

## Fairness boundary

The public sequence representation used here does not provide a complete, ethically appropriate basis for demographic fairness claims.

The absence of a subgroup analysis must not be interpreted as evidence that errors are evenly distributed. A real deployment would require a separately justified fairness study using lawful, appropriate attributes and a clearly specified educational use case.

## Privacy

Use only the public, de-identified research representation documented in `DATA.md`. Do not attempt re-identification or join learner traces to external identities.

A real-user extension would require a separate data-governance plan covering lawful basis or consent, data minimization, access control, retention, and deletion.

## Educational-risk boundary

Predictive improvement does not show that acting on the prediction improves learning.

Before using a knowledge-tracing signal to change instruction, a separate intervention study should establish whether the resulting action benefits learners and whether educators and learners can understand and challenge the decision process.

## Excluded uses

This research bundle does not validate:

- autonomous grading;
- admissions or disciplinary decisions;
- permanent ability labeling;
- employment decisions;
- psychological or medical diagnosis;
- covert learner monitoring;
- automated intervention without meaningful educator oversight.

## Minimum evidence before deployment

Any consequential use would require, at minimum:

1. a clearly defined instructional decision and stakeholder;
2. external validation on the intended population and context;
3. subgroup and calibration analysis where ethically and legally appropriate;
4. privacy and data-governance review;
5. accessibility review;
6. educator and learner oversight mechanisms;
7. an intervention-impact study;
8. monitoring and rollback criteria.
