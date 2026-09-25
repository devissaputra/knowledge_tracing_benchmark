# Ethics and Responsible-Use Boundary

This repository evaluates predictive learner-state models. It does not validate automated grading, tracking, ability labeling, admissions, discipline, or intervention decisions.

A knowledge-tracing probability is a model-dependent prediction about the next observed response under this dataset and protocol. It is not a direct measurement of intelligence, effort, motivation, disability, future potential, or stable mastery.

## Main risks

Historical tutoring logs may encode curriculum order, item-selection policies, missing opportunities, uneven skill tagging, and population-specific patterns. A model that predicts well can still be wrong for particular learners or groups, and predictive improvement does not establish instructional benefit.

## Safeguards in this bundle

- learner-disjoint train/validation/test partitions;
- no current-response leakage into the prediction emitted for that interaction;
- transparent prior, BKT, and PFA baselines alongside the GRU;
- first-seen versus repeated-skill reporting;
- learner-level uncertainty resampling;
- error analysis rather than a single headline score;
- explicit non-claims about learner ability and educational effectiveness.

Any consequential deployment would require separate validity, fairness, privacy, accessibility, teacher-oversight, and intervention-impact studies.
