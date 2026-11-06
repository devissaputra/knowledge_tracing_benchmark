# Frozen stage-2, disjoint-cohort confirmatory replication

**Reason for independent rerun.** The predeclared first external-cohort execution (GitHub run 39754046492; source SHA 75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0) generated estimates, but PFA `saga` emitted a nonconvergence warning and the planned diagnostic gate marked the test `completed_with_unresolved_diagnostics`. That result is retained without a confirmatory label. The correction addresses only numerical solver convergence; it must not select hyperparameters based on the observed direction or magnitude of the first test result.

## New learner cohort and data provenance

Use the **same official ASSISTments 2015 raw CSV bytes**, SHA-256 above, linked from the official ASSISTments portal. Apply exactly the same predeclared validity filters and chronological order (`log_id`) as in stage 1. Lexicographically sort eligible learner IDs, reconstruct the 5,000 participants originally sampled with `random.Random(20261022).sample(ids, 5000)`, exclude ALL 5,000 of those previously analysed learners, and sample 5,000 learners from the remaining eligible IDs with `random.Random(20261023).sample(remaining, 5000)`. Use a learner-disjoint 70/15/15 train/validation/test split with seed `20261023`. Assert zero overlap with stage 1 participants. No stage-2 outcomes have been inspected when this protocol was specified.

## Fixed models and criterion

- **PFA**: the same enriched information features (`skill`, within-skill previous correct/incorrect counts, across-skill previous correct/incorrect counts), StandardScaler for counts, OneHotEncoder for skill, logistic regression regularization `C=1.0`. Replace the nonconverged `saga` numerical solver by fixed `lbfgs` with `max_iter=2000`, `tol=1e-8`, to estimate the *same regularized logistic objective*. Record and reject nonconvergence warnings; **no alternative solvers, seeds, or hyperparameters after this check**.
- **GRU**: same implementation as first execution; train only on stage-2 training learners, select epoch only on stage-2 validation learners. Seed 42, up to 25 epochs, patience 4, Adam LR .002, token dim 32, query dim 16, hidden 64.
- Primary estimand: paired pooled interaction-weighted Brier loss GRU minus PFA across stage-2 test learners. Learner-cluster percentile bootstrap 2,000 replicates, seed 20261023. Directional result counts as *confirmed on this external 2015 subset* only if upper 95% percentile endpoint < 0 and all data/provenance/convergence guards pass. Otherwise it is a failed or indeterminate test; publish unchanged. Do not compare different PFA solvers on these stage-2 test outcomes to pick the favorable one.
- Secondary descriptive measures: ROC-AUC, log loss, ECE, sample sizes and observed correctness prevalence; not selection criteria.

## Scope

A successful stage-2 test would independently replicate the direction of a predictive difference on a different-year ASSISTments cohort, with a numerically converged baseline. It does not prove that GRU architecture alone causes the difference, demonstrate source parity between the original 2009 mirror and corrected official data, or establish generalizability outside the analyzed contexts. This is not a registered external trial or blind third-party peer review.