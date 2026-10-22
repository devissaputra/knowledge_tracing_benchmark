# Paper 1: source-parity and contribution audit (22 October 2026)

## Evidentiary boundary

The existing 2009 experiment uses SHA-256-pinned transformed `Atomi/ASSISTments2009` parquet, revision `c72a664a9693547fb206652ed2ce18e62d320c7d`, digest `2053b7ea63f1d5ea953834bf82ed89543e1f11bae1e3e7535c2f9d7f49d4dda7`. The mirrored record exposes `user_id`, `skill_ids`, `skill_names`, `grades`, `attempt_counts`, and `answer_types` as *sequences* (Hugging Face dataset card, checked 22 October 2026). It has 4,148 serialized learner rows. The data adapter verifies unique learner IDs and aligned binary responses but does not reconstruct raw record identities or validated event chronology.

The official corrected 2009-2010 Skill Builder release documents that duplicate records existed in the earlier dataset and publishes a corrected representation with **one row per student-problem**, collapsing multiple skills into one row. The official source supplies row-level identifiers in its original release that are not present in the Atomi sequence mirror. No pre-transformation crosswalk or data lineage statement establishing that the Atomi parquet came from that corrected exact release was located in its exposed card. Exact source equivalence is therefore **NOT VERIFIED**. Similar variable names or comparable learner counts would not suffice as proof. The paper must continue using *pinned transformed mirror*, not *verified corrected 2009 dataset*, in empirical claims.

Official 2009 corrected source and correction notice: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

Public transformed mirror metadata and version: https://huggingface.co/datasets/Atomi/ASSISTments2009 and https://huggingface.co/datasets/Atomi/ASSISTments2009/tree/c72a664a9693547fb206652ed2ce18e62d320c7d

The official 2015 Skill Builder source has a different schema and includes `user_id`, `log_id`, `sequence_id`, `correct`. Its 2015 event-log ID is used as a within-learner ordering proxy in the new analysis. Replicating the *direction* of a model gap on disjoint 2015 learners is a separate finding, not validation of historical 2009 source parity.

Official 2015 source: https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data

## Literature and novelty positioning

- Abdelrahman, Wang and Nunes, *Knowledge Tracing: A Survey* (ACM CSUR; https://doi.org/10.1145/3569576) already synthesizes model families and evaluation practices.
- Gervet et al., *When is Deep Learning the Best Approach to Knowledge Tracing?* (2020; https://zenodo.org/records/4143614) directly cautions against generic recurrent-vs-classical superiority claims.
- Polydoros (2026; https://doi.org/10.5281/zenodo.22131530) offers a corrected-2009 / official-2015 multi-model benchmark; therefore this manuscript cannot claim to introduce cross-cohort knowledge tracing, learner splits, or calibration evaluation.
- Padi et al. (2026; https://arxiv.org/abs/2609.28502) addresses duplicated histories and information symmetry, so a claim that this paper first discovers either issue is unsupportable.

**Defensible incremental contribution:** a fully pinned reproduction of the *specific* 2009 transformed mirror with two explicitly distinguished Brier-weighting estimands, a transparent post-inspection enhanced-baseline/GRU optimization audit, and an independently sampled 2015 directional replication that retains its initial PFA-convergence failure and disjoint corrective cohort. This is a contribution in research auditability and evidence hierarchy, not new model theory or a state-of-the-art predictor. Its publication significance ultimately requires human editorial/peer judgment.

## Release-gate determination

- Public documentation / source-schema audit: completed.
- Independent 2015 direction-of-predictive-difference replication: **passed narrow numerical gate** per GitHub Actions run 39754854426; first pilot run 39754046492 retained as nonconverged.
- Corrected official 2009 row equivalence / verified chronology: **not established**; impossible to infer from the mirror columns alone.
- Architecture-only effect and general applicability: **not established**.
- Author sign-off, contribution originality, arXiv metadata and submission: pending human author.

The final manuscript is a carefully bounded working preprint candidate; do not state that corrected-source parity or peer review has been achieved.