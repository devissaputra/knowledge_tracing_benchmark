# Paper 1: proposed arXiv metadata and final author checklist

**Title:** Knowledge-Tracing Reproducibility and Information-Budget Auditing: A Pinned 2009 Sequence Mirror and Fresh 2015 Learner-Cohort Replication

**Author:** Devis Wawan Saputra (independent researcher; author must check name and ORCID: 0000-0002-7133-9410)

**Suggested primary subject:** cs.CY (Computers and Society), for the emphasis on research evaluation and educational impacts. **Alternative:** cs.LG (Machine Learning) if the arXiv classifier/moderator judges the predictive-model evaluation central. This is a suggestion, not a guarantee of category eligibility. Category selection and any endorsement remain the submitting author's responsibility.

**Proposed arXiv abstract (copy-paste, under the 1,920-character limit):**

Knowledge-tracing model comparisons are sensitive to preprocessing, learner partitioning, and differences in the information available to predictors. We reproduce a six-model benchmark on a pinned transformed ASSISTments 2009 sequence mirror (4,148 learners; 274,331 interactions). Direct comparison with the official corrected 2009 source shows non-equivalence: among 4,148 shared learners, 817 have different ordered skill-response sequences, and the mirror has 8,774 fewer labeled events. On 623 mirror-based held-out learners, a GRU achieves AUC 0.7470 and Brier 0.1800, versus standard PFA AUC 0.6982 and Brier 0.1976. Exploratory analyses with richer PFA histories and longer GRU training remain conditional on the previously inspected split. In an independent-year ASSISTments 2015 evaluation, after discarding a nonconverged first pilot and excluding all its learners, the GRU has Brier 0.18227 versus 0.18695 for enriched PFA on 750 new held-out learners. The interaction-weighted Brier difference is -0.00468 (95% learner-cluster bootstrap interval [-0.00607, -0.00335]). This supports directional predictive replication under the tested protocol, not an isolated effect of recurrent architecture, universal generalization, or improved educational outcomes.

**Comments:** Empirical research working preprint; 11 pages, three figures; source/data audit and replication code linked via https://github.com/devissaputra/knowledge_tracing_benchmark/pull/2. Update exact page count from the FINAL PDF before submission. Not peer reviewed.

**Report identifier/DOI:** Leave blank; do not invent an arXiv ID, DOI, journal identifier, peer-review status or affiliation. The arXiv service assigns its ID only after announcement.

## Author-only release checklist

- [ ] Personally verify name/ORCID, provenance, models, exclusion rules, raw-data comparison, exact numerical results, and the corrected-source caveat.
- [ ] Confirm authorship and originality, permissions, acknowledgments and accurate contribution/AI-assistance disclosures under current archive/institutional policy.
- [ ] Decide whether to seek **optional** external human methodological review. This is recommended for stronger scientific credibility but is not arXiv's peer-review requirement; arXiv moderation is not independent peer review.
- [ ] Log into own account at https://arxiv.org/submit ; resolve registration, endorsement or rate-limit eligibility if asked.
- [ ] Upload **only the minimal TeX/source ZIP** (not the larger research archive); inspect arXiv's generated PDF after processing.
- [ ] Verify arXiv title, abstract, authors, subject category and license; approve and click final Submit Article.

## Technical references

- Official source-file guidance: https://info.arxiv.org/help/submit_tex.html . Includes the required figure formats and guidance to exclude PDFs, .aux files, old logs and unrelated files.
- Official submission verification: https://info.arxiv.org/help/submit/ . Preview the generated PDF and metadata before the final submission action.
- Official abstract limit and metadata guidance: https://info.arxiv.org/help/prep.html (1,920 characters).
- arXiv moderation is **not** academic peer review: https://info.arxiv.org/help/moderation/ .
- Updated 2026 arXiv new-submission limits: https://blog.arxiv.org/2026/10/01/updated-rate-limit-policy/ .

**Status:** All technical content prepared by assistant remains a draft until the named author performs the release checks. No author endorsement, account status, consent or publication acceptance is asserted by this document.