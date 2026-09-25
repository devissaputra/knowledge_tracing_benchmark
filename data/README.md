# Data Policy

The empirical study uses real ASSISTments 2009 learner sequences retrieved at run time from the pinned public mirror documented in `DATA.md`.

No source learner dataset is committed to this repository.

The runner records the upstream revision, file path, SHA-256, serialized learner count, unique learner count, and interaction count. It rejects duplicate learner rows and misaligned or non-binary response sequences before model evaluation.

Small synthetic sequences appear only inside automated tests as software fixtures. They are not used to produce research results.

See:
- `../DATA.md` for scientific source and provenance;
- `../docs/dataset_card.md` for schema, privacy, and validity boundaries;
- `../REPRODUCIBILITY.md` for the frozen execution protocol.
