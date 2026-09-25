# Dataset Card — ASSISTments 2009

## Scientific source

This empirical benchmark uses learner-response sequences derived from the ASSISTments 2009 dataset. The original ASSISTments dataset is the scientific source; the executable adapter retrieves a public Hugging Face mirror only as a convenient serialized representation.

Original data information: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data  
Executable mirror: https://huggingface.co/datasets/Atomi/ASSISTments2009  
Pinned mirror revision: `c72a664a9693547fb206652ed2ce18e62d320c7d`  
Pinned file: `data/train-00000-of-00001.parquet`

The raw parquet file is not committed to this repository.

For publication, the original ASSISTments data page requests citation of the specific 2009–2010 dataset URL and the ASSISTments system paper:

Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). *Addressing the assessment challenge in an Intelligent Tutoring System that tutors as it assesses*. User Modeling and User-Adapted Interaction, 19, 243–266.

Because this repository executes against a transformed sequence mirror, a publication should additionally report the mirror repository, pinned revision, file path, and SHA-256 below.

## Provenance recorded by the runner

Each empirical run records:

- mirror repository and exact Git revision;
- mirror file path;
- SHA-256 of the downloaded parquet bytes;
- serialized learner-row count;
- unique `user_id` count;
- total number of aligned interactions.

The revision pin prevents an unnoticed upstream mirror update from silently changing the benchmark.

## Sequence fields used

The runner requires aligned `user_id`, `skill_ids`, and binary `grades` sequence fields. It rejects learner rows whose skill and grade sequences have different lengths and rejects response values outside 0/1.

## Learner-row invariant

The benchmark assumes one serialized row per learner. Before any split, the runner checks that `user_id` values are unique and raises an error if duplicate learner rows are present.

This check matters because splitting rows without first enforcing learner uniqueness could place interactions from the same learner into different partitions.

## Split boundary

Learners, rather than individual interactions, are split into 70% train, 15% validation, and 15% final test with seed 42. After the uniqueness check, a learner therefore cannot appear in more than one partition.

All stateful predictions are emitted before the current response is observed.

## Transformation boundary

The pinned mirror is already a sequence-formatted transformation. It exposes learner IDs, skill sequences, skill names, binary grades, attempt counts, and answer types, but it does not expose the full original row-level identifiers needed to reconstruct every raw interaction.

The repository can therefore verify sequence alignment, binary outcomes, learner uniqueness, split integrity, and the exact transformed bytes it used. It does **not** claim to independently reproduce or audit every preprocessing and raw-row deduplication decision that occurred before this mirror was serialized. That upstream transformation is a validity limitation and should be reported in any publication.

## Scientific interpretation

The mirror is not treated as an independent dataset or as authoritative documentation of ASSISTments. Any publication using this bundle should cite the original ASSISTments data source and describe the mirror revision used for reproducibility.

## Limitations

Skill labels can be incomplete, noisy, or composite; opportunity order can encode item-selection policies; and predictive mastery states are not direct measurements of knowledge. Results from this historical tutoring dataset do not establish current classroom validity, fairness, or pedagogical benefit.
