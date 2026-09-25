# Dataset Card — ASSISTments 2009

## Scientific source

This empirical benchmark uses learner-response sequences derived from the ASSISTments 2009 dataset. The original ASSISTments dataset is the scientific source; the executable adapter retrieves a public Hugging Face mirror only as a convenient serialized representation.

Original data information: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data  
Executable mirror: https://huggingface.co/datasets/Atomi/ASSISTments2009  
Pinned mirror revision: `c72a664a9693547fb206652ed2ce18e62d320c7d`  
Pinned file: `data/train-00000-of-00001.parquet`

The raw parquet file is not committed to this repository.

## Provenance recorded by the runner

Each empirical run records:

- mirror repository and exact Git revision;
- mirror file path;
- SHA-256 of the downloaded parquet bytes;
- number of learner rows;
- total number of aligned interactions.

The revision pin prevents an unnoticed upstream mirror update from silently changing the benchmark.

## Sequence fields used

The runner requires aligned `user_id`, `skill_ids`, and binary `grades` sequence fields. It rejects learner rows whose skill and grade sequences have different lengths and rejects response values outside 0/1.

## Split boundary

Learners, rather than individual interactions, are split into 70% train, 15% validation, and 15% final test with seed 42. A learner therefore cannot appear in more than one partition.

All stateful predictions are emitted before the current response is observed.

## Scientific interpretation

The mirror is not treated as an independent dataset or as authoritative documentation of ASSISTments. Any publication using this bundle should cite the original ASSISTments data source and describe the mirror revision used for reproducibility.

## Limitations

Skill labels can be incomplete or noisy, opportunity order can encode item selection policies, and predictive mastery states are not direct measurements of knowledge. Results from this historical tutoring dataset do not establish current classroom validity or pedagogical benefit.
