# Dataset Card — ASSISTments 2009

## Lineage
Original system/data lineage: ASSISTments, Worcester Polytechnic Institute and collaborating researchers.  
Convenience research mirror used by the executable adapter: https://huggingface.co/datasets/Atomi/ASSISTments2009

The mirror is not treated as the scientific authority for the original collection. Publications should cite the original ASSISTments dataset/papers as appropriate and state that the sequence-formatted mirror was used for retrieval.

## Mirror schema
The public representation exposes one learner sequence per row with aligned fields including:
`user_id`, `skill_ids`, `skill_names`, `grades`, `attempt_counts`, and `answer_types`.

`grades` contains binary strings such as `"0"` and `"1"`, which the adapter converts to integer correctness labels.

## Study unit and executable invariant
The intended split unit is the learner. The runner verifies that every serialized row has a unique `user_id` before splitting. Duplicate learner rows are rejected because row-level splitting would otherwise risk leakage across train, validation, and test partitions.

## Skill representation
Some skill identifiers are composite strings. This benchmark treats the observed identifier string as the state key instead of inventing a decomposition rule.

The evaluation label `first_seen_skill_for_learner` refers only to the first occurrence of that observed skill key within one learner's sequence. It does not mean the skill is globally unseen in training.

## Transformation boundary
The sequence mirror does not expose all original row-level identifiers. This bundle can verify the exact mirror bytes, learner-row uniqueness, aligned sequences, and evaluation split, but it cannot independently reconstruct every preprocessing or raw-interaction deduplication decision made upstream of the mirror.

That distinction matters for replication: the pinned mirror is the executable research artifact, while the original ASSISTments release remains the scientific source.

## Privacy
The executable study uses the public, de-identified sequence representation supplied by the mirror. Do not join these data to external identities or attempt re-identification.

## Limitations
Skill tagging, item difficulty, opportunity ordering, school/course context, missing metadata, and the transformation from the original source into a sequence mirror can all affect interpretation.
