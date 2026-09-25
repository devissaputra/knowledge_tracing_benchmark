# Dataset Card — ASSISTments 2009

## Lineage
Original system/data lineage: ASSISTments, Worcester Polytechnic Institute and collaborating researchers.  
Convenience research mirror used by the executable adapter: https://huggingface.co/datasets/Atomi/ASSISTments2009

The mirror is not treated as the scientific authority for the original collection. Publications should cite the original ASSISTments dataset/papers as appropriate and state that the sequence-formatted mirror was used for retrieval.

## Mirror schema
The public viewer exposes one learner per row with aligned sequence columns:
`user_id`, `skill_ids`, `skill_names`, `grades`, `attempt_counts`, and `answer_types`.

`grades` contains binary strings such as `"0"` and `"1"`, which the adapter converts to integer correctness labels.

## Study unit
The split unit is the learner row. Interactions from one learner never cross train/test partitions.

## Skill representation
Some skill identifiers are composite strings. This baseline treats the observed identifier string as the state key instead of inventing a decomposition rule.

## Privacy
The executable study uses the public, de-identified sequence representation supplied by the mirror. Do not join these data to external identities or attempt re-identification.

## Limitations
Skill tagging, item difficulty, opportunity ordering, school/course context and missing metadata can all affect interpretation. The sequence mirror is also a transformed representation rather than the raw original table.
