# Analytic system card

## System

Knowledge Tracing Benchmark

## Purpose

Bayesian Knowledge Tracing baseline with a benchmark scaffold for future recurrent and attention based models.

## Current maturity

Working research prototype. The bundled example checks the software path with synthetic inputs. It does not establish validity for real learners, instructors, courses, or workplaces.

## Inputs

See `../data/README.md` for the current synthetic schema and the documentation expected before real data are connected.

## Outputs

The current code produces a mastery probability after each observed binary response. These outputs are research signals and should be interpreted with the educational context that produced them.

## Evidence needed before real use

Use learner disjoint splits for generalization to new learners and temporal evaluation for future responses. Compare BKT with stronger models only after all methods share the same data preprocessing and split logic.

## Main limitation

BKT makes strong assumptions about skill independence, stationarity, and the meaning of correct responses. The present repository is a baseline and benchmark scaffold, not a completed comparison of BKT, DKT, and attention models.

## Human oversight

A person must review any output before it can affect a learner, instructor, applicant, or employee.
