"""One-shot, frozen independent ASSISTments-2015 cohort replication.

Run from repository root:
  PYTHONPATH=src python scripts/confirmatory_2015.py
Must not be retuned after inspecting holdout outcomes. See CONFIRMATORY_PROTOCOL_2015.md.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustness_extension import paired_cluster_bootstrap, stronger_pfa
import run_experiment as kt

SOURCE_URL = "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data"
GOOGLE_FILE_ID = "0B_hO8cnpcIMgUGZzRnh3bHJrSjQ"
SEED = 20261022
N_SAMPLE = 5000
BOOTS = 2000
OUTPUT = Path("results/confirmatory_assist2015.json")


def download_official_csv(path: Path) -> None:
    """Only the official ASSISTments-linked file ID is accepted for network retrieval."""
    if path.exists():
        print(f"Using locally supplied file {path}; source authenticity requires author audit")
        return
    import gdown
    path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=GOOGLE_FILE_ID, output=str(path), quiet=False)
    if result is None or not path.exists():
        raise RuntimeError("Official Google Drive CSV download failed; no fallback to an unverified mirror")
    with path.open("rb") as handle:
        if handle.read(128).lstrip().startswith((b"<!doctype html", b"<html", b"<HTML")):
            raise RuntimeError("Retrieved HTML, not official dataset CSV")


def load_rows(csv_path: Path) -> tuple[list[dict], dict]:
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    raw = pd.read_csv(csv_path, low_memory=False)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = ("user_id", "log_id", "sequence_id", "correct")
    if len(set(raw.columns)) != len(raw.columns) or any(c not in raw for c in required):
        raise ValueError(f"Expected unique columns {required}, found {list(raw.columns)}")
    source_n = len(raw)
    df = raw[list(required)].copy()
    invalid_null = df.isna().any(axis=1)
    for name in ("log_id", "correct"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    invalid = invalid_null | df.isna().any(axis=1) | ~df["correct"].isin([0, 1])
    bad_n = int(invalid.sum())
    if bad_n > 0:
        # Source data may include missing records; the exclusion rule is declared ex ante.
        df = df.loc[~invalid].copy()
    if len(df) == 0 or bad_n / source_n > 0.05:
        raise ValueError(f"Too few valid records: {bad_n}/{source_n} excluded (5% ceiling)")
    if df.duplicated(["user_id", "log_id"]).any():
        raise ValueError("Duplicate learner-event key in 2015 raw file; manual provenance audit needed")
    if not np.isfinite(df["log_id"]).all():
        raise ValueError("Nonfinite chronology field")
    if (df["log_id"] != df["log_id"].astype("int64")).any():
        raise ValueError("Nonintegral chronology ID")
    df["correct"] = df["correct"].astype(int)
    df["user_id"] = df["user_id"].astype(str)
    df["sequence_id"] = df["sequence_id"].astype(str)
    df = df.sort_values(["user_id", "log_id"], kind="stable")
    unique_users = sorted(df["user_id"].unique())
    if len(unique_users) < N_SAMPLE:
        raise ValueError(f"Need {N_SAMPLE} learners, found {len(unique_users)}")
    sampled = set(random.Random(SEED).sample(unique_users, N_SAMPLE))
    chosen = df[df["user_id"].isin(sampled)]
    rows = [
        {"user_id": str(uid), "skill_ids": group["sequence_id"].tolist(),
         "grades": group["correct"].tolist()}
        for uid, group in chosen.groupby("user_id", sort=True)
    ]
    assert kt.validate_unique_learner_rows(rows) == N_SAMPLE
    assert all(len(r["grades"]) == len(r["skill_ids"]) for r in rows)
    metadata = {
        "official_portal": SOURCE_URL,
        "official_linked_drive_file_id": GOOGLE_FILE_ID,
        "raw_filename": csv_path.name,
        "source_sha256": digest,
        "source_total_rows": source_n,
        "excluded_invalid_rows": bad_n,
        "source_unique_eligible_learners": len(unique_users),
        "sample_seed": SEED,
        "sample_learners": N_SAMPLE,
        "sample_interactions": int(sum(len(r["grades"]) for r in rows)),
        "chronology": "ascending log_id within user_id",
        "concept": "sequence_id (skill builder; not 2009 mirror skill_id)",
    }
    return rows, metadata


def evaluate(csv_path: Path) -> dict:
    rows, dataset = load_rows(csv_path)
    train, validation, test = kt.split_rows(rows, seed=SEED)
    ids = [set(str(r["user_id"]) for r in block) for block in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    global_prior, skill_prior, _ = kt.train_priors(train)
    # These fixed protocols were written before seeing any 2015 test outcomes.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        reference = stronger_pfa(train, test)
    convergence_warnings = [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]
    vocab = kt.skill_vocabulary(train)
    kt.GRU_PATIENCE = 4
    model, history, best_epoch = kt.train_gru(train, validation, vocab, seed=42, epochs=25)
    candidate = kt.predict_gru(model, test, vocab, global_prior, skill_prior)
    comparison = paired_cluster_bootstrap(reference, candidate, repetitions=BOOTS, seed=SEED)
    metrics = {"enriched_pfa": kt.metric_bundle(reference["y"], reference["p"]),
               "extended_gru": kt.metric_bundle(candidate["y"], candidate["p"])}
    positive = comparison["bootstrap_95_interval"][1] < 0
    validity_gate = (
        not convergence_warnings
        and len(train) > 0 and len(validation) > 0 and len(test) > 0
        and not (best_epoch == 25)
        and np.isfinite(comparison["difference_candidate_minus_reference"])
    )
    result = {
        "status": "completed" if validity_gate else "completed_with_unresolved_diagnostics",
        "directional_replication_confirmed": bool(positive and validity_gate),
        "test_is_external_cohort_not_prior_2009_test": True,
        "source_integrity_scope": "Official-source URL/id and downloaded-byte SHA; no independent publisher checksum available",
        "data": dataset,
        "protocol": {"sample_learners": N_SAMPLE, "sample_seed": SEED,
                     "split_seed": SEED,
                     "train_learners": len(train), "validation_learners": len(validation), "test_learners": len(test),
                     "gru_seed": 42, "gru_max_epochs": 25, "gru_patience": 4,
                     "bootstrap_seed": SEED, "bootstrap_draws": BOOTS,
                     "predeclared_primary_estimand": "GRU - enhanced PFA pooled interaction-weighted Brier",
                     "directional_rule": "Upper endpoint of 95% percentile learner-cluster bootstrap CI < 0; must pass diagnostic gates"},
        "metrics": metrics,
        "brier_comparison": comparison,
        "gru_best_validation_epoch": int(best_epoch),
        "gru_epochs_completed": len(history),
        "pfa_convergence_warnings": convergence_warnings,
        "conclusion_scope": "Different-year ASSISTments learner-subset replication. Not source parity for 2009 and not a causal attribution of architecture.",
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cache/assistments_2015_official.csv")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    source = Path(args.csv)
    download_official_csv(source)
    result = evaluate(source)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "confirmed": result["directional_replication_confirmed"],
                      "n_test_learners": result["protocol"]["test_learners"],
                      "metrics": result["metrics"],
                      "brier_difference": result["brier_comparison"]}, indent=2))


if __name__ == "__main__":
    main()