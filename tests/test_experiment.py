from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_tracing_benchmark.core import BKTParams, predict_correct_probability, predictive_trace
from run_experiment import (
    GRUKnowledgeTracer,
    aligned,
    expected_calibration_error,
    learner_block_bootstrap_brier_delta,
    pfa_features,
    sliced_metrics,
    split_rows,
    summarize_gru_seed_runs,
    validate_unique_learner_rows,
)


def fixture_rows(n=20):
    return [
        {"user_id": i, "skill_ids": ["a", "a", "b"], "grades": ["0", "1", str(i % 2)]}
        for i in range(n)
    ]


def test_repository_is_research_bundle():
    for p in [
        "README.md", "RESEARCH_BUNDLE.md", "DATA.md", "REPRODUCIBILITY.md", "ETHICS.md",
        "src/run_experiment.py", "paper/paper.md", ".github/workflows/ci.yml", ".github/workflows/empirical.yml",
        "requirements-repro.txt", "docs/research_protocol.md", "reports/model_card.md",
    ]:
        assert (ROOT / p).exists(), p


def test_bkt_is_pre_response_and_validated():
    params = BKTParams()
    expected = predict_correct_probability(params.p_init, params)
    assert predictive_trace([1])[0] == expected
    with pytest.raises(ValueError):
        BKTParams(p_guess=.9, p_slip=.2)


def test_aligned_rejects_bad_rows():
    with pytest.raises(ValueError):
        aligned({"user_id": 1, "skill_ids": ["a"], "grades": ["0", "1"]})


def test_split_is_deterministic_and_disjoint():
    rows = fixture_rows(20)
    a1, b1, c1 = split_rows(rows, 42)
    a2, _, _ = split_rows(rows, 42)
    assert [x["user_id"] for x in a1] == [x["user_id"] for x in a2]
    assert set(x["user_id"] for x in a1).isdisjoint(x["user_id"] for x in c1)
    assert len(a1) + len(b1) + len(c1) == 20


def test_duplicate_learner_rows_are_rejected_before_split():
    rows = fixture_rows(4)
    rows.append({"user_id": rows[0]["user_id"], "skill_ids": ["z"], "grades": ["1"]})
    with pytest.raises(ValueError, match="duplicate learner rows"):
        validate_unique_learner_rows(rows)
    with pytest.raises(ValueError, match="duplicate learner rows"):
        split_rows(rows, 42)


def test_first_seen_slice_name_is_unambiguous():
    pred = pd.DataFrame({
        "y": [0, 1, 1],
        "p": [.2, .7, .8],
        "learner": ["a", "a", "a"],
        "skill": ["x", "x", "y"],
        "cold_start": [True, False, True],
    })
    out = sliced_metrics(pred)
    assert "first_seen_skill_for_learner" in out
    assert "cold_start_skill" not in out
    assert out["first_seen_skill_for_learner"]["n"] == 2


def test_gru_seed_summary_reports_mean_and_sample_sd():
    runs = {
        "13": {"test_all": {"roc_auc": .70, "average_precision": .80, "brier": .20, "log_loss": .60, "ece_10": .03}},
        "42": {"test_all": {"roc_auc": .72, "average_precision": .82, "brier": .18, "log_loss": .56, "ece_10": .01}},
        "73": {"test_all": {"roc_auc": .74, "average_precision": .84, "brier": .16, "log_loss": .52, "ece_10": .02}},
    }
    out = summarize_gru_seed_runs(runs)
    assert out["seeds"] == [13, 42, 73]
    assert out["metrics"]["roc_auc"]["mean"] == pytest.approx(.72)
    assert out["metrics"]["roc_auc"]["std"] > 0


def test_pfa_features_use_only_prior_counts():
    X, y, _ = pfa_features([{"user_id": 1, "skill_ids": ["a", "a", "a"], "grades": ["1", "0", "1"]}])
    assert X[["successes", "failures"]].values.tolist() == [[0, 0], [1, 0], [1, 1]]
    assert y.tolist() == [1, 0, 1]


def test_gru_shape():
    model = GRUKnowledgeTracer(n_skills=3)
    logits = model(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 2, 0]]))
    assert logits.shape == (1, 3)


def test_ece_perfect_probabilities():
    assert expected_calibration_error([0, 0, 1, 1], [0, 0, 1, 1]) == 0


def test_learner_block_bootstrap_alignment_and_direction():
    ref = pd.DataFrame({"y": [0, 1, 0, 1], "p": [.4, .6, .4, .6], "learner": ["a", "a", "b", "b"]})
    cand = pd.DataFrame({"y": [0, 1, 0, 1], "p": [.1, .9, .1, .9], "learner": ["a", "a", "b", "b"]})
    out = learner_block_bootstrap_brier_delta(ref, cand, n_boot=50, seed=1)
    assert out["mean_learner_brier_delta_candidate_minus_skill_prior"] < 0
