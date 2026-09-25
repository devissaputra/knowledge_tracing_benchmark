from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from knowledge_tracing_benchmark.core import BKTParams, predict_correct_probability, update_mastery

SEED = 42
DATASET_REPO = "Atomi/ASSISTments2009"
DATASET_REVISION = "c72a664a9693547fb206652ed2ce18e62d320c7d"
DATASET_FILE = "data/train-00000-of-00001.parquet"
BKT_GRID = {
    "p_init": (0.10, 0.25, 0.40),
    "p_learn": (0.05, 0.15, 0.30),
    "p_guess": (0.10, 0.20, 0.30),
    "p_slip": (0.05, 0.10, 0.20),
}
GRU_SEEDS = (13, 42, 73)
GRU_EPOCHS = 8
GRU_PATIENCE = 2


def aligned(row) -> tuple[list[str], list[int]]:
    skills = [str(x) for x in list(row["skill_ids"])]
    grades = [int(str(x).strip()) for x in list(row["grades"])]
    if len(skills) != len(grades):
        raise ValueError(f"misaligned learner row {row.get('user_id')}")
    if any(v not in (0, 1) for v in grades):
        raise ValueError("grades must be binary")
    return skills, grades


def validate_unique_learner_rows(rows) -> int:
    """Require exactly one serialized row per learner before any partitioning."""
    seen, duplicates = set(), set()
    for row in rows:
        if "user_id" not in row:
            raise ValueError("every learner row must contain user_id")
        uid = str(row["user_id"])
        if uid in seen:
            duplicates.add(uid)
        seen.add(uid)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(f"duplicate learner rows detected for user_id: {preview}{suffix}")
    return len(seen)


def load_real_rows(data_path: str | Path | None = None, cache_dir: str | Path = "data/cache"):
    if data_path is None:
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
        source = f"hf:{DATASET_REPO}@{DATASET_REVISION}/{DATASET_FILE}"
    else:
        local = str(data_path)
        source = f"local:{data_path}"
    payload = Path(local).read_bytes()
    frame = pd.read_parquet(local)
    rows = frame.to_dict(orient="records")
    unique_learners = validate_unique_learner_rows(rows)
    interactions = 0
    for row in rows:
        _, grades = aligned(row)
        interactions += len(grades)
    metadata = {
        "mirror_repo": DATASET_REPO,
        "mirror_revision": DATASET_REVISION,
        "mirror_file": DATASET_FILE,
        "source": source,
        "parquet_sha256": hashlib.sha256(payload).hexdigest(),
        "n_learners": int(len(rows)),
        "n_unique_learners": int(unique_learners),
        "n_interactions": int(interactions),
    }
    return rows, metadata


def split_rows(rows, seed: int = SEED):
    rows = list(rows)
    validate_unique_learner_rows(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    a, b = int(.70 * n), int(.85 * n)
    return rows[:a], rows[a:b], rows[b:]


def train_priors(rows):
    total_correct = total_n = 0
    skill_correct, skill_n = defaultdict(int), defaultdict(int)
    for row in rows:
        skills, grades = aligned(row)
        for skill, y in zip(skills, grades):
            total_correct += y
            total_n += 1
            skill_correct[skill] += y
            skill_n[skill] += 1
    if total_n == 0:
        raise ValueError("empty training interactions")
    global_rate = total_correct / total_n
    skill_rate = {s: skill_correct[s] / skill_n[s] for s in skill_n}
    return global_rate, skill_rate, total_n


def expected_calibration_error(y, p, n_bins: int = 10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(p, edges[1:-1], right=True)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def metric_bundle(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "ece_10": expected_calibration_error(y, p, 10),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out


def prediction_frame(y, p, learner, skill, cold):
    return pd.DataFrame({
        "y": np.asarray(y, dtype=int),
        "p": np.asarray(p, dtype=float),
        "learner": learner,
        "skill": skill,
        "cold_start": np.asarray(cold, dtype=bool),
    })


def prior_predictions(rows, global_rate, skill_rate):
    y, global_p, skill_p, learner, skills_out, cold = [], [], [], [], [], []
    for row in rows:
        skills, grades = aligned(row)
        seen = set()
        uid = str(row["user_id"])
        for skill, grade in zip(skills, grades):
            y.append(grade)
            global_p.append(global_rate)
            skill_p.append(skill_rate.get(skill, global_rate))
            learner.append(uid)
            skills_out.append(skill)
            cold.append(skill not in seen)
            seen.add(skill)
    return (
        prediction_frame(y, global_p, learner, skills_out, cold),
        prediction_frame(y, skill_p, learner, skills_out, cold),
    )


def bkt_predictions(rows, params: BKTParams):
    y, p, learner, skill_out, cold = [], [], [], [], []
    for row in rows:
        skills, grades = aligned(row)
        mastery, seen = {}, set()
        uid = str(row["user_id"])
        for skill, grade in zip(skills, grades):
            current = mastery.get(skill, params.p_init)
            y.append(grade)
            p.append(predict_correct_probability(current, params))
            learner.append(uid)
            skill_out.append(skill)
            cold.append(skill not in seen)
            mastery[skill] = update_mastery(current, bool(grade), params)
            seen.add(skill)
    return prediction_frame(y, p, learner, skill_out, cold)


def tune_bkt(validation_rows):
    best = None
    for p_init in BKT_GRID["p_init"]:
        for p_learn in BKT_GRID["p_learn"]:
            for p_guess in BKT_GRID["p_guess"]:
                for p_slip in BKT_GRID["p_slip"]:
                    if p_guess >= 1 - p_slip:
                        continue
                    params = BKTParams(p_init, p_learn, p_guess, p_slip)
                    pred = bkt_predictions(validation_rows, params)
                    score = metric_bundle(pred["y"], pred["p"])["log_loss"]
                    if best is None or score < best[0]:
                        best = (score, params)
    return best[1], float(best[0])


def pfa_features(rows):
    records, labels, metadata = [], [], []
    for row in rows:
        successes, failures, seen = defaultdict(int), defaultdict(int), set()
        uid = str(row["user_id"])
        skills, grades = aligned(row)
        for skill, grade in zip(skills, grades):
            records.append({"skill": skill, "successes": successes[skill], "failures": failures[skill]})
            labels.append(grade)
            metadata.append((uid, skill, skill not in seen))
            if grade:
                successes[skill] += 1
            else:
                failures[skill] += 1
            seen.add(skill)
    return pd.DataFrame(records), np.asarray(labels, dtype=int), metadata


def fit_pfa(train_rows, seed: int = SEED):
    X, y, _ = pfa_features(train_rows)
    pre = ColumnTransformer([
        ("skill", OneHotEncoder(handle_unknown="ignore"), ["skill"]),
        ("counts", "passthrough", ["successes", "failures"]),
    ])
    model = LogisticRegression(solver="saga", max_iter=500, C=1.0, random_state=seed)
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe


def predict_pfa(model, rows):
    X, y, metadata = pfa_features(rows)
    p = model.predict_proba(X)[:, 1]
    return prediction_frame(
        y, p,
        [m[0] for m in metadata],
        [m[1] for m in metadata],
        [m[2] for m in metadata],
    )


class KTSequenceDataset(Dataset):
    def __init__(self, rows, skill_to_idx):
        self.samples = []
        for row in rows:
            skills, grades = aligned(row)
            if len(grades) < 2:
                continue
            idx = [skill_to_idx.get(s, 0) for s in skills]
            tokens = [1 + idx[t] * 2 + grades[t] for t in range(len(grades) - 1)]
            self.samples.append((
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(idx[1:], dtype=torch.long),
                torch.tensor(grades[1:], dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_kt(batch):
    tokens = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    queries = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    targets = pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(x[0]) for x in batch])
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return tokens, queries, targets, mask


class GRUKnowledgeTracer(nn.Module):
    def __init__(self, n_skills: int, token_dim: int = 32, query_dim: int = 16, hidden: int = 64):
        super().__init__()
        token_vocab = 1 + 2 * (n_skills + 1)
        self.token_embedding = nn.Embedding(token_vocab, token_dim, padding_idx=0)
        self.query_embedding = nn.Embedding(n_skills + 1, query_dim)
        self.gru = nn.GRU(token_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden + query_dim, 1)

    def forward(self, tokens, queries):
        h, _ = self.gru(self.token_embedding(tokens))
        q = self.query_embedding(queries)
        return self.head(torch.cat([h, q], dim=-1)).squeeze(-1)


def skill_vocabulary(train_rows):
    skills = sorted({skill for row in train_rows for skill in aligned(row)[0]})
    return {skill: i + 1 for i, skill in enumerate(skills)}


def sequence_loss(model, loader, optimizer=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    training = optimizer is not None
    model.train(training)
    total_loss = total_n = 0.0
    for tokens, queries, targets, mask in loader:
        if training:
            optimizer.zero_grad()
        logits = model(tokens, queries)
        losses = loss_fn(logits, targets)
        loss = (losses * mask).sum() / mask.sum()
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float((losses * mask).sum().detach().cpu())
        total_n += int(mask.sum())
    return total_loss / max(1, total_n)


def train_gru(train_rows, validation_rows, skill_to_idx, seed: int, epochs: int = GRU_EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    train_ds = KTSequenceDataset(train_rows, skill_to_idx)
    val_ds = KTSequenceDataset(validation_rows, skill_to_idx)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_kt, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_kt)
    model = GRUKnowledgeTracer(len(skill_to_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best_state, best_val, best_epoch, stale, history = None, math.inf, 0, 0, []
    for epoch in range(1, epochs + 1):
        train_loss = sequence_loss(model, train_loader, optimizer)
        with torch.no_grad():
            val_loss = sequence_loss(model, val_loader)
        history.append({"epoch": epoch, "train_bce": train_loss, "validation_bce": val_loss})
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= GRU_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def predict_gru(model, rows, skill_to_idx, global_rate, skill_rate):
    y_all, p_all, learner_all, skill_all, cold_all = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            skills, grades = aligned(row)
            if not skills:
                continue
            uid = str(row["user_id"])
            seen = set()
            probs = [skill_rate.get(skills[0], global_rate)]
            if len(grades) > 1:
                idx = [skill_to_idx.get(s, 0) for s in skills]
                tokens = torch.tensor([[1 + idx[t] * 2 + grades[t] for t in range(len(grades) - 1)]], dtype=torch.long)
                queries = torch.tensor([idx[1:]], dtype=torch.long)
                neural = torch.sigmoid(model(tokens, queries).squeeze(0)).cpu().numpy()
                for t, probability in enumerate(neural, start=1):
                    probs.append(float(probability) if idx[t] != 0 else skill_rate.get(skills[t], global_rate))
            for skill, grade, probability in zip(skills, grades, probs):
                y_all.append(grade)
                p_all.append(probability)
                learner_all.append(uid)
                skill_all.append(skill)
                cold_all.append(skill not in seen)
                seen.add(skill)
    return prediction_frame(y_all, p_all, learner_all, skill_all, cold_all)


def sliced_metrics(pred: pd.DataFrame):
    out = {"all": metric_bundle(pred["y"], pred["p"])}
    for name, mask in {
        "first_seen_skill_for_learner": pred["cold_start"],
        "repeated_skill": ~pred["cold_start"],
    }.items():
        if int(mask.sum()):
            out[name] = metric_bundle(pred.loc[mask, "y"], pred.loc[mask, "p"])
    return out


def summarize_gru_seed_runs(gru_runs: dict) -> dict:
    """Summarize full-test GRU metrics across prespecified random seeds."""
    metric_names = ("roc_auc", "average_precision", "brier", "log_loss", "ece_10")
    seeds = sorted(gru_runs, key=int)
    summary = {"n_seeds": len(seeds), "seeds": [int(seed) for seed in seeds], "metrics": {}}
    for metric in metric_names:
        values = [gru_runs[seed]["test_all"][metric] for seed in seeds if gru_runs[seed]["test_all"][metric] is not None]
        if values:
            summary["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                "values": [float(v) for v in values],
            }
    return summary


def learner_block_bootstrap_brier_delta(reference: pd.DataFrame, candidate: pd.DataFrame, n_boot: int = 2000, seed: int = 20260925):
    if not np.array_equal(reference["y"].to_numpy(), candidate["y"].to_numpy()):
        raise ValueError("prediction frames are not aligned")
    if not np.array_equal(reference["learner"].to_numpy(), candidate["learner"].to_numpy()):
        raise ValueError("learner frames are not aligned")
    frame = pd.DataFrame({
        "learner": reference["learner"].to_numpy(),
        "delta": (candidate["p"].to_numpy() - candidate["y"].to_numpy()) ** 2
        - (reference["p"].to_numpy() - reference["y"].to_numpy()) ** 2,
    })
    learner_means = frame.groupby("learner")["delta"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(learner_means, size=len(learner_means), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_learner_brier_delta_candidate_minus_skill_prior": float(learner_means.mean()),
        "learner_block_bootstrap_95_interval": [float(lo), float(hi)],
        "n_test_learners": int(len(learner_means)),
        "interpretation": "Negative values favor the candidate. Learners, not interactions, are the bootstrap resampling unit.",
    }


def skill_error_analysis(pred: pd.DataFrame, min_n: int = 100, top_n: int = 10):
    rows = []
    for skill, group in pred.groupby("skill"):
        if len(group) < min_n:
            continue
        rows.append({"skill": str(skill), "n": int(len(group)), "brier": float(brier_score_loss(group["y"], group["p"]))})
    return sorted(rows, key=lambda x: x["brier"], reverse=True)[:top_n]


def write_calibration_figure(predictions: dict[str, pd.DataFrame], outdir: Path):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    edges = np.linspace(0, 1, 11)
    for name, pred in predictions.items():
        p = pred["p"].to_numpy()
        y = pred["y"].to_numpy()
        bins = np.digitize(p, edges[1:-1], right=True)
        xs, ys = [], []
        for b in range(10):
            mask = bins == b
            if mask.any():
                xs.append(float(p[mask].mean()))
                ys.append(float(y[mask].mean()))
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", label="ideal")
    ax.set_xlabel("Mean predicted correctness")
    ax.set_ylabel("Observed correctness")
    ax.set_title("ASSISTments 2009 held-out calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "calibration.png", dpi=170)
    plt.close(fig)


def write_summary(results: dict, path: Path):
    lines = [
        "# Empirical Results Summary", "",
        "Generated by src/run_experiment.py; numerical values should not be hand-edited.", "",
        "| Model | ROC-AUC | Average precision | Brier ↓ | Log loss ↓ | ECE-10 ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results["test_metrics"].items():
        m = metrics["all"]
        fmt = lambda x: "NA" if x is None else f"{x:.4f}"
        lines.append(f"| {name} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['brier'])} | {fmt(m['log_loss'])} | {fmt(m['ece_10'])} |")
    seed_summary = results.get("gru_seed_summary", {})
    if seed_summary.get("metrics"):
        lines += ["", "## GRU repeated-seed stability", ""]
        lines.append(f"Full-run seeds: {', '.join(str(x) for x in seed_summary['seeds'])}. Mean ± sample SD:")
        lines.append("")
        lines.append("| Metric | Mean | SD |")
        lines.append("|---|---:|---:|")
        labels = {
            "roc_auc": "ROC-AUC",
            "average_precision": "Average precision",
            "brier": "Brier",
            "log_loss": "Log loss",
            "ece_10": "ECE-10",
        }
        for key, label in labels.items():
            if key in seed_summary["metrics"]:
                item = seed_summary["metrics"][key]
                lines.append(f"| {label} | {item['mean']:.4f} | {item['std']:.4f} |")
    lines += [
        "", "## Interpretation guardrail", "",
        "Knowledge-tracing probabilities are model states and predictive summaries, not direct measurements of a learner's knowledge. Prediction quality on ASSISTments 2009 does not establish instructional benefit or justify high-stakes learner labeling.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(results_dir: str | Path = "results", data_path: str | Path | None = None, quick: bool = False):
    rows, dataset = load_real_rows(data_path=data_path)
    train, validation, test = split_rows(rows, SEED)
    global_rate, skill_rate, train_interactions = train_priors(train)

    fixed_params = BKTParams()
    tuned_params, validation_log_loss = tune_bkt(validation)
    pfa = fit_pfa(train, SEED)

    skill_to_idx = skill_vocabulary(train)
    gru_seeds = (SEED,) if quick else GRU_SEEDS
    gru_predictions, gru_runs = {}, {}
    for seed in gru_seeds:
        model, history, best_epoch = train_gru(train, validation, skill_to_idx, seed, epochs=3 if quick else GRU_EPOCHS)
        pred = predict_gru(model, test, skill_to_idx, global_rate, skill_rate)
        gru_predictions[str(seed)] = pred
        gru_runs[str(seed)] = {
            "best_epoch": int(best_epoch),
            "validation_history": history,
            "test_all": metric_bundle(pred["y"], pred["p"]),
        }

    global_pred, skill_pred = prior_predictions(test, global_rate, skill_rate)
    predictions = {
        "global_rate": global_pred,
        "skill_prior": skill_pred,
        "bkt_fixed": bkt_predictions(test, fixed_params),
        "bkt_validation_tuned": bkt_predictions(test, tuned_params),
        "pfa_logistic": predict_pfa(pfa, test),
        "gru_kt_seed_42": gru_predictions[str(SEED)],
    }
    test_metrics = {name: sliced_metrics(pred) for name, pred in predictions.items()}
    uncertainty = {
        name: learner_block_bootstrap_brier_delta(skill_pred, pred)
        for name, pred in predictions.items()
        if name != "skill_prior"
    }
    results = {
        "research_bundle": True,
        "status": "quick_smoke_run" if quick else "complete",
        "dataset": dataset,
        "protocol": {
            "learner_split_seed": SEED,
            "learner_split": {"train": len(train), "validation": len(validation), "test": len(test)},
            "train_interactions": int(train_interactions),
            "training_skills": int(len(skill_rate)),
            "bkt_fixed": asdict(fixed_params),
            "bkt_validation_tuned": asdict(tuned_params),
            "bkt_validation_log_loss": validation_log_loss,
            "pfa_features": ["skill intercept", "prior successes on learner-skill", "prior failures on learner-skill"],
            "gru_seeds": list(gru_seeds),
            "gru_known_skill_vocab": int(len(skill_to_idx)),
            "gru_first_interaction_fallback": "training skill prior with global fallback",
            "slice_definition": {
                "first_seen_skill_for_learner": "first occurrence of the observed skill key within that learner sequence; not globally unseen in training",
                "repeated_skill": "later occurrence of the observed skill key within that learner sequence",
            },
        },
        "test_metrics": test_metrics,
        "gru_repeated_seed_runs": gru_runs,
        "gru_seed_summary": summarize_gru_seed_runs(gru_runs),
        "learner_block_brier_uncertainty_vs_skill_prior": uncertainty,
        "skill_error_analysis": {name: skill_error_analysis(pred) for name, pred in predictions.items()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out / "summary.md")
    write_calibration_figure(predictions, out)
    Path("paper").mkdir(exist_ok=True)
    Path("paper/results.md").write_text(
        "# Results\n\n" + (out / "summary.md").read_text(encoding="utf-8").replace("# Empirical Results Summary\n\n", "", 1),
        encoding="utf-8",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the ASSISTments 2009 knowledge-tracing research bundle")
    parser.add_argument("--data-path", default=None, help="Optional local pinned mirror parquet")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="One GRU seed and three epochs")
    args = parser.parse_args()
    result = run_experiment(args.results_dir, args.data_path, args.quick)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"]}, indent=2))


if __name__ == "__main__":
    main()
