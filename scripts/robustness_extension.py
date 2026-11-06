"""Independent real-data robustness extension for the ASSISTments knowledge tracing benchmark.

Execution:
  PYTHONPATH=src python scripts/robustness_extension.py --epochs 25 --patience 4 --boots 2000

This reuses the pinned dataset adapter in src/run_experiment.py. The original test
partition has been inspected already; this analysis is EXPLORATORY, not independent
confirmation. No synthetic observations or fabricated experiment outcomes.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_experiment as kt

def paired_cluster_bootstrap(reference, candidate, repetitions=2000, seed=20261022):
    """Resample learners, preserve all responses, estimate pooled interaction-weighted Brier difference."""
    if repetitions < 200:
        raise ValueError("Use at least 200 bootstrap repetitions")
    for col in ("learner", "y", "skill"):
        if not np.array_equal(reference[col].to_numpy(), candidate[col].to_numpy()):
            raise ValueError("Misaligned prediction frames: " + col)
    diff = (candidate["p"].to_numpy()-candidate["y"].to_numpy())**2 - (reference["p"].to_numpy()-reference["y"].to_numpy())**2
    records = pd.DataFrame({"learner":reference["learner"].astype(str).to_numpy(), "loss_diff":diff})
    grp = records.groupby("learner", sort=False)["loss_diff"].agg(["sum","count"])
    sums = grp["sum"].to_numpy(float)
    counts = grp["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(repetitions)
    for i in range(repetitions):
        index = rng.integers(0,len(sums),size=len(sums))
        sampled[i] = sums[index].sum()/counts[index].sum()
    return {"difference_candidate_minus_reference":float(sums.sum()/counts.sum()),
            "bootstrap_95_interval":[float(x) for x in np.quantile(sampled,[.025,.975])],
            "resampling_unit":"learner", "estimand_unit":"interaction",
            "n_learners":len(sums), "n_interactions":int(counts.sum()), "n_bootstrap":repetitions}

def pfa_plus_all_skill_history_frame(rows):
    features=[]; y=[]; users=[]; skills=[]; first=[]
    for row in rows:
        skill_success=defaultdict(int);skill_failure=defaultdict(int)
        total_success=0;total_failure=0;seen=set()
        sequence, outcomes=kt.aligned(row)
        for s,response in zip(sequence,outcomes):
            features.append({"skill":s,"skill_success":skill_success[s],"skill_failure":skill_failure[s],
                             "all_success":total_success,"all_failure":total_failure})
            y.append(response);users.append(str(row["user_id"]));skills.append(s);first.append(s not in seen)
            if response:
                skill_success[s]+=1;total_success+=1
            else:
                skill_failure[s]+=1;total_failure+=1
            seen.add(s)
    return pd.DataFrame(features),np.asarray(y,dtype=int),users,skills,first

def stronger_pfa(train, test):
    train_x,train_y,*_=pfa_plus_all_skill_history_frame(train)
    test_x,test_y,users,skills,first=pfa_plus_all_skill_history_frame(test)
    model=Pipeline([
      ("prep",ColumnTransformer([
        ("skill",OneHotEncoder(handle_unknown="ignore"),["skill"]),
        ("history",StandardScaler(),["skill_success","skill_failure","all_success","all_failure"])])),
      ("clf",LogisticRegression(solver="saga",max_iter=1000,C=1.0,random_state=42))])
    model.fit(train_x,train_y)
    predictions=model.predict_proba(test_x)[:,1]
    return kt.prediction_frame(test_y,predictions,users,skills,first)

def run(args):
    rows, data_record=kt.load_real_rows()
    train, validation, test=kt.split_rows(rows,kt.SEED)
    global_rate,skill_prior,_=kt.train_priors(train)
    pfa=kt.predict_pfa(kt.fit_pfa(train,kt.SEED),test)
    enhanced=stronger_pfa(train,test)
    vocabulary=kt.skill_vocabulary(train)
    kt.GRU_PATIENCE=args.patience
    output={}
    for seed in args.seeds:
        model,history,best_epoch=kt.train_gru(train,validation,vocabulary,seed,epochs=args.epochs)
        prediction=kt.predict_gru(model,test,vocabulary,global_rate,skill_prior)
        output[str(seed)]={"best_epoch":best_epoch, "epochs_completed":len(history),
                          "best_epoch_at_training_cap":best_epoch==args.epochs,
                          "training_history":history,
                          "metrics":kt.metric_bundle(prediction["y"],prediction["p"]),
                          "first_and_repeated":kt.sliced_metrics(prediction),
                          "paired_vs_pfa":paired_cluster_bootstrap(pfa,prediction,args.boots,seed+100),
                          "paired_vs_stronger_pfa":paired_cluster_bootstrap(enhanced,prediction,args.boots,seed+200)}
    record={"status":"complete_exploratory_real_data","dataset":data_record,
            "qualification":"Prior final test data were already inspected; this is not an untouched holdout confirmation.",
            "source_code":"scripts/robustness_extension.py",
            "split_seed":kt.SEED, "train_learners":len(train),"validation_learners":len(validation),"test_learners":len(test),
            "max_epochs":args.epochs,"patience":args.patience,"seeds":args.seeds,
            "pfa":kt.metric_bundle(pfa["y"],pfa["p"]),
            "stronger_pfa":kt.metric_bundle(enhanced["y"],enhanced["p"]),
            "stronger_minus_original_pfa":paired_cluster_bootstrap(pfa,enhanced,args.boots),
            "extended_gru":output}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(path),"seed_results":{s:{"auc":v["metrics"]["roc_auc"],"brier":v["metrics"]["brier"],
               "best_epoch":v["best_epoch"]} for s,v in output.items()}},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--patience",type=int,default=4)
    parser.add_argument("--boots",type=int,default=2000)
    parser.add_argument("--seeds",type=int,nargs="+",default=[13,42,73])
    parser.add_argument("--output",default="results/robustness_extension.json")
    args=parser.parse_args()
    if args.epochs<=8 or args.patience<2: parser.error("Set epochs > 8 and patience >= 2")
    run(args)
