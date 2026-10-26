#!/usr/bin/env python3
"""Frozen 2026 methodology study: matched history budgets on unused ASSISTments 2015 learners.

Read METHODOLOGICAL_2026_PROTOCOL.md. Source- and algorithm-level gates reject
unverified substitutions. Only aggregate results are written to the repository.
"""
from __future__ import annotations

import argparse, hashlib, json, random, sys, time, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
import run_experiment as kt
from robustness_extension import paired_cluster_bootstrap
from confirmatory_2015 import download_official_csv
from confirmatory_2015_stage2 import SOURCE_SHA256

FIRST=20261022
SECOND=20261023
THIRD=20261026
SAMPLE=4000
HORIZONS=[1,20]
BOOTSTRAPS=2000
SKILL_COL="skill"
COUNT_COLS=["within_success","within_failure","all_success","all_failure"]
ORDER_COLS=COUNT_COLS+["prev_outcome","prev_same_outcome","lag_to_same"]

def official_unused_cohort(path:Path):
    observed_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash!=SOURCE_SHA256: raise RuntimeError("official 2015 SHA-256 differs")
    df=pd.read_csv(path,low_memory=False)
    df.columns=[str(c).strip().lower() for c in df.columns]
    required=["user_id","log_id","sequence_id","correct"]
    if any(c not in df for c in required) or len(set(df.columns))!=len(df.columns):
        raise ValueError("source columns differ")
    n_original=len(df)
    df=df[required].copy()
    missing=df.isna().any(axis=1)
    df["log_id"]=pd.to_numeric(df["log_id"],errors="coerce")
    df["correct"]=pd.to_numeric(df["correct"],errors="coerce")
    invalid=missing|df.isna().any(axis=1)|~df["correct"].isin([0,1])
    excluded=int(invalid.sum())
    if n_original!=708631 or excluded!=24830:
        raise ValueError("source row-validation totals differ from prior audits")
    df=df.loc[~invalid].copy()
    if df.duplicated(["user_id","log_id"]).any():
        raise ValueError("Duplicate learner-log identifiers")
    if not np.isfinite(df["log_id"]).all() or (df["log_id"]!=df["log_id"].astype("int64")).any():
        raise ValueError("invalid chronological event IDs")
    df["user_id"]=df["user_id"].astype(str)
    df["sequence_id"]=df["sequence_id"].astype(str)
    df["correct"]=df["correct"].astype(int)
    users=sorted(df["user_id"].unique())
    if len(users)!=19840:raise ValueError("eligible learner population changed")
    pilot=set(random.Random(FIRST).sample(users,5000))
    after_pilot=[x for x in users if x not in pilot]
    stage2=set(random.Random(SECOND).sample(after_pilot,5000))
    untouched=[x for x in after_pilot if x not in stage2]
    chosen=set(random.Random(THIRD).sample(untouched,SAMPLE))
    if chosen&pilot or chosen&stage2 or len(chosen)!=SAMPLE:raise ValueError("Cohort reused!")
    df=df[df["user_id"].isin(chosen)].sort_values(["user_id","log_id"],kind="stable")
    rows=[{"user_id":str(uid),"skill_ids":g["sequence_id"].tolist(),"grades":g["correct"].tolist()}
          for uid,g in df.groupby("user_id",sort=True)]
    if len(rows)!=SAMPLE or kt.validate_unique_learner_rows(rows)!=SAMPLE:
        raise ValueError("Unmatched learner count")
    cohort_fingerprint=hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest()
    return rows,{"source_sha256":observed_hash, "eligible_learners":len(users),
                 "stage1_excluded":len(pilot),"stage2_excluded":len(stage2),
                 "eligible_unused_pool":len(untouched),"stage3_learners":len(chosen),
                 "stage3_interactions":int(len(df)), "learner_set_sha256":cohort_fingerprint,
                 "cohort_overlap_previous":0,"invalid_source_rows_excluded":excluded}

def frame(rows,k):
    """Every target uses only the k preceding labeled events, never its own label."""
    feats=[];labels=[];learners=[];skills=[];positions=[]
    for row in rows:
        uid=str(row["user_id"])
        ss,ys=kt.aligned(row)
        for t in range(1,len(ys)):
            recent=list(zip(ss[max(0,t-k):t],ys[max(0,t-k):t]))
            current=ss[t]
            within=[int(y) for skill,y in recent if skill==current]
            total=[int(y) for skill,y in recent]
            lag=next((j for j,(skill,y) in enumerate(reversed(recent),1) if skill==current),k+1)
            last_same=next((2*int(y)-1 for skill,y in reversed(recent) if skill==current),0)
            feats.append({"skill":current,"within_success":sum(within),
                          "within_failure":len(within)-sum(within),
                          "all_success":sum(total),"all_failure":len(total)-sum(total),
                          "prev_outcome":2*int(recent[-1][1])-1,
                          "prev_same_outcome":last_same,"lag_to_same":lag})
            labels.append(ys[t]);learners.append(uid);skills.append(current);positions.append(t)
    X=pd.DataFrame(feats)
    ids=pd.MultiIndex.from_arrays([learners,positions],names=["learner","target_position"])
    if ids.has_duplicates:raise ValueError("duplicate target identifiers")
    return X,np.asarray(labels,dtype=int),np.asarray(learners),np.asarray(skills),np.asarray(positions)

def fit_logit(train_f,test_f,which):
    cols=COUNT_COLS if which=="window_count" else ORDER_COLS
    X,y,*_=train_f; xx,yy,learner,skill,positions=test_f
    pre=ColumnTransformer([("skill",OneHotEncoder(handle_unknown="ignore"),[SKILL_COL]),
                           ("hist",StandardScaler(),cols)])
    estimator=LogisticRegression(C=1.,solver="lbfgs",max_iter=1600,tol=1e-7)
    model=Pipeline([("pre",pre),("model",estimator)])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(X[[SKILL_COL]+cols],y)
    warn=[str(q.message) for q in seen if issubclass(q.category,ConvergenceWarning)]
    prob=model.predict_proba(xx[[SKILL_COL]+cols])[:,1]
    return predictions(yy,prob,learner,skill,positions),{"iterations":int(model["model"].n_iter_[0]),"warnings":warn}

def predictions(y,p,learners,skills,positions):
    result=kt.prediction_frame(y,p,learners,skills,np.zeros(len(y),dtype=bool))
    result["target_position"]=np.asarray(positions,dtype=int)
    return result

class HistoryTargets(Dataset):
    def __init__(self,rows,k,vocab):
        self.data=[]
        self.index=[]
        for ri,row in enumerate(rows):
            ss,yy=kt.aligned(row)
            idx=np.array([vocab.get(s,0) for s in ss],dtype=np.int32)
            lab=np.array(yy,dtype=np.int8)
            self.data.append((idx,lab,str(row["user_id"]),ss))
            self.index.extend((ri,t) for t in range(1,len(yy)))
        self.k=k
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        ri,t=self.index[i]
        skills,answers,uid,sknames=self.data[ri]
        lo=max(0,t-self.k)
        # Encoding of previous events only; t's outcome appears only as label.
        tokens=(1+2*skills[lo:t]+answers[lo:t]).astype(np.int64)
        if not len(tokens):raise AssertionError("no history on valid target")
        return torch.from_numpy(tokens),int(skills[t]),int(answers[t])
def collate(batch):
    tokens=pad_sequence([z[0] for z in batch],batch_first=True,padding_value=0)
    q=torch.tensor([z[1] for z in batch],dtype=torch.long)
    y=torch.tensor([z[2] for z in batch],dtype=torch.float32)
    lengths=torch.tensor([len(z[0]) for z in batch],dtype=torch.long)
    return tokens,q,y,lengths

class WindowGRU(nn.Module):
    def __init__(self,nskills):
        super().__init__()
        self.tokens=nn.Embedding(2*(nskills+1)+1,24,padding_idx=0)
        self.query=nn.Embedding(nskills+1,12)
        self.gru=nn.GRU(24,48,batch_first=True)
        self.head=nn.Linear(48+12,1)
    def forward(self,x,q,length):
        h,_=self.gru(self.tokens(x))
        idx=(length-1).to(h.device)
        final=h[torch.arange(len(idx),device=h.device),idx]
        return self.head(torch.cat([final,self.query(q)],dim=-1)).squeeze(-1)

def fit_gru(train,val,test,k,vocab,epochs=12):
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    td=HistoryTargets(train,k,vocab)
    vd=HistoryTargets(val,k,vocab)
    ed=HistoryTargets(test,k,vocab)
    gen=torch.Generator().manual_seed(42)
    tl=DataLoader(td,batch_size=256,shuffle=True,collate_fn=collate,generator=gen,num_workers=0)
    vl=DataLoader(vd,batch_size=512,shuffle=False,collate_fn=collate,num_workers=0)
    el=DataLoader(ed,batch_size=512,shuffle=False,collate_fn=collate,num_workers=0)
    m=WindowGRU(len(vocab));opt=torch.optim.Adam(m.parameters(),lr=0.002)
    criterion=nn.BCEWithLogitsLoss()
    best=float("inf");best_epoch=0;best_weights=None;stale=0;history=[]
    def evaluate(loader,trainmode):
        m.train(trainmode);loss_sum=0;num=0
        ps=[]
        for x,q,y,lengths in loader:
            if trainmode:opt.zero_grad()
            with torch.set_grad_enabled(trainmode):
                logits=m(x,q,lengths);loss=criterion(logits,y)
                if trainmode:loss.backward();opt.step()
            loss_sum+=float(loss.detach())*len(y);num+=len(y)
            if not trainmode:ps.extend(torch.sigmoid(logits).detach().numpy().tolist())
        return loss_sum/max(num,1),ps
    for epoch in range(1,epochs+1):
        tr,_=evaluate(tl,True)
        with torch.no_grad():v,_=evaluate(vl,False)
        history.append({"epoch":epoch,"train_bce":tr,"validation_bce":v})
        if v<best-1e-5:
            best=v;best_epoch=epoch;stale=0;best_weights={name:x.detach().clone() for name,x in m.state_dict().items()}
        else:
            stale+=1
            if stale>=3:break
    m.load_state_dict(best_weights)
    with torch.no_grad():_,prob=evaluate(el,False)
    ff=frame(test,k)
    if len(ff[1])!=len(prob):raise RuntimeError("GRU targets misaligned with logistic")
    out=predictions(ff[1],np.asarray(prob),ff[2],ff[3],ff[4])
    return out,{"best_epoch":best_epoch,"epochs_completed":len(history),
                "history":history, "train_targets":len(td),"validation_targets":len(vd),
                "test_targets":len(ed)}

def align(*preds):
    first=preds[0]
    for p in preds[1:]:
        for col in ["learner","target_position","skill","y"]:
            if not np.array_equal(first[col].to_numpy(),p[col].to_numpy()):
                raise ValueError(f"Models do not evaluate same targets: {col}")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",default="data/cache/assistments_2015_official.csv")
    parser.add_argument("--output",default="results/matched_history_budget_2026.json")
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args()
    if a.smoke:
        synthetic=[{"user_id":"A","skill_ids":["x","y","x","x"],"grades":[0,1,1,0]},
                   {"user_id":"B","skill_ids":["y","x","y"],"grades":[1,0,1]}]
        one=frame(synthetic,1)
        twenty=frame(synthetic,20)
        # Target at t=2: previous last event is y=1, current x=1 must not leak.
        assert one[0].iloc[1]["within_success"]==0
        assert one[0].iloc[1]["prev_outcome"]==1
        assert twenty[0].iloc[1]["within_failure"]==1
        assert one[1].tolist()==twenty[1].tolist()
        altered=[dict(r) for r in synthetic]
        altered[0]=dict(synthetic[0],grades=[0,1,0,0])
        assert frame(altered,1)[0].iloc[1].equals(one[0].iloc[1])
        vocab={"x":1,"y":2}
        ds=HistoryTargets(synthetic,1,vocab)
        assert ds[1][0].tolist()==[1+2*2+1]
        print("SMOKE PASS: k restrictions, label exclusion, same targets")
        return
    source=Path(a.csv)
    download_official_csv(source)
    rows,manifest=official_unused_cohort(source)
    train,val,test=kt.split_rows(rows,seed=THIRD)
    train_ids={r["user_id"] for r in train};val_ids={r["user_id"] for r in val};test_ids={r["user_id"] for r in test}
    if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids:raise ValueError("learner leakage")
    if (len(train),len(val),len(test))!=(2800,600,600):raise ValueError("split size wrong")
    vocab=kt.skill_vocabulary(train)
    output={"status":"started","source":manifest,"design":{
        "predeclared_protocol":"METHODOLOGICAL_2026_PROTOCOL.md",
        "stage3_sampling_seed":THIRD,"learner_partition_seed":THIRD,
        "train_learners":len(train),"validation_learners":len(val),"test_learners":len(test),
        "history_budgets":HORIZONS,"model_types":["window_count","ordered_logistic","window_gru"],
        "GRU_seed":42,"bootstrap_replicates":BOOTSTRAPS,
        "same_targets":"Every model at every K predicts responses t >= 1 for identical 600 test learners.",
        "selection":"Fixed before new cohort test; prior research decisions were informed by other 2009/2015 data.",
        "inference":"Descriptive exploratory bootstrap contrasts, not multiplicity-corrected confirmatory tests."
    },"comparisons":{}}
    preds_by_k={}
    guards=[]
    for k in HORIZONS:
        ts=time.monotonic()
        train_f=frame(train,k)
        test_f=frame(test,k)
        all_preds={};diagnostics={}
        for which in ["window_count","ordered_logistic"]:
            all_preds[which],diagnostics[which]=fit_logit(train_f,test_f,which)
            if diagnostics[which]["warnings"]:guards.append(which+"_"+str(k)+"_nonconverged")
        all_preds["window_gru"],diagnostics["window_gru"]=fit_gru(train,val,test,k,vocab)
        align(*all_preds.values())
        if len(all_preds["window_gru"])==0:raise ValueError("zero test events")
        vals={name:kt.metric_bundle(z["y"],z["p"]) for name,z in all_preds.items()}
        contrasts={ref:paired_cluster_bootstrap(all_preds[ref],all_preds["window_gru"],
                    repetitions=BOOTSTRAPS,seed=THIRD+int(k))
                   for ref in ["window_count","ordered_logistic"]}
        output["comparisons"][str(k)]={"models":vals,"paired_GRU_minus":contrasts,
                                       "diagnostics":diagnostics,
                                       "elapsed_seconds":round(time.monotonic()-ts,2)}
        preds_by_k[k]=all_preds
        print("FINISHED_HORIZON",k,json.dumps({"brier":{key:value["brier"] for key,value in vals.items()},
              "gru_minus_ordered_ci":contrasts["ordered_logistic"]["bootstrap_95_interval"]}),flush=True)
    # Same exact learners and target ordinals between history budgets, with identical label stream.
    align(*[p for k in HORIZONS for p in preds_by_k[k].values()])
    output["within_model_brier_k20_minus_k1"]={
        name:float(output["comparisons"]["20"]["models"][name]["brier"]-
                   output["comparisons"]["1"]["models"][name]["brier"])
        for name in ["window_count","ordered_logistic","window_gru"]}
    first=output["comparisons"]["1"]["paired_GRU_minus"]["ordered_logistic"]["difference_candidate_minus_reference"]
    last=output["comparisons"]["20"]["paired_GRU_minus"]["ordered_logistic"]["difference_candidate_minus_reference"]
    output["gru_minus_ordered_gap_k20_minus_k1"]=float(last-first)
    output["status"]="passed_all_computational_gates" if not guards else "inconclusive_diagnostic_fail"
    output["diagnostic_failures"]=guards
    output["interpretation"]="Same target set and same last-K learner response information; representations/capacities still unequal. Single stage-3 sample after prior research; no familywise or causal claim."
    out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("FINAL_RESULTS",json.dumps({"status":output["status"],
          "delta_k20_minus_k1":output["within_model_brier_k20_minus_k1"],
          "gap_change":output["gru_minus_ordered_gap_k20_minus_k1"],
          "path":str(out)},indent=2),flush=True)
    if guards:sys.exit(3)
if __name__=="__main__":main()
