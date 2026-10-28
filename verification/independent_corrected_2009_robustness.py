"""Fresh code path for the source-corrected ASSISTments 2009 robustness analysis.
No imports from the original research scripts or src modules.
Uses public corrected raw source, tests a frozen learner split, refits logistic
PFA+ and a sequence GRU, and outputs aggregate metrics + learner-cluster CI only.
Prior test set has been viewed: this is exploratory reproduction, not confirmation.
"""
from __future__ import annotations
import hashlib, io, json, random, re, warnings, zipfile, platform
from collections import defaultdict
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

OFFICIAL_FILE_ID="1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
OFFICIAL_HASH="162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
SPLIT_SEED=20261024
BASELINE={
  "enriched_pfa":{"brier":0.19675493213137696,"auc":0.6950809111386503},
  "extended_gru":{"brier":0.18507380934628698,"auc":0.7324336482108271},
  "gru_minus_pfa":-0.011681122785090007,
  "bootstrap_95ci":[-0.013717998422616664,-0.009862452037847193]}
FROZEN_ABS_TOLERANCE=2e-5

def checksum(path):
    sha=hashlib.sha256()
    with open(path,"rb") as fp:
        for chunk in iter(lambda:fp.read(1<<20),b""):sha.update(chunk)
    return sha.hexdigest()

def norm_id(x):
    t=str(x).strip()
    if t.endswith(".0") and t[:-2].lstrip("-").isdigit():return t[:-2]
    return t

def norm_skill(x):
    t=str(x).strip()
    if t.lower() in ("nan","null","none",""):return ""
    return "_".join(norm_id(piece) for piece in re.split(r"[_;]",t) if piece.strip())

def fetch_cohort():
    folder=Path("data/cache/independent_2009_robustness")
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/"official_corrected_2009_source"
    if not file.exists():
        if not gdown.download(id=OFFICIAL_FILE_ID,output=str(file),quiet=False):
            raise RuntimeError("Official corrected 2009 source is unavailable")
    observed=checksum(file)
    if observed!=OFFICIAL_HASH:
        raise RuntimeError("SOURCE SHA-256 DOES NOT MATCH ORIGINAL RECORD "+observed)
    if zipfile.is_zipfile(file):
        with zipfile.ZipFile(file) as z:
            files=[n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if not files:raise RuntimeError("Source archive contains no CSV")
            files.sort(key=lambda n:z.getinfo(n).file_size,reverse=True)
            with z.open(files[0]) as fp:raw=pd.read_csv(fp,low_memory=False,encoding="latin-1")
    else:
        raw=pd.read_csv(file,low_memory=False,encoding="latin-1")
    raw.columns=[str(c).strip().lower().replace(" ","_") for c in raw.columns]
    cols=("user_id","order_id","skill_id","correct")
    if any(c not in raw for c in cols):raise RuntimeError("Required source fields absent")
    count=len(raw)
    tab=raw[list(cols)].dropna(subset=cols).copy()
    tab["user_id"]=tab["user_id"].map(norm_id)
    tab["skill_id"]=tab["skill_id"].map(norm_skill)
    tab["correct"]=pd.to_numeric(tab["correct"],errors="coerce")
    tab["order_id"]=pd.to_numeric(tab["order_id"],errors="coerce")
    mask=tab["correct"].isin([0,1]) & tab["order_id"].notna() & (tab["skill_id"]!="")
    tab=tab.loc[mask].copy()
    tab["correct"]=tab["correct"].astype(int)
    if tab.duplicated(["user_id","order_id"]).any():raise RuntimeError("Duplicate learner order IDs")
    tab=tab.sort_values(["user_id","order_id"],kind="stable")
    people=[(str(uid),group["skill_id"].tolist(),group["correct"].tolist())
            for uid,group in tab.groupby("user_id",sort=True)]
    if (count,len(tab),len(people))!=(346860,283105,4163):raise RuntimeError("Data accounting mismatch")
    rng=random.Random(SPLIT_SEED);rng.shuffle(people)
    a,b=int(.70*len(people)),int(.85*len(people))
    train,validate,test=people[:a],people[a:b],people[b:]
    if (len(train),len(validate),len(test))!=(2914,624,625):
        raise RuntimeError("Partition sizes differ from archival protocol")
    all_ids=[set(p[0] for p in cohort) for cohort in (train,validate,test)]
    if any(all_ids[x]&all_ids[y] for x,y in ((0,1),(0,2),(1,2))):
        raise RuntimeError("Learner leakage in split")
    return train,validate,test,{"official_sha256":observed,"original_rows":count,
      "labeled_events":len(tab),"learner_count":len(people),
      "train":len(train),"validation":len(validate),"test":len(test),
      "test_interactions":sum(len(x[2]) for x in test)}

def feature_view(cohort):
    items=[];labels=[];uids=[]
    for uid,skills,answers in cohort:
        used=defaultdict(int);failed=defaultdict(int)
        wins=losses=0
        for skill,answer in zip(skills,answers):
            items.append({"skill":skill,"skill_success":used[skill],
              "skill_failure":failed[skill],"all_success":wins,"all_failure":losses})
            labels.append(answer);uids.append(uid)
            if answer:used[skill]+=1;wins+=1
            else:failed[skill]+=1;losses+=1
    return pd.DataFrame(items),np.asarray(labels,dtype=int),uids

def train_logistic(train,test):
    x,labels,_=feature_view(train)
    xx,y,ids=feature_view(test)
    numeric=["skill_success","skill_failure","all_success","all_failure"]
    pre=ColumnTransformer([
      ("skill_one_hot",OneHotEncoder(handle_unknown="ignore"),["skill"]),
      ("numeric_standardized",StandardScaler(),numeric)])
    model=Pipeline([
      ("transform",pre),
      ("logistic",LogisticRegression(solver="lbfgs",max_iter=2000,tol=1e-8,C=1.0))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(x,labels)
    problems=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if problems:raise RuntimeError("Logistic convergence warnings: "+str(problems))
    return np.asarray(model.predict_proba(xx)[:,1],dtype=float),y,ids,{
      "solver":"lbfgs","iterations":int(model.named_steps["logistic"].n_iter_[0])}

class EventSequences(Dataset):
    def __init__(self,cohort,vocab):
        self.rows=[]
        for uid,skills,labels in cohort:
            if len(labels)<2:continue
            ids=[vocab.get(s,0) for s in skills]
            tokens=[1+ids[i]*2+labels[i] for i in range(len(labels)-1)]
            self.rows.append((torch.tensor(tokens,dtype=torch.long),
                 torch.tensor(ids[1:],dtype=torch.long),
                 torch.tensor(labels[1:],dtype=torch.float32)))
    def __len__(self):return len(self.rows)
    def __getitem__(self,j):return self.rows[j]

def pack(batch):
    token=pad_sequence([x[0] for x in batch],batch_first=True,padding_value=0)
    query=pad_sequence([x[1] for x in batch],batch_first=True,padding_value=0)
    true=pad_sequence([x[2] for x in batch],batch_first=True,padding_value=0.)
    sizes=torch.tensor([len(x[0]) for x in batch],dtype=torch.long)
    mask=torch.arange(token.shape[1])[None,:]<sizes[:,None]
    return token,query,true,mask

class SequenceNet(nn.Module):
    def __init__(self,n_skills):
        super().__init__()
        self.history=nn.Embedding(1+2*(n_skills+1),32,padding_idx=0)
        self.question=nn.Embedding(n_skills+1,16)
        self.recurrent=nn.GRU(32,64,batch_first=True)
        self.head=nn.Linear(80,1)
    def forward(self,tokens,queries):
        sequence,_=self.recurrent(self.history(tokens))
        side=self.question(queries)
        return self.head(torch.cat([sequence,side],dim=-1)).squeeze(-1)

def one_pass(model,loader,optimizer=None):
    training=optimizer is not None
    model.train(training)
    objective=nn.BCEWithLogitsLoss(reduction="none")
    total=0.
    steps=0
    for token,query,actual,mask in loader:
        if training:optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            pred=model(token,query)
            losses=objective(pred,actual)
            loss=(losses*mask).sum()/mask.sum()
            if training:
                loss.backward()
                optimizer.step()
        total+=float((losses*mask).sum().detach().cpu())
        steps+=int(mask.sum())
    return total/max(steps,1)

def predict_network(train,val,test):
    vocab={skill:j+1 for j,skill in enumerate(sorted({x for _,skills,_ in train for x in skills}))}
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(42)
    train_loader=DataLoader(EventSequences(train,vocab),batch_size=32,shuffle=True,collate_fn=pack,generator=generator)
    val_loader=DataLoader(EventSequences(val,vocab),batch_size=64,shuffle=False,collate_fn=pack)
    net=SequenceNet(len(vocab))
    optimizer=torch.optim.Adam(net.parameters(),lr=.002)
    best=float("inf");best_epoch=0;stale=0;weights=None;history=[]
    for epoch in range(1,26):
        train_loss=one_pass(net,train_loader,optimizer)
        with torch.no_grad():validation_loss=one_pass(net,val_loader)
        history.append((epoch,train_loss,validation_loss))
        if validation_loss<best-1e-5:
            best=validation_loss;best_epoch=epoch;stale=0
            weights={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:
            stale+=1
            if stale>=4:break
    net.load_state_dict(weights);net.eval()
    count=0;total=0;skill_counts=defaultdict(int);skill_correct=defaultdict(int)
    for _,skills,answers in train:
        for s,a in zip(skills,answers):
            total+=a;count+=1;skill_counts[s]+=1;skill_correct[s]+=a
    overall=total/count;prior={s:skill_correct[s]/skill_counts[s] for s in skill_counts}
    preds=[];truth=[];ids=[]
    with torch.no_grad():
        for uid,skills,answers in test:
            if not skills:continue
            q=[vocab.get(s,0) for s in skills]
            probabilities=[prior.get(skills[0],overall)]
            if len(answers)>1:
                history_tokens=torch.tensor([[1+q[t]*2+answers[t] for t in range(len(answers)-1)]],dtype=torch.long)
                queries=torch.tensor([q[1:]],dtype=torch.long)
                neural=torch.sigmoid(net(history_tokens,queries).squeeze(0)).cpu().numpy()
                for t,p in enumerate(neural,start=1):
                    probabilities.append(float(p) if q[t]!=0 else prior.get(skills[t],overall))
            preds.extend(probabilities);truth.extend(answers);ids.extend([uid]*len(answers))
    return np.array(preds,float),np.array(truth,int),ids,{"gru_best_epoch":best_epoch,
      "epochs_completed":len(history),"gru_training_last_validation":float(history[-1][2]),
      "vocabulary":len(vocab)}

def summarize(y,p):
    n=len(y);p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return {"n":int(n),"positive_rate":float(y.mean()),"brier":float(np.mean((p-y)**2)),
      "log_loss":float(log_loss(y,p,labels=[0,1])),
      "roc_auc":float(roc_auc_score(y,p)),
      "average_precision":float(average_precision_score(y,p))}

def independent_cluster_interval(y,reference,candidate,learners):
    groups={}
    for idx,uid in enumerate(learners):
        row=groups.setdefault(uid,[0.,0])
        delta=(float(candidate[idx])-int(y[idx]))**2-(float(reference[idx])-int(y[idx]))**2
        row[0]+=delta;row[1]+=1
    values=np.array(list(groups.values()),float)
    losses,counts=values[:,0],values[:,1]
    rng=np.random.default_rng(SPLIT_SEED)
    draws=np.empty(2000)
    for i in range(2000):
        ix=rng.integers(0,len(groups),size=len(groups))
        draws[i]=losses[ix].sum()/counts[ix].sum()
    return {"gru_minus_pfa":float(losses.sum()/counts.sum()),
      "percentile_interval":np.quantile(draws,[.025,.975]).tolist(),
      "n_learners":int(len(groups)),"n_events":int(sum(counts)),"bootstrap_draws":int(len(draws))}

def main():
    tr,va,te,data=fetch_cohort()
    pp,y,ids,logistic=train_logistic(tr,te)
    neural,yy,nn_ids,training=predict_network(tr,va,te)
    if not np.array_equal(y,yy) or ids!=nn_ids:raise RuntimeError("Different target identities across methods")
    if len(y)!=48842:raise RuntimeError("Test event sample changed")
    metrics={"enriched_pfa":summarize(y,pp),"extended_gru":summarize(y,neural)}
    paired=independent_cluster_interval(y,pp,neural,ids)
    deviations={key:{kind:float(metrics[key][kind]-BASELINE[key][kind]) for kind in ("brier","roc_auc") if kind in BASELINE[key]} for key in metrics}
    # Compare explicitly and separately to the archived outputs; never substitute baseline metrics.
    compare={
       "pfa_brier":abs(metrics["enriched_pfa"]["brier"]-BASELINE["enriched_pfa"]["brier"])<FROZEN_ABS_TOLERANCE,
       "gru_brier":abs(metrics["extended_gru"]["brier"]-BASELINE["extended_gru"]["brier"])<FROZEN_ABS_TOLERANCE,
       "pfa_auc":abs(metrics["enriched_pfa"]["roc_auc"]-BASELINE["enriched_pfa"]["auc"])<FROZEN_ABS_TOLERANCE,
       "gru_auc":abs(metrics["extended_gru"]["roc_auc"]-BASELINE["extended_gru"]["auc"])<FROZEN_ABS_TOLERANCE,
       "delta_brier":abs(paired["gru_minus_pfa"]-BASELINE["gru_minus_pfa"])<FROZEN_ABS_TOLERANCE,
       "ci_lower":abs(paired["percentile_interval"][0]-BASELINE["bootstrap_95ci"][0])<FROZEN_ABS_TOLERANCE,
       "ci_upper":abs(paired["percentile_interval"][1]-BASELINE["bootstrap_95ci"][1])<FROZEN_ABS_TOLERANCE
    }
    status="PASS" if all(compare.values()) else "DISAGREEMENT"
    report={"status":status,"scope":"independent source-corrected 2009 exploratory computational reproduction",
       "data":data,"protocol":{"split_seed":SPLIT_SEED,"gru_seed":42,"max_gru_epochs":25,
          "gru_patience":4,"bootstrap_draws":2000,"resampling_unit":"learner"},
       "training":{"logistic":logistic,"gru":training},
       "metrics":metrics,"paired":paired,"deviations_from_original":deviations,
       "concordance":compare,"environment":{"python":platform.python_version(),
          "pandas":pd.__version__,"numpy":np.__version__,
          "sklearn":sklearn.__version__,"torch":torch.__version__},
       "interpretation":"Independent computational reproduction of an exploratory comparison; does not establish causal architecture effects or untouched confirmation."}
    destination=Path("results/independent_corrected_2009_robustness_20261028.json")
    destination.parent.mkdir(exist_ok=True,parents=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("INDEPENDENT_2009_ROBUSTNESS",json.dumps({"status":status,"metrics":metrics,
       "paired":paired,"concordance":compare,"training":training,"source_sha256":data["official_sha256"]}),flush=True)
    if status!="PASS":raise SystemExit(3)

if __name__=="__main__":main()
