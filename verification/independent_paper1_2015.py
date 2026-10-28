"""Separate implementation of the 2026 ASSISTments-2015 K=1/K=20 study.
No imports from the original project. Writes aggregates only; no student records.
This is independent computational verification of a previously inspected analysis,
NOT a fresh holdout or an identified causal architecture effect.
"""
import hashlib,json,random,datetime,platform,warnings
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

SEEDS=(20261022,20261023,20261026)
SOURCE_SHA="75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0"
GDRIVE_ID="0B_hO8cnpcIMgUGZzRnh3bHJrSjQ"
BASIC=["within_success","within_failure","all_success","all_failure"]
EXTRA=["prev_outcome","prev_same_outcome","lag_to_same"]
BASELINE={"1":{"window_count":0.17788199203683247,"ordered_logistic":0.1778819783584268,"window_gru":0.18000310877303297},
"20":{"window_count":0.17667353711786812,"ordered_logistic":0.17388534919067833,"window_gru":0.17514020613173276}}

def source():
    import gdown
    p=Path("data/cache/paper1_independent_assist2015.csv");p.parent.mkdir(exist_ok=True,parents=True)
    if not p.exists():
        if not gdown.download(id=GDRIVE_ID,output=str(p),quiet=False):
            raise RuntimeError("Official 2015 file could not be downloaded")
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=SOURCE_SHA:raise RuntimeError("Official file checksum mismatch: "+h)
    data=pd.read_csv(p,low_memory=False)
    data.columns=[str(c).lower().strip() for c in data.columns]
    required=["user_id","log_id","sequence_id","correct"]
    if len(set(data.columns))!=len(data.columns) or any(c not in data for c in required):
        raise RuntimeError("Unexpected CSV columns")
    z=data[required].copy()
    missing=z.isna().any(axis=1)
    z["log_id"]=pd.to_numeric(z["log_id"],errors="coerce")
    z["correct"]=pd.to_numeric(z["correct"],errors="coerce")
    drop=missing|z.isna().any(axis=1)|~z["correct"].isin([0,1])
    nbad=int(drop.sum());z=z.loc[~drop].copy()
    if z.duplicated(["user_id","log_id"]).any():raise RuntimeError("Duplicated event ID")
    if not np.isfinite(z["log_id"]).all() or not np.all(z["log_id"].to_numpy()==z["log_id"].astype("int64").to_numpy()):
        raise RuntimeError("Invalid ordering keys")
    z["user_id"]=z["user_id"].astype(str)
    z["sequence_id"]=z["sequence_id"].astype(str)
    z["correct"]=z["correct"].astype(int)
    population=sorted(z["user_id"].unique().tolist())
    s1=set(random.Random(SEEDS[0]).sample(population,5000))
    left=[x for x in population if x not in s1]
    s2=set(random.Random(SEEDS[1]).sample(left,5000))
    left=[x for x in left if x not in s2]
    chosen=set(random.Random(SEEDS[2]).sample(left,4000))
    if chosen&s1 or chosen&s2:raise RuntimeError("Prior cohort overlap")
    selected=z.loc[z["user_id"].isin(chosen)].sort_values(["user_id","log_id"],kind="stable")
    students=[(str(u),q["sequence_id"].tolist(),q["correct"].tolist()) for u,q in selected.groupby("user_id",sort=True)]
    if (len(data),nbad,len(population),len(students),len(selected))!=(708631,24830,19840,4000,138129):
        raise RuntimeError("Unexpected population accounting")
    random.Random(SEEDS[2]).shuffle(students)
    info={"sha256":h,"raw_rows":len(data),"excluded_invalid":nbad,"eligible_learners":len(population),
          "prior_1_excluded":len(s1),"prior_2_excluded":len(s2),"unused_pool":len(left),
          "new_learners":len(students),"new_interactions":len(selected),"previous_overlap":len(chosen&(s1|s2)),
          "cohort_id_sha256":hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest(),
          "train":2800,"validation":600,"test":600}
    return students[:2800],students[2800:3400],students[3400:],info

def records(cohort,k):
    f=[];y=[];ids=[]
    for uid,skills,ans in cohort:
        for t in range(1,len(ans)):
            earlier=list(zip(skills[max(0,t-k):t],ans[max(0,t-k):t]))
            same=[r for s,r in earlier if s==skills[t]]
            pos=next(((j,r) for j,(s,r) in enumerate(reversed(earlier),1) if s==skills[t]),None)
            f.append({"skill":skills[t],"within_success":sum(same),"within_failure":len(same)-sum(same),
                      "all_success":sum(r for _,r in earlier),"all_failure":len(earlier)-sum(r for _,r in earlier),
                      "prev_outcome":2*earlier[-1][1]-1,"prev_same_outcome":2*pos[1]-1 if pos else 0,
                      "lag_to_same":pos[0] if pos else k+1})
            y.append(ans[t]);ids.append((uid,t,skills[t],ans[t]))
    return pd.DataFrame(f),np.array(y,dtype=int),ids

def regression(tr,te,k,ordered):
    a,y,_=records(tr,k);b,yt,ids=records(te,k)
    numeric=BASIC+EXTRA if ordered else BASIC
    pre=ColumnTransformer([("onehot",OneHotEncoder(handle_unknown="ignore"),["skill"]),
                           ("numeric",StandardScaler(),numeric)])
    clf=Pipeline([("prep",pre),("fit",LogisticRegression(C=1.,solver="lbfgs",max_iter=1600,tol=1e-7))])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always",ConvergenceWarning)
        clf.fit(a[["skill"]+numeric],y)
    if any(issubclass(e.category,ConvergenceWarning) for e in w):
        raise RuntimeError("Unconverged logistic regression")
    return clf.predict_proba(b[["skill"]+numeric])[:,1],yt,ids,int(clf["fit"].n_iter_[0])

class Prefixes(Dataset):
    def __init__(self,cohort,k,vocab):
        self.students=[([vocab.get(s,0) for s in skills],ans) for uid,skills,ans in cohort]
        self.targets=[(i,t) for i,(_,ys) in enumerate(self.students) for t in range(1,len(ys))]
        self.k=k
    def __len__(self):return len(self.targets)
    def __getitem__(self,n):
        i,t=self.targets[n];skills,ans=self.students[i]
        history=[1+2*skills[j]+ans[j] for j in range(max(0,t-self.k),t)]
        return torch.tensor(history,dtype=torch.long),skills[t],float(ans[t])

def batch(rows):
    x=pad_sequence([r[0] for r in rows],batch_first=True,padding_value=0)
    return x,torch.tensor([r[1] for r in rows]),torch.tensor([r[2] for r in rows]),torch.tensor([len(r[0]) for r in rows])

class IndependentGRU(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.tokens=nn.Embedding(2*(n+1)+1,24,padding_idx=0)
        self.queries=nn.Embedding(n+1,12)
        self.core=nn.GRU(24,48,batch_first=True)
        self.last=nn.Linear(60,1)
    def forward(self,x,q,length):
        h,_=self.core(self.tokens(x))
        h=h[torch.arange(x.size(0)),length-1]
        return self.last(torch.cat((h,self.queries(q)),dim=-1)).squeeze(-1)

def gru(train,valid,test,k):
    vocab={s:i+1 for i,s in enumerate(sorted(set(s for _,ss,_ in train for s in ss)))}
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    parts=[Prefixes(z,k,vocab) for z in (train,valid,test)]
    loaders=[DataLoader(parts[0],batch_size=256,shuffle=True,collate_fn=batch,generator=torch.Generator().manual_seed(42)),
             DataLoader(parts[1],batch_size=512,shuffle=False,collate_fn=batch),
             DataLoader(parts[2],batch_size=512,shuffle=False,collate_fn=batch)]
    m=IndependentGRU(len(vocab))
    opt=torch.optim.Adam(m.parameters(),lr=.002)
    criterion=nn.BCEWithLogitsLoss()
    best=float("inf");bestepoch=0;state=None;stale=0;epochs=[]
    for epoch in range(1,13):
        losses=[]
        for mode,loader in ((True,loaders[0]),(False,loaders[1])):
            m.train(mode);summed=0.;n=0
            for x,q,y,l in loader:
                if mode:opt.zero_grad()
                with torch.set_grad_enabled(mode):
                    error=criterion(m(x,q,l),y)
                    if mode:error.backward();opt.step()
                summed+=float(error.detach())*len(y);n+=len(y)
            losses.append(summed/n)
        epochs.append({"epoch":epoch,"train_bce":losses[0],"validation_bce":losses[1]})
        if losses[1]<best-1e-5:
            best=losses[1];bestepoch=epoch;stale=0;state={key:value.detach().clone() for key,value in m.state_dict().items()}
        else:
            stale+=1
            if stale>=3:break
    m.load_state_dict(state);m.eval();probs=[]
    with torch.no_grad():
        for x,q,y,l in loaders[2]:
            probs.extend(torch.sigmoid(m(x,q,l)).cpu().numpy().tolist())
    return np.array(probs),{"best_epoch":bestepoch,"epochs_completed":len(epochs),"history":epochs}

def boot(y,ids,ref,trial,seed):
    group={}
    for (uid,t,skill,label),actual,p,q in zip(ids,y,ref,trial):
        record=group.setdefault(uid,[0.,0])
        record[0]+=(float(q)-actual)**2-(float(p)-actual)**2
        record[1]+=1
    values=np.array(list(group.values()))
    totals,counts=values[:,0],values[:,1]
    rng=np.random.default_rng(seed);samples=[]
    for j in range(2000):
        pick=rng.integers(len(group),size=len(group))
        samples.append(np.sum(totals[pick])/np.sum(counts[pick]))
    return {"difference":float(np.sum(totals)/np.sum(counts)),
            "ci_95":np.quantile(samples,[.025,.975]).tolist(),
            "bootstrap_draws":2000,"evaluable_learners":len(group),"targets":int(np.sum(counts))}

def test_prefix():
    s=[("A",["x","y","x","x"],[0,1,1,0]),("B",["y","x","y"],[1,0,1])]
    a=records(s,1);b=records(s,20)
    assert np.array_equal(a[1],b[1]) and a[2]==b[2]
    assert a[0].iloc[1]["within_success"]==0 and a[0].iloc[1]["prev_outcome"]==1
    assert b[0].iloc[1]["within_failure"]==1
    mutated=[("A",["x","y","x","x"],[0,1,0,0]),s[1]]
    assert a[0].iloc[1].equals(records(mutated,1)[0].iloc[1])
    print("INDEPENDENT PREFIX SMOKE TEST PASSED",flush=True)

def main():
    test_prefix()
    train,valid,test,manifest=source()
    result={"date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "purpose":"independent reimplementation of fixed 2015 experiment","data":manifest,
            "environment":{"python":platform.python_version(),"numpy":np.__version__,
                           "pandas":pd.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__},
            "windows":{}}
    streams=[]
    for k in (1,20):
        predicted={};iterations={}
        for name,ordered in (("window_count",False),("ordered_logistic",True)):
            p,y,ids,it=regression(train,test,k,ordered)
            predicted[name]=p;iterations[name]=it
        net,diagnostic=gru(train,valid,test,k)
        if len(net)!=len(y):raise RuntimeError("Target length does not agree across methods")
        predicted["window_gru"]=net
        metrics={kind:{"n":len(y),"brier":float(np.mean((p-y)**2)),"positive_rate":float(y.mean())} for kind,p in predicted.items()}
        ci={name:boot(y,ids,predicted[name],net,SEEDS[2]+k) for name in ("window_count","ordered_logistic")}
        diffs={name:metrics[name]["brier"]-BASELINE[str(k)][name] for name in metrics}
        # Fixed tolerance is only a numerical reproduction threshold, not an inference test.
        concordance={name:abs(delta)<0.0002 for name,delta in diffs.items()}
        target_hash=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
        streams.append(target_hash)
        result["windows"][str(k)]={"metrics":metrics,"bootstraps":ci,"gru_training":diagnostic,
            "logistic_iterations":iterations,"target_stream_sha256":target_hash,
            "test_evaluable_learners":len(set(uid for uid,*_ in ids)),
            "baseline_brier_deltas":diffs,"brier_concordance_with_original":concordance}
        print("INDEPENDENT_K_COMPLETE",k,json.dumps({"brier":metrics,"bootstrap":ci,"agreement":concordance}),flush=True)
    result["targets_identical_across_k"]=streams[0]==streams[1]
    result["same_test_counts"]=all(result["windows"][str(k)]["metrics"]["window_gru"]["n"]==20592 and
                                  result["windows"][str(k)]["test_evaluable_learners"]==582 for k in (1,20))
    result["overall_reproduced"]=(result["targets_identical_across_k"] and result["same_test_counts"] and
        all(all(z["brier_concordance_with_original"].values()) for z in result["windows"].values()))
    out=Path("results/independent_paper1_2015_20261028.json");out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("INDEPENDENT_REPRODUCTION_RESULT",result["overall_reproduced"],"evidence",out,flush=True)
    if not result["overall_reproduced"]:raise SystemExit(4)

if __name__=="__main__":main()
