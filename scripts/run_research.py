import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from knowledge_tracing_benchmark.core import BKTParams, predict_correct_probability, update_mastery

SEED=42
DATASET="Atomi/ASSISTments2009"

def aligned(row):
    skills=list(row["skill_ids"])
    grades=[int(str(x)) for x in row["grades"]]
    if len(skills)!=len(grades):
        raise ValueError(f"misaligned learner row {row['user_id']}")
    if any(v not in (0,1) for v in grades):
        raise ValueError("grades must be binary")
    return [str(s) for s in skills],grades

def split_rows(rows, seed=SEED):
    rows=list(rows)
    rng=random.Random(seed)
    rng.shuffle(rows)
    n=len(rows)
    a=int(.70*n); b=int(.85*n)
    return rows[:a],rows[a:b],rows[b:]

def train_priors(rows):
    total_correct=0; total_n=0
    skill_correct=defaultdict(int); skill_n=defaultdict(int)
    for row in rows:
        skills,grades=aligned(row)
        for skill,y in zip(skills,grades):
            total_correct+=y; total_n+=1
            skill_correct[skill]+=y; skill_n[skill]+=1
    global_rate=total_correct/total_n
    skill_rate={s:skill_correct[s]/skill_n[s] for s in skill_n}
    return global_rate,skill_rate,total_n

def metrics(y,p):
    out={
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "n":len(y),
        "positive_rate":sum(y)/len(y),
    }
    out["roc_auc"]=float(roc_auc_score(y,p)) if len(set(y))==2 else None
    return out

def evaluate(rows,global_rate,skill_rate,params):
    y=[]; global_p=[]; skill_p=[]; bkt_p=[]; cold=[]
    for row in rows:
        skills,grades=aligned(row)
        mastery={}
        seen=set()
        for skill,grade in zip(skills,grades):
            current=mastery.get(skill,params.p_init)
            y.append(grade)
            global_p.append(global_rate)
            skill_p.append(skill_rate.get(skill,global_rate))
            bkt_p.append(predict_correct_probability(current,params))
            cold.append(skill not in seen)
            mastery[skill]=update_mastery(current,bool(grade),params)
            seen.add(skill)
    result={
        "all":{
            "global_rate":metrics(y,global_p),
            "skill_prior":metrics(y,skill_p),
            "bkt":metrics(y,bkt_p),
        }
    }
    for name,flag in [("cold_start",True),("repeated_skill",False)]:
        idx=[i for i,v in enumerate(cold) if v is flag]
        if idx:
            yy=[y[i] for i in idx]
            result[name]={
                "global_rate":metrics(yy,[global_p[i] for i in idx]),
                "skill_prior":metrics(yy,[skill_p[i] for i in idx]),
                "bkt":metrics(yy,[bkt_p[i] for i in idx]),
            }
    return result

def main():
    ds=load_dataset(DATASET,split="train")
    rows=[dict(row) for row in ds]
    train,validation,test=split_rows(rows)
    global_rate,skill_rate,n_train_interactions=train_priors(train)
    params=BKTParams()
    result={
        "research_bundle":True,
        "dataset":DATASET,
        "n_learners":len(rows),
        "splits":{"train":len(train),"validation":len(validation),"test":len(test)},
        "train_interactions":n_train_interactions,
        "training_global_correctness":global_rate,
        "training_skills":len(skill_rate),
        "bkt_params":params.__dict__,
        "test_evaluation":evaluate(test,global_rate,skill_rate,params),
        "seed":SEED,
    }
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    (out/"research_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
