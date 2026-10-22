"""Exploratory, source-corrected 2009 learning-sequence analysis, uses official raw data.
The source hash and protocol are fixed in CORRECTED_2009_REANALYSIS_PROTOCOL.md.
Outputs aggregate metadata only, not learner records.
"""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from audit_corrected_2009 import get_source,load_csv_or_archive,identifier,skill_token
from robustness_extension import paired_cluster_bootstrap, pfa_plus_all_skill_history_frame
import run_experiment as kt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

SOURCE_HASH='162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da'
SEED=20261024

def load():
    p=get_source()
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha!=SOURCE_HASH:raise RuntimeError('Corrected 2009 official raw source hash does not match audited file')
    raw=load_csv_or_archive(p)
    raw.columns=[str(c).strip().lower().replace(' ','_') for c in raw.columns]
    required=('user_id','order_id','skill_id','correct')
    if any(c not in raw.columns for c in required):raise ValueError('Corrected source lacks essential fields')
    n_all=len(raw)
    source=raw[list(required)].copy()
    source=source.dropna(subset=['user_id','order_id','skill_id','correct']).copy()
    source['user_id']=source['user_id'].map(identifier)
    source['skill_id']=source['skill_id'].map(skill_token)
    source['correct']=pd.to_numeric(source['correct'],errors='coerce')
    source['order_id']=pd.to_numeric(source['order_id'],errors='coerce')
    valid=source['correct'].isin([0,1]) & source['order_id'].notna() & (source['skill_id']!='')
    source=source.loc[valid].copy()
    source['correct']=source['correct'].astype(int)
    if source.duplicated(['user_id','order_id']).any():raise ValueError('Duplicated learner-event order_id')
    source=source.sort_values(['user_id','order_id'],kind='stable')
    rows=[{'user_id':str(user),'skill_ids':g['skill_id'].tolist(),'grades':g['correct'].tolist()}
          for user,g in source.groupby('user_id',sort=True)]
    if len(rows)!=4163 or len(source)!=283105 or n_all!=346860:
        raise ValueError('Corrected source aggregate record count differs from independent provenance audit')
    return rows,{'corrected_sha256':sha,'original_source_rows':n_all,'labeled_rows':len(source),
                 'excluded_unlabeled_or_invalid':n_all-len(source),
                 'learner_count':len(rows),'chronology':'numeric order_id; source-identified sequence order'}

def pfa(train,test):
    X,y,*_=pfa_plus_all_skill_history_frame(train)
    XX,yy,learners,skills,first=pfa_plus_all_skill_history_frame(test)
    pipe=Pipeline([('prep',ColumnTransformer([
         ('skill',OneHotEncoder(handle_unknown='ignore'),['skill']),
         ('counts',StandardScaler(),['skill_success','skill_failure','all_success','all_failure'])])),
         ('clf',LogisticRegression(solver='lbfgs',max_iter=2000,tol=1e-8,C=1.0))])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        pipe.fit(X,y)
    messages=[str(q.message) for q in captured if issubclass(q.category,ConvergenceWarning)]
    return kt.prediction_frame(yy,pipe.predict_proba(XX)[:,1],learners,skills,first),messages,int(pipe.named_steps['clf'].n_iter_[0])

def run():
    rows,source=load()
    train,val,test=kt.split_rows(rows,seed=SEED)
    ids=[{str(q['user_id']) for q in group} for group in (train,val,test)]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    pfa_pred,warnings_list,pfa_n=pfa(train,test)
    global_rate,skills,_=kt.train_priors(train)
    vocab=kt.skill_vocabulary(train)
    kt.GRU_PATIENCE=4
    model,history,best=kt.train_gru(train,val,vocab,seed=42,epochs=25)
    gru_pred=kt.predict_gru(model,test,vocab,global_rate,skills)
    delta=paired_cluster_bootstrap(pfa_pred,gru_pred,repetitions=2000,seed=SEED)
    guards=not warnings_list and len(history)>=1 and delta['n_learners']==len(test)
    return {'status':'completed_exploratory_corrected_source' if guards else 'inconclusive_diagnostics',
            'analysis_type':'exploratory source-sensitivity, not confirmatory',
            'source':source,
            'protocol':{'split_seed':SEED,'gru_seed':42,'train_learners':len(train),'validation_learners':len(val),
                        'test_learners':len(test),'pfa_solver':'lbfgs','pfa_C':1.0,'pfa_max_iter':2000,'pfa_tol':1e-8,
                        'gru_max_epochs':25,'gru_patience':4,'bootstrap_resamples':2000,'bootstrap_seed':SEED},
            'pfa_convergence_warnings':warnings_list,'pfa_iterations':pfa_n,'gru_epochs_completed':len(history),
            'gru_best_epoch':best,
            'metrics':{'enriched_pfa':kt.metric_bundle(pfa_pred['y'],pfa_pred['p']),
                       'extended_gru':kt.metric_bundle(gru_pred['y'],gru_pred['p'])},
            'paired_brier':delta,
            'interpretation':'Numbers are conditional on corrected source and selected splitting/representation. This is not a new untouched 2009 holdout or causal architecture effect.'}

def main():
    out=Path('results/corrected_2009_sensitivity.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    try:r=run()
    except Exception as e:
        r={'status':'blocked_or_failed','error_type':type(e).__name__,
           'diagnostic':str(e)[:700],'interpretation':'Never substitute expected results or claim success on a blocked source-corrected run.'}
    out.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()