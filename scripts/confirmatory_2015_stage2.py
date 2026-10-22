"""Untouched 2015 learner cohort, disjoint from the nonconverged stage-1 pilot.

Protocol frozen in CONFIRMATORY_STAGE2_PROTOCOL_2015.md before any stage-2 outcomes.
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
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robustness_extension import paired_cluster_bootstrap, pfa_plus_all_skill_history_frame
from confirmatory_2015 import download_official_csv, GOOGLE_FILE_ID, SOURCE_URL
import run_experiment as kt

FIRST_SEED=20261022
SECOND_SEED=20261023
SAMPLE_SIZE=5000
SOURCE_SHA256='75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0'


def cohort_from_official_csv(path:Path):
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum!=SOURCE_SHA256:
        raise ValueError('Official ASSISTments-2015 source checksum changed since stage 1; stop for provenance audit')
    df=pd.read_csv(path,low_memory=False)
    df.columns=[str(c).strip().lower() for c in df.columns]
    columns=['user_id','log_id','sequence_id','correct']
    if len(set(df.columns))!=len(df.columns) or any(c not in df for c in columns):
        raise ValueError('Missing/nonunique required source column')
    orig_n=len(df)
    df=df[columns].copy()
    missing=df.isna().any(axis=1)
    df['log_id']=pd.to_numeric(df['log_id'],errors='coerce')
    df['correct']=pd.to_numeric(df['correct'],errors='coerce')
    invalid=missing|df.isna().any(axis=1)|~df['correct'].isin([0,1])
    exclusions=int(invalid.sum())
    if orig_n==0 or exclusions/orig_n>0.05:
        raise ValueError('Invalid record rate exceeds frozen threshold')
    df=df.loc[~invalid].copy()
    if df.duplicated(['user_id','log_id']).any():
        raise ValueError('Duplicate learner-event identifiers')
    if not np.isfinite(df['log_id']).all() or (df['log_id']!=df['log_id'].astype('int64')).any():
        raise ValueError('Malformed chronological log IDs')
    df['correct']=df['correct'].astype(int)
    df['user_id']=df['user_id'].astype(str)
    df['sequence_id']=df['sequence_id'].astype(str)
    df=df.sort_values(['user_id','log_id'],kind='stable')
    users=sorted(df['user_id'].unique())
    stage1=set(random.Random(FIRST_SEED).sample(users,SAMPLE_SIZE))
    remaining=[u for u in users if u not in stage1]
    if len(remaining)<SAMPLE_SIZE:raise ValueError('Insufficient distinct untouched learners')
    stage2=set(random.Random(SECOND_SEED).sample(remaining,SAMPLE_SIZE))
    assert not stage1.intersection(stage2)
    chosen=df[df['user_id'].isin(stage2)]
    rows=[{'user_id':str(uid),'skill_ids':group['sequence_id'].tolist(),
           'grades':group['correct'].tolist()}
          for uid,group in chosen.groupby('user_id',sort=True)]
    if kt.validate_unique_learner_rows(rows)!=SAMPLE_SIZE:raise ValueError('Learner count mismatch')
    return rows,{'source_url':SOURCE_URL,'drive_id':GOOGLE_FILE_ID,'sha256':checksum,
                 'total_source_rows':orig_n,'excluded_invalid_rows':exclusions,
                 'eligible_learners':len(users),'stage1_excluded_learners':len(stage1),
                 'stage2_sampling_pool':len(remaining),'stage2_sampled_learners':len(stage2),
                 'stage2_sample_interactions':sum(len(r['grades']) for r in rows),
                 'stage2_intersection_stage1':0,
                 'sorting':'log_id ascending per learner, sequence_id as skill-builder concept'}


def converged_enriched_pfa(train,test):
    X,y,*_=pfa_plus_all_skill_history_frame(train)
    XX,yy,users,skills,first=pfa_plus_all_skill_history_frame(test)
    clf=Pipeline([
        ('prep',ColumnTransformer([
            ('skill',OneHotEncoder(handle_unknown='ignore'),['skill']),
            ('history',StandardScaler(),['skill_success','skill_failure','all_success','all_failure'])])),
        ('clf',LogisticRegression(solver='lbfgs',max_iter=2000,tol=1e-8,C=1.0))])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(X,y)
    warns=[str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning)]
    pred=kt.prediction_frame(yy,clf.predict_proba(XX)[:,1],users,skills,first)
    return pred,warns,int(clf.named_steps['clf'].n_iter_[0])


def run(path:Path):
    rows,source=cohort_from_official_csv(path)
    train,val,test=kt.split_rows(rows,seed=SECOND_SEED)
    sets=[set(str(r['user_id']) for r in split) for split in (train,val,test)]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise ValueError('Student overlap across partitions')
    reference,warnings_list,pfa_n_iter=converged_enriched_pfa(train,test)
    global_prior,skill_prior,_=kt.train_priors(train)
    vocab=kt.skill_vocabulary(train)
    kt.GRU_PATIENCE=4
    model,history,best_epoch=kt.train_gru(train,val,vocab,seed=42,epochs=25)
    candidate=kt.predict_gru(model,test,vocab,global_prior,skill_prior)
    test_stat=paired_cluster_bootstrap(reference,candidate,repetitions=2000,seed=SECOND_SEED)
    metrics={'enriched_pfa':kt.metric_bundle(reference['y'],reference['p']),
             'extended_gru':kt.metric_bundle(candidate['y'],candidate['p'])}
    guards=(not warnings_list and best_epoch<25
            and len(test)==750
            and test_stat['n_learners']==750
            and np.isfinite(test_stat['difference_candidate_minus_reference']))
    supports=test_stat['bootstrap_95_interval'][1]<0
    return {'status':'passed_all_computational_gates' if guards else 'inconclusive_diagnostic_fail',
            'directional_replication_confirmed':bool(guards and supports),
            'test_supports_direction':bool(supports),
            'stage1_result_used_for_model_selection':False,
            'data':source,
            'protocol':{'stage1_sample_seed':FIRST_SEED,'stage2_sample_seed':SECOND_SEED,
                        'sampled_learners':SAMPLE_SIZE,'train_learners':len(train),
                        'validation_learners':len(val),'test_learners':len(test),
                        'gru_seed':42,'gru_max_epochs':25,'gru_patience':4,
                        'pfa_solver':'lbfgs','pfa_C':1.0,'pfa_max_iter':2000,'pfa_tol':1e-8,
                        'bootstrap_draws':2000,'bootstrap_seed':SECOND_SEED,
                        'primary_estimand':'GRU minus PFA pooled interaction-weighted Brier',
                        'success':'upper endpoint of 95% learner-cluster percentile interval <0; diagnostic gates pass'},
            'test_metrics':metrics,'paired_brier':test_stat,
            'pfa_iterations':pfa_n_iter,'pfa_convergence_warnings':warnings_list,
            'gru_best_validation_epoch':best_epoch,'gru_epochs_completed':len(history),
            'limitation':'Replication of the directional performance difference on a newly sampled 2015 population; not causal architecture identification or universal transportability.'}


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/cache/assistments_2015_official.csv')
    p.add_argument('--output',default='results/confirmatory_assist2015_stage2.json')
    args=p.parse_args()
    source=Path(args.csv)
    download_official_csv(source)
    result=run(source)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],
                      'confirmed':result['directional_replication_confirmed'],
                      'n_test':result['test_metrics']['enriched_pfa']['n'],
                      'metrics':result['test_metrics'],
                      'brier_difference':result['paired_brier']},indent=2))

if __name__=='__main__':main()