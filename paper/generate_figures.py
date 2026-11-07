"""Rebuild manuscript figures from previously archived real-data GitHub Actions JSON.

Run from repository root: python paper/generate_figures.py
Requires matplotlib and numpy. Does not download learner-level data or refit models.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent

def read(rel):
    path=ROOT/rel
    if not path.is_file(): raise FileNotFoundError(f'Required executed output: {path}')
    return json.loads(path.read_text(encoding='utf-8'))

def bars(items,metric,title,filename,ylim=None):
    fig,ax=plt.subplots(figsize=(7,3.35))
    names=[str(n) for n,_ in items]
    values=[float(d[metric]) for _,d in items]
    x=np.arange(len(items))
    ax.bar(x,values,width=.68)
    ax.set_xticks(x,names,rotation=18,ha='right')
    ax.set_ylabel(metric.replace('_',' ').upper() if metric=='roc_auc' else 'Brier score (lower is better)')
    ax.set_title(title,fontsize=10)
    if ylim is not None:ax.set_ylim(*ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT/filename,dpi=180)
    plt.close(fig)

original=read('results/metrics.json')
robust=read('results/robustness_extension.json')
new=read('results/confirmatory_assist2015_stage2.json')
assert original['dataset']['n_learners']==4148
assert robust['status']=='complete_exploratory_real_data'
assert new['status']=='passed_all_computational_gates' and new['directional_replication_confirmed']
assert new['paired_brier']['n_interactions']==26697

choices=[('Global prior','global_rate'),('Skill prior','skill_prior'),('Fixed BKT','bkt_fixed'),('Tuned BKT','bkt_validation_tuned'),('PFA','pfa_logistic'),('GRU','gru_kt_seed_42')]
m=[(name,original['test_metrics'][key]['all']) for name,key in choices]
bars(m,'brier','2009 mirror: original six-model test','kt_brier.png',ylim=(0,.25))
bars(m,'roc_auc','2009 mirror: original six-model test','kt_auc.png',ylim=(0,.85))

fig,axes=plt.subplots(1,2,figsize=(8,3.1))
labels=['PFA','Enriched PFA','GRU 13','GRU 42','GRU 73']
rs=[robust['pfa'],robust['stronger_pfa']]+[robust['extended_gru'][str(s)]['metrics'] for s in [13,42,73]]
for ax,key,desc in zip(axes,['roc_auc','brier'],['ROC-AUC (higher is better)','Brier (lower is better)']):
    vals=[float(t[key]) for t in rs]
    ax.bar(range(len(rs)),vals,width=.68)
    ax.set_xticks(range(len(rs)),labels,rotation=28,ha='right')
    ax.set_ylabel(desc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.suptitle('2009 exploratory same-test information-budget sensitivity',fontsize=10)
fig.tight_layout(pad=1.2)
fig.savefig(OUT/'robust_comparison.png',dpi=180)
plt.close(fig)

ms=new['test_metrics']
bars([('Enriched PFA (converged)',ms['enriched_pfa']),('Extended GRU',ms['extended_gru'])],
     'brier','Fresh ASSISTments 2015: 750 learners; 26,697 held-out responses',
     'cohort2015_brier.png',ylim=(0,.23))
print('Generated four manuscript figures from executed JSON.')

# 2026 matched-history-budget figure, recomputed from executed aggregate evidence.
budget=read("results/matched_history_budget_2026.json")
assert budget["status"]=="passed_all_computational_gates"
assert budget["source"]["cohort_overlap_previous"]==0
fig,ax=plt.subplots(figsize=(7.25,3.35))
for name,label in [("window_count","Window-count PFA+"),("ordered_logistic","Order-aware logistic"),("window_gru","Window-GRU")]:
    y=[budget["comparisons"][str(k)]["models"][name]["brier"] for k in (1,20)]
    ax.plot([1,20],y,marker="o",linewidth=1.8,markersize=6,label=label)
    for k,v in zip((1,20),y):
        ax.annotate(f"{v:.4f}",(k,v),xytext=(3,7 if name!="window_count" else -14),textcoords="offset points",fontsize=7)
ax.set_xticks([1,20],["1 prior event","20 prior events"])
ax.set_ylabel("Interaction-pooled test Brier score (lower better)")
ax.set_xlabel("Information budget per next-response prediction")
ax.grid(axis="y",alpha=.27)
ax.legend(loc="upper right",fontsize=8,frameon=False)
fig.tight_layout()
fig.savefig(OUT/"history_budget_brier.png",dpi=190)
plt.close(fig)
print("Generated new 2026 matched-budget figure.")
