"""Independent source-identity test of corrected ASSISTments 2009 versus a pinned 2009 mirror.
Reimplements source comparison without importing any original research module.
Only aggregate counts and provenance are exported; no learner-level records.
"""
import hashlib,json,re,zipfile,io
from collections import Counter
from pathlib import Path
import gdown,pandas as pd
from huggingface_hub import hf_hub_download

OFFICIAL_ID="1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
OFFICIAL_SHA="162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
MIRROR_SHA="2053b7ea63f1d5ea953834bf82ed89543e1f11bae1e3e7535c2f9d7f49d4dda7"
REVISION="c72a664a9693547fb206652ed2ce18e62d320c7d"

def digest(fp):
    h=hashlib.sha256()
    with open(fp,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def normalize_id(value):
    s=str(value).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].lstrip("-").isdigit() else s

def normalize_skill(value):
    string=str(value).strip()
    if string.lower() in ("none","nan","null",""):return ""
    return "_".join(normalize_id(p) for p in re.split(r"[_;]",string) if p.strip())

def official_rows(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members=[x for x in z.namelist() if x.lower().endswith(".csv") and not x.startswith("__MACOSX/")]
            if not members:raise ValueError("Corrected official source lacks CSV")
            members.sort(key=lambda f:z.getinfo(f).file_size,reverse=True)
            with z.open(members[0]) as f:
                tab=pd.read_csv(f,encoding="latin-1",low_memory=False)
    else:tab=pd.read_csv(path,encoding="latin-1",low_memory=False)
    tab.columns=[re.sub(r"[^a-z0-9]+","_",str(x).strip().lower()).strip("_") for x in tab.columns]
    if "skill_id" not in tab and "list_skill_ids" in tab:tab["skill_id"]=tab["list_skill_ids"]
    for c in ("user_id","skill_id","correct"):
        if c not in tab:raise ValueError("Missing official column "+c)
    raw_count=len(tab)
    cols=["user_id","skill_id","correct"]+(["order_id"] if "order_id" in tab else [])
    tab=tab[cols].dropna(subset=["user_id","correct"]).copy()
    tab["correct"]=pd.to_numeric(tab["correct"],errors="coerce")
    tab=tab.loc[tab["correct"].isin([0,1])].copy()
    tab["user_id"]=tab["user_id"].map(normalize_id)
    tab["skill_id"]=tab["skill_id"].map(normalize_skill)
    tab["correct"]=tab["correct"].astype(int)
    valid=len(tab)
    labeled=tab[tab["skill_id"]!=""].copy()
    original={str(uid):list(zip(g["skill_id"],g["correct"])) for uid,g in labeled.groupby("user_id",sort=False)}
    if "order_id" in labeled:
        temp=labeled.copy()
        temp["numeric_order"]=pd.to_numeric(temp["order_id"],errors="coerce")
        if temp["numeric_order"].isna().any():raise ValueError("Non-numeric order_id")
        temp=temp.sort_values(["user_id","numeric_order"],kind="stable")
        chronological={uid:list(zip(g["skill_id"],g["correct"])) for uid,g in temp.groupby("user_id",sort=False)}
    else:raise ValueError("Official chronological order column missing")
    return original,chronological,{"original_rows":raw_count,"binary_valid":valid,
        "labeled_rows":len(labeled),"n_unique_learners":len(original)}

def mirror_rows(path):
    table=pd.read_parquet(path)
    for c in ("user_id","skill_ids","grades"):
        if c not in table:raise ValueError("Pinned mirror column missing: "+c)
    out={}
    events=0
    for rec in table.itertuples(index=False):
        uid=normalize_id(getattr(rec,"user_id"))
        if uid in out:raise ValueError("Duplicate mirrored learner")
        pairs=list(zip([normalize_skill(x) for x in rec.skill_ids],[int(x) for x in rec.grades]))
        if len(pairs)!=len(rec.skill_ids) or len(pairs)!=len(rec.grades):raise ValueError("Misaligned sequences")
        if any(y not in (0,1) for _,y in pairs):raise ValueError("Invalid mirror label")
        events+=len(pairs)
        out[uid]=pairs
    return out,events

def main():
    folder=Path("data/cache/independent_provenance_2009");folder.mkdir(parents=True,exist_ok=True)
    official_file=folder/"corrected_original"
    if not official_file.exists():
        if not gdown.download(id=OFFICIAL_ID,output=str(official_file),quiet=False):
            raise RuntimeError("Official 2009 corrected source not accessible")
    mirror_file=Path(hf_hub_download(repo_id="Atomi/ASSISTments2009",repo_type="dataset",
         filename="data/train-00000-of-00001.parquet",revision=REVISION,cache_dir=str(folder/"mirror_cache")))
    oh,mh=digest(official_file),digest(mirror_file)
    if oh!=OFFICIAL_SHA or mh!=MIRROR_SHA:raise RuntimeError("Source bytes changed: "+str((oh,mh)))
    first,ordered,summary=official_rows(official_file)
    second,n_events=mirror_rows(mirror_file)
    intersection=set(first)&set(second)
    equivalent_sizes=sum(len(first[u])==len(second[u]) for u in intersection)
    equivalent_multisets=sum(Counter(first[u])==Counter(second[u]) for u in intersection)
    equivalent_native=sum(first[u]==second[u] for u in intersection)
    equivalent_numeric=sum(ordered[u]==second[u] for u in intersection)
    result={"provenance":{"corrected_original_sha256":oh,"pinned_mirror_sha256":mh,
                           "mirror_revision":REVISION,"official_file_id":OFFICIAL_ID},
            "official":summary,
            "mirror":{"serialized_learners":len(second),"interactions":n_events},
            "comparison":{"shared_learners":len(intersection),
                          "official_only":len(first.keys()-second.keys()),
                          "mirror_only":len(second.keys()-first.keys()),
                          "same_event_counts":equivalent_sizes,
                          "same_skill_response_multisets":equivalent_multisets,
                          "same_native_file_order":equivalent_native,
                          "same_numeric_order":equivalent_numeric,
                          "different_shared_sequences":len(intersection)-equivalent_numeric},
            "scientific_scope":"Fails exact parity; source has event IDs that transformed mirror does not. No proof of complete learner-event identity."}
    expected=(346860,346860,283105,4163,4148,274331,4148,3331,3331,1441,3331,817)
    observed=(summary["original_rows"],summary["binary_valid"],summary["labeled_rows"],summary["n_unique_learners"],
              len(second),n_events,len(intersection),equivalent_sizes,equivalent_multisets,equivalent_native,equivalent_numeric,
              result["comparison"]["different_shared_sequences"])
    result["matches_original_aggregate"]=(observed==expected)
    result["expected"]=expected
    result["observed"]=observed
    path=Path("results/independent_paper1_2009_provenance_20261028.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("INDEPENDENT_2009_PROVENANCE",json.dumps(result),flush=True)
    if observed!=expected:raise SystemExit(3)

if __name__=="__main__":main()
