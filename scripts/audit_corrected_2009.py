"""Public-data provenance audit; never stores or prints student-level records.

Compares the pinned Atomi serialized sequences to the original-data maintainers'
corrected 2009 Skill Builder release. The corrected release retains student/problem
identifiers that the third-party transformed mirror omits, so this can reject
exact parity but cannot in general prove event-by-event identity or chronology.

Run: python scripts/audit_corrected_2009.py
Data downloads stay in the untracked data/cache folder.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

CORRECTED_FILE_ID = '1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE'
SOURCE_PAGE = 'https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010'
MIRROR_REPO = 'Atomi/ASSISTments2009'
MIRROR_REV = 'c72a664a9693547fb206652ed2ce18e62d320c7d'
MIRROR_FILE = 'data/train-00000-of-00001.parquet'
MIRROR_SHA = '2053b7ea63f1d5ea953834bf82ed89543e1f11bae1e3e7535c2f9d7f49d4dda7'


def identifier(x):
    """Normalize simple integer/decimal IDs, but not compound skill strings."""
    s = str(x).strip()
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        return s[:-2]
    return s


def skill_token(x):
    s = str(x).strip()
    if s.lower() in {'nan', 'none', '', 'null'}:
        return ''
    # Corrected data use skill1_skill2; some public CSVs use semicolon lists.
    parts = re.split(r'[_;]', s)
    return '_'.join(identifier(z) for z in parts if z.strip())


def load_csv_or_archive(path):
    import io
    import zipfile
    import pandas as pd
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members = [x for x in z.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX/')]
            if not members:
                raise ValueError('Official corrected archive lacks a CSV')
            members.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
            with z.open(members[0]) as f:
                return pd.read_csv(f, low_memory=False, encoding='latin-1')
    if path.read_bytes()[:32].lower().lstrip().startswith((b'<html', b'<!doctype')):
        raise ValueError('The corrected link returned HTML rather than a dataset')
    try:
        return pd.read_csv(path, low_memory=False, encoding='latin-1')
    except Exception as e:
        raise RuntimeError('Downloaded corrected source is not parseable CSV/ZIP') from e


def get_source():
    import gdown
    dest = Path('data/cache/assistments_2009_corrected_official')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        got = gdown.download(id=CORRECTED_FILE_ID, output=str(dest), quiet=False)
        if got is None or not dest.exists():
            raise RuntimeError('Corrected official 2009 release not accessible by the published Drive ID')
    return dest


def get_mirror():
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=MIRROR_REPO, repo_type='dataset', filename=MIRROR_FILE,
                                revision=MIRROR_REV, cache_dir='data/cache/hf2009'))


def collect(source_path, mirror_path):
    import pandas as pd
    official_bytes = source_path.read_bytes()
    mirror_digest = hashlib.sha256(mirror_path.read_bytes()).hexdigest()
    if mirror_digest != MIRROR_SHA:
        raise ValueError('Pinned mirror SHA mismatch: refusing dataset substitution')
    raw = load_csv_or_archive(source_path)
    raw.columns = [re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_') for c in raw.columns]
    if len(set(raw.columns)) != len(raw.columns):
        raise ValueError('Duplicate normalized source column names')
    aliases = {'user':'user_id', 'student_id':'user_id', 'order_id':'order_id',
               'list_skill_ids':'skill_id', 'correct':'correct'}
    if 'skill_id' not in raw and 'list_skill_ids' in raw:
        raw['skill_id'] = raw['list_skill_ids']
    cols = set(raw.columns)
    missing = sorted({'user_id', 'correct'} - cols)
    if missing:
        raise ValueError('Corrected source lacks expected columns: '+','.join(missing))
    if 'skill_id' not in cols:
        raise ValueError('Corrected source lacks skill_id/list_skill_ids')
    before = len(raw)
    raw = raw[['user_id', 'skill_id', 'correct'] + (['order_id'] if 'order_id' in raw else [])].copy()
    raw = raw.dropna(subset=['user_id', 'correct'])
    raw['correct'] = pd.to_numeric(raw['correct'], errors='coerce')
    raw = raw[raw['correct'].isin([0, 1])].copy()
    raw['correct'] = raw['correct'].astype('int8')
    raw['user_id'] = raw['user_id'].map(identifier)
    raw['skill_id'] = raw['skill_id'].map(skill_token)
    # Count all even if skill missing; mirror evidently uses only labeled steps.
    raw_valid = raw[raw['skill_id'] != ''].copy()
    mirror = pd.read_parquet(mirror_path)
    required_mirror = {'user_id', 'skill_ids', 'grades'}
    if not required_mirror.issubset(mirror):
        raise ValueError('Mirror missing expected fields')
    mirror_ids = [identifier(x) for x in mirror['user_id']]
    if len(mirror_ids) != len(set(mirror_ids)):
        raise ValueError('Mirror learner rows not unique')
    mirror_skills = {}
    mirror_total = 0
    for i, rec in enumerate(mirror.itertuples(index=False)):
        # itertuples preserves original dataframe column order; direct field access.
        uid = identifier(getattr(rec, 'user_id'))
        skills = [skill_token(x) for x in getattr(rec,'skill_ids')]
        grades = [int(x) for x in getattr(rec,'grades')]
        if len(skills) != len(grades):
            raise ValueError('Mirror skill/response arrays misaligned')
        if not set(grades).issubset({0, 1}):
            raise ValueError('Mirror responses outside binary support')
        mirror_total += len(grades)
        mirror_skills[uid] = list(zip(skills, grades))

    # The source is corrected to student-problem level. Preserve all those rows;
    # repeated (skill, outcome) pairs in different exercises are not deduplicated.
    official = {}
    for uid, group in raw_valid.groupby('user_id', sort=False):
        official[uid] = list(zip(group['skill_id'].tolist(), group['correct'].tolist()))
    both = sorted(set(official) & set(mirror_skills))
    count_equal = 0
    multiset_equal = 0
    exact_order_equal = 0
    n_mirror_interactions_overlap = 0
    n_source_interactions_overlap = 0
    for uid in both:
        a = mirror_skills[uid]
        b = official[uid]
        n_mirror_interactions_overlap += len(a)
        n_source_interactions_overlap += len(b)
        count_equal += len(a) == len(b)
        multiset_equal += Counter(a) == Counter(b)
        exact_order_equal += a == b  # Native CSV row order, not a validated chronology.
    dup_event_pairs = None
    if 'order_id' in raw:
        # An event identifier duplicated across distinct source rows requires scrutiny.
        dup_event_pairs = int(raw.duplicated(['user_id','order_id'],keep=False).sum())
    by_order_equal = None
    if 'order_id' in raw and both:
        s = raw_valid.copy()
        s['order_num'] = pd.to_numeric(s['order_id'],errors='coerce')
        if s['order_num'].notna().all():
            sorted_source = {uid:list(zip(g['skill_id'],g['correct'])) for uid,g in
                             s.sort_values(['user_id','order_num'],kind='mergesort').groupby('user_id',sort=False)}
            by_order_equal = sum(mirror_skills[uid]==sorted_source[uid] for uid in both)
    findings = {
        'source': {'official_documentation_url':SOURCE_PAGE,'official_corrected_drive_id':CORRECTED_FILE_ID,
                   'corrected_file_sha256':hashlib.sha256(official_bytes).hexdigest(),
                   'corrected_file_bytes':len(official_bytes),'source_columns':list(raw.columns),
                   'corrected_source_records_total':before,'records_valid_binary':len(raw),
                   'records_valid_binary_and_labeled_skill':len(raw_valid),
                   'source_unique_learners_with_skills':len(official),
                   'source_rows_repeating_student_order_key':dup_event_pairs},
        'mirror':{'repository':MIRROR_REPO,'revision':MIRROR_REV,'sha256':mirror_digest,
                  'serialized_learners':len(mirror_skills),'total_interactions':mirror_total},
        'comparison':{'shared_learner_ids':len(both),
                      'official_only_learner_ids':len(set(official)-set(mirror_skills)),
                      'mirror_only_learner_ids':len(set(mirror_skills)-set(official)),
                      'shared_learner_ids_same_number_of_interactions':count_equal,
                      'shared_learner_ids_identical_skill_response_multisets':multiset_equal,
                      'shared_learner_ids_identical_native_CSV_order':exact_order_equal,
                      'shared_learner_ids_identical_numeric_order_id_order':by_order_equal,
                      'mirror_interactions_shared_learners':n_mirror_interactions_overlap,
                      'corrected_interactions_shared_learners':n_source_interactions_overlap},
        'parity_status':'not_proven',
        'interpretation':'Necessary count and (skill, correctness) multiset conditions only; no mirror original event IDs, no independent chronology guarantee. Equal aggregates cannot prove row identity; unequal totals/multisets disprove exact full parity.',
        'privacy':'No learner IDs, rows, or raw source bytes are written to output.'
    }
    if (len(official)==len(mirror_skills) and len(both)==len(official) and
        n_mirror_interactions_overlap==n_source_interactions_overlap and
        multiset_equal==len(both)):
        findings['parity_status']='aggregate_necessary_conditions_pass_but_event_parity_not_proven'
    else:
        findings['parity_status']='exact_full_dataset_parity_disproved_by_necessary_conditions'
    return findings


def main():
    dst = Path('results/official_2009_mirror_parity_audit.json')
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:
        source = get_source()
        mirror = get_mirror()
        result = collect(source,mirror)
    except Exception as e:
        result={'parity_status':'blocked_not_established', 'error_type':type(e).__name__,
                'diagnostic':str(e)[:700],
                'official_linked_file_id':CORRECTED_FILE_ID,
                'mirror_repository':MIRROR_REPO,
                'recommendation':'Do not claim official-source parity. Retain mirror-specific claims until source is accessible and results are inspected.'}
    dst.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    # Return success to preserve diagnostic artifacts if download blocked.

if __name__=='__main__':
    main()