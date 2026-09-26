"""Pure helpers for the prospective specification of a retrospective extension."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
BLOCKS = ('evidence_balance', 'ecology_snapshot', 'calibration_snapshot')
FORBIDDEN = {'sample_id', 'patient_id', 'group_id', 'dataset', 'dataset_name', 'task_name', 'prompt_version', 'duration', 'survival_time', 'event', 'event_status', 'fold', 'true_label', 'outcome', 'label', 'response_support_score', 'risk_score'}

def patient_stratified_folds(outcomes, groups):
    frame = pd.DataFrame({'group': np.asarray(groups, dtype=str), 'outcome': np.asarray(outcomes, dtype=int)})
    if frame.groupby('group').outcome.nunique().max() != 1:
        raise ValueError('Patient outcome conflict')
    units = frame.drop_duplicates('group').reset_index(drop=True)
    if units.outcome.value_counts().min() < 5:
        raise ValueError('Fewer than five patients in a stratum')
    assignment = {}
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(units, units.outcome), 1):
        assignment.update({g: fold for g in units.iloc[test].group})
    return frame.group.map(assignment).to_numpy(int)

def request_payload(request):
    text = request.get('user_prompt', '')
    decoder = json.JSONDecoder()
    for match in re.finditer('\\{', text):
        try:
            value, end = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and (not text[match.start() + end:].strip()):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError('No complete JSON payload in saved user prompt')

def numeric_summary(payload):
    result = {}

    def walk(value, prefix=''):
        if isinstance(value, dict):
            for k, v in sorted(value.items()):
                if k in FORBIDDEN:
                    raise ValueError(f'Forbidden field in feature block: {k}')
                if k in ('program', 'activity_level', 'representative_genes'):
                    continue
                walk(v, f'{prefix}.{k}' if prefix else k)
        elif isinstance(value, list):
            for record in value:
                if isinstance(record, dict) and 'program' in record:
                    walk(record, f"{prefix}[{record['program']}]")
        elif isinstance(value, (float, int, bool)):
            if not np.isfinite(value):
                raise ValueError('Nonfinite summary field')
            result[prefix] = float(value)
            result[f'{prefix}.__present'] = 1.0
    for block in BLOCKS:
        if block in payload:
            walk(payload[block], block)
    if 'dominant_programs' in payload:
        walk(payload['dominant_programs'], 'dominant_programs')
    if not result:
        raise ValueError('No retained numeric summary features')
    return result

def load_saved_summaries(path: Path):
    records, audit = ({}, [])
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sid = str(row['sample_id'])
            result = row['result']
            if sid in records:
                raise ValueError('Duplicate request sample')
            payload = request_payload(result['request'])
            if 'sample_id' in payload and str(payload['sample_id']) != sid:
                raise ValueError('Request sample differs from record')
            records[sid] = {'features': numeric_summary(payload), 'payload': payload, 'parsed': result.get('parsed_response', {})}
            audit.append({'sample_id': sid, 'sample_id_in_payload': 'sample_id' in payload, 'cohort_in_payload': bool({'dataset', 'dataset_name'} & payload.keys()), 'outcome_field_in_payload': bool({'duration', 'survival_time', 'event', 'event_status', 'true_label', 'outcome', 'label', 'fold'} & payload.keys())})
    return (records, pd.DataFrame(audit))

def summary_oof(features, y, meta, task, folds, only_folds=None):
    rows = []
    for fold in sorted(set(folds) if only_folds is None else only_folds):
        tr = np.flatnonzero(folds != fold)
        te = np.flatnonzero(folds == fold)
        if set(meta.iloc[tr].group_id) & set(meta.iloc[te].group_id):
            raise ValueError('Patient leakage')
        vectorizer = DictVectorizer(sparse=False)
        train = vectorizer.fit_transform([features[i] for i in tr])
        test = vectorizer.transform([features[i] for i in te])
        keep = np.ptp(train, axis=0) > 1e-12
        train, test = (train[:, keep], test[:, keep])
        block = meta.iloc[te][['sample_id', 'group_id']].copy()
        block['fold'] = fold
        if train.shape[1]:
            scaler = StandardScaler()
            train = scaler.fit_transform(train)
            test = scaler.transform(test)
        if task == 'task3':
            outcome = y.iloc[:, 0].to_numpy(int)
            block['outcome'] = outcome[te]
            if train.shape[1]:
                model = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
                model.fit(train, outcome[tr])
                score = model.predict_proba(test)[:, 1]
            else:
                score = np.repeat(outcome[tr].mean(), len(te))
            block['score'] = score
            block['predicted'] = (score >= 0.5).astype(int)
        else:
            block['duration'] = y.duration.to_numpy(float)[te]
            block['event'] = y.event.to_numpy(int)[te]
            if train.shape[1]:
                pca = PCA(n_components=min(10, len(tr) - 1, train.shape[1]), svd_solver='full')
                train = pca.fit_transform(train)
                test = pca.transform(test)
                cols = [f'PC{i}' for i in range(train.shape[1])]
                training = pd.DataFrame(train, columns=cols)
                training['duration'] = y.duration.to_numpy(float)[tr]
                training['event'] = y.event.to_numpy(int)[tr]
                model = CoxPHFitter(penalizer=0.1).fit(training, 'duration', 'event')
                score = model.predict_log_partial_hazard(pd.DataFrame(test, columns=cols)).to_numpy(float)
            else:
                score = np.zeros(len(te))
            block['risk'] = score
        rows.append(block)
    return pd.concat(rows, ignore_index=True)

def metric_value(frame, task, metric=None):
    if task == 'task3':
        if frame.outcome.nunique() != 2:
            return np.nan
        return (average_precision_score if metric == 'AUPRC' else roc_auc_score)(frame.outcome, frame.score)
    t = frame.duration.to_numpy(float)
    e = frame.event.to_numpy(int)
    r = frame.risk.to_numpy(float)
    permissible = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    if metric != 'CINDEX_POOLED':
        f = frame.fold.to_numpy(int)
        permissible &= f[:, None] == f[None, :]
    n = permissible.sum()
    return float((((r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])) * permissible).sum() / n) if n else np.nan

def bootstrap(frame, task, metric, n=1000, reference=None):
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(frame), len(frame))
        v = metric_value(frame.iloc[idx], task, metric)
        if reference is not None:
            v -= metric_value(reference.iloc[idx], task, metric)
        if np.isfinite(v):
            values.append(v)
    if len(values) < 0.8 * n:
        raise ValueError('Too few valid bootstrap draws')
    lo, hi = np.quantile(values, [0.025, 0.975])
    return (float(lo), float(hi), len(values))
