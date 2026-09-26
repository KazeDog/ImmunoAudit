"""Versioned temporal refits and fixed-input summary controls; no network code."""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PROJECT = _submission_path('project', '')
sys.path.insert(0, str(PROJECT / 'analysis/grouped_tasks/scripts'))
import run_task135_audit as old
from task135_core import make_grouped_folds, survival_oof_predictions, aggregate_classification_by_group, aggregate_survival_by_group
from audit_core import load_saved_summaries, summary_oof, metric_value, bootstrap, patient_stratified_folds

def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')

def write_csv(path, frame):
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)

def summary_path(project, task, dataset):
    prediction = old.qwen_prediction_path(project, task, dataset)
    path = project / 'results/strategy3/qwen' / prediction.parent.name / prediction.name.replace('_predictions.csv', '_raw.jsonl')
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def metadata(args, task, ds):
    x, y = old.load_xy(args.project_root / 'processed_data_strategy1', f'{task}_{ds}')
    clinical = old.align_group_metadata(x.index, old.raw_clinical_mapping(args.raw_root, ds), ds)
    clinical = clinical.reset_index().rename(columns={'index': 'sample_id'})
    clinical['sample_id'] = x.index.to_numpy()
    clinical['task'] = task
    clinical['dataset'] = ds
    timing = clinical.treatment_time.str.upper().str.strip()
    clinical['timing_class'] = np.where(timing.eq('PRE'), 'pretreatment', np.where(timing.isin(['EDT', 'ON']), 'on_treatment', 'unverified'))
    if ds == 'IMvigor210':
        clinical['timing_class'] = 'baseline_assigned_by_previous_loader'
    if ds == 'Liu_phs000452':
        raw = pd.read_csv(args.raw_root / '3-' / old.TASK3_CLINICAL_FILES[ds], sep='\t').set_index('sample_id')
        names = raw.loc[x.index, 'patient_name'].astype(str).str.strip()
        source = np.where(names.str.startswith('IPI_'), 'VanAllen', np.where(names.str.startswith('Patient'), 'Liu', 'unresolved'))
        if 'unresolved' in source:
            raise ValueError('Unresolved combined-cohort provenance')
        clinical['study_source'] = source
    else:
        clinical['study_source'] = ds
    return (x, y, clinical)

def prepare(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    sources = {PROJECT / 'analysis/temporal_summary/analysis_contract.md', Path(__file__), Path(__file__).with_name('audit_core.py'), PROJECT / 'analysis/grouped_tasks/scripts/run_task135_audit.py', PROJECT / 'analysis/grouped_tasks/scripts/task135_core.py'}
    oldhash = pd.read_csv(args.project_root / 'analysis/grouped_tasks/full_v1/metadata/source_file_hashes.csv')
    for r in oldhash.itertuples():
        if old.sha256_file(Path(r.path)) != r.sha256:
            raise ValueError(f'Historical input changed: {r.path}')
    inventories = []
    for task, datasets in [('task3', old.TASK3_DATASETS), ('task5', old.TASK5_DATASETS)]:
        for ds in datasets:
            x, y, meta = metadata(args, task, ds)
            inventories.append(meta)
            sources.update((args.project_root / 'processed_data_strategy1' / f'{task}_{ds}_{v}_final.csv' for v in ['X', 'y']))
            sources.add(old.qwen_prediction_path(args.project_root, task, ds))
            sources.add(summary_path(args.project_root, task, ds))
            if ds == 'IMvigor210':
                sources.add(args.raw_root / '3-/IMvigor210/pData_IMvigor210.csv')
            else:
                sources.add(args.raw_root / '3-' / old.TASK3_CLINICAL_FILES[ds])
            for emb in ['geneformer', 'scfoundation', 'scgpt'] if task == 'task3' else ['scfoundation']:
                for suffix in ['X_emb.npy', 'samples.csv']:
                    sources.add(args.project_root / 'processed_data_strategy2' / f'{task}_{ds}_{emb}_{suffix}')
    olddir = args.project_root / 'analysis/grouped_tasks/full_v1'
    sources.update((olddir / 'predictions').glob('*.csv'))
    sources.update((olddir / 'fold_assignments').glob('*.csv'))
    hashes = [{'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': old.sha256_file(p)} for p in sorted(sources)]
    target = root / 'source_hashes.json'
    if target.exists():
        if json.loads(target.read_text()) != hashes:
            raise ValueError('Resume source/contract hash mismatch')
    else:
        write_json(target, hashes)
        inv = pd.concat(inventories, ignore_index=True)
        write_csv(root / 'sample_timing_inventory.csv', inv)
        counts = inv.groupby(['task', 'dataset', 'timing_class'], sort=True).agg(n_samples=('sample_id', 'size'), n_patients=('group_id', 'nunique')).reset_index()
        write_csv(root / 'timing_counts.csv', counts)
        import sklearn, scipy, xgboost, lifelines
        write_json(root / 'run_manifest.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'bootstrap': args.bootstrap, 'random_state': 42, 'model_jobs': args.model_jobs, 'blas_threads': args.blas_threads, 'python': platform.python_version(), 'versions': {m.__name__: m.__version__ for m in [np, pd, sklearn, scipy, xgboost, lifelines]}, 'command': sys.argv, 'external_inference': False, 'embedding_regeneration': False, 'old_hashes_verified': len(oldhash)})

def canonical(args, task, ds):
    name = 'classification' if task == 'task3' else 'survival'
    data = pd.read_csv(args.project_root / f'analysis/grouped_tasks/full_v1/predictions/{name}_predictions.csv')
    frame = data[(data.task == task) & (data.dataset == ds)].copy()
    frame.sample_id = frame.sample_id.astype(str)
    frame.group_id = frame.group_id.astype(str)
    return frame

def evaluate(root, pred, task, nboot, smoke=False):
    aggregate = aggregate_classification_by_group if task == 'task3' else aggregate_survival_by_group
    primary = pred[pred.model.isin(['nested_selected', 'Qwen', 'Summary-LR', 'Summary-Cox', 'Biology-score', 'CoxPH(PCA)', 'scFoundation+CoxPH(PCA)', 'Source-only'])]
    evaluated = {(s, m): aggregate(b).sort_values('group_id').reset_index(drop=True) for (s, m), b in primary.groupby(['strategy', 'model'], sort=True)}
    qwen = evaluated['Fixed LLM', 'Qwen']
    metrics = []
    paired = []
    for (strategy, model), frame in evaluated.items():
        if not frame.group_id.equals(qwen.group_id):
            raise ValueError('Nonmatched comparison')
        for column in ['outcome'] if task == 'task3' else ['duration', 'event']:
            if not np.allclose(frame[column], qwen[column]):
                raise ValueError('Mismatched outcomes')
        names = ['AUROC', 'AUPRC'] if task == 'task3' else ['CINDEX_FOLD_RESTRICTED']
        if task == 'task5' and strategy == 'Fixed LLM':
            names.append('CINDEX_POOLED')
        for name in names:
            lo, hi, valid = bootstrap(frame, task, name, nboot)
            metrics.append({'strategy': strategy, 'model': model, 'metric': name, 'estimate': metric_value(frame, task, name), 'ci_lower': lo, 'ci_upper': hi, 'n_patients': len(frame), 'positive_or_events': int(frame['outcome' if task == 'task3' else 'event'].sum()), 'valid_bootstrap': valid})
            if model != 'Qwen' and name != 'AUPRC':
                lo, hi, valid = bootstrap(frame, task, name, nboot, reference=qwen)
                paired.append({'strategy': strategy, 'model': model, 'reference': 'Qwen', 'metric': name, 'difference': metric_value(frame, task, name) - metric_value(qwen, task, name), 'ci_lower': lo, 'ci_upper': hi, 'valid_bootstrap': valid, 'n_patients': len(frame)})
    write_csv(root / 'metrics.csv', pd.DataFrame(metrics))
    write_csv(root / 'paired_vs_qwen.csv', pd.DataFrame(paired))

def run_unit(args, task, ds, scheme):
    root = args.output_dir / f'{task}__{ds}__{scheme}'
    if (root / 'COMPLETE.json').exists():
        print(f'[resume verified sources] {root.name}', flush=True)
        return
    if root.exists():
        raise FileExistsError(f'Incomplete unit; do not overwrite: {root}')
    root.mkdir()
    start = time.monotonic()
    print(f'[start] {root.name}', flush=True)
    x, y, meta = metadata(args, task, ds)
    records, request_audit = load_saved_summaries(summary_path(args.project_root, task, ds))
    hist = canonical(args, task, ds)
    hq = hist[hist.model == 'Qwen'].set_index('sample_id').loc[x.index]
    key = 'response_support_score' if task == 'task3' else 'risk_score'
    saved = np.array([float(records[s]['parsed'][key]) for s in x.index])
    if not np.allclose(saved, hq['score' if task == 'task3' else 'risk'], atol=1e-12, rtol=0):
        raise ValueError('Raw request/output does not match canonical prediction')
    if scheme == 'pretreatment':
        keep = meta.timing_class.eq('pretreatment').to_numpy()
        x = x.iloc[keep]
        y = y.iloc[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        folds = patient_stratified_folds(y.iloc[:, 0] if task == 'task3' else y.event, meta.group_id)
    else:
        folds = hq.fold.to_numpy(int)
    meta['fold'] = folds
    if meta.groupby('group_id').fold.nunique().max() != 1:
        raise ValueError('Patient fold mismatch')
    write_csv(root / 'folds_and_metadata.csv', meta)
    write_csv(root / 'request_field_audit.csv', request_audit[request_audit.sample_id.isin(x.index)])
    features = [records[s]['features'] for s in x.index]
    write_json(root / 'retained_numeric_features.json', dict(zip(x.index, features)))
    only = [1] if args.mode == 'smoke' else None
    predictions = []
    if scheme == 'pretreatment':
        if task == 'task3':
            pools = old.task3_candidate_pools(args.project_root, x, ds, args)
            for pool in pools:
                if only is not None:
                    key = 'LR' if pool['strategy'] == 'Strategy 1' else 'scfoundation+LR'
                    pool['features'] = {key: pool['features'][key]}
                    pool['factories'] = {key: pool['factories'][key]}
                print(f"[refit] {ds} {pool['strategy']} ({len(pool['factories'])} candidates)", flush=True)
                pred, _, selection = old.evaluate_candidate_pool(task=task, dataset=ds, **pool, y=y.iloc[:, 0].to_numpy(int), sample_ids=x.index, groups=meta.group_id.to_numpy(), folds=folds, patient_level_metric=True, only_folds=only)
                write_csv(root / f"{pool['strategy'].replace(' ', '')}_predictions.csv", pred)
                if not selection.empty:
                    write_csv(root / f"{pool['strategy'].replace(' ', '')}_selection.csv", selection)
                predictions.append(pred)
        else:
            stem = f'task5_{ds}_scfoundation'
            embroot = args.project_root / 'processed_data_strategy2'
            emb = old.load_embedding(embroot / f'{stem}_X_emb.npy', x.index, embroot / f'{stem}_samples.csv')
            for strategy, model, values in [('Strategy 1', 'CoxPH(PCA)', x.to_numpy(np.float32)), ('Strategy 2', 'scFoundation+CoxPH(PCA)', emb)]:
                p = survival_oof_predictions(values, y.duration, y.event, x.index, meta.group_id, folds, only_folds=only)
                p['strategy'] = strategy
                p['model'] = model
                predictions.append(p)
    else:
        retained = hist[hist.model.isin(['nested_selected', 'CoxPH(PCA)', 'scFoundation+CoxPH(PCA)'])].copy()
        if only is not None:
            retained = retained[retained.fold.isin(only)]
        predictions.append(retained)
    q = hq.loc[x.index].rename_axis('sample_id').reset_index()
    q['fold'] = folds
    if only is not None:
        q = q[q.fold.isin(only)]
    predictions.append(q)
    summary = summary_oof(features, y, meta, task, folds, only)
    summary['strategy'] = 'Summary control'
    summary['model'] = 'Summary-LR' if task == 'task3' else 'Summary-Cox'
    predictions.append(summary)
    biology = q.copy()
    biology['strategy'] = 'Summary control'
    biology['model'] = 'Biology-score'
    if task == 'task3':
        biology['score'] = [records[s]['payload']['evidence_balance']['net_response_minus_resistance'] for s in biology.sample_id]
        biology['predicted'] = (biology.score >= 0).astype(int)
    else:
        biology['risk'] = [records[s]['payload']['calibration_snapshot']['calibrated_net_risk'] for s in biology.sample_id]
    predictions.append(biology)
    if ds == 'Liu_phs000452':
        sourcefeatures = [{'Liu_source': float(v == 'Liu')} for v in meta.study_source]
        source = summary_oof(sourcefeatures, y, meta, task, folds, only)
        score_column = 'score' if task == 'task3' else 'risk'
        source[score_column] = source[score_column].round(12)
        source['strategy'] = 'Source context'
        source['model'] = 'Source-only'
        predictions.append(source)
    pred = pd.concat(predictions, ignore_index=True)
    pred['task'] = task
    pred['dataset'] = ds
    pred['scheme'] = scheme
    write_csv(root / 'predictions.csv', pred)
    evaluate(root, pred, task, args.bootstrap, args.mode == 'smoke')
    if ds == 'Liu_phs000452' and args.mode == 'full':
        for source, group in meta.groupby('study_source'):
            sub = root / f'source_{source}'
            sub.mkdir()
            evaluate(sub, pred[pred.sample_id.isin(group.sample_id)], task, args.bootstrap)
    write_json(root / 'COMPLETE.json', {'elapsed_seconds': time.monotonic() - start, 'n_samples': len(x), 'n_patients': meta.group_id.nunique(), 'mode': args.mode, 'scheme': scheme, 'prediction_sha256': old.sha256_file(root / 'predictions.csv')})
    print(f'[complete] {root.name} {time.monotonic() - start:.1f}s', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--raw-root', type=Path, default=Path(str(_submission_path('data', ''))))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='full')
    parser.add_argument('--stage', choices=['preflight', 'pretreatment', 'summary', 'all'], default='all')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--model-jobs', type=int, default=16)
    parser.add_argument('--blas-threads', type=int, default=2)
    args = parser.parse_args()
    prepare(args)
    if args.stage == 'preflight':
        return
    with threadpool_limits(limits=args.blas_threads, user_api='blas'):
        for scheme in ['pretreatment', 'all_samples'] if args.stage == 'all' else ['pretreatment'] if args.stage == 'pretreatment' else ['all_samples']:
            for task, datasets in [('task3', old.TASK3_DATASETS), ('task5', old.TASK5_DATASETS)]:
                for ds in datasets:
                    if scheme == 'pretreatment' and ds not in ['Gide_PRJEB23709', 'Riaz_GSE91061']:
                        continue
                    if args.mode == 'smoke' and ds != 'Gide_PRJEB23709':
                        continue
                    run_unit(args, task, ds, scheme)
if __name__ == '__main__':
    main()
