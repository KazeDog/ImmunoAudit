"""Run the deterministic, patient-grouped audit for Tasks 1, 3 and 5."""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from task135_core import AuditInvariantError, aggregate_classification_by_group, aggregate_survival_by_group, canonical_task1_group, classification_metrics, classifier_oof_predictions, cluster_bootstrap_metric, fold_restricted_cindex, harrell_cindex, make_grouped_folds, nested_select_from_outer_predictions, percentile_interval, set_deterministic_seed, survival_oof_predictions
RANDOM_STATE = 42
TASK3_DATASETS = ('Gide_PRJEB23709', 'IMvigor210', 'Kim_PRJEB25780', 'Liu_phs000452', 'Riaz_GSE91061')
TASK5_DATASETS = ('Gide_PRJEB23709', 'Liu_phs000452', 'Riaz_GSE91061')
TASK3_PATIENT_AGGREGATED = {'Gide_PRJEB23709', 'Riaz_GSE91061'}
TASK3_CLINICAL_FILES = {'Gide_PRJEB23709': 'Gide_PRJEB23709/Melanoma-PRJEB23709.Response.Clinical Data.tsv', 'Kim_PRJEB25780': 'Kim_PRJEB25780/STAD-PRJEB25780.Response.Clinical Data.tsv', 'Liu_phs000452': 'Liu_phs000452/Melanoma-phs000452.Response.Clinical Data.tsv', 'Riaz_GSE91061': 'Riaz_GSE91061/Melanoma-GSE91061.Response.Clinical Data.tsv'}
LOCKED_STRATEGY1 = {'task1': 'SVM', 'task3_Gide_PRJEB23709': 'RF', 'task3_IMvigor210': 'XGB', 'task3_Kim_PRJEB25780': 'XGB', 'task3_Liu_phs000452': 'XGB', 'task3_Riaz_GSE91061': 'SVM'}
LOCKED_STRATEGY2 = {'task1': 'Geneformer+XGB', 'task3_Gide_PRJEB23709': 'geneformer+SVM', 'task3_IMvigor210': 'scfoundation+LR', 'task3_Kim_PRJEB25780': 'geneformer+LR', 'task3_Liu_phs000452': 'scfoundation+RF', 'task3_Riaz_GSE91061': 'scfoundation+SVM'}

def parse_args() -> argparse.Namespace:
    project_root = _submission_path('project', '')
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=project_root)
    parser.add_argument('--raw-root', type=Path, default=Path(str(_submission_path('data', ''))))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='full')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--model-jobs', type=int, default=16)
    return parser.parse_args()

def ensure_dirs(root: Path) -> None:
    for name in ('fold_assignments', 'predictions', 'metrics', 'metadata', 'logs'):
        (root / name).mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def model_factories(model_jobs: int) -> dict[str, Callable[[], Any]]:
    return {'LR': lambda: Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]), 'SVM': lambda: Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True, random_state=RANDOM_STATE))]), 'RF': lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=model_jobs), 'MLP': lambda: Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE))]), 'XGB': lambda: XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=model_jobs, verbosity=0)}

def load_xy(processed: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_path = processed / f'{stem}_X_final.csv'
    y_path = processed / f'{stem}_y_final.csv'
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f'Missing X/y for {stem}')
    x = pd.read_csv(x_path, index_col=0).fillna(0.0)
    y = pd.read_csv(y_path, index_col=0)
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    common = x.index.intersection(y.index, sort=False)
    x = x.loc[common]
    y = y.loc[common]
    if not x.index.equals(y.index) or not x.index.is_unique:
        raise AuditInvariantError(f'X/y alignment failed for {stem}')
    return (x, y)

def raw_clinical_mapping(raw_root: Path, dataset: str) -> pd.DataFrame:
    task3_root = raw_root / '3-'
    if dataset == 'IMvigor210':
        frame = pd.read_csv(task3_root / 'IMvigor210/pData_IMvigor210.csv', index_col=0)
        result = pd.DataFrame({'sample_id': frame.index.astype(str), 'group_id': frame['ANONPT_ID'].astype(str).to_numpy(), 'treatment_time': 'baseline', 'raw_response': frame['Best.Confirmed.Overall.Response'].astype(str).to_numpy()})
    else:
        frame = pd.read_csv(task3_root / TASK3_CLINICAL_FILES[dataset], sep='\t')
        result = pd.DataFrame({'sample_id': frame['sample_id'].astype(str), 'group_id': frame['patient_name'].astype(str), 'treatment_time': frame['Treatment'].fillna('unspecified').astype(str), 'raw_response': frame['response'].fillna('missing').astype(str)})
    if result['sample_id'].duplicated().any():
        raise AuditInvariantError(f'Raw clinical sample IDs are duplicated for {dataset}')
    return result.set_index('sample_id')

def align_group_metadata(sample_ids: pd.Index, mapping: pd.DataFrame, dataset: str) -> pd.DataFrame:
    missing = sample_ids.difference(mapping.index)
    if len(missing):
        raise AuditInvariantError(f'Missing patient mapping for {dataset}: {missing[:5].tolist()}')
    return mapping.loc[sample_ids].copy()

def fold_assignment_frame(*, task: str, dataset: str, sample_ids: pd.Index, groups: np.ndarray, outcome: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({'task': task, 'dataset': dataset, 'sample_id': sample_ids.astype(str), 'group_id': groups.astype(str), 'stratification_outcome': outcome.astype(int), 'fold': folds.astype(int)})
    for fold in sorted(frame['fold'].unique()):
        train_groups = set(frame.loc[frame['fold'] != fold, 'group_id'])
        test_groups = set(frame.loc[frame['fold'] == fold, 'group_id'])
        if train_groups.intersection(test_groups):
            raise AuditInvariantError(f'Fold assignment leaks groups for {task}/{dataset}')
    return frame

def load_task1(processed: Path) -> dict[str, Any]:
    x, y = load_xy(processed, 'task1')
    outcome = y.iloc[:, 0].astype(int).to_numpy()
    groups = np.asarray([canonical_task1_group(value) for value in x.index], dtype=object)
    folds = make_grouped_folds(outcome, groups)
    return {'x': x, 'y': outcome, 'groups': groups, 'folds': folds}

def load_embedding(path: Path, sample_ids: pd.Index, sample_path: Path | None=None) -> np.ndarray:
    matrix = np.load(path).astype(np.float32)
    if sample_path is None:
        if matrix.shape[0] != len(sample_ids):
            raise AuditInvariantError(f'Embedding rows do not match samples: {path}')
        return matrix
    order = pd.read_csv(sample_path)['sample_id'].astype(str)
    if len(order) != matrix.shape[0] or order.duplicated().any():
        raise AuditInvariantError(f'Invalid sample order file: {sample_path}')
    locations = pd.Series(np.arange(len(order)), index=order)
    missing = sample_ids.difference(locations.index)
    if len(missing):
        raise AuditInvariantError(f'Embedding missing samples: {missing[:5].tolist()}')
    return matrix[locations.loc[sample_ids].to_numpy(dtype=int)]

def evaluate_candidate_pool(*, task: str, dataset: str, strategy: str, features: dict[str, np.ndarray], factories: dict[str, Callable[[], Any]], y: np.ndarray, sample_ids: pd.Index, groups: np.ndarray, folds: np.ndarray, patient_level_metric: bool, only_folds: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    for model in sorted(factories):
        frame = classifier_oof_predictions(features[model], y, sample_ids, groups, folds, factories[model], only_folds=only_folds)
        frame['task'] = task
        frame['dataset'] = dataset
        frame['strategy'] = strategy
        frame['model'] = model
        predictions.append(frame)
    all_predictions = pd.concat(predictions, ignore_index=True)
    if only_folds is None:
        nested, selection = nested_select_from_outer_predictions(features=features, factories=factories, y=y, sample_ids=sample_ids, group_ids=groups, outer_folds=folds, outer_predictions=all_predictions, patient_level_metric=patient_level_metric)
        nested['task'] = task
        nested['dataset'] = dataset
        nested['strategy'] = strategy
        selection['task'] = task
        selection['dataset'] = dataset
        selection['strategy'] = strategy
        all_predictions = pd.concat([all_predictions, nested], ignore_index=True)
    else:
        selection = pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    for model, frame in all_predictions.groupby('model', sort=True):
        evaluation = aggregate_classification_by_group(frame) if patient_level_metric else frame
        metrics = classification_metrics(evaluation['outcome'], evaluation['score'], evaluation['predicted'])
        for name in ('auroc', 'auprc', 'accuracy', 'macro_f1'):
            metric_rows.append({'task': task, 'dataset': dataset, 'strategy': strategy, 'model': model, 'analysis_level': 'patient' if patient_level_metric else 'sample_grouped_by_patient', 'metric': name.upper(), 'estimate': getattr(metrics, name), 'n': metrics.n})
    return (all_predictions, pd.DataFrame(metric_rows), selection)

def task1_candidate_pools(project: Path, data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    base_factories = model_factories(args.model_jobs)
    x = data['x'].to_numpy(dtype=np.float32)
    strategy1_names = ('LR', 'SVM', 'RF', 'XGB')
    strategy1_features = {name: x for name in strategy1_names}
    strategy1_factories = {name: base_factories[name] for name in strategy1_names}
    emb_root = project / 'processed_data_strategy2'
    feature_paths = {'Geneformer': emb_root / 'task1_geneformer_X_emb.npy', 'Nicheformer': emb_root / 'task1_nicheformer_X_emb.npy', 'scLong_1B': emb_root / 'task1_sclong_X_emb.npy', 'scGPT': emb_root / 'task1_scgpt_X_emb.npy', 'scFoundation': emb_root / 'task1_scfoundation_X_emb.npy'}
    strategy2_features: dict[str, np.ndarray] = {}
    strategy2_factories: dict[str, Callable[[], Any]] = {}
    classifier_names = ('LR', 'SVM', 'RF', 'MLP', 'XGB')
    for feature, path in feature_paths.items():
        matrix = load_embedding(path, data['x'].index)
        for classifier in classifier_names:
            name = f'{feature}+{classifier}'
            strategy2_features[name] = matrix
            strategy2_factories[name] = base_factories[classifier]
    return [{'strategy': 'Strategy 1', 'features': strategy1_features, 'factories': strategy1_factories}, {'strategy': 'Strategy 2', 'features': strategy2_features, 'factories': strategy2_factories}]

def task3_candidate_pools(project: Path, x: pd.DataFrame, dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    base_factories = model_factories(args.model_jobs)
    raw = x.to_numpy(dtype=np.float32)
    strategy1_names = ('LR', 'SVM', 'RF', 'XGB')
    strategy1_features = {name: raw for name in strategy1_names}
    strategy1_factories = {name: base_factories[name] for name in strategy1_names}
    emb_root = project / 'processed_data_strategy2'
    strategy2_features: dict[str, np.ndarray] = {}
    strategy2_factories: dict[str, Callable[[], Any]] = {}
    for feature in ('geneformer', 'scfoundation', 'scgpt'):
        stem = f'task3_{dataset}_{feature}'
        matrix = load_embedding(emb_root / f'{stem}_X_emb.npy', x.index, emb_root / f'{stem}_samples.csv')
        for classifier in ('LR', 'SVM', 'RF', 'MLP', 'XGB'):
            name = f'{feature}+{classifier}'
            strategy2_features[name] = matrix
            strategy2_factories[name] = base_factories[classifier]
    return [{'strategy': 'Strategy 1', 'features': strategy1_features, 'factories': strategy1_factories}, {'strategy': 'Strategy 2', 'features': strategy2_features, 'factories': strategy2_factories}]

def qwen_prediction_path(project: Path, task: str, dataset: str) -> Path:
    root = project / 'final/strategy3/results/strategy3'
    if task == 'task1':
        return root / 'task1_prompt_v2_full/task1_n48_predictions.csv'
    if task == 'task3':
        n = {'Gide_PRJEB23709': 91, 'IMvigor210': 298, 'Kim_PRJEB25780': 78, 'Liu_phs000452': 150, 'Riaz_GSE91061': 98}[dataset]
        return root / f'task3_dual_score_batch_full/task3_{dataset}_n{n}_predictions.csv'
    if task == 'task5':
        n = {'Gide_PRJEB23709': 91, 'Liu_phs000452': 153, 'Riaz_GSE91061': 101}[dataset]
        return root / f'task5_v021_full/task5_{dataset}_n{n}_predictions.csv'
    raise ValueError(task)

def load_qwen_classification(path: Path, *, task: str, dataset: str, sample_ids: pd.Index, groups: np.ndarray, folds: np.ndarray, outcomes: np.ndarray) -> pd.DataFrame:
    source = pd.read_csv(path)
    source['sample_id'] = source['sample_id'].astype(str)
    source = source.set_index('sample_id')
    missing = sample_ids.difference(source.index)
    extra = source.index.difference(sample_ids)
    if len(missing) or len(extra):
        raise AuditInvariantError(f'Qwen IDs differ for {task}/{dataset}')
    source = source.loc[sample_ids]
    if not np.array_equal(source['true_label'].astype(int).to_numpy(), outcomes):
        raise AuditInvariantError(f'Qwen labels differ for {task}/{dataset}')
    pred_col = 'pred_label' if 'pred_label' in source.columns else 'pred_binary'
    frame = pd.DataFrame({'sample_id': sample_ids, 'group_id': groups, 'outcome': outcomes, 'fold': folds, 'score': pd.to_numeric(source['response_support_score'], errors='raise').to_numpy(), 'predicted': source[pred_col].astype(int).to_numpy(), 'task': task, 'dataset': dataset, 'strategy': 'Fixed LLM', 'model': 'Qwen', 'variant': 'prompt_v2' if task == 'task1' else 'dual_score_v3'})
    return frame

def load_qwen_survival(path: Path, *, dataset: str, sample_ids: pd.Index, groups: np.ndarray, folds: np.ndarray, duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    source = pd.read_csv(path)
    source['sample_id'] = source['sample_id'].astype(str)
    source = source.set_index('sample_id').loc[sample_ids]
    if not np.allclose(source['duration'].astype(float), duration) or not np.array_equal(source['event'].astype(int), event):
        raise AuditInvariantError(f'Qwen survival labels differ for {dataset}')
    return pd.DataFrame({'sample_id': sample_ids, 'group_id': groups, 'duration': duration, 'event': event, 'fold': folds, 'risk': pd.to_numeric(source['risk_score'], errors='raise').to_numpy(), 'task': 'task5', 'dataset': dataset, 'strategy': 'Fixed LLM', 'model': 'Qwen', 'variant': 'v0.2.1_previous_analysis'})

def add_classification_bootstrap(rows: list[dict[str, Any]], frame: pd.DataFrame, *, task: str, dataset: str, strategy: str, model: str, patient_level: bool, n_bootstrap: int, seed: int) -> None:
    evaluation = aggregate_classification_by_group(frame) if patient_level else frame.copy()
    for metric_name in ('AUROC', 'AUPRC'):

        def metric(block: pd.DataFrame, name: str=metric_name) -> float:
            values = classification_metrics(block['outcome'], block['score'], block.get('predicted'))
            return values.auroc if name == 'AUROC' else values.auprc
        point = metric(evaluation)
        bootstrap = cluster_bootstrap_metric(evaluation, metric, n_bootstrap=n_bootstrap, random_state=seed)
        lower, upper = percentile_interval(bootstrap)
        rows.append({'task': task, 'dataset': dataset, 'strategy': strategy, 'model': model, 'analysis_level': 'patient' if patient_level else 'sample_grouped_by_patient', 'metric': metric_name, 'estimate': point, 'ci_lower': lower, 'ci_upper': upper, 'bootstrap_replicates': len(bootstrap), 'n': evaluation['group_id'].nunique() if patient_level else len(evaluation), 'n_groups': evaluation['group_id'].nunique()})

def classification_paired_bootstrap(left: pd.DataFrame, right: pd.DataFrame, *, patient_level: bool, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    left_eval = aggregate_classification_by_group(left) if patient_level else left.copy()
    right_eval = aggregate_classification_by_group(right) if patient_level else right.copy()
    merged = left_eval[['group_id', 'outcome', 'score']].rename(columns={'score': 'left_score'}).merge(right_eval[['group_id', 'outcome', 'score']].rename(columns={'score': 'right_score'}), on=['group_id', 'outcome'], how='inner', validate='one_to_one' if patient_level else 'many_to_many')
    if not patient_level:
        merged = left[['sample_id', 'group_id', 'outcome', 'score']].rename(columns={'score': 'left_score'}).merge(right[['sample_id', 'group_id', 'outcome', 'score']].rename(columns={'score': 'right_score'}), on=['sample_id', 'group_id', 'outcome'], validate='one_to_one')

    def difference(block: pd.DataFrame) -> float:
        return float(classification_metrics(block['outcome'], block['left_score']).auroc - classification_metrics(block['outcome'], block['right_score']).auroc)
    point = difference(merged)
    values = cluster_bootstrap_metric(merged, difference, n_bootstrap=n_bootstrap, random_state=seed)
    lower, upper = percentile_interval(values)
    return (point, lower, upper)

def fixed_llm_availability(project: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in ('task1', 'task3', 'task5'):
        datasets = ('task1',) if task == 'task1' else TASK3_DATASETS if task == 'task3' else TASK5_DATASETS
        for dataset in datasets:
            qwen = qwen_prediction_path(project, task, '' if dataset == 'task1' else dataset)
            rows.append({'task': task, 'dataset': dataset, 'model': 'Qwen', 'variant': 'auditable_saved_variant', 'path': str(qwen), 'available': qwen.exists()})
            for model in ('Kimi-2.6', 'MiniMax-M2.7'):
                rows.append({'task': task, 'dataset': dataset, 'model': model, 'variant': 'manuscript_reported', 'path': 'not_present_on_audit_server', 'available': False})
    return pd.DataFrame(rows)

def run(args: argparse.Namespace) -> None:
    set_deterministic_seed()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    ensure_dirs(output)
    processed = project / 'processed_data_strategy1'
    only_folds = [1] if args.mode == 'smoke' else None
    n_bootstrap = min(args.bootstrap, 50) if args.mode == 'smoke' else args.bootstrap
    fold_frames: list[pd.DataFrame] = []
    classification_predictions: list[pd.DataFrame] = []
    classification_metrics_rows: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    survival_predictions: list[pd.DataFrame] = []
    uncertainty_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    print('[task1] loading and freezing patient-grouped folds', flush=True)
    task1 = load_task1(processed)
    fold_frames.append(fold_assignment_frame(task='task1', dataset='task1', sample_ids=task1['x'].index, groups=task1['groups'], outcome=task1['y'], folds=task1['folds']))
    source_paths.update({processed / 'task1_X_final.csv', processed / 'task1_y_final.csv'})
    pools = task1_candidate_pools(project, task1, args)
    if args.mode == 'smoke':
        pools[0]['features'] = {'SVM': pools[0]['features']['SVM']}
        pools[0]['factories'] = {'SVM': pools[0]['factories']['SVM']}
        pools[1]['features'] = {'Geneformer+XGB': pools[1]['features']['Geneformer+XGB']}
        pools[1]['factories'] = {'Geneformer+XGB': pools[1]['factories']['Geneformer+XGB']}
    for pool in pools:
        print(f"[task1] {pool['strategy']} candidate evaluation", flush=True)
        pred, metric, selection = evaluate_candidate_pool(task='task1', dataset='task1', strategy=pool['strategy'], features=pool['features'], factories=pool['factories'], y=task1['y'], sample_ids=task1['x'].index, groups=task1['groups'], folds=task1['folds'], patient_level_metric=False, only_folds=only_folds)
        classification_predictions.append(pred)
        classification_metrics_rows.append(metric)
        if not selection.empty:
            selection_frames.append(selection)
    qwen1_path = qwen_prediction_path(project, 'task1', '')
    qwen1 = load_qwen_classification(qwen1_path, task='task1', dataset='task1', sample_ids=task1['x'].index, groups=task1['groups'], folds=task1['folds'], outcomes=task1['y'])
    source_paths.add(qwen1_path)
    classification_predictions.append(qwen1)
    print('[task3] loading five response cohorts', flush=True)
    for dataset_index, dataset in enumerate(TASK3_DATASETS, start=1):
        stem = f'task3_{dataset}'
        x, y_frame = load_xy(processed, stem)
        outcome = y_frame.iloc[:, 0].astype(int).to_numpy()
        clinical = align_group_metadata(x.index, raw_clinical_mapping(args.raw_root, dataset), dataset)
        groups = clinical['group_id'].astype(str).to_numpy()
        folds = make_grouped_folds(outcome, groups)
        fold_frames.append(fold_assignment_frame(task='task3', dataset=dataset, sample_ids=x.index, groups=groups, outcome=outcome, folds=folds))
        source_paths.update({processed / f'{stem}_X_final.csv', processed / f'{stem}_y_final.csv'})
        pools = task3_candidate_pools(project, x, dataset, args)
        if args.mode == 'smoke' and dataset != 'Gide_PRJEB23709':
            continue
        if args.mode == 'smoke':
            pools[0]['features'] = {'RF': pools[0]['features']['RF']}
            pools[0]['factories'] = {'RF': pools[0]['factories']['RF']}
            pools[1]['features'] = {'scfoundation+LR': pools[1]['features']['scfoundation+LR']}
            pools[1]['factories'] = {'scfoundation+LR': pools[1]['factories']['scfoundation+LR']}
        patient_level = True
        for pool in pools:
            print(f"[task3:{dataset}] {pool['strategy']} candidate evaluation", flush=True)
            pred, metric, selection = evaluate_candidate_pool(task='task3', dataset=dataset, strategy=pool['strategy'], features=pool['features'], factories=pool['factories'], y=outcome, sample_ids=x.index, groups=groups, folds=folds, patient_level_metric=patient_level, only_folds=only_folds)
            classification_predictions.append(pred)
            classification_metrics_rows.append(metric)
            if not selection.empty:
                selection_frames.append(selection)
        qwen_path = qwen_prediction_path(project, 'task3', dataset)
        qwen = load_qwen_classification(qwen_path, task='task3', dataset=dataset, sample_ids=x.index, groups=groups, folds=folds, outcomes=outcome)
        classification_predictions.append(qwen)
        source_paths.add(qwen_path)
        cohort_rows.append({'task': 'task3', 'dataset': dataset, 'n_samples': len(x), 'n_patients': len(np.unique(groups)), 'positive_samples': int(outcome.sum()), 'positive_patients': int(pd.DataFrame({'group': groups, 'outcome': outcome}).drop_duplicates('group')['outcome'].sum()), 'repeated_patients': int((pd.Series(groups).value_counts() > 1).sum())})
    print('[task5] loading three survival cohorts', flush=True)
    for dataset_index, dataset in enumerate(TASK5_DATASETS, start=1):
        stem = f'task5_{dataset}'
        x, y_frame = load_xy(processed, stem)
        duration = y_frame['duration'].astype(float).to_numpy()
        event = y_frame['event'].astype(int).to_numpy()
        clinical = align_group_metadata(x.index, raw_clinical_mapping(args.raw_root, dataset), dataset)
        groups = clinical['group_id'].astype(str).to_numpy()
        folds = make_grouped_folds(event, groups)
        fold_frames.append(fold_assignment_frame(task='task5', dataset=dataset, sample_ids=x.index, groups=groups, outcome=event, folds=folds))
        source_paths.update({processed / f'{stem}_X_final.csv', processed / f'{stem}_y_final.csv'})
        if args.mode == 'smoke' and dataset != 'Gide_PRJEB23709':
            continue
        print(f'[task5:{dataset}] raw CoxPH(PCA)', flush=True)
        raw_pred = survival_oof_predictions(x.to_numpy(dtype=np.float32), duration, event, x.index, groups, folds, only_folds=only_folds)
        raw_pred['task'] = 'task5'
        raw_pred['dataset'] = dataset
        raw_pred['strategy'] = 'Strategy 1'
        raw_pred['model'] = 'CoxPH(PCA)'
        raw_pred['variant'] = 'patient_grouped'
        survival_predictions.append(raw_pred)
        emb_root = project / 'processed_data_strategy2'
        emb_stem = f'task5_{dataset}_scfoundation'
        embedding = load_embedding(emb_root / f'{emb_stem}_X_emb.npy', x.index, emb_root / f'{emb_stem}_samples.csv')
        print(f'[task5:{dataset}] scFoundation CoxPH(PCA)', flush=True)
        scf_pred = survival_oof_predictions(embedding, duration, event, x.index, groups, folds, only_folds=only_folds)
        scf_pred['task'] = 'task5'
        scf_pred['dataset'] = dataset
        scf_pred['strategy'] = 'Strategy 2'
        scf_pred['model'] = 'scFoundation+CoxPH(PCA)'
        scf_pred['variant'] = 'patient_grouped'
        survival_predictions.append(scf_pred)
        qwen_path = qwen_prediction_path(project, 'task5', dataset)
        qwen = load_qwen_survival(qwen_path, dataset=dataset, sample_ids=x.index, groups=groups, folds=folds, duration=duration, event=event)
        survival_predictions.append(qwen)
        source_paths.add(qwen_path)
        patient_outcomes = pd.DataFrame({'group': groups, 'event': event}).drop_duplicates('group')
        cohort_rows.append({'task': 'task5', 'dataset': dataset, 'n_samples': len(x), 'n_patients': len(np.unique(groups)), 'events_samples': int(event.sum()), 'events_patients': int(patient_outcomes['event'].sum()), 'repeated_patients': int((pd.Series(groups).value_counts() > 1).sum()), 'duration_min_days': float(duration.min()), 'duration_max_days': float(duration.max())})
    folds_all = pd.concat(fold_frames, ignore_index=True)
    for (task, dataset), frame in folds_all.groupby(['task', 'dataset'], sort=True):
        frame.to_csv(output / 'fold_assignments' / f'{task}_{dataset}_patient_grouped_folds.csv', index=False)
    class_all = pd.concat(classification_predictions, ignore_index=True)
    class_all.to_csv(output / 'predictions/classification_predictions.csv', index=False)
    candidate_metrics = pd.concat(classification_metrics_rows, ignore_index=True)
    candidate_metrics.to_csv(output / 'metrics/classification_candidate_metrics.csv', index=False)
    if selection_frames:
        pd.concat(selection_frames, ignore_index=True).to_csv(output / 'metadata/nested_model_selection.csv', index=False)
    survival_all = pd.concat(survival_predictions, ignore_index=True)
    survival_all.to_csv(output / 'predictions/survival_predictions.csv', index=False)
    pd.DataFrame(cohort_rows).to_csv(output / 'metadata/cohort_audit.csv', index=False)
    if args.mode == 'full':
        print('[uncertainty] patient-clustered bootstrap', flush=True)
        primary_classification: list[tuple[str, str, str, str, pd.DataFrame, bool]] = []
        for (task, dataset, strategy), frame in class_all.groupby(['task', 'dataset', 'strategy'], sort=True):
            patient_level = task == 'task3'
            if strategy == 'Fixed LLM':
                selected = frame[frame['model'] == 'Qwen'].copy()
                primary_classification.append((task, dataset, strategy, 'Qwen', selected, patient_level))
            else:
                nested = frame[frame['model'] == 'nested_selected'].copy()
                if not nested.empty:
                    primary_classification.append((task, dataset, strategy, 'nested_selected', nested, patient_level))
                key = task if task == 'task1' else f'task3_{dataset}'
                locked = LOCKED_STRATEGY1.get(key) if strategy == 'Strategy 1' else LOCKED_STRATEGY2.get(key)
                if locked is not None:
                    locked_frame = frame[frame['model'] == locked].copy()
                    if not locked_frame.empty:
                        primary_classification.append((task, dataset, strategy, f'historical_locked:{locked}', locked_frame, patient_level))
        for index, (task, dataset, strategy, model, frame, patient_level) in enumerate(primary_classification):
            add_classification_bootstrap(uncertainty_rows, frame, task=task, dataset=dataset, strategy=strategy, model=model, patient_level=patient_level, n_bootstrap=n_bootstrap, seed=RANDOM_STATE + index)
        for task, dataset in sorted(class_all[['task', 'dataset']].drop_duplicates().itertuples(index=False, name=None)):
            qwen = class_all[(class_all['task'] == task) & (class_all['dataset'] == dataset) & (class_all['strategy'] == 'Fixed LLM')]
            patient_level = task == 'task3'
            for strategy in ('Strategy 1', 'Strategy 2'):
                fitted = class_all[(class_all['task'] == task) & (class_all['dataset'] == dataset) & (class_all['strategy'] == strategy) & (class_all['model'] == 'nested_selected')]
                if fitted.empty or qwen.empty:
                    continue
                point, lower, upper = classification_paired_bootstrap(fitted, qwen, patient_level=patient_level, n_bootstrap=n_bootstrap, seed=RANDOM_STATE + len(paired_rows))
                paired_rows.append({'task': task, 'dataset': dataset, 'left_model': f'{strategy}:nested_selected', 'right_model': 'Fixed LLM:Qwen', 'metric': 'AUROC_difference', 'estimate': point, 'ci_lower': lower, 'ci_upper': upper, 'analysis_level': 'patient' if patient_level else 'sample_grouped_by_patient'})
        survival_metric_rows: list[dict[str, Any]] = []
        for index, ((dataset, strategy, model), frame) in enumerate(survival_all.groupby(['dataset', 'strategy', 'model'], sort=True)):
            patient = aggregate_survival_by_group(frame)
            metric_mode = 'full_cohort' if strategy == 'Fixed LLM' else 'fold_restricted_oof'
            metric_function = (lambda block: harrell_cindex(block['duration'], block['risk'], block['event'])) if strategy == 'Fixed LLM' else fold_restricted_cindex
            point = metric_function(patient)
            bootstrap = cluster_bootstrap_metric(patient, metric_function, n_bootstrap=n_bootstrap, random_state=RANDOM_STATE + 100 + index)
            lower, upper = percentile_interval(bootstrap)
            survival_metric_rows.append({'task': 'task5', 'dataset': dataset, 'strategy': strategy, 'model': model, 'analysis_level': 'patient', 'comparison_pairs': metric_mode, 'metric': 'C_INDEX', 'estimate': point, 'ci_lower': lower, 'ci_upper': upper, 'n_patients': len(patient), 'events': int(patient['event'].sum()), 'bootstrap_replicates': len(bootstrap)})
            if strategy == 'Fixed LLM':
                restricted = fold_restricted_cindex(patient)
                survival_metric_rows.append({'task': 'task5', 'dataset': dataset, 'strategy': strategy, 'model': model, 'analysis_level': 'patient', 'comparison_pairs': 'fold_restricted_for_paired_comparison', 'metric': 'C_INDEX', 'estimate': restricted, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_patients': len(patient), 'events': int(patient['event'].sum()), 'bootstrap_replicates': 0})
        for dataset in TASK5_DATASETS:
            qwen = aggregate_survival_by_group(survival_all[(survival_all['dataset'] == dataset) & (survival_all['strategy'] == 'Fixed LLM')]).rename(columns={'risk': 'right_risk'})
            for strategy, model in (('Strategy 1', 'CoxPH(PCA)'), ('Strategy 2', 'scFoundation+CoxPH(PCA)')):
                fitted = aggregate_survival_by_group(survival_all[(survival_all['dataset'] == dataset) & (survival_all['strategy'] == strategy) & (survival_all['model'] == model)]).rename(columns={'risk': 'left_risk'})
                merged = fitted.merge(qwen[['group_id', 'duration', 'event', 'fold', 'right_risk']], on=['group_id', 'duration', 'event', 'fold'], validate='one_to_one')

                def difference(block: pd.DataFrame) -> float:
                    left_block = block.rename(columns={'left_risk': 'risk'})
                    right_block = block.rename(columns={'right_risk': 'risk'})
                    return fold_restricted_cindex(left_block) - fold_restricted_cindex(right_block)
                point = difference(merged)
                bootstrap = cluster_bootstrap_metric(merged, difference, n_bootstrap=n_bootstrap, random_state=RANDOM_STATE + 200 + len(paired_rows))
                lower, upper = percentile_interval(bootstrap)
                paired_rows.append({'task': 'task5', 'dataset': dataset, 'left_model': f'{strategy}:{model}', 'right_model': 'Fixed LLM:Qwen', 'metric': 'C_INDEX_difference_fold_restricted', 'estimate': point, 'ci_lower': lower, 'ci_upper': upper, 'analysis_level': 'patient'})
        pd.DataFrame(survival_metric_rows).to_csv(output / 'metrics/survival_primary_metrics.csv', index=False)
        pd.DataFrame(uncertainty_rows).to_csv(output / 'metrics/classification_primary_metrics.csv', index=False)
        pd.DataFrame(paired_rows).to_csv(output / 'metrics/paired_differences.csv', index=False)
    availability = fixed_llm_availability(project)
    availability.to_csv(output / 'metadata/fixed_llm_source_availability.csv', index=False)
    source_rows = []
    for path in sorted(source_paths):
        source_rows.append({'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    pd.DataFrame(source_rows).to_csv(output / 'metadata/source_file_hashes.csv', index=False)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'random_state': RANDOM_STATE, 'bootstrap_requested': n_bootstrap, 'model_jobs': args.model_jobs, 'project_root': str(project), 'raw_root': str(args.raw_root.resolve()), 'output_dir': str(output), 'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'critical_rules': {'independent_group': 'patient/subject', 'classification_split': 'fivefold StratifiedGroupKFold with a fixed seed', 'survival_split': 'fivefold event-stratified StratifiedGroupKFold with a fixed seed', 'preprocessing': 'all learned scaling and PCA operations fitted in the training fold', 'fixed_llm': 'single saved full-cohort inference; no API calls and no CV claim', 'task3_primary': 'patient-averaged AUROC for cohorts with longitudinal biopsies', 'task5_primary': 'patient-averaged risk and fold-restricted OOF C-index'}}
    manifest_path = output / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Completed {args.mode} audit at {output}', flush=True)
if __name__ == '__main__':
    run(parse_args())
