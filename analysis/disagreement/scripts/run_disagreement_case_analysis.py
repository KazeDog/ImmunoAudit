"""Deterministic disagreement and prespecified case audit.

This script consumes only previously validated held-out predictions. It does not
fit a model, regenerate an embedding, or call an LLM.
"""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
STRATEGIES = ('conventional', 'representation', 'qwen')
STRATEGY_LABELS = {'conventional': 'Conventional learning', 'representation': 'Representation learning', 'qwen': 'Fixed Qwen'}
PAIR_ORDER = (('conventional', 'representation'), ('conventional', 'qwen'), ('representation', 'qwen'))
DATASET_LABELS = {'task1': 'Single-cell response', 'Gide_PRJEB23709': 'Gide response', 'IMvigor210': 'IMvigor210 response', 'Kim_PRJEB25780': 'Kim response', 'Liu_phs000452': 'Liu response', 'Riaz_GSE91061': 'Riaz response', 'task5:Gide_PRJEB23709': 'Gide survival', 'task5:Liu_phs000452': 'Liu survival', 'task5:Riaz_GSE91061': 'Riaz survival'}

@dataclass(frozen=True)
class InputPaths:
    classification: Path
    survival: Path
    task4_repeated_samples: Path
    task4_time_audit: Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def metric_seed(base_seed: int, identifier: str) -> int:
    token = hashlib.sha256(identifier.encode('utf-8')).digest()
    return (base_seed + int.from_bytes(token[:4], 'little')) % (2 ** 32 - 1)

def midrank_percentile(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError('midrank_percentile requires a non-empty one-dimensional vector')
    if not np.isfinite(array).all():
        raise ValueError('midrank_percentile received non-finite values')
    return (rankdata(array, method='average') - 0.5) / len(array)

def cluster_members(groups: Iterable[object]) -> tuple[np.ndarray, list[np.ndarray]]:
    group_array = np.asarray(list(groups), dtype=object)
    unique = pd.unique(group_array)
    members = [np.flatnonzero(group_array == group) for group in unique]
    if any((len(index) == 0 for index in members)):
        raise AssertionError('empty bootstrap cluster')
    return (np.asarray(unique, dtype=object), members)

def cluster_bootstrap(arrays: tuple[np.ndarray, ...], groups: Iterable[object], statistic: Callable[..., float], n_bootstrap: int, seed: int) -> np.ndarray:
    arrays = tuple((np.asarray(array) for array in arrays))
    n = len(arrays[0])
    if any((len(array) != n for array in arrays)):
        raise ValueError('bootstrap arrays must have the same length')
    unique, members = cluster_members(groups)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[position] for position in selected])
        value = float(statistic(*(array[index] for array in arrays)))
        if math.isfinite(value):
            values.append(value)
    if len(values) < max(20, int(0.9 * n_bootstrap)):
        raise RuntimeError(f'only {len(values)} of {n_bootstrap} bootstrap replicates were finite')
    return np.asarray(values, dtype=float)

def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple((float(value) for value in np.quantile(values, [0.025, 0.975])))

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(x, y)
    return float(result.statistic)

def binary_mean(values: np.ndarray) -> float:
    return float(np.mean(values.astype(float)))

def majority_generated_call(values: Iterable[int]) -> int:
    array = np.asarray(list(values), dtype=int)
    if len(array) == 0 or not np.isin(array, [0, 1]).all():
        raise ValueError('generated calls must be a non-empty binary vector')
    return int(array.mean() >= 0.5)

def checked_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {path}')
    frame.to_csv(path, index=False)

def checked_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {path}')
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def assert_single_value(frame: pd.DataFrame, group_columns: list[str], column: str) -> None:
    maximum = frame.groupby(group_columns, dropna=False)[column].nunique(dropna=False).max()
    if maximum != 1:
        raise AssertionError(f'{column} is not constant within {group_columns}')

def selected_classification_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[frame['strategy'].isin(['Strategy 1', 'Strategy 2']) & (frame['model'] == 'nested_selected') | (frame['strategy'] == 'Fixed LLM') & (frame['model'] == 'Qwen')].copy()
    selected['strategy_key'] = selected['strategy'].map({'Strategy 1': 'conventional', 'Strategy 2': 'representation', 'Fixed LLM': 'qwen'})
    if selected['strategy_key'].isna().any():
        raise AssertionError('unmapped classification strategy')
    if not np.isfinite(selected['score']).all():
        raise AssertionError('non-finite classification score')
    if selected.duplicated(['task', 'dataset', 'strategy_key', 'sample_id']).any():
        raise AssertionError('duplicate classification prediction')
    assert_single_value(selected, ['task', 'dataset', 'sample_id'], 'outcome')
    assert_single_value(selected, ['task', 'dataset', 'group_id'], 'fold')
    for _, block in selected.groupby(['task', 'dataset']):
        sets = [set(group['sample_id']) for _, group in block.groupby('strategy_key')]
        if len(sets) != 3 or any((item != sets[0] for item in sets[1:])):
            raise AssertionError('classification strategies do not share identical observations')
    return selected

def selected_survival_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[(frame['strategy'] == 'Strategy 1') & (frame['model'] == 'CoxPH(PCA)') | (frame['strategy'] == 'Strategy 2') & (frame['model'] == 'scFoundation+CoxPH(PCA)') | (frame['strategy'] == 'Fixed LLM') & (frame['model'] == 'Qwen')].copy()
    selected['strategy_key'] = selected['strategy'].map({'Strategy 1': 'conventional', 'Strategy 2': 'representation', 'Fixed LLM': 'qwen'})
    if not np.isfinite(selected['risk']).all():
        raise AssertionError('non-finite survival risk')
    if selected.duplicated(['dataset', 'strategy_key', 'sample_id']).any():
        raise AssertionError('duplicate survival prediction')
    assert_single_value(selected, ['dataset', 'sample_id'], 'duration')
    assert_single_value(selected, ['dataset', 'sample_id'], 'event')
    assert_single_value(selected, ['dataset', 'group_id'], 'fold')
    for _, block in selected.groupby('dataset'):
        sets = [set(group['sample_id']) for _, group in block.groupby('strategy_key')]
        if len(sets) != 3 or any((item != sets[0] for item in sets[1:])):
            raise AssertionError('survival strategies do not share identical observations')
    return selected

def prepare_analysis_units(classification: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    task1 = classification[classification['task'] == 'task1'].copy()
    task1_units = task1.rename(columns={'sample_id': 'unit_id', 'group_id': 'cluster_id', 'score': 'raw_score'})
    task1_units['endpoint_type'] = 'classification'
    task1_units['n_samples'] = 1
    task1_units['sample_ids'] = task1_units['unit_id'].astype(str)
    task1_units['call'] = task1_units['predicted'].astype(int)
    task1_units['duration'] = np.nan
    task1_units['event'] = np.nan
    records.append(task1_units[['task', 'dataset', 'endpoint_type', 'unit_id', 'cluster_id', 'outcome', 'duration', 'event', 'fold', 'strategy_key', 'raw_score', 'call', 'n_samples', 'sample_ids']])
    task3 = classification[classification['task'] == 'task3'].copy()
    assert_single_value(task3, ['dataset', 'group_id'], 'outcome')
    grouped_task3 = task3.groupby(['task', 'dataset', 'group_id', 'strategy_key'], as_index=False).agg(outcome=('outcome', 'first'), fold=('fold', 'first'), raw_score=('score', 'mean'), generated_call=('predicted', majority_generated_call), n_samples=('sample_id', 'nunique'), sample_ids=('sample_id', lambda values: ';'.join(sorted(map(str, values))))).rename(columns={'group_id': 'unit_id'})
    grouped_task3['cluster_id'] = grouped_task3['unit_id'].astype(str)
    grouped_task3['endpoint_type'] = 'classification'
    grouped_task3['call'] = grouped_task3['generated_call'].astype(int)
    grouped_task3['duration'] = np.nan
    grouped_task3['event'] = np.nan
    records.append(grouped_task3[['task', 'dataset', 'endpoint_type', 'unit_id', 'cluster_id', 'outcome', 'duration', 'event', 'fold', 'strategy_key', 'raw_score', 'call', 'n_samples', 'sample_ids']])
    assert_single_value(survival, ['dataset', 'group_id'], 'duration')
    assert_single_value(survival, ['dataset', 'group_id'], 'event')
    grouped_survival = survival.groupby(['task', 'dataset', 'group_id', 'strategy_key'], as_index=False).agg(duration=('duration', 'first'), event=('event', 'first'), fold=('fold', 'first'), raw_score=('risk', 'mean'), n_samples=('sample_id', 'nunique'), sample_ids=('sample_id', lambda values: ';'.join(sorted(map(str, values))))).rename(columns={'group_id': 'unit_id'})
    grouped_survival['cluster_id'] = grouped_survival['unit_id'].astype(str)
    grouped_survival['endpoint_type'] = 'survival'
    grouped_survival['outcome'] = np.nan
    grouped_survival['call'] = np.nan
    records.append(grouped_survival[['task', 'dataset', 'endpoint_type', 'unit_id', 'cluster_id', 'outcome', 'duration', 'event', 'fold', 'strategy_key', 'raw_score', 'call', 'n_samples', 'sample_ids']])
    units = pd.concat(records, ignore_index=True)
    units['unit_id'] = units['unit_id'].astype(str)
    units['cluster_id'] = units['cluster_id'].astype(str)
    units['percentile'] = np.nan
    for _, index in units.groupby(['task', 'dataset', 'strategy_key'], sort=False).groups.items():
        units.loc[index, 'percentile'] = midrank_percentile(units.loc[index, 'raw_score'].to_numpy())
    if units['percentile'].isna().any():
        raise AssertionError('missing percentile')
    return units.sort_values(['task', 'dataset', 'unit_id', 'strategy_key']).reset_index(drop=True)

def wide_analysis_units(units: pd.DataFrame) -> pd.DataFrame:
    id_columns = ['task', 'dataset', 'endpoint_type', 'unit_id', 'cluster_id']
    metadata = units.groupby(id_columns, as_index=False).agg(outcome=('outcome', 'first'), duration=('duration', 'first'), event=('event', 'first'), fold=('fold', 'first'), n_samples=('n_samples', 'first'), sample_ids=('sample_ids', 'first'))
    wide = metadata.copy()
    for value_column, prefix in [('raw_score', 'score'), ('percentile', 'percentile'), ('call', 'call')]:
        pivot = units.pivot(index=id_columns, columns='strategy_key', values=value_column).reset_index()
        pivot = pivot.rename(columns={key: f'{prefix}_{key}' for key in STRATEGIES})
        wide = wide.merge(pivot, on=id_columns, how='inner', validate='one_to_one')
    required = [f'score_{key}' for key in STRATEGIES] + [f'percentile_{key}' for key in STRATEGIES]
    if wide[required].isna().any().any():
        raise AssertionError('incomplete strategy matrix')
    percentile_columns = [f'percentile_{key}' for key in STRATEGIES]
    wide['rank_dispersion'] = wide[percentile_columns].max(axis=1) - wide[percentile_columns].min(axis=1)
    wide['dataset_key'] = np.where(wide['task'] == 'task5', 'task5:' + wide['dataset'], wide['dataset'])
    wide['dataset_label'] = wide['dataset_key'].map(DATASET_LABELS)
    if wide['dataset_label'].isna().any():
        raise AssertionError('missing dataset label')
    return wide.sort_values(['task', 'dataset', 'unit_id']).reset_index(drop=True)

def correlation_job(task: str, dataset: str, endpoint_type: str, dataset_label: str, strategy_a: str, strategy_b: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_bootstrap: int, base_seed: int) -> dict:
    identifier = f'correlation|{task}|{dataset}|{strategy_a}|{strategy_b}'
    seed = metric_seed(base_seed, identifier)
    point = safe_spearman(x, y)
    boot = cluster_bootstrap((x, y), groups, safe_spearman, n_bootstrap, seed)
    lower, upper = percentile_interval(boot)
    return {'task': task, 'dataset': dataset, 'dataset_label': dataset_label, 'endpoint_type': endpoint_type, 'strategy_a': strategy_a, 'strategy_b': strategy_b, 'strategy_a_label': STRATEGY_LABELS[strategy_a], 'strategy_b_label': STRATEGY_LABELS[strategy_b], 'pair_label': f'{STRATEGY_LABELS[strategy_a]} vs {STRATEGY_LABELS[strategy_b]}', 'n_units': len(x), 'n_patients': len(pd.unique(groups)), 'estimate': point, 'ci_lower': lower, 'ci_upper': upper, 'bootstrap_replicates': n_bootstrap, 'finite_bootstrap_replicates': len(boot), 'bootstrap_seed': seed, 'interval': 'patient-level or patient-clustered percentile 95% CI'}

def compute_pairwise_correlations(wide: pd.DataFrame, n_bootstrap: int, base_seed: int, jobs: int) -> pd.DataFrame:
    work = []
    for (task, dataset), block in wide.groupby(['task', 'dataset'], sort=False):
        for strategy_a, strategy_b in PAIR_ORDER:
            work.append(joblib.delayed(correlation_job)(task, dataset, block['endpoint_type'].iloc[0], block['dataset_label'].iloc[0], strategy_a, strategy_b, block[f'score_{strategy_a}'].to_numpy(float), block[f'score_{strategy_b}'].to_numpy(float), block['cluster_id'].to_numpy(object), n_bootstrap, base_seed))
    results = joblib.Parallel(n_jobs=jobs, prefer='threads')(work)
    return pd.DataFrame(results)

def compute_dispersion_summary(wide: pd.DataFrame, n_bootstrap: int, base_seed: int) -> pd.DataFrame:
    rows = []
    for (task, dataset), block in wide.groupby(['task', 'dataset'], sort=False):
        values = block['rank_dispersion'].to_numpy(float)
        groups = block['cluster_id'].to_numpy(object)
        identifier = f'median_dispersion|{task}|{dataset}'
        seed = metric_seed(base_seed, identifier)
        boot = cluster_bootstrap((values,), groups, lambda x: float(np.median(x)), n_bootstrap, seed)
        lower, upper = percentile_interval(boot)
        rows.append({'task': task, 'dataset': dataset, 'dataset_label': block['dataset_label'].iloc[0], 'endpoint_type': block['endpoint_type'].iloc[0], 'n_units': len(block), 'n_patients': block['cluster_id'].nunique(), 'median_rank_dispersion': float(np.median(values)), 'iqr_lower': float(np.quantile(values, 0.25)), 'iqr_upper': float(np.quantile(values, 0.75)), 'ci_lower': lower, 'ci_upper': upper, 'bootstrap_replicates': n_bootstrap, 'bootstrap_seed': seed})
    return pd.DataFrame(rows)

def add_classification_patterns(wide: pd.DataFrame) -> pd.DataFrame:
    frame = wide[wide['endpoint_type'] == 'classification'].copy()
    calls = frame[[f'call_{key}' for key in STRATEGIES]].to_numpy(int)
    truth = frame['outcome'].to_numpy(int)
    correct = calls == truth[:, None]
    frame['unanimous_call'] = np.all(calls == calls[:, [0]], axis=1)
    frame['n_correct'] = correct.sum(axis=1)
    frame['all_correct'] = frame['n_correct'] == 3
    frame['all_wrong'] = frame['n_correct'] == 0
    frame['exactly_one_correct'] = frame['n_correct'] == 1
    frame['exactly_two_correct'] = frame['n_correct'] == 2
    for index, key in enumerate(STRATEGIES):
        frame[f'correct_{key}'] = correct[:, index]
        frame[f'unique_correct_{key}'] = correct[:, index] & (frame['n_correct'].to_numpy() == 1)
    frame['call_pattern'] = [''.join(map(str, row)) for row in calls]
    frame['correctness_pattern'] = [''.join(('C' if value else 'W' for value in row)) for row in correct]
    return frame

def compute_decision_agreement(classified: pd.DataFrame, n_bootstrap: int, base_seed: int) -> pd.DataFrame:
    metrics = ['unanimous_call', 'all_correct', 'all_wrong', 'exactly_one_correct', 'exactly_two_correct', 'unique_correct_conventional', 'unique_correct_representation', 'unique_correct_qwen']
    rows = []
    for (task, dataset), block in classified.groupby(['task', 'dataset'], sort=False):
        groups = block['cluster_id'].to_numpy(object)
        for metric in metrics:
            values = block[metric].to_numpy(bool)
            identifier = f'decision|{task}|{dataset}|{metric}'
            seed = metric_seed(base_seed, identifier)
            boot = cluster_bootstrap((values,), groups, binary_mean, n_bootstrap, seed)
            lower, upper = percentile_interval(boot)
            rows.append({'task': task, 'dataset': dataset, 'dataset_label': block['dataset_label'].iloc[0], 'metric': metric, 'n_units': len(block), 'n_patients': block['cluster_id'].nunique(), 'count': int(values.sum()), 'rate': float(values.mean()), 'ci_lower': lower, 'ci_upper': upper, 'bootstrap_replicates': n_bootstrap, 'bootstrap_seed': seed})
    return pd.DataFrame(rows)

def correctness_pattern_counts(classified: pd.DataFrame) -> pd.DataFrame:
    counts = classified.groupby(['task', 'dataset', 'dataset_label', 'correctness_pattern'], as_index=False).size().rename(columns={'size': 'count'})
    totals = counts.groupby(['task', 'dataset'])['count'].transform('sum')
    counts['rate'] = counts['count'] / totals
    return counts

def sample_sort_key(sample_id: str) -> tuple[int, int, str]:
    text = str(sample_id)
    prefix = 0 if text.lower().startswith('pre') else 1
    suffix_match = re.search('_(\\d+)$', text)
    suffix = int(suffix_match.group(1)) if suffix_match else 1
    return (prefix, suffix, text)

def select_task1_cases(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task1 = classified[classified['task'] == 'task1'].copy()
    label_counts = task1.groupby('cluster_id')['outcome'].nunique()
    selected_patients = sorted(label_counts[label_counts > 1].index, key=lambda value: int(re.sub('\\D', '', value)))
    task1['selected'] = task1['cluster_id'].isin(selected_patients)
    selected = task1[task1['selected']].copy()
    selected['patient_order'] = selected['cluster_id'].map({patient: i for i, patient in enumerate(selected_patients)})
    selected['sample_order'] = selected['unit_id'].map(sample_sort_key)
    selected = selected.sort_values(['patient_order', 'sample_order']).drop(columns='sample_order')
    selected['selection_rule'] = 'all patients with discordant biopsy-level outcome labels'
    return (task1, selected)

def select_task3_cases(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task3 = classified[classified['task'] == 'task3'].copy()
    task3['selected'] = False
    selection_rows = []
    for (dataset, outcome), block in task3.groupby(['dataset', 'outcome'], sort=True):
        chosen = block.sort_values(['rank_dispersion', 'unit_id'], ascending=[False, True]).iloc[0]
        selection_rows.append(chosen)
        task3.loc[chosen.name, 'selected'] = True
    selected = pd.DataFrame(selection_rows).reset_index(drop=True)
    selected = selected.sort_values(['dataset', 'outcome', 'unit_id']).reset_index(drop=True)
    selected['selection_rule'] = 'maximum rank dispersion within cohort and observed outcome; lexical ID tie-break'
    return (task3, selected)

def timepoint_day(label: str, event_day: float | None=None) -> float:
    text = str(label).strip()
    if text.lower() == 'bl':
        return 0.0
    if text.upper() == 'TOX':
        return float(event_day) if event_day is not None and math.isfinite(float(event_day)) else np.nan
    week = re.fullmatch('(\\d+(?:\\.\\d+)?)W', text.upper())
    if week:
        return float(week.group(1)) * 7.0
    week_range = re.fullmatch('(\\d+(?:\\.\\d+)?)-(\\d+(?:\\.\\d+)?)W', text.upper())
    if week_range:
        return (float(week_range.group(1)) + float(week_range.group(2))) * 3.5
    month = re.fullmatch('(\\d+(?:\\.\\d+)?)M', text.upper())
    if month:
        return float(month.group(1)) * 30.4375
    return np.nan

def choose_max_and_lower_median(block: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    ascending = block.sort_values(['score_range', 'patient_id'], ascending=[True, True]).reset_index(drop=True)
    median = ascending.iloc[(len(ascending) - 1) // 2]
    maximum = block.sort_values(['score_range', 'patient_id'], ascending=[False, True]).iloc[0]
    return [('maximum range', maximum), ('median range', median)]

def prepare_task4_cases(predictions_path: Path, time_audit_path: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(predictions_path)
    frame = frame[(frame['analysis'] == 'all_samples_case_post_ici_incident') & (frame['model_key'] == 'strategy1|BCR_3mer|XGBoost')].copy()
    seeds = sorted(frame['repeat_seed'].unique())
    expected_seeds = seeds[:3] if smoke else seeds
    frame = frame[frame['repeat_seed'].isin(expected_seeds)].copy()
    if len(expected_seeds) != (3 if smoke else 20):
        raise AssertionError('unexpected Task 4 repeat count')
    if frame.duplicated(['repeat_seed', 'sample_id']).any():
        raise AssertionError('duplicate Task 4 repeated prediction')
    assert_single_value(frame, ['sample_id'], 'true_label')
    assert_single_value(frame, ['sample_id'], 'collection_time')
    sample = frame.groupby(['sample_id', 'patient_id', 'collection_time', 'cancer_type', 'ici_treatment', 'true_label'], as_index=False).agg(mean_score=('risk_score', 'mean'), sd_across_partitions=('risk_score', 'std'), min_score=('risk_score', 'min'), max_score=('risk_score', 'max'), n_partitions=('repeat_seed', 'nunique'))
    if not (sample['n_partitions'] == len(expected_seeds)).all():
        raise AssertionError('incomplete Task 4 repeated predictions')
    sample['is_baseline'] = sample['collection_time'].astype(str).str.lower().eq('bl')
    patient = sample.groupby(['patient_id', 'true_label'], as_index=False).agg(n_samples=('sample_id', 'nunique'), n_baseline=('is_baseline', 'sum'), score_min=('mean_score', 'min'), score_max=('mean_score', 'max'))
    patient['score_range'] = patient['score_max'] - patient['score_min']
    patient = patient[(patient['n_samples'] >= 2) & (patient['n_baseline'] >= 1)].copy()
    patient['selected'] = False
    selected_rows = []
    for label, block in patient.groupby('true_label', sort=True):
        for role, row in choose_max_and_lower_median(block):
            selected_rows.append({'patient_id': row['patient_id'], 'true_label': label, 'selection_role': role})
            patient.loc[patient['patient_id'] == row['patient_id'], 'selected'] = True
    selection = pd.DataFrame(selected_rows)
    selected = sample.merge(selection, on=['patient_id', 'true_label'], how='inner', validate='many_to_one')
    time = pd.read_csv(time_audit_path)
    if time.duplicated('patient_id').any():
        raise AssertionError('duplicate Task 4 time-audit patient')
    selected = selected.merge(time[['patient_id', 'earliest_post_ici_irae_day', 'max_recorded_grade', 'post_ici_incident_label']], on='patient_id', how='left', validate='many_to_one')
    if selected['post_ici_incident_label'].isna().any():
        raise AssertionError('Task 4 case lacks time-audit linkage')
    if not (selected['post_ici_incident_label'].astype(int) == selected['true_label'].astype(int)).all():
        raise AssertionError('Task 4 incident label mismatch')
    selected['approximate_collection_day'] = [timepoint_day(label, event_day) for label, event_day in zip(selected['collection_time'], selected['earliest_post_ici_irae_day'])]
    selected['selection_rule'] = 'maximum or lower-median within-patient range of 20-partition mean 3-mer XGBoost score, stratified by post-ICI incident outcome'
    selected = selected.merge(patient[['patient_id', 'score_range']], on='patient_id', how='left', validate='many_to_one')
    selected = selected.sort_values(['true_label', 'selection_role', 'patient_id', 'approximate_collection_day', 'collection_time']).reset_index(drop=True)
    return (patient.sort_values(['true_label', 'score_range', 'patient_id']), selected)

def input_paths(project_root: Path) -> InputPaths:
    return InputPaths(classification=project_root / 'analysis/grouped_tasks/full_v1/predictions/classification_predictions.csv', survival=project_root / 'analysis/grouped_tasks/full_v1/predictions/survival_predictions.csv', task4_repeated_samples=project_root / 'analysis/bcr/bcr_case_input.csv', task4_time_audit=project_root / 'analysis/endpoint_controls/metadata/task4_patient_label_time_audit.csv')

def output_inventory(root: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'run_manifest.json':
            inventory[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    return inventory

def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    paths = input_paths(project_root)
    paths = InputPaths(classification=Path(args.classification) if args.classification else paths.classification,
                       survival=Path(args.survival) if args.survival else paths.survival,
                       task4_repeated_samples=Path(args.bcr_predictions) if args.bcr_predictions else paths.task4_repeated_samples,
                       task4_time_audit=Path(args.bcr_time_audit) if args.bcr_time_audit else paths.task4_time_audit)
    for path in paths.__dict__.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    bootstrap = min(args.bootstrap, 50) if args.smoke else args.bootstrap
    jobs = min(args.jobs, 2) if args.smoke else args.jobs
    classification = selected_classification_predictions(paths.classification)
    survival = selected_survival_predictions(paths.survival)
    units = prepare_analysis_units(classification, survival)
    if args.smoke:
        keep = (units['task'] == 'task1') & (units['dataset'] == 'task1') | (units['task'] == 'task3') & (units['dataset'] == 'Gide_PRJEB23709') | (units['task'] == 'task5') & (units['dataset'] == 'Gide_PRJEB23709')
        units = units[keep].copy()
    wide = wide_analysis_units(units)
    correlations = compute_pairwise_correlations(wide, bootstrap, args.random_state, jobs)
    dispersion = compute_dispersion_summary(wide, bootstrap, args.random_state)
    classified = add_classification_patterns(wide)
    agreement = compute_decision_agreement(classified, bootstrap, args.random_state)
    patterns = correctness_pattern_counts(classified)
    task1_population, task1_cases = select_task1_cases(classified)
    task3_population, task3_cases = select_task3_cases(classified)
    task4_population, task4_cases = prepare_task4_cases(paths.task4_repeated_samples, paths.task4_time_audit, args.smoke)
    files = {'metadata/analysis_units_long.csv': units, 'metadata/analysis_units_wide.csv': wide, 'metrics/pairwise_rank_correlations.csv': correlations, 'metrics/rank_dispersion_summary.csv': dispersion, 'metrics/classification_decision_agreement.csv': agreement, 'metrics/classification_correctness_patterns.csv': patterns, 'cases/task1_case_eligible_population.csv': task1_population, 'cases/task1_label_discordant_cases.csv': task1_cases, 'cases/task3_case_eligible_population.csv': task3_population, 'cases/task3_prespecified_high_disagreement_cases.csv': task3_cases, 'cases/task4_temporal_case_eligible_population.csv': task4_population, 'cases/task4_prespecified_temporal_cases.csv': task4_cases, 'source_data/fig8_panel_a_rank_correlations.csv': correlations, 'source_data/fig8_panel_b_decision_agreement.csv': agreement, 'source_data/fig8_panel_c_task1_cases.csv': task1_cases, 'source_data/fig8_panel_d_task3_cases.csv': task3_cases, 'source_data/fig8_panel_e_task4_cases.csv': task4_cases}
    for relative, frame in files.items():
        checked_write_csv(frame, output_root / relative)
    summary = {'status': 'PASS', 'run_kind': 'smoke' if args.smoke else 'formal', 'bootstrap_replicates': bootstrap, 'random_state': args.random_state, 'jobs': jobs, 'dataset_count': int(wide[['task', 'dataset']].drop_duplicates().shape[0]), 'analysis_unit_count': int(len(wide)), 'correlation_rows': int(len(correlations)), 'decision_agreement_rows': int(len(agreement)), 'task1_case_patients': int(task1_cases['cluster_id'].nunique()), 'task3_cases': int(len(task3_cases)), 'task4_case_patients': int(task4_cases['patient_id'].nunique())}
    checked_write_json(summary, output_root / 'logs/analysis_summary.json')
    manifest = {'analysis': 'quantitative disagreement and prespecified case audit', 'created_utc': datetime.now(timezone.utc).isoformat(), 'run_kind': 'smoke' if args.smoke else 'formal', 'project_root': str(project_root), 'output_root': str(output_root), 'bootstrap_replicates': bootstrap, 'random_state': args.random_state, 'metric_seed_rule': 'SHA-256 metric identifier plus base random state', 'jobs': jobs, 'python': platform.python_version(), 'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__}, 'inputs': {str(path): sha256_file(path) for path in paths.__dict__.values()}, 'outputs': output_inventory(output_root), 'analysis_script': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(Path(__file__).resolve())}, 'no_model_refit': True, 'no_embedding_extraction': True, 'no_llm_call': True, 'case_rules_fixed_before_selected_ids_were_inspected': True, 'external_validation_claim': False}
    checked_write_json(manifest, output_root / 'run_manifest.json')
    print(json.dumps(summary, sort_keys=True))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=_submission_path('project'))
    parser.add_argument('--classification', type=Path)
    parser.add_argument('--survival', type=Path)
    parser.add_argument('--bcr-predictions', type=Path, help='Corrected repeated BCR case input, not the pre-correction cache.')
    parser.add_argument('--bcr-time-audit', type=Path)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error('--bootstrap must be at least 20')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    return args
if __name__ == '__main__':
    run(parse_args())
