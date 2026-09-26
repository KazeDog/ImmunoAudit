"""Audit cancer context in the clinical-mutation survival analysis.

This program is deliberately prediction-only: it reuses frozen out-of-fold (OOF)
or fixed LLM scores and never trains a model, calls an API, or recomputes an
embedding.  Every output path is published with no-overwrite semantics.
"""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from lifelines.utils.concordance import _concordance_summary_statistics
from scipy.stats import kruskal
RANDOM_STATE = 42
N_PATIENTS = 1610
N_SPLITS = 5
BASELINE_MODEL = 'mutation_only_coxph_pca'
EXPECTED_COX_MODELS = (BASELINE_MODEL, 'cancer_type_only', 'clinical_tmb', 'mutation_pcs_cancer_type', 'mutation_pcs_clinical_tmb_cancer_type', 'mutation_pcs_clinical_tmb_stratified_cancer')
EXPECTED_CANCER_COUNTS = {'Non-Small Cell Lung Cancer': 344, 'Melanoma': 313, 'Bladder Cancer': 211, 'Renal Cell Carcinoma': 143, 'Head and Neck Cancer': 129, 'Esophagogastric Cancer': 118, 'Glioma': 116, 'Colorectal Cancer': 109, 'Cancer of Unknown Primary': 85, 'Breast Cancer': 41, 'Skin Cancer, Non-Melanoma': 1}
MIN_CANCER_N = 40
MIN_CANCER_EVENTS = 20
MIN_CANCER_PAIRS = 100
MIN_FOLDS_WITH_PAIRS = 4
OUTPUT_FILES = {'patient_cancer': 'metadata/task2_patient_cancer.csv', 'cohort_by_cancer': 'metadata/task2_cohort_by_cancer.csv', 'input_discovery': 'metadata/task2_input_discovery.csv', 'model_availability': 'metadata/task2_model_availability.csv', 'component_manifest': 'metadata/task2_component_manifest.json', 'folds': 'fold_assignments/task2_frozen_fold_assignments.csv', 'predictions': 'predictions/task2_existing_predictions_long.csv', 'fold_metrics': 'metrics/task2_fold_metrics.csv', 'global_summary': 'metrics/task2_global_summary.csv', 'per_cancer': 'metrics/task2_per_cancer_metrics.csv', 'risk_by_cancer': 'metrics/task2_risk_distribution_by_cancer.csv', 'risk_tests': 'metrics/task2_risk_distribution_tests.csv', 'bootstrap_absolute': 'metrics/task2_bootstrap_absolute_cindex.csv', 'bootstrap_paired': 'metrics/task2_bootstrap_paired_differences.csv', 'report_payload': 'metrics/task2_report_payload.json', 'log': 'logs/task2_cancer_context_audit.log'}

@dataclass(frozen=True)
class ConcordanceResult:
    c_index: float
    permissible_pairs: int
    concordant_pairs: int
    tied_prediction_pairs: int

@dataclass(frozen=True)
class DiscoveredInputs:
    x_path: Path
    y_path: Path
    clinical_sample_path: Path
    fold_path: Path
    cox_oof_path: Path
    fixed_prediction_paths: tuple[Path, ...]
    traditional_summary_paths: tuple[Path, ...]
    traditional_prediction_paths: tuple[Path, ...] = ()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _atomic_publish_no_overwrite(path: Path, writer: Any) -> None:
    """Write a complete temporary file, then hard-link it without clobbering."""
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        writer(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f'Refusing to overwrite existing output: {path}') from exc
    finally:
        temporary.unlink(missing_ok=True)

def write_csv_new(frame: pd.DataFrame, path: Path) -> None:
    _atomic_publish_no_overwrite(path, lambda tmp: frame.to_csv(tmp, index=False))

def write_json_new(payload: Any, path: Path) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    _atomic_publish_no_overwrite(path, lambda tmp: tmp.write_text(text, encoding='utf-8'))

def write_text_new(text: str, path: Path) -> None:
    _atomic_publish_no_overwrite(path, lambda tmp: tmp.write_text(text, encoding='utf-8'))

def assert_output_targets_are_new(output_root: Path) -> None:
    existing = [output_root / relative for relative in OUTPUT_FILES.values() if (output_root / relative).exists()]
    if existing:
        preview = '\n'.join((str(path) for path in existing[:10]))
        raise FileExistsError('Task 2 output preflight refused to overwrite existing files:\n' + preview)

def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Could not discover {label}; checked {len(seen)} candidates')

def _glob_files(roots_and_patterns: Sequence[tuple[Path, str]]) -> list[Path]:
    found: set[Path] = set()
    for root, pattern in roots_and_patterns:
        if root.exists():
            found.update((path.resolve() for path in root.glob(pattern) if path.is_file()))
    return sorted(found, key=lambda path: str(path))

def discover_inputs(project_root: Path, output_root: Path, *, x_path: Path | None=None, y_path: Path | None=None, clinical_sample_path: Path | None=None, fold_path: Path | None=None, cox_oof_path: Path | None=None) -> DiscoveredInputs:
    """Discover only narrowly scoped Task 2 inputs, in deterministic priority order."""
    project_root = project_root.resolve()
    x_candidates = [project_root / 'processed_data_strategy1/task2_X_final.csv', project_root / 'final/shared_data/processed_data_strategy1/task2_X_final.csv'] + _glob_files([(project_root / 'final', '**/task2_X_final.csv')])
    y_candidates = [project_root / 'processed_data_strategy1/task2_y_final.csv', project_root / 'final/shared_data/processed_data_strategy1/task2_y_final.csv'] + _glob_files([(project_root / 'final', '**/task2_y_final.csv')])
    clinical_candidates = [project_root / 'data_clinical_sample.txt', project_root / 'final/shared_data/data_clinical_sample.txt', Path(str(_submission_path('data', '2-/tmb_mskcc_2018/data_clinical_sample.txt'))), Path(str(_submission_path('data', '2-/tmb_mskcc_2018/data_clinical_sample.txt')))] + _glob_files([(project_root / 'final/shared_data', '**/data_clinical_sample.txt'), (project_root, 'cBioPortal/**/data_clinical_sample.txt')])
    fold_candidates = [project_root / 'results/task2_cancer_context/fold_assignments.csv', project_root / 'final/shared_data/task2_cancer_context/fold_assignments.csv'] + _glob_files([(project_root / 'results/task2_cancer_context', '**/fold_assignments.csv')])
    cox_candidates = [project_root / 'results/task2_cancer_context/full_v2_fixed_bootstrap/oof_predictions.csv', project_root / 'results/task2_cancer_context/oof_predictions.csv'] + _glob_files([(project_root / 'results/task2_cancer_context', '**/oof_predictions.csv')])
    resolved_x = x_path.resolve() if x_path else _first_existing(x_candidates, 'Task 2 X')
    resolved_y = y_path.resolve() if y_path else _first_existing(y_candidates, 'Task 2 y')
    resolved_clinical = clinical_sample_path.resolve() if clinical_sample_path else _first_existing(clinical_candidates, 'data_clinical_sample.txt')
    resolved_folds = fold_path.resolve() if fold_path else _first_existing(fold_candidates, 'frozen Task 2 folds')
    resolved_cox = cox_oof_path.resolve() if cox_oof_path else _first_existing(cox_candidates, 'six-model Cox OOF predictions')
    fixed_paths = _glob_files([(project_root / 'results', '**/task2_n1610_predictions.csv'), (project_root / 'final', '**/results/**/task2_n1610_predictions.csv'), (project_root / 'analysis/cancer_context/metadata/source_predictions', '**/*task2*predictions*.csv'), (output_root / 'metadata/source_predictions', '**/*task2*predictions*.csv')])
    traditional_paths = _glob_files([(project_root / 'results/strategy2', 'strategy2_*transformer_results.csv'), (project_root / 'results/strategy2', 'strategy2_tabpfn_results.csv'), (project_root / 'final/strategy2/results/strategy2', 'strategy2_*transformer_results.csv'), (project_root / 'final/strategy2/results/strategy2', 'strategy2_tabpfn_results.csv')])
    traditional_prediction_paths = _glob_files([(project_root / 'analysis/cancer_context/predictions', 'task2_strategy2_*_oof_predictions.csv'), (output_root / 'predictions', 'task2_strategy2_*_oof_predictions.csv'), (project_root / 'results', '**/task2_strategy2_*_oof_predictions.csv')])
    return DiscoveredInputs(x_path=resolved_x, y_path=resolved_y, clinical_sample_path=resolved_clinical, fold_path=resolved_folds, cox_oof_path=resolved_cox, fixed_prediction_paths=tuple(fixed_paths), traditional_summary_paths=tuple(traditional_paths), traditional_prediction_paths=tuple(traditional_prediction_paths))

def read_cbioportal_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', comment='#', low_memory=False)

def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing columns: {missing}')

def load_and_validate_metadata(inputs: DiscoveredInputs, *, expected_n: int=N_PATIENTS, expected_counts: dict[str, int] | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical labels/cancer/folds and enforce the frozen 1610-person cohort."""
    if expected_counts is None:
        expected_counts = EXPECTED_CANCER_COUNTS
    x = pd.read_csv(inputs.x_path, usecols=['PATIENT_ID'])
    y = pd.read_csv(inputs.y_path)
    clinical = read_cbioportal_table(inputs.clinical_sample_path)
    folds = pd.read_csv(inputs.fold_path)
    _require_columns(y, ['PATIENT_ID', 'duration', 'event'], 'Task 2 y')
    _require_columns(clinical, ['PATIENT_ID', 'CANCER_TYPE'], 'clinical sample')
    _require_columns(folds, ['PATIENT_ID', 'fold'], 'frozen folds')
    for label, frame in (('X', x), ('y', y), ('folds', folds)):
        if len(frame) != expected_n:
            raise AssertionError(f'Task 2 {label} has {len(frame)} rows, expected {expected_n}')
        if frame['PATIENT_ID'].astype(str).duplicated().any():
            raise AssertionError(f'Task 2 {label} contains duplicate PATIENT_ID values')
    x_ids = x['PATIENT_ID'].astype(str).tolist()
    y_ids = y['PATIENT_ID'].astype(str).tolist()
    if x_ids != y_ids:
        raise AssertionError('Canonical Task 2 X/y patient order differs')
    if set(folds['PATIENT_ID'].astype(str)) != set(y_ids):
        raise AssertionError('Frozen folds do not cover exactly the canonical Task 2 patients')
    if 'original_row_position_0based' in folds:
        ordered = folds.sort_values('original_row_position_0based')['PATIENT_ID'].astype(str).tolist()
        if ordered != y_ids:
            raise AssertionError('Frozen fold original-row positions do not reproduce canonical order')
    fold_values = pd.to_numeric(folds['fold'], errors='raise').astype(int)
    if set(fold_values) != set(range(1, N_SPLITS + 1)):
        raise AssertionError(f'Frozen folds are not exactly 1..{N_SPLITS}')
    if 'n_splits' in folds and set(pd.to_numeric(folds['n_splits']).astype(int)) != {N_SPLITS}:
        raise AssertionError('Frozen fold file does not declare n_splits=5')
    if 'random_state' in folds and set(pd.to_numeric(folds['random_state']).astype(int)) != {RANDOM_STATE}:
        raise AssertionError('Frozen fold file does not declare random_state=42')
    clinical = clinical[[column for column in ['PATIENT_ID', 'SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'ONCOTREE_CODE', 'PRIMARY_SITE'] if column in clinical.columns]].copy()
    clinical['PATIENT_ID'] = clinical['PATIENT_ID'].astype(str)
    cancer_conflicts = clinical.groupby('PATIENT_ID')['CANCER_TYPE'].nunique(dropna=False)
    cancer_conflicts = cancer_conflicts[cancer_conflicts > 1]
    if not cancer_conflicts.empty:
        raise AssertionError('Clinical sample table assigns multiple cancer types to Task 2 patients: ' + ', '.join(cancer_conflicts.index[:5]))
    clinical_patient = clinical.drop_duplicates('PATIENT_ID', keep='first')
    metadata = y.copy()
    metadata['PATIENT_ID'] = metadata['PATIENT_ID'].astype(str)
    metadata['duration'] = pd.to_numeric(metadata['duration'], errors='raise').astype(float)
    metadata['event'] = pd.to_numeric(metadata['event'], errors='raise').astype(int)
    if not set(metadata['event'].unique()).issubset({0, 1}):
        raise AssertionError('Task 2 event is not binary')
    if (~np.isfinite(metadata['duration']) | (metadata['duration'] < 0)).any():
        raise AssertionError('Task 2 duration contains invalid values')
    metadata = metadata.merge(clinical_patient, on='PATIENT_ID', how='left', validate='one_to_one')
    metadata = metadata.merge(folds[['PATIENT_ID', 'fold']].assign(PATIENT_ID=lambda d: d.PATIENT_ID.astype(str)), on='PATIENT_ID', how='left', validate='one_to_one')
    if metadata['CANCER_TYPE'].isna().any():
        missing = metadata.loc[metadata['CANCER_TYPE'].isna(), 'PATIENT_ID'].tolist()
        raise AssertionError(f'Not all Task 2 patients matched cancer type: {missing[:5]}')
    observed_counts = metadata['CANCER_TYPE'].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise AssertionError(f'Task 2 cancer counts differ from the prespecified cohort; observed={observed_counts}; expected={expected_counts}')
    if len(metadata) != expected_n or metadata['PATIENT_ID'].nunique() != expected_n:
        raise AssertionError(f'Final Task 2 cohort is not exactly {expected_n} patients')
    metadata['fold'] = metadata['fold'].astype(int)
    return (metadata, folds.copy(), clinical_patient)

def concordance_with_pairs(duration: Sequence[float] | np.ndarray | pd.Series, event: Sequence[int] | np.ndarray | pd.Series, risk_score: Sequence[float] | np.ndarray | pd.Series) -> ConcordanceResult:
    """Harrell C where a larger score means earlier event / higher risk."""
    duration_array = np.asarray(duration, dtype=float)
    event_array = np.asarray(event, dtype=int)
    risk_array = np.asarray(risk_score, dtype=float)
    if not len(duration_array) == len(event_array) == len(risk_array):
        raise ValueError('duration, event and risk_score lengths differ')
    if len(duration_array) == 0:
        return ConcordanceResult(math.nan, 0, 0, 0)
    if not (np.isfinite(duration_array).all() and np.isfinite(event_array).all() and np.isfinite(risk_array).all()):
        raise ValueError('Concordance inputs contain non-finite values')
    if not set(np.unique(event_array)).issubset({0, 1}):
        raise ValueError('event must be binary')
    correct, tied, pairs = _concordance_summary_statistics(duration_array, -risk_array, event_array)
    pairs_int = int(pairs)
    c_index = math.nan if pairs_int == 0 else float((float(correct) + 0.5 * float(tied)) / float(pairs))
    return ConcordanceResult(c_index, pairs_int, int(correct), int(tied))

def legacy_fixed_llm_c_index(duration: Sequence[float] | np.ndarray | pd.Series, event: Sequence[int] | np.ndarray | pd.Series, risk_score: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Exactly reproduce the historical Strategy 3 Task 2 pair loop.

    The legacy implementation omitted equal-time event/censor pairs.  We retain
    this value only for exact old-result reproduction; new cancer-context
    estimates use the lifelines-compatible Harrell implementation above.
    """
    times = np.asarray(duration, dtype=float)
    status = np.asarray(event, dtype=int)
    scores = np.asarray(risk_score, dtype=float)
    concordant = 0.0
    permissible = 0.0
    tied = 0.0
    for first in range(len(times)):
        later_time = times[first + 1:]
        later_status = status[first + 1:]
        later_score = scores[first + 1:]
        first_earlier = (times[first] < later_time) & (status[first] == 1)
        later_earlier = (later_time < times[first]) & (later_status == 1)
        permissible += float(first_earlier.sum() + later_earlier.sum())
        concordant += float(np.sum(first_earlier & (scores[first] > later_score)))
        concordant += float(np.sum(later_earlier & (later_score > scores[first])))
        tied += float(np.sum(first_earlier & (scores[first] == later_score)))
        tied += float(np.sum(later_earlier & (later_score == scores[first])))
    return math.nan if permissible == 0 else float((concordant + 0.5 * tied) / permissible)

def aggregate_concordance(parts: Iterable[ConcordanceResult]) -> ConcordanceResult:
    parts = list(parts)
    pairs = int(sum((part.permissible_pairs for part in parts)))
    correct = int(sum((part.concordant_pairs for part in parts)))
    tied = int(sum((part.tied_prediction_pairs for part in parts)))
    c_index = math.nan if pairs == 0 else float((correct + 0.5 * tied) / pairs)
    return ConcordanceResult(c_index, pairs, correct, tied)

def grouped_concordance(frame: pd.DataFrame, group_columns: Sequence[str]) -> ConcordanceResult:
    if not group_columns:
        return concordance_with_pairs(frame.duration, frame.event, frame.risk_score)
    parts = [concordance_with_pairs(group.duration, group.event, group.risk_score) for _, group in frame.groupby(list(group_columns), sort=True, dropna=False)]
    return aggregate_concordance(parts)

def _km_median(duration: pd.Series, observed: pd.Series) -> float:
    """Kaplan-Meier median; returns NA when the curve never reaches 0.5."""
    times = np.asarray(duration, dtype=float)
    status = np.asarray(observed, dtype=int)
    survival = 1.0
    for time in np.sort(np.unique(times)):
        at_risk = int(np.sum(times >= time))
        events = int(np.sum((times == time) & (status == 1)))
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return float(time)
    return math.nan

def cohort_by_cancer(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cancer, group in metadata.groupby('CANCER_TYPE', sort=False):
        events = int(group.event.sum())
        rows.append({'cancer_type': cancer, 'n': int(len(group)), 'events': events, 'censored': int(len(group) - events), 'event_rate': float(events / len(group)), 'median_observed_time': float(group.duration.median()), 'median_survival_time_km': _km_median(group.duration, group.event), 'median_follow_up_time_reverse_km': _km_median(group.duration, 1 - group.event)})
    return pd.DataFrame(rows)

def _read_first_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}

def _prompt_version_from_path(path: Path) -> str | None:
    text = str(path).lower()
    matches = re.findall('prompt[_-]?v(\\d+)', text)
    if not matches:
        return None
    digits = matches[-1]
    if digits == '21':
        return 'v2.1'
    if digits == '221':
        return 'v2.2.1'
    return 'v' + '.'.join(digits)

def infer_fixed_prediction_identity(path: Path) -> dict[str, Any]:
    text = str(path).lower()
    raw_path = path.with_name(path.name.replace('_predictions.csv', '_raw.jsonl'))
    raw = _read_first_jsonl(raw_path)
    result = raw.get('result', {}) if isinstance(raw.get('result'), dict) else {}
    provider = str(result.get('provider', '')).strip().lower()
    model = str(result.get('model', '')).strip()
    if not provider:
        provider = next((name for name in ('qwen', 'kimi', 'minimax') if name in text), 'unknown')
    summary_path = path.with_name(path.name.replace('_predictions.csv', '_summary.json'))
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding='utf-8'))
            summary = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            summary = {}
    if not model:
        model = {'qwen': 'Qwen3.5-Plus' if 'qwen35' in text or 'qwen3' in text else 'qwen_unknown', 'kimi': 'Kimi-K2.6' if 'k26' in text or 'k2.6' in text else 'kimi_unknown', 'minimax': 'MiniMax-M2.7' if 'm27' in text or 'm2.7' in text else 'minimax_unknown'}.get(provider, 'unknown_model')
    prompt_version = summary.get('prompt_version') or _prompt_version_from_path(path)
    if prompt_version is None:
        compact_version = re.search('(?:^|[_-])v(21|221)(?:[_-]|$)', path.stem.lower())
        if compact_version:
            prompt_version = {'21': 'v2.1', '221': 'v2.2.1'}[compact_version.group(1)]
    run_name = path.parent.name
    clean_model = re.sub('[^a-z0-9]+', '_', model.lower()).strip('_')
    clean_run = re.sub('[^a-z0-9]+', '_', run_name.lower()).strip('_')
    model_id = 'fixed_llm__' + '__'.join((part for part in (provider, clean_model, prompt_version, clean_run) if part))
    return {'provider': provider, 'llm_model': model, 'prompt_version': prompt_version, 'run_name': run_name, 'model_id': model_id, 'raw_metadata_path': str(raw_path) if raw_path.is_file() else None}

def _companion_reference_cindex(path: Path) -> tuple[float, str | None]:
    summary_path = path.with_name(path.name.replace('_predictions.csv', '_summary.json'))
    if not summary_path.is_file():
        return (math.nan, None)
    try:
        payload = json.loads(summary_path.read_text(encoding='utf-8'))
        value = float(payload['c_index'])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return (math.nan, str(summary_path))
    return (value, str(summary_path))

def _normalise_prediction_frame(raw: pd.DataFrame, metadata: pd.DataFrame, *, model_id: str, source_path: Path, source_kind: str) -> pd.DataFrame:
    id_column = next((column for column in ('PATIENT_ID', 'sample_id', 'patient_id') if column in raw), None)
    if id_column is None or 'risk_score' not in raw:
        raise ValueError('patient ID or risk_score column is absent')
    selected = raw[[id_column, 'risk_score']].rename(columns={id_column: 'PATIENT_ID'}).copy()
    selected['PATIENT_ID'] = selected['PATIENT_ID'].astype(str)
    if selected['PATIENT_ID'].duplicated().any():
        raise ValueError('duplicate patient predictions')
    selected['risk_score'] = pd.to_numeric(selected['risk_score'], errors='raise').astype(float)
    if not np.isfinite(selected['risk_score']).all():
        raise ValueError('non-finite risk scores')
    expected_ids = set(metadata.PATIENT_ID)
    observed_ids = set(selected.PATIENT_ID)
    if observed_ids != expected_ids:
        raise ValueError(f'patient coverage differs: n={len(observed_ids)}, missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}')
    selected = metadata[['PATIENT_ID', 'duration', 'event', 'fold', 'CANCER_TYPE']].merge(selected, on='PATIENT_ID', how='left', validate='one_to_one')
    selected['model'] = model_id
    selected['source_kind'] = source_kind
    selected['source_path'] = str(source_path)
    selected['score_orientation'] = 'higher_is_earlier_event_higher_risk'
    return selected[['PATIENT_ID', 'fold', 'model', 'risk_score', 'score_orientation', 'duration', 'event', 'CANCER_TYPE', 'source_kind', 'source_path']]

def load_existing_predictions(inputs: DiscoveredInputs, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Load six Cox OOF and every distinct complete fixed Task 2 prediction file."""
    availability: list[dict[str, Any]] = []
    discovery: list[dict[str, Any]] = []
    pieces: list[pd.DataFrame] = []
    cox = pd.read_csv(inputs.cox_oof_path)
    _require_columns(cox, ['PATIENT_ID', 'fold', 'model', 'risk_score', 'duration', 'event', 'CANCER_TYPE'], 'six-model Cox OOF')
    observed_cox_models = set(cox.model.astype(str))
    if observed_cox_models != set(EXPECTED_COX_MODELS):
        raise AssertionError(f'Cox OOF model set differs; observed={sorted(observed_cox_models)}')
    for model in EXPECTED_COX_MODELS:
        raw_model = cox.loc[cox.model.astype(str) == model, ['PATIENT_ID', 'risk_score']]
        normalised = _normalise_prediction_frame(raw_model, metadata, model_id=model, source_path=inputs.cox_oof_path, source_kind='existing_frozen_oof')
        source_check = cox.loc[cox.model.astype(str) == model].copy()
        source_check['PATIENT_ID'] = source_check.PATIENT_ID.astype(str)
        check = source_check.merge(metadata[['PATIENT_ID', 'fold', 'duration', 'event', 'CANCER_TYPE']], on='PATIENT_ID', suffixes=('_source', '_canonical'), validate='one_to_one')
        for column in ('fold', 'duration', 'event', 'CANCER_TYPE'):
            if not np.all(check[f'{column}_source'].astype(str) == check[f'{column}_canonical'].astype(str)):
                raise AssertionError(f'Cox OOF {model} has noncanonical {column}')
        pieces.append(normalised)
        availability.append({'model': model, 'model_family': 'cox_ablation', 'status': 'available_complete_prediction', 'reason': None, 'source_path': str(inputs.cox_oof_path), 'source_sha256': sha256_file(inputs.cox_oof_path), 'n_predictions': len(normalised), 'provider': None, 'llm_model': None, 'prompt_version': None})
    seen_hashes: dict[str, Path] = {}
    seen_model_ids: set[str] = set(EXPECTED_COX_MODELS)
    observed_providers: set[str] = set()
    for path in inputs.fixed_prediction_paths:
        file_hash = sha256_file(path)
        identity = infer_fixed_prediction_identity(path)
        base_record = {'artifact_type': 'fixed_task2_prediction', 'path': str(path), 'sha256': file_hash, **identity}
        if file_hash in seen_hashes:
            discovery.append({**base_record, 'status': 'duplicate_identical_content', 'reason': f'identical to {seen_hashes[file_hash]}'})
            continue
        seen_hashes[file_hash] = path
        model_id = identity['model_id']
        if model_id in seen_model_ids:
            model_id += '__sha_' + file_hash[:8]
        seen_model_ids.add(model_id)
        try:
            raw = pd.read_csv(path)
            normalised = _normalise_prediction_frame(raw, metadata, model_id=model_id, source_path=path, source_kind='existing_fixed_llm_prediction')
            if 'duration' in raw and 'event' in raw:
                id_column = next((column for column in ('PATIENT_ID', 'sample_id', 'patient_id') if column in raw))
                source_labels = raw[[id_column, 'duration', 'event']].rename(columns={id_column: 'PATIENT_ID'})
                source_labels.PATIENT_ID = source_labels.PATIENT_ID.astype(str)
                label_check = metadata[['PATIENT_ID', 'duration', 'event']].merge(source_labels, on='PATIENT_ID', suffixes=('_canonical', '_source'), validate='one_to_one')
                if not np.allclose(label_check.duration_canonical, label_check.duration_source):
                    raise ValueError('embedded duration differs from canonical y')
                if not np.array_equal(label_check.event_canonical.astype(int), label_check.event_source.astype(int)):
                    raise ValueError('embedded event differs from canonical y')
            pieces.append(normalised)
            observed_providers.add(identity['provider'])
            reference_c, reference_path = _companion_reference_cindex(path)
            availability.append({'model': model_id, 'model_family': 'fixed_llm', 'status': 'available_complete_prediction', 'reason': None, 'source_path': str(path), 'source_sha256': file_hash, 'n_predictions': len(normalised), 'provider': identity['provider'], 'llm_model': identity['llm_model'], 'prompt_version': identity['prompt_version'], 'existing_reference_c_index': reference_c, 'existing_reference_path': reference_path})
            discovery.append({**base_record, 'model_id': model_id, 'status': 'selected', 'reason': None})
        except Exception as exc:
            discovery.append({**base_record, 'status': 'incompatible_prediction', 'reason': str(exc)})
            availability.append({'model': model_id, 'model_family': 'fixed_llm', 'status': 'unavailable_incompatible_prediction', 'reason': str(exc), 'source_path': str(path), 'source_sha256': file_hash, 'n_predictions': 0, 'provider': identity['provider'], 'llm_model': identity['llm_model'], 'prompt_version': identity['prompt_version']})
    for provider in ('qwen', 'kimi', 'minimax'):
        if provider not in observed_providers:
            availability.append({'model': f'fixed_llm__{provider}__not_found', 'model_family': 'fixed_llm', 'status': 'unavailable_source_prediction_not_found', 'reason': 'No distinct complete task2_n1610 patient-level prediction was discovered', 'source_path': None, 'source_sha256': None, 'n_predictions': 0, 'provider': provider, 'llm_model': None, 'prompt_version': None})
    traditional_with_oof: set[str] = set()
    for path in inputs.traditional_prediction_paths:
        file_hash = sha256_file(path)
        try:
            raw = pd.read_csv(path)
            _require_columns(raw, ['PATIENT_ID', 'fold', 'duration', 'event', 'model', 'corrected_risk_score_negative_predicted_duration'], 'Strategy 2 frozen-fold OOF')
            slugs = raw.model.astype(str).str.lower().unique().tolist()
            if len(slugs) != 1 or slugs[0] not in {'fttransformer', 'tabpfn'}:
                raise ValueError(f'unexpected Strategy 2 model values: {slugs}')
            slug = slugs[0]
            availability_key = 'ft_transformer' if slug == 'fttransformer' else 'tabpfn'
            model_id = f'strategy2_{slug}_corrected_duration_risk'
            source_scores = raw[['PATIENT_ID', 'corrected_risk_score_negative_predicted_duration']].rename(columns={'corrected_risk_score_negative_predicted_duration': 'risk_score'})
            normalised = _normalise_prediction_frame(source_scores, metadata, model_id=model_id, source_path=path, source_kind='rerun_frozen_oof')
            source_check = raw.assign(PATIENT_ID=lambda d: d.PATIENT_ID.astype(str)).merge(metadata[['PATIENT_ID', 'fold', 'duration', 'event']], on='PATIENT_ID', suffixes=('_source', '_canonical'), validate='one_to_one')
            for column in ('fold', 'duration', 'event'):
                if not np.allclose(pd.to_numeric(source_check[f'{column}_source']), pd.to_numeric(source_check[f'{column}_canonical'])):
                    raise ValueError(f'Strategy 2 OOF has noncanonical {column}')
            pieces.append(normalised)
            traditional_with_oof.add(availability_key)
            availability.append({'model': model_id, 'model_family': 'traditional_downstream', 'status': 'available_complete_prediction', 'reason': None, 'source_path': str(path), 'source_sha256': file_hash, 'n_predictions': len(normalised), 'provider': None, 'llm_model': None, 'prompt_version': None, 'score_note': 'corrected risk = -predicted duration; historical reversed orientation remains in the source artifact'})
            discovery.append({'artifact_type': 'traditional_frozen_oof_prediction', 'path': str(path), 'sha256': file_hash, 'model_id': model_id, 'status': 'selected', 'reason': None})
        except Exception as exc:
            discovery.append({'artifact_type': 'traditional_frozen_oof_prediction', 'path': str(path), 'sha256': file_hash, 'model_id': path.stem, 'status': 'incompatible_prediction', 'reason': str(exc)})
            availability.append({'model': path.stem, 'model_family': 'traditional_downstream', 'status': 'unavailable_incompatible_or_incomplete_oof', 'reason': str(exc), 'source_path': str(path), 'source_sha256': file_hash, 'n_predictions': 0, 'provider': None, 'llm_model': None, 'prompt_version': None})
    traditional_seen: set[str] = set()
    for path in inputs.traditional_summary_paths:
        lower = path.name.lower()
        model = 'ft_transformer' if 'fttransformer' in lower else 'tabpfn' if 'tabpfn' in lower else path.stem
        if model in traditional_seen:
            discovery.append({'artifact_type': 'traditional_fold_summary', 'path': str(path), 'sha256': sha256_file(path), 'model_id': model, 'status': 'duplicate_summary_location', 'reason': 'a preferred summary for this model was already recorded'})
            continue
        traditional_seen.add(model)
        if model in traditional_with_oof:
            discovery.append({'artifact_type': 'traditional_fold_summary', 'path': str(path), 'sha256': sha256_file(path), 'model_id': model, 'status': 'legacy_summary_supplemented_by_frozen_oof', 'reason': 'patient-level frozen-fold rerun is available for cancer-context analysis'})
            continue
        availability.append({'model': model, 'model_family': 'traditional_downstream', 'status': 'unavailable_patient_level_oof_not_saved', 'reason': 'Only legacy fold-level C-index summary exists; no patient-level risk score', 'source_path': str(path), 'source_sha256': sha256_file(path), 'n_predictions': 0, 'provider': None, 'llm_model': None, 'prompt_version': None})
        discovery.append({'artifact_type': 'traditional_fold_summary', 'path': str(path), 'sha256': sha256_file(path), 'model_id': model, 'status': 'summary_only_no_patient_predictions', 'reason': 'cannot calculate cancer-stratified or patient-bootstrap metrics'})
    predictions = pd.concat(pieces, ignore_index=True)
    if predictions.duplicated(['PATIENT_ID', 'model']).any():
        raise AssertionError('Combined Task 2 predictions are not unique per patient/model')
    counts = predictions.groupby('model').PATIENT_ID.nunique()
    if not (counts == N_PATIENTS).all():
        raise AssertionError(f'Loaded models are not complete: {counts.to_dict()}')
    return (predictions, pd.DataFrame(availability), discovery)

def build_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, fold), group in predictions.groupby(['model', 'fold'], sort=True):
        global_result = concordance_with_pairs(group.duration, group.event, group.risk_score)
        within_result = grouped_concordance(group, ['CANCER_TYPE'])
        rows.append({'model': model, 'fold': int(fold), 'n': int(len(group)), 'events': int(group.event.sum()), 'global_c_index': global_result.c_index, 'global_permissible_pairs': global_result.permissible_pairs, 'within_cancer_pair_pooled_c_index': within_result.c_index, 'within_cancer_permissible_pairs': within_result.permissible_pairs})
    return pd.DataFrame(rows)

def build_per_cancer_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, cancer), group in predictions.groupby(['model', 'CANCER_TYPE'], sort=True):
        concatenated = concordance_with_pairs(group.duration, group.event, group.risk_score)
        crossfitted_parts = [concordance_with_pairs(fold_group.duration, fold_group.event, fold_group.risk_score) for _, fold_group in group.groupby('fold', sort=True)]
        crossfitted = aggregate_concordance(crossfitted_parts)
        folds_with_pairs = int(sum((part.permissible_pairs > 0 for part in crossfitted_parts)))
        n = int(len(group))
        events = int(group.event.sum())
        is_fixed_score = str(model).startswith('fixed_llm__')
        primary = concatenated if is_fixed_score else crossfitted
        reportable = bool(n >= MIN_CANCER_N and events >= MIN_CANCER_EVENTS and (primary.permissible_pairs >= MIN_CANCER_PAIRS) and (is_fixed_score or folds_with_pairs >= MIN_FOLDS_WITH_PAIRS) and np.isfinite(primary.c_index))
        rows.append({'model': model, 'cancer_type': cancer, 'n': n, 'events': events, 'censored': n - events, 'c_index': primary.c_index if reportable else math.nan, 'c_index_reportable': reportable, 'score_scope': 'single_fixed_full_cohort_score_scale_same_cancer_pairs' if is_fixed_score else 'same_test_fold_and_same_cancer_pairs', 'cross_fold_comparability': 'comparable_single_fixed_score_scale' if is_fixed_score else 'not_assumed_fold_specific_fitted_score_scales', 'permissible_pairs': primary.permissible_pairs, 'raw_crossfitted_c_index': crossfitted.c_index, 'crossfitted_permissible_pairs': crossfitted.permissible_pairs, 'folds_with_permissible_pairs': folds_with_pairs, 'raw_concatenated_oof_c_index': concatenated.c_index, 'concatenated_oof_permissible_pairs': concatenated.permissible_pairs, 'status': 'ok' if reportable else 'not_reportable_low_support', 'reporting_rule': f'n>={MIN_CANCER_N};events>={MIN_CANCER_EVENTS};pairs>={MIN_CANCER_PAIRS}' + ('' if is_fixed_score else f';folds_with_pairs>={MIN_FOLDS_WITH_PAIRS}')})
    return pd.DataFrame(rows)

def _legacy_reference_for_model(model: str, availability: pd.DataFrame) -> tuple[float, str]:
    if model == BASELINE_MODEL:
        return (0.5541684629731131, 'legacy_mean_of_five_fold_c_index')
    matched = availability[(availability.model == model) & availability.existing_reference_c_index.notna()] if 'existing_reference_c_index' in availability else pd.DataFrame()
    if not matched.empty:
        return (float(matched.iloc[0].existing_reference_c_index), 'legacy_task2_pair_loop_global_c_index')
    return (math.nan, 'not_available')

def build_global_summary(predictions: pd.DataFrame, fold_metrics: pd.DataFrame, per_cancer: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby('model', sort=True):
        concatenated = concordance_with_pairs(group.duration, group.event, group.risk_score)
        crossfitted = grouped_concordance(group, ['fold'])
        within_all = grouped_concordance(group, ['CANCER_TYPE'])
        within_crossfitted = grouped_concordance(group, ['fold', 'CANCER_TYPE'])
        fold_rows = fold_metrics[fold_metrics.model == model]
        cancer_rows = per_cancer[(per_cancer.model == model) & per_cancer.c_index_reportable]
        if cancer_rows.empty:
            reportable_pair_weighted = math.nan
            reportable_sample_weighted = math.nan
        else:
            reportable_pair_weighted = float(np.average(cancer_rows.c_index, weights=cancer_rows.permissible_pairs))
            reportable_sample_weighted = float(np.average(cancer_rows.c_index, weights=cancer_rows.n))
        reference, reference_scope = _legacy_reference_for_model(model, availability)
        legacy_llm_value = legacy_fixed_llm_c_index(group.duration, group.event, group.risk_score) if str(model).startswith('fixed_llm__') else math.nan
        reproduced_value = float(fold_rows.global_c_index.mean()) if reference_scope == 'legacy_mean_of_five_fold_c_index' else legacy_llm_value
        tolerance = 0.001 if np.isfinite(reference) else math.nan
        is_fixed_score = str(model).startswith('fixed_llm__')
        is_stratified_cox = model == 'mutation_pcs_clinical_tmb_stratified_cancer'
        primary_global = concatenated.c_index if is_fixed_score else math.nan if is_stratified_cox else crossfitted.c_index
        primary_within = within_all.c_index if is_fixed_score else within_crossfitted.c_index
        rows.append({'model': model, 'n': int(len(group)), 'events': int(group.event.sum()), 'n_folds': int(group.fold.nunique()), 'score_orientation': 'higher_is_earlier_event_higher_risk', 'global_concatenated_c_index': concatenated.c_index, 'global_concatenated_permissible_pairs': concatenated.permissible_pairs, 'global_crossfitted_pair_pooled_c_index': crossfitted.c_index, 'global_crossfitted_permissible_pairs': crossfitted.permissible_pairs, 'legacy_fixed_llm_global_c_index': legacy_llm_value, 'fold_mean_c_index': float(fold_rows.global_c_index.mean()), 'fold_sd_c_index_ddof1': float(fold_rows.global_c_index.std(ddof=1)), 'within_cancer_all_oof_pair_pooled_c_index': within_all.c_index, 'within_cancer_all_oof_permissible_pairs': within_all.permissible_pairs, 'within_cancer_crossfitted_pair_pooled_c_index': within_crossfitted.c_index, 'within_cancer_crossfitted_permissible_pairs': within_crossfitted.permissible_pairs, 'within_cancer_reportable_pair_weighted_c_index': reportable_pair_weighted, 'within_cancer_reportable_sample_weighted_c_index': reportable_sample_weighted, 'within_cancer_reportable_cancers': int(len(cancer_rows)), 'primary_global_c_index': primary_global, 'primary_global_score_scope': 'single_fixed_full_cohort_score_scale_all_comparable_pairs' if is_fixed_score else 'not_applicable_cross_stratum_baseline_hazards' if is_stratified_cox else 'same_test_fold_pairs_only', 'primary_within_cancer_c_index': primary_within, 'primary_within_cancer_score_scope': 'single_fixed_full_cohort_score_scale_same_cancer_pairs' if is_fixed_score else 'same_test_fold_and_same_cancer_pairs', 'cross_fold_comparability': 'comparable_single_fixed_score_scale' if is_fixed_score else 'not_assumed_fold_specific_fitted_score_scales', 'existing_reference_c_index': reference, 'existing_reference_scope': reference_scope, 'reproduction_value': reproduced_value, 'reproduction_absolute_difference': abs(reproduced_value - reference) if np.isfinite(reference) else math.nan, 'reproduction_tolerance': tolerance, 'existing_result_reproduced': bool(abs(reproduced_value - reference) <= tolerance) if np.isfinite(reference) else pd.NA, 'global_score_caveat': 'cross-stratum Cox scores are not globally comparable; global C is descriptive' if model == 'mutation_pcs_clinical_tmb_stratified_cancer' else 'concatenated scores may be fold-scale sensitive; crossfitted result is primary' if not model.startswith('fixed_llm__') else 'fixed score scale; concatenated C reproduces the original fixed-prediction analysis'})
    return pd.DataFrame(rows)

def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted

def build_risk_distribution_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for model, model_group in predictions.groupby('model', sort=True):
        arrays: list[np.ndarray] = []
        for cancer, group in model_group.groupby('CANCER_TYPE', sort=True):
            values = group.risk_score.to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            rows.append({'model': model, 'cancer_type': cancer, 'n': len(values), 'risk_mean': float(values.mean()), 'risk_sd_ddof1': float(values.std(ddof=1)) if len(values) > 1 else math.nan, 'risk_median': float(median), 'risk_q1': float(q1), 'risk_q3': float(q3), 'risk_iqr': float(q3 - q1), 'risk_min': float(values.min()), 'risk_max': float(values.max())})
            arrays.append(values)
        try:
            if np.unique(model_group.risk_score).size <= 1:
                h_statistic, p_value = (0.0, 1.0)
            else:
                h_statistic, p_value = (float(value) for value in kruskal(*arrays))
            k = len(arrays)
            n = len(model_group)
            epsilon_squared = max(0.0, float((h_statistic - k + 1) / (n - k)))
            status = 'ok_exploratory'
        except ValueError as exc:
            h_statistic = p_value = epsilon_squared = math.nan
            status = f'not_estimable: {exc}'
        tests.append({'model': model, 'test': 'Kruskal-Wallis risk score by cancer type', 'n': int(len(model_group)), 'cancer_groups': int(len(arrays)), 'h_statistic': h_statistic, 'p_value': p_value, 'epsilon_squared': epsilon_squared, 'status': status, 'interpretation': 'exploratory omnibus distribution test'})
    tests_frame = pd.DataFrame(tests)
    tests_frame['p_value_bh_fdr_across_models'] = _bh_adjust(tests_frame.p_value)
    return (pd.DataFrame(rows), tests_frame)
BOOTSTRAP_ESTIMANDS = ('global', 'within_cancer')

def _model_scope(model: str, estimand: str) -> tuple[list[str], str, bool]:
    """Return the scientifically valid pair scope for each score source."""
    is_fixed_score = str(model).startswith('fixed_llm__')
    if estimand == 'global':
        if model == 'mutation_pcs_clinical_tmb_stratified_cancer':
            return ([], 'not_applicable_cross_stratum_baseline_hazards', False)
        if is_fixed_score:
            return ([], 'single_fixed_full_cohort_score_scale_all_comparable_pairs', True)
        return (['fold'], 'same_test_fold_pairs_only', True)
    if estimand == 'within_cancer':
        if is_fixed_score:
            return (['CANCER_TYPE'], 'single_fixed_full_cohort_score_scale_same_cancer_pairs', True)
        return (['fold', 'CANCER_TYPE'], 'same_test_fold_and_same_cancer_pairs', True)
    raise KeyError(estimand)

def _estimand_score(frame: pd.DataFrame, model: str, estimand: str) -> float:
    group_columns, _, appropriate = _model_scope(model, estimand)
    return grouped_concordance(frame, group_columns).c_index if appropriate else math.nan

def _paired_common_scope(estimand: str) -> tuple[list[str], str]:
    """Common permissible-pair set required for a valid paired model delta."""
    if estimand == 'global':
        return (['fold'], 'common_same_test_fold_pairs_only')
    if estimand == 'within_cancer':
        return (['fold', 'CANCER_TYPE'], 'common_same_test_fold_and_same_cancer_pairs')
    raise KeyError(estimand)

def patient_bootstrap(predictions: pd.DataFrame, *, n_bootstrap: int, random_state: int=RANDOM_STATE, baseline_model: str=BASELINE_MODEL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired patient bootstrap; each patient (not row/pair) is one sampling unit."""
    if n_bootstrap < 1:
        return (pd.DataFrame(), pd.DataFrame())
    model_names = sorted(predictions.model.unique())
    if baseline_model not in model_names:
        raise ValueError(f'Bootstrap baseline is absent: {baseline_model}')
    metadata_columns = ['PATIENT_ID', 'duration', 'event', 'fold', 'CANCER_TYPE']
    metadata = predictions[metadata_columns].drop_duplicates().sort_values('PATIENT_ID').reset_index(drop=True)
    if not metadata.PATIENT_ID.is_unique:
        raise ValueError('Cross-model patient metadata is inconsistent')
    risk_wide = predictions.pivot(index='PATIENT_ID', columns='model', values='risk_score')
    wide = metadata.merge(risk_wide, left_on='PATIENT_ID', right_index=True, validate='one_to_one')
    if wide[model_names].isna().any().any():
        raise ValueError('Models do not share complete patient coverage')
    point: dict[tuple[str, str], float] = {}
    paired_point: dict[tuple[str, str], float] = {}
    for model in model_names:
        model_frame = wide[metadata_columns + [model]].rename(columns={model: 'risk_score'})
        for estimand in BOOTSTRAP_ESTIMANDS:
            point[model, estimand] = _estimand_score(model_frame, model, estimand)
            common_columns, _ = _paired_common_scope(estimand)
            paired_point[model, estimand] = grouped_concordance(model_frame, common_columns).c_index
    draws: dict[tuple[str, str], list[float]] = {(model, estimand): [] for model in model_names for estimand in BOOTSTRAP_ESTIMANDS if _model_scope(model, estimand)[2]}
    deltas: dict[tuple[str, str], list[float]] = {(model, estimand): [] for model in model_names if model != baseline_model for estimand in BOOTSTRAP_ESTIMANDS if _model_scope(model, estimand)[2]}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        sampled_positions = rng.integers(0, len(wide), size=len(wide))
        sample = wide.iloc[sampled_positions]
        replicate_scores: dict[tuple[str, str], float] = {}
        common_replicate_scores: dict[tuple[str, str], float] = {}
        for model in model_names:
            model_frame = sample[metadata_columns + [model]].rename(columns={model: 'risk_score'})
            for estimand in BOOTSTRAP_ESTIMANDS:
                common_columns, _ = _paired_common_scope(estimand)
                common_replicate_scores[model, estimand] = grouped_concordance(model_frame, common_columns).c_index
                if not _model_scope(model, estimand)[2]:
                    continue
                score = _estimand_score(model_frame, model, estimand)
                replicate_scores[model, estimand] = score
                if np.isfinite(score):
                    draws[model, estimand].append(float(score))
        for model in model_names:
            if model == baseline_model:
                continue
            for estimand in BOOTSTRAP_ESTIMANDS:
                if not _model_scope(model, estimand)[2]:
                    continue
                model_score = common_replicate_scores[model, estimand]
                baseline_score = common_replicate_scores[baseline_model, estimand]
                if np.isfinite(model_score) and np.isfinite(baseline_score):
                    deltas[model, estimand].append(float(model_score - baseline_score))
    absolute_rows: list[dict[str, Any]] = []
    for model in model_names:
        for estimand in BOOTSTRAP_ESTIMANDS:
            group_columns, score_scope, appropriate = _model_scope(model, estimand)
            if not appropriate:
                continue
            values = np.asarray(draws[model, estimand], dtype=float)
            low, high = np.quantile(values, [0.025, 0.975]) if len(values) else (math.nan, math.nan)
            absolute_rows.append({'model': model, 'estimand': estimand, 'score_scope': score_scope, 'cross_fold_comparability': 'comparable_single_fixed_score_scale' if not group_columns or str(model).startswith('fixed_llm__') else 'not_assumed_pairs_restricted_to_same_test_fold', 'point_c_index': point[model, estimand], 'bootstrap_mean_c_index': float(values.mean()) if len(values) else math.nan, 'bootstrap_sd_ddof1': float(values.std(ddof=1)) if len(values) > 1 else math.nan, 'bootstrap_ci95_low': float(low), 'bootstrap_ci95_high': float(high), 'bootstrap_reps_requested': n_bootstrap, 'bootstrap_reps_valid': int(len(values)), 'bootstrap_unit': 'patient', 'bootstrap_scheme': 'paired iid patient resampling with replacement; percentile 95% CI', 'random_state': random_state})
    paired_rows: list[dict[str, Any]] = []
    for model in model_names:
        if model == baseline_model:
            continue
        for estimand in BOOTSTRAP_ESTIMANDS:
            _, _, appropriate = _model_scope(model, estimand)
            if not appropriate:
                continue
            _, common_scope = _paired_common_scope(estimand)
            values = np.asarray(deltas[model, estimand], dtype=float)
            low, high = np.quantile(values, [0.025, 0.975]) if len(values) else (math.nan, math.nan)
            paired_rows.append({'model': model, 'baseline_model': baseline_model, 'estimand': estimand, 'common_pair_scope': common_scope, 'model_score_scope': common_scope, 'baseline_score_scope': common_scope, 'pair_restriction_identical': True, 'model_point_c_index': paired_point[model, estimand], 'baseline_point_c_index': paired_point[baseline_model, estimand], 'point_delta': paired_point[model, estimand] - paired_point[baseline_model, estimand], 'bootstrap_mean_delta': float(values.mean()) if len(values) else math.nan, 'bootstrap_ci95_low': float(low), 'bootstrap_ci95_high': float(high), 'bootstrap_reps_requested': n_bootstrap, 'bootstrap_reps_valid': int(len(values)), 'bootstrap_unit': 'patient', 'bootstrap_scheme': 'paired iid patient resampling with replacement; percentile 95% CI', 'random_state': random_state})
    return (pd.DataFrame(absolute_rows), pd.DataFrame(paired_rows))

def build_input_discovery_rows(inputs: DiscoveredInputs, prediction_discovery: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for artifact, path in (('canonical_task2_X', inputs.x_path), ('canonical_task2_y', inputs.y_path), ('clinical_sample', inputs.clinical_sample_path), ('frozen_fold_assignments', inputs.fold_path), ('six_cox_oof_predictions', inputs.cox_oof_path)):
        rows.append({'artifact_type': artifact, 'path': str(path), 'sha256': sha256_file(path), 'status': 'selected', 'reason': None})
    rows.extend(prediction_discovery)
    return pd.DataFrame(rows)

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value

def build_report_payload(global_summary: pd.DataFrame, availability: pd.DataFrame) -> dict[str, Any]:

    def model_row(model: str) -> dict[str, Any] | None:
        match = global_summary[global_summary.model == model]
        return None if match.empty else _json_safe(match.iloc[0].to_dict())
    return {'analysis_title': 'Clinical and mutation-feature survival: cancer-context audit', 'random_state': RANDOM_STATE, 'n_patients': N_PATIENTS, 'baseline': model_row(BASELINE_MODEL), 'cancer_only': model_row('cancer_type_only'), 'full_context': model_row('mutation_pcs_clinical_tmb_cancer_type'), 'stratified_context': model_row('mutation_pcs_clinical_tmb_stratified_cancer'), 'available_models': availability.loc[availability.status == 'available_complete_prediction', 'model'].astype(str).tolist(), 'unavailable_models': _json_safe(availability.loc[availability.status != 'available_complete_prediction', ['model', 'status', 'reason', 'source_path']].to_dict('records')), 'interpretation_guardrails': ['Overall pan-cancer concordance includes between-cancer comparisons.', 'Within-cancer concordance excludes all patient pairs from different cancer types.', 'Fold-pooled concordance avoids comparing fold-specific Cox score scales.', 'A cancer-stratified Cox model has no globally shared baseline hazard; its global C-index is descriptive only.', 'Kruskal-Wallis tests are exploratory and BH-FDR adjusted across models.', 'Bootstrap resampling unit is the patient.']}

def run_audit(*, project_root: Path, output_root: Path, bootstrap_reps: int, x_path: Path | None=None, y_path: Path | None=None, clinical_sample_path: Path | None=None, fold_path: Path | None=None, cox_oof_path: Path | None=None, run_command: str | None=None) -> dict[str, Path]:
    if bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be positive')
    output_root = output_root.resolve()
    assert_output_targets_are_new(output_root)
    inputs = discover_inputs(project_root, output_root, x_path=x_path, y_path=y_path, clinical_sample_path=clinical_sample_path, fold_path=fold_path, cox_oof_path=cox_oof_path)
    metadata, folds, _ = load_and_validate_metadata(inputs)
    predictions, availability, prediction_discovery = load_existing_predictions(inputs, metadata)
    fold_metrics = build_fold_metrics(predictions)
    per_cancer = build_per_cancer_metrics(predictions)
    global_summary = build_global_summary(predictions, fold_metrics, per_cancer, availability)
    risk_by_cancer, risk_tests = build_risk_distribution_metrics(predictions)
    bootstrap_absolute, bootstrap_paired = patient_bootstrap(predictions, n_bootstrap=bootstrap_reps, random_state=RANDOM_STATE)
    cohort_summary = cohort_by_cancer(metadata)
    discovery_frame = build_input_discovery_rows(inputs, prediction_discovery)
    report_payload = build_report_payload(global_summary, availability)
    outputs = {name: output_root / relative for name, relative in OUTPUT_FILES.items()}
    write_csv_new(metadata, outputs['patient_cancer'])
    write_csv_new(cohort_summary, outputs['cohort_by_cancer'])
    write_csv_new(discovery_frame, outputs['input_discovery'])
    write_csv_new(availability, outputs['model_availability'])
    write_csv_new(folds, outputs['folds'])
    write_csv_new(predictions, outputs['predictions'])
    write_csv_new(fold_metrics, outputs['fold_metrics'])
    write_csv_new(global_summary, outputs['global_summary'])
    write_csv_new(per_cancer, outputs['per_cancer'])
    write_csv_new(risk_by_cancer, outputs['risk_by_cancer'])
    write_csv_new(risk_tests, outputs['risk_tests'])
    write_csv_new(bootstrap_absolute, outputs['bootstrap_absolute'])
    write_csv_new(bootstrap_paired, outputs['bootstrap_paired'])
    write_json_new(report_payload, outputs['report_payload'])
    dependency_versions = {}
    for package in ('numpy', 'pandas', 'scipy', 'lifelines', 'scikit-learn'):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    manifest = {'analysis_title': 'Clinical and mutation-feature survival: cancer-context audit', 'created_utc': datetime.now(timezone.utc).isoformat(), 'random_state': RANDOM_STATE, 'bootstrap_reps': bootstrap_reps, 'bootstrap_unit': 'patient', 'python': sys.version, 'dependency_versions': dependency_versions, 'run_command': run_command, 'code': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(Path(__file__).resolve())}, 'inputs': {row['artifact_type']: {'path': row['path'], 'sha256': row['sha256']} for row in discovery_frame.to_dict('records') if row.get('status') == 'selected' and row.get('artifact_type') in {'canonical_task2_X', 'canonical_task2_y', 'clinical_sample', 'frozen_fold_assignments', 'six_cox_oof_predictions'}}, 'model_availability': _json_safe(availability.to_dict('records')), 'outputs': {name: {'path': str(path), 'sha256': sha256_file(path) if path.is_file() else None} for name, path in outputs.items()}, 'rules': {'expected_n': N_PATIENTS, 'expected_cancer_counts': EXPECTED_CANCER_COUNTS, 'per_cancer_reporting': {'min_n': MIN_CANCER_N, 'min_events': MIN_CANCER_EVENTS, 'min_pairs': MIN_CANCER_PAIRS, 'min_folds_with_pairs': MIN_FOLDS_WITH_PAIRS}, 'score_orientation': 'higher score means earlier event / higher risk', 'no_training_or_api_calls': True}}
    write_json_new(manifest, outputs['component_manifest'])
    log_lines = ['Clinical and mutation-feature survival: cancer-context audit', f"created_utc={manifest['created_utc']}", f'project_root={project_root.resolve()}', f'output_root={output_root}', f'random_state={RANDOM_STATE}', f'bootstrap_reps={bootstrap_reps}', 'bootstrap_unit=patient', f'run_command={run_command}', 'dependency_versions=' + json.dumps(dependency_versions, sort_keys=True), f'n_patients={len(metadata)}', f"n_models_available={(availability.status == 'available_complete_prediction').sum()}", 'api_calls=0', 'model_training=0']
    write_text_new('\n'.join(log_lines) + '\n', outputs['log'])
    return outputs

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = _submission_path('project', '')
    default_output = Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root', type=Path, default=default_project)
    parser.add_argument('--output-root', type=Path, default=default_output)
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    parser.add_argument('--x-path', type=Path)
    parser.add_argument('--y-path', type=Path)
    parser.add_argument('--clinical-sample-path', type=Path)
    parser.add_argument('--fold-path', type=Path)
    parser.add_argument('--cox-oof-path', type=Path)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = parse_args(argv)
    command = ' '.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        outputs = run_audit(project_root=args.project_root, output_root=args.output_root, bootstrap_reps=args.bootstrap_reps, x_path=args.x_path, y_path=args.y_path, clinical_sample_path=args.clinical_sample_path, fold_path=args.fold_path, cox_oof_path=args.cox_oof_path, run_command=command)
    except Exception:
        print(f'command={command}', file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f'command={command}')
    for name, path in outputs.items():
        print(f'{name}={path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
