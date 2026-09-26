"""Task 2 priority-2 transportability stress test.

Compare frozen fivefold cross-validation with leave-one-cancer-out (LOCO)
evaluation using models that do not contain cancer identity as a covariate.
All preprocessing and Cox fitting are confined to the corresponding training
partition. Outputs are append-only and the singleton cancer is retained in the
prediction set while being prespecified as not estimable for within-cancer C.
"""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(str(_submission_path('project', '')))
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments/task2_cancer_context'
AUDIT_SCRIPT_DIR = PROJECT_ROOT / 'analysis/cancer_context/scripts'
for module_dir in (EXPERIMENT_DIR, AUDIT_SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
import run_task2_cancer_context_cox as core
from task2_cancer_context_audit import grouped_concordance
DATASET_PATH = PROJECT_ROOT / 'results/task2_cancer_context/task2_cancer_context_v1.csv'
FOLD_PATH = PROJECT_ROOT / 'analysis/cancer_context/fold_assignments/task2_frozen_fold_assignments.csv'
RANDOM_STATE = 42
BASELINE_MODEL = 'mutation_only_coxph_pca'
MODEL_SPECS = (core.ModelSpec(BASELINE_MODEL, True, (), False), core.ModelSpec('clinical_tmb_no_cancer', False, ('SEX', 'AGE_GROUP', 'DRUG_TYPE'), True), core.ModelSpec('mutation_pcs_clinical_tmb_no_cancer', True, ('SEX', 'AGE_GROUP', 'DRUG_TYPE'), True))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'analysis/transportability')
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_new(path: Path, writer: Callable[[Path], None]) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_csv_new(frame: pd.DataFrame, path: Path) -> None:
    write_new(path, lambda temporary: frame.to_csv(temporary, index=False))

def write_json_new(payload: Any, path: Path) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    write_new(path, lambda temporary: temporary.write_text(rendered, encoding='utf-8'))

def load_inputs() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    folds = pd.read_csv(FOLD_PATH)[['PATIENT_ID', 'fold']]
    data['PATIENT_ID'] = data['PATIENT_ID'].astype(str)
    folds['PATIENT_ID'] = folds['PATIENT_ID'].astype(str)
    data = data.merge(folds, on='PATIENT_ID', how='left', validate='one_to_one')
    if len(data) != 1610 or data['PATIENT_ID'].nunique() != 1610:
        raise AssertionError('Expected 1,610 unique Task 2 patients')
    if int(data['event'].sum()) != 802 or data['fold'].isna().any():
        raise AssertionError('Task 2 event/fold invariants changed')
    if data[core.ORIGINAL_CANCER_COLUMN].nunique() != 11:
        raise AssertionError('Expected 11 original cancer groups')
    data.index = data['PATIENT_ID']
    return data

def partition_definitions(data: pd.DataFrame, smoke: bool) -> list[tuple[str, str, np.ndarray]]:
    definitions: list[tuple[str, str, np.ndarray]] = []
    if smoke:
        smoke_cancers = ('Renal Cell Carcinoma', 'Skin Cancer, Non-Melanoma')
        for cancer in smoke_cancers:
            definitions.append(('leave_one_cancer_out', cancer, data[core.ORIGINAL_CANCER_COLUMN].eq(cancer).to_numpy()))
        definitions.append(('frozen_fivefold', 'fold_1', data['fold'].eq(1).to_numpy()))
        return definitions
    for cancer in sorted(data[core.ORIGINAL_CANCER_COLUMN].astype(str).unique()):
        definitions.append(('leave_one_cancer_out', cancer, data[core.ORIGINAL_CANCER_COLUMN].eq(cancer).to_numpy()))
    for fold in range(1, 6):
        definitions.append(('frozen_fivefold', f'fold_{fold}', data['fold'].eq(fold).to_numpy()))
    return definitions

def fit_predictions(data: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    mutation_columns = [column for column in data.columns if column not in set(core.CONTEXT_COLUMNS) | {'fold'}]
    if len(mutation_columns) != 468:
        raise AssertionError(f'Expected 468 mutation features, found {len(mutation_columns)}')
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    specs = MODEL_SPECS[:2] if smoke else MODEL_SPECS
    for scheme, partition, test_mask in partition_definitions(data, smoke):
        train = data.loc[~test_mask].copy()
        test = data.loc[test_mask].copy()
        if not len(test) or set(train.index).intersection(test.index):
            raise AssertionError(f'Invalid partition {scheme}/{partition}')
        mutation_train, mutation_test, mutation_audit = core.fit_mutation_transformer(train, test, mutation_columns)
        for spec in specs:
            risk, preprocessing = core.fit_one_model(train, test, spec, mutation_train if spec.mutation_pcs else pd.DataFrame(index=train.index), mutation_test if spec.mutation_pcs else pd.DataFrame(index=test.index))
            for position, patient_id in enumerate(test.index):
                predictions.append({'PATIENT_ID': patient_id, 'evaluation_scheme': scheme, 'test_partition': partition, 'model': spec.name, 'risk_score': float(risk[position]), 'duration': float(test.loc[patient_id, 'duration']), 'event': int(test.loc[patient_id, 'event']), 'cancer_type': str(test.loc[patient_id, core.ORIGINAL_CANCER_COLUMN])})
            audits.append({'evaluation_scheme': scheme, 'test_partition': partition, 'model': spec.name, 'n_train': len(train), 'n_test': len(test), 'events_train': int(train['event'].sum()), 'events_test': int(test['event'].sum()), 'patient_overlap': 0, 'fit_scope': 'training_partition_only', **mutation_audit, **{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in preprocessing.items()}})
            print(f'{scheme} {partition} {spec.name}: train={len(train)} test={len(test)}')
    return (pd.DataFrame(predictions), pd.DataFrame(audits))

def concordance_summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    cancer_rows: list[dict[str, Any]] = []
    for (scheme, model), frame in predictions.groupby(['evaluation_scheme', 'model'], sort=False):
        estimate = grouped_concordance(frame, ['cancer_type'])
        overall_rows.append({'evaluation_scheme': scheme, 'model': model, 'n_patients': len(frame), 'n_cancers': frame['cancer_type'].nunique(), 'within_cancer_c_index': estimate.c_index, 'permissible_pairs': estimate.permissible_pairs, 'estimand': 'pair-weighted Harrell C restricted within original cancer type'})
        for cancer, group in frame.groupby('cancer_type', sort=True):
            estimate = grouped_concordance(group, [])
            cancer_rows.append({'evaluation_scheme': scheme, 'model': model, 'cancer_type': cancer, 'n_patients': len(group), 'n_events': int(group['event'].sum()), 'c_index': estimate.c_index, 'permissible_pairs': estimate.permissible_pairs, 'status': 'ok' if estimate.permissible_pairs else 'not_estimable_no_permissible_pairs'})
    return (pd.DataFrame(overall_rows), pd.DataFrame(cancer_rows))

def resample_within_cancer(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby('cancer_type', sort=True):
        positions = rng.integers(0, len(group), size=len(group))
        pieces.append(group.iloc[positions])
    return pd.concat(pieces, ignore_index=True)

def c_index(frame: pd.DataFrame, score_column: str) -> float:
    work = frame[['duration', 'event', 'cancer_type', score_column]].rename(columns={score_column: 'risk_score'})
    return grouped_concordance(work, ['cancer_type']).c_index

def paired_bootstrap(predictions: pd.DataFrame, reps: int, smoke: bool) -> pd.DataFrame:
    if smoke:
        return pd.DataFrame()
    wide = predictions.pivot(index=['evaluation_scheme', 'PATIENT_ID', 'duration', 'event', 'cancer_type'], columns='model', values='risk_score').reset_index()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_STATE)
    cached: dict[tuple[str, str], list[float]] = {}
    for scheme, frame in wide.groupby('evaluation_scheme', sort=True):
        frame = frame.reset_index(drop=True)
        for model in [column for column in MODEL_SPECS if column.name in frame.columns]:
            cached[scheme, model.name] = []
        for _ in range(reps):
            sample = resample_within_cancer(frame, rng)
            for spec in MODEL_SPECS:
                cached[scheme, spec.name].append(c_index(sample, spec.name))
        baseline_point = c_index(frame, BASELINE_MODEL)
        for spec in MODEL_SPECS:
            point = c_index(frame, spec.name)
            values = np.asarray(cached[scheme, spec.name], dtype=float)
            base_values = np.asarray(cached[scheme, BASELINE_MODEL], dtype=float)
            delta = values - base_values
            rows.append({'comparison': 'model_minus_mutation_only_within_scheme', 'evaluation_scheme': scheme, 'model': spec.name, 'reference': BASELINE_MODEL, 'point_estimate': point, 'ci95_low': float(np.nanquantile(values, 0.025)), 'ci95_high': float(np.nanquantile(values, 0.975)), 'reference_point_estimate': baseline_point, 'point_delta': point - baseline_point, 'delta_ci95_low': float(np.nanquantile(delta, 0.025)), 'delta_ci95_high': float(np.nanquantile(delta, 0.975)), 'bootstrap_reps': reps, 'bootstrap_unit': 'patient resampled within original cancer type'})
    for spec in MODEL_SPECS:
        loco = wide[wide['evaluation_scheme'] == 'leave_one_cancer_out'].reset_index(drop=True)
        fivefold = wide[wide['evaluation_scheme'] == 'frozen_fivefold'].reset_index(drop=True)
        point_loco = c_index(loco, spec.name)
        point_fivefold = c_index(fivefold, spec.name)
        values = np.asarray(cached['leave_one_cancer_out', spec.name])
        references = np.asarray(cached['frozen_fivefold', spec.name])
        rng_pair = np.random.default_rng(RANDOM_STATE + 1000)
        paired_delta: list[float] = []
        merged = loco.merge(fivefold[['PATIENT_ID', spec.name]], on='PATIENT_ID', suffixes=('_loco', '_fivefold'), validate='one_to_one')
        for _ in range(reps):
            sample = resample_within_cancer(merged, rng_pair)
            paired_delta.append(c_index(sample, f'{spec.name}_loco') - c_index(sample, f'{spec.name}_fivefold'))
        rows.append({'comparison': 'loco_minus_frozen_fivefold', 'evaluation_scheme': 'paired_schemes', 'model': spec.name, 'reference': 'same_model_frozen_fivefold', 'point_estimate': point_loco, 'ci95_low': float(np.nanquantile(values, 0.025)), 'ci95_high': float(np.nanquantile(values, 0.975)), 'reference_point_estimate': point_fivefold, 'point_delta': point_loco - point_fivefold, 'delta_ci95_low': float(np.nanquantile(paired_delta, 0.025)), 'delta_ci95_high': float(np.nanquantile(paired_delta, 0.975)), 'bootstrap_reps': reps, 'bootstrap_unit': 'paired patient resampling within original cancer type'})
    return pd.DataFrame(rows)

def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    suffix = '_smoke' if args.smoke else ''
    targets = {'predictions': output_root / f'predictions/task2_priority2_predictions{suffix}.csv', 'fit_audit': output_root / f'metadata/task2_priority2_fit_audit{suffix}.csv', 'summary': output_root / f'metrics/task2_priority2_concordance{suffix}.csv', 'per_cancer': output_root / f'metrics/task2_priority2_per_cancer{suffix}.csv', 'bootstrap': output_root / f'metrics/task2_priority2_bootstrap{suffix}.csv', 'manifest': output_root / f'metadata/task2_priority2_manifest{suffix}.json'}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f'Planned outputs already exist: {existing}')
    data = load_inputs()
    predictions, fit_audit = fit_predictions(data, args.smoke)
    summary, per_cancer = concordance_summaries(predictions)
    bootstrap = paired_bootstrap(predictions, args.bootstrap_reps, args.smoke)
    for key, frame in (('predictions', predictions), ('fit_audit', fit_audit), ('summary', summary), ('per_cancer', per_cancer), ('bootstrap', bootstrap)):
        write_csv_new(frame, targets[key])
    manifest = {'analysis': 'Task 2 priority-2 leave-one-cancer-out transportability stress test', 'created_utc': datetime.now(timezone.utc).isoformat(), 'run_kind': 'smoke' if args.smoke else 'formal', 'models': [asdict(spec) for spec in (MODEL_SPECS[:2] if args.smoke else MODEL_SPECS)], 'cancer_identity_excluded_from_all_model_covariates': True, 'preprocessing_scope': 'training partition only', 'primary_estimand': 'within-cancer pair-weighted Harrell C', 'singleton_policy': 'predict and retain; report cancer-specific C as not estimable when no permissible pairs', 'external_validation_claim': False, 'interpretation': 'internal transportability stress test across represented cancer cohorts', 'bootstrap_reps': 0 if args.smoke else args.bootstrap_reps, 'bootstrap_seed': RANDOM_STATE, 'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'lifelines']}, 'inputs': {str(path): sha256(path) for path in [DATASET_PATH, FOLD_PATH]}, 'outputs': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in targets.items() if key != 'manifest'}}
    write_json_new(manifest, targets['manifest'])
    print(json.dumps({'status': 'ok', 'targets': {key: str(path) for key, path in targets.items()}}, indent=2))
if __name__ == '__main__':
    main()
