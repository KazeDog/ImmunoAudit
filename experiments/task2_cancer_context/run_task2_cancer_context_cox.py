"""Run six fixed-fold, leakage-controlled Cox ablations for Task 2."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import lifelines
import numpy as np
import pandas as pd
import sklearn
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from build_task2_cancer_context import build_context_dataset, load_cbioportal_table
from evaluate_task2_cancer_context import BASELINE_MODEL, STRATIFIED_MODEL, build_fold_results, evaluate_oof
from task2_context_common import BASELINE_TARGET_MEAN, CANONICAL_X_PATH, CANONICAL_Y_PATH, CONTEXT_COLUMNS, DEFAULT_RESULTS_DIR, EXPERIMENT_VERSION, MODEL_CANCER_COLUMN, N_SPLITS, ORIGINAL_CANCER_COLUMN, PROJECT_ROOT, RANDOM_SEED, RAW_PATIENT_PATH, RAW_SAMPLE_PATH, protected_file_hashes, require_new_path, sha256_file, sha256_lines, write_csv_new, write_json_new

@dataclass(frozen=True)
class ModelSpec:
    name: str
    mutation_pcs: bool
    categorical_columns: tuple[str, ...]
    canonical_tmb: bool
    cancer_strata: bool = False
    penalizer: float = 0.0
MODEL_SPECS = (ModelSpec(BASELINE_MODEL, True, (), False), ModelSpec('cancer_type_only', False, (MODEL_CANCER_COLUMN,), False), ModelSpec('clinical_tmb', False, ('SEX', 'AGE_GROUP', 'DRUG_TYPE'), True), ModelSpec('mutation_pcs_cancer_type', True, (MODEL_CANCER_COLUMN,), False), ModelSpec('mutation_pcs_clinical_tmb_cancer_type', True, ('SEX', 'AGE_GROUP', 'DRUG_TYPE', MODEL_CANCER_COLUMN), True), ModelSpec(STRATIFIED_MODEL, True, ('SEX', 'AGE_GROUP', 'DRUG_TYPE'), True, cancer_strata=True))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_RESULTS_DIR / f'{EXPERIMENT_VERSION}.csv')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--fold-assignments', type=Path, default=DEFAULT_RESULTS_DIR / 'fold_assignments.csv')
    parser.add_argument('--run-kind', choices=['smoke', 'full'], default='full')
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--freeze-folds-only', action='store_true')
    return parser.parse_args()

def build_fold_assignments(patient_ids: pd.Series | pd.Index) -> pd.DataFrame:
    patient_ids = pd.Index(patient_ids).astype(str)
    if not patient_ids.is_unique:
        raise ValueError('Patient IDs must be unique before splitting')
    fold = np.zeros(len(patient_ids), dtype=int)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    for fold_number, (_, test_idx) in enumerate(splitter.split(np.arange(len(patient_ids))), start=1):
        fold[test_idx] = fold_number
    if set(fold.tolist()) != set(range(1, N_SPLITS + 1)):
        raise RuntimeError('Fold construction failed')
    return pd.DataFrame({'PATIENT_ID': patient_ids, 'original_row_position_0based': np.arange(len(patient_ids), dtype=int), 'fold': fold, 'splitter': 'KFold', 'n_splits': N_SPLITS, 'shuffle': True, 'random_state': RANDOM_SEED})

def load_or_freeze_fold_assignments(patient_ids: pd.Series | pd.Index, path: Path) -> pd.DataFrame:
    expected = build_fold_assignments(patient_ids)
    if path.exists():
        observed = pd.read_csv(path)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
        return observed
    write_csv_new(expected, path)
    return expected

def _safe_feature_names(names: list[str]) -> list[str]:
    output: list[str] = []
    seen: dict[str, int] = {}
    for raw_name in names:
        base = re.sub('[^0-9A-Za-z_]+', '_', str(raw_name)).strip('_')
        base = base or 'feature'
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        output.append(base if occurrence == 0 else f'{base}_{occurrence + 1}')
    return output

def fit_mutation_transformer(train: pd.DataFrame, test: pd.DataFrame, mutation_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_mutation = train[mutation_columns].to_numpy(dtype=float)
    test_mutation = test[mutation_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_mutation)
    test_scaled = scaler.transform(test_mutation)
    n_components = min(10, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    columns = [f'PC{i + 1}' for i in range(n_components)]
    return (pd.DataFrame(train_pca, index=train.index, columns=columns), pd.DataFrame(test_pca, index=test.index, columns=columns), {'mutation_n_input_features': len(mutation_columns), 'mutation_n_components': n_components, 'mutation_explained_variance_sum': float(pca.explained_variance_ratio_.sum()), 'mutation_scaler_mean_sha256': sha256_lines(np.asarray(scaler.mean_, dtype=float).tolist()), 'mutation_pca_components_sha256': sha256_lines(np.asarray(pca.components_, dtype=float).ravel().tolist())})

def fit_categorical_block(train: pd.DataFrame, test: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not columns:
        return (pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), {'categorical_columns': [], 'categorical_categories': {}, 'unknown_test_levels': {}})
    imputer = SimpleImputer(strategy='most_frequent')
    train_imputed = imputer.fit_transform(train[list(columns)])
    test_imputed = imputer.transform(test[list(columns)])
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False, dtype=float)
    train_encoded = encoder.fit_transform(train_imputed)
    test_encoded = encoder.transform(test_imputed)
    names = _safe_feature_names(encoder.get_feature_names_out(list(columns)).tolist())
    categories: dict[str, list[str]] = {}
    references: dict[str, str] = {}
    unknown: dict[str, list[str]] = {}
    for column, values in zip(columns, encoder.categories_, strict=True):
        categories[column] = [str(value) for value in values.tolist()]
        references[column] = str(values[0])
        train_levels = set(train[column].dropna().astype(str))
        test_levels = set(test[column].dropna().astype(str))
        unknown[column] = sorted(test_levels.difference(train_levels))
    if any(unknown.values()):
        raise ValueError(f'Unseen test category after singleton collapse: {unknown}')
    return (pd.DataFrame(train_encoded, index=train.index, columns=names), pd.DataFrame(test_encoded, index=test.index, columns=names), {'categorical_columns': list(columns), 'categorical_categories': categories, 'categorical_reference_levels': references, 'categorical_imputer_statistics': [str(value) for value in imputer.statistics_], 'unknown_test_levels': unknown})

def fit_tmb_block(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Use a prespecified log1p transform of canonical nonsynonymous TMB."""
    imputer = SimpleImputer(strategy='median')
    train_raw = imputer.fit_transform(train[['TMB_NONSYNONYMOUS']]).astype(float)
    test_raw = imputer.transform(test[['TMB_NONSYNONYMOUS']]).astype(float)
    if (train_raw < 0).any() or (test_raw < 0).any():
        raise ValueError('Canonical TMB must be non-negative for log1p')
    train_log = np.log1p(train_raw)
    test_log = np.log1p(test_raw)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_log)
    test_scaled = scaler.transform(test_log)
    return (pd.DataFrame(train_scaled, index=train.index, columns=['log1p_TMB_scaled']), pd.DataFrame(test_scaled, index=test.index, columns=['log1p_TMB_scaled']), {'tmb_source': 'data_clinical_sample.txt:TMB_NONSYNONYMOUS', 'tmb_transform': 'train-median imputation; log1p; train StandardScaler', 'tmb_train_median': float(imputer.statistics_[0]), 'tmb_log_train_mean': float(scaler.mean_[0]), 'tmb_log_train_scale': float(scaler.scale_[0])})

def assemble_covariates(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec, mutation_train: pd.DataFrame, mutation_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_blocks: list[pd.DataFrame] = []
    test_blocks: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    if spec.mutation_pcs:
        train_blocks.append(mutation_train)
        test_blocks.append(mutation_test)
    categorical_train, categorical_test, categorical_audit = fit_categorical_block(train, test, spec.categorical_columns)
    if not categorical_train.empty:
        train_blocks.append(categorical_train)
        test_blocks.append(categorical_test)
    audit.update(categorical_audit)
    if spec.canonical_tmb:
        tmb_train, tmb_test, tmb_audit = fit_tmb_block(train, test)
        train_blocks.append(tmb_train)
        test_blocks.append(tmb_test)
        audit.update(tmb_audit)
    if not train_blocks:
        raise ValueError(f'Model {spec.name} has no covariates')
    train_covariates = pd.concat(train_blocks, axis=1)
    test_covariates = pd.concat(test_blocks, axis=1)
    if train_covariates.columns.duplicated().any():
        raise ValueError(f'Duplicate covariate names for {spec.name}')
    if list(train_covariates.columns) != list(test_covariates.columns):
        raise ValueError(f'Train/test covariate columns differ for {spec.name}')
    if not (np.isfinite(train_covariates.to_numpy(dtype=float)).all() and np.isfinite(test_covariates.to_numpy(dtype=float)).all()):
        raise ValueError(f'Non-finite covariates for {spec.name}')
    audit['n_covariates'] = int(train_covariates.shape[1])
    audit['covariate_names'] = train_covariates.columns.tolist()
    return (train_covariates, test_covariates, audit)

def fit_one_model(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec, mutation_train: pd.DataFrame, mutation_test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    train_covariates, test_covariates, preprocessing = assemble_covariates(train, test, spec, mutation_train, mutation_test)
    fit_frame = train_covariates.copy()
    fit_frame['duration'] = train['duration'].to_numpy(dtype=float)
    fit_frame['event'] = train['event'].to_numpy(dtype=int)
    strata_column = None
    if spec.cancer_strata:
        strata_column = '__cancer_stratum'
        fit_frame[strata_column] = train[MODEL_CANCER_COLUMN].astype(str).to_numpy()
        stratum_events = fit_frame.groupby(strata_column)['event'].sum()
        if (stratum_events < 1).any():
            raise ValueError(f'Training stratum without an event: {stratum_events.to_dict()}')
    fitter = CoxPHFitter(penalizer=spec.penalizer)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fitter.fit(fit_frame, duration_col='duration', event_col='event', strata=strata_column)
    warning_rows = [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in captured]
    convergence = [warning for warning in captured if issubclass(warning.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f'Convergence warning for {spec.name}: {warning_rows}')
    if not np.isfinite(fitter.params_.to_numpy(dtype=float)).all():
        raise RuntimeError(f'Non-finite Cox coefficients for {spec.name}')
    partial_hazard = fitter.predict_partial_hazard(test_covariates).to_numpy(dtype=float)
    if (partial_hazard <= 0).any() or not np.isfinite(partial_hazard).all():
        raise RuntimeError(f'Invalid partial hazard for {spec.name}')
    risk_score = np.log(partial_hazard)
    covariate_signature = pd.util.hash_pandas_object(test_covariates, index=False)
    risk_score = pd.Series(risk_score, index=test_covariates.index).groupby(covariate_signature.to_numpy()).transform('mean').to_numpy(dtype=float)
    preprocessing.update({'cox_penalizer': spec.penalizer, 'cox_l1_ratio': 0.0, 'cox_strata': strata_column, 'cox_fit_warnings': warning_rows, 'coefficient_count': int(len(fitter.params_)), 'coefficient_max_abs': float(np.max(np.abs(fitter.params_.to_numpy(dtype=float)))), 'identical_covariate_prediction_tie_groups': int(covariate_signature.duplicated(keep=False).groupby(covariate_signature).any().sum())})
    return (risk_score, preprocessing)

def run_models(dataset: pd.DataFrame, assignments: pd.DataFrame, *, folds_to_run: list[int], model_specs: tuple[ModelSpec, ...]=MODEL_SPECS) -> tuple[pd.DataFrame, pd.DataFrame]:
    mutation_columns = [column for column in dataset.columns if column not in CONTEXT_COLUMNS]
    if len(mutation_columns) != 468:
        raise ValueError(f'Expected 468 mutation features, found {len(mutation_columns)}')
    fold_map = assignments.set_index('PATIENT_ID')['fold']
    dataset = dataset.copy()
    dataset.index = dataset['PATIENT_ID'].astype(str)
    dataset['fold'] = dataset.index.map(fold_map)
    if dataset['fold'].isna().any():
        raise ValueError('Dataset patient missing from frozen fold assignments')
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    mutation_transform_consumers = int(sum((spec.mutation_pcs for spec in model_specs)))
    for fold in folds_to_run:
        train = dataset[dataset['fold'] != fold].copy()
        test = dataset[dataset['fold'] == fold].copy()
        if set(train.index).intersection(test.index):
            raise RuntimeError('Train/test patient overlap')
        if len(train) + len(test) != len(dataset):
            raise RuntimeError('Train/test coverage error')
        mutation_train, mutation_test, mutation_audit = fit_mutation_transformer(train, test, mutation_columns)
        for spec in model_specs:
            risk_score, preprocessing = fit_one_model(train, test, spec, mutation_train if spec.mutation_pcs else pd.DataFrame(index=train.index), mutation_test if spec.mutation_pcs else pd.DataFrame(index=test.index))
            for position, patient_id in enumerate(test.index):
                prediction_rows.append({'PATIENT_ID': patient_id, 'fold': int(fold), 'model': spec.name, 'risk_score': float(risk_score[position]), 'score_type': 'log_partial_hazard', 'score_orientation': 'higher_is_earlier_event_higher_risk', 'duration': float(test.loc[patient_id, 'duration']), 'event': int(test.loc[patient_id, 'event']), ORIGINAL_CANCER_COLUMN: test.loc[patient_id, ORIGINAL_CANCER_COLUMN], MODEL_CANCER_COLUMN: test.loc[patient_id, MODEL_CANCER_COLUMN], 'prediction_status': 'ok'})
            audit_rows.append({'model': spec.name, 'fold': int(fold), 'n_train': int(len(train)), 'n_test': int(len(test)), 'events_train': int(train['event'].sum()), 'events_test': int(test['event'].sum()), 'train_patient_id_sha256': sha256_lines(train.index), 'test_patient_id_sha256': sha256_lines(test.index), 'fit_scope': 'outer_training_fold_only', 'mutation_transform_reused_within_fold': mutation_transform_consumers > 1, 'mutation_transform_consumers_within_pass': mutation_transform_consumers, **mutation_audit, **{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in preprocessing.items()}})
            print(f"fold={fold} model={spec.name} n_train={len(train)} n_test={len(test)} events_test={int(test['event'].sum())}")
    return (pd.DataFrame(prediction_rows), pd.DataFrame(audit_rows))

def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True, timeout=20)
        return {'command': command, 'returncode': completed.returncode, 'stdout': completed.stdout.strip(), 'stderr': completed.stderr.strip()}
    except Exception as exc:
        return {'command': command, 'error': repr(exc)}

def build_manifest(args: argparse.Namespace, output_paths: list[Path], folds_to_run: list[int], protected_before: dict[str, str | None]) -> dict[str, Any]:
    package_names = ['numpy', 'pandas', 'scikit-learn', 'lifelines', 'scipy']
    packages = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'experiment_version': EXPERIMENT_VERSION, 'run_kind': args.run_kind, 'started_and_completed_utc': datetime.now(timezone.utc).isoformat(), 'command': shlex.join([sys.executable, *sys.argv]), 'working_directory': str(PROJECT_ROOT), 'python_executable': sys.executable, 'python_version': platform.python_version(), 'platform': platform.platform(), 'packages': packages, 'numpy_version_runtime': np.__version__, 'sklearn_version_runtime': sklearn.__version__, 'lifelines_version_runtime': lifelines.__version__, 'random_seed': RANDOM_SEED, 'folds_run': folds_to_run, 'splitter': {'class': 'KFold', 'n_splits': N_SPLITS, 'shuffle': True, 'random_state': RANDOM_SEED}, 'bootstrap': {'reps': int(args.bootstrap_reps), 'seed': int(args.bootstrap_seed), 'ci': 'percentile 2.5/97.5', 'scheme': 'paired patient bootstrap stratified by original cancer type and outer fold', 'within_cancer_eligibility': 'eligible cancer set frozen on the original OOF cohort before resampling'}, 'model_specs': [asdict(spec) for spec in MODEL_SPECS], 'tmb_transform': 'prespecified log1p canonical TMB, train-median imputation and scaling', 'c_index_orientation': 'concordance_index(duration, -log_partial_hazard, event)', 'stratified_model_global_metric': 'not_applicable', 'input_paths': {'versioned_dataset': str(args.dataset.resolve()), 'canonical_x': str(CANONICAL_X_PATH.resolve()), 'canonical_y': str(CANONICAL_Y_PATH.resolve()), 'fold_assignments': str(args.fold_assignments.resolve()), 'raw_patient_table': str(RAW_PATIENT_PATH.resolve()), 'raw_sample_table': str(RAW_SAMPLE_PATH.resolve())}, 'input_sha256': {str(path.resolve()): sha256_file(path) for path in [args.dataset, CANONICAL_X_PATH, CANONICAL_Y_PATH, RAW_PATIENT_PATH, RAW_SAMPLE_PATH, args.fold_assignments]}, 'protected_python_settings_sha256_before': protected_before, 'protected_python_settings_sha256_after': protected_file_hashes(), 'git_status': _command_output(['git', 'status', '--short', '--branch']), 'gpu_query': _command_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']), 'output_paths': [str(path.resolve()) for path in output_paths], 'output_sha256': {str(path.resolve()): sha256_file(path) for path in output_paths if path.exists()}, 'methodological_notes': ['All imputers, encoders, scalers, and PCA transforms were fitted on outer training folds only.', 'Within each model pass, mutation PCA was fitted once per outer training fold. The baseline gate and extension pass refitted it deterministically; extension mutation-containing models reused their pass-specific transform.', 'No penalization was used; convergence warnings are fatal rather than silently ignored.', 'Algebraically identical test covariate rows were canonicalized to identical risk scores before concordance evaluation.', 'Cancer-stratified Cox partial hazards are compared only within original cancer groups.', 'Within-cancer bootstrap eligibility was frozen from the original OOF cohort.', 'Bootstrap intervals condition on the fixed folds and fitted OOF predictions.', 'The upstream provenance of the frozen 468-gene feature matrix is unavailable.']}

def preflight_outputs(output_dir: Path) -> list[Path]:
    names = ['oof_predictions.csv', 'model_fit_audit.csv', 'fold_results.csv', 'summary.csv', 'per_cancer_results.csv', 'paired_comparison.csv', 'bootstrap_comparison.csv', 'baseline_reproduction.json', 'run_manifest.json']
    paths = [output_dir / name for name in names]
    for path in paths:
        require_new_path(path)
    return paths

def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.run_kind == 'full' and args.max_folds is not None:
        raise ValueError('--max-folds is only allowed for smoke runs')
    dataset = pd.read_csv(args.dataset)
    missing_context = sorted(set(CONTEXT_COLUMNS).difference(dataset.columns))
    if missing_context:
        raise ValueError(f'Versioned dataset missing context columns: {missing_context}')
    if dataset['PATIENT_ID'].duplicated().any():
        raise ValueError('Versioned dataset contains duplicate patients')
    canonical_x = pd.read_csv(CANONICAL_X_PATH, index_col=0)
    canonical_y = pd.read_csv(CANONICAL_Y_PATH, index_col=0)
    expected_dataset, _ = build_context_dataset(canonical_x, canonical_y, load_cbioportal_table(RAW_PATIENT_PATH), load_cbioportal_table(RAW_SAMPLE_PATH))
    pd.testing.assert_frame_equal(dataset.reset_index(drop=True), expected_dataset.reset_index(drop=True), check_dtype=False, check_exact=True)
    canonical_ids = pd.Series(canonical_x.index.astype(str), name='PATIENT_ID')
    if dataset['PATIENT_ID'].astype(str).tolist() != canonical_ids.tolist():
        raise ValueError('Versioned dataset no longer has canonical X patient order')
    assignments = load_or_freeze_fold_assignments(canonical_ids, args.fold_assignments)
    if args.freeze_folds_only:
        print(f'fold_assignments={args.fold_assignments}')
        return
    output_dir = args.output_dir.resolve()
    output_paths = preflight_outputs(output_dir)
    protected_before = protected_file_hashes()
    all_folds = list(range(1, N_SPLITS + 1))
    if args.run_kind == 'smoke':
        max_folds = args.max_folds or 1
        folds_to_run = all_folds[:max_folds]
    else:
        folds_to_run = all_folds
    if args.run_kind == 'full':
        baseline_oof, baseline_audit = run_models(dataset, assignments, folds_to_run=folds_to_run, model_specs=(MODEL_SPECS[0],))
        baseline_folds = build_fold_results(baseline_oof)
        reproduced_mean = float(baseline_folds['primary_c_index'].mean())
        difference = abs(reproduced_mean - BASELINE_TARGET_MEAN)
        exact_tolerance = 1e-10
        material_tolerance = 0.0001
        if difference > material_tolerance:
            raise RuntimeError(f'Material baseline reproduction failure before extensions: mean={reproduced_mean}, target={BASELINE_TARGET_MEAN}, abs_diff={difference}, material_tolerance={material_tolerance}')
        print(f'baseline_gate_passed mean={reproduced_mean:.16f} abs_diff={difference:.3g} exact={difference <= exact_tolerance}')
        variant_oof, variant_audit = run_models(dataset, assignments, folds_to_run=folds_to_run, model_specs=MODEL_SPECS[1:])
        oof = pd.concat([baseline_oof, variant_oof], ignore_index=True)
        fit_audit = pd.concat([baseline_audit, variant_audit], ignore_index=True)
    else:
        oof, fit_audit = run_models(dataset, assignments, folds_to_run=folds_to_run, model_specs=MODEL_SPECS)
        reproduced_mean = math.nan
        difference = math.nan
        exact_tolerance = 1e-10
        material_tolerance = 0.0001
    evaluations = evaluate_oof(oof, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed)
    if args.run_kind == 'full':
        baseline_summary = evaluations['summary'].set_index('model').loc[BASELINE_MODEL]
        if int(baseline_summary['n_folds']) != N_SPLITS:
            raise RuntimeError('Baseline reproduction did not produce five valid folds')
    else:
        reproduced_mean = float(evaluations['summary'].set_index('model').loc[BASELINE_MODEL, 'fold_mean_c_index'])
        difference = math.nan
    if protected_before != protected_file_hashes():
        raise RuntimeError('A protected python_settings file changed during Cox execution')
    path_by_name = {path.name: path for path in output_paths}
    write_csv_new(oof, path_by_name['oof_predictions.csv'])
    write_csv_new(fit_audit, path_by_name['model_fit_audit.csv'])
    for key in ['fold_results', 'summary', 'per_cancer_results', 'paired_comparison', 'bootstrap_comparison']:
        write_csv_new(evaluations[key], path_by_name[f'{key}.csv'])
    write_json_new({'run_kind': args.run_kind, 'target_old_mean': BASELINE_TARGET_MEAN, 'reproduced_fold_mean': reproduced_mean, 'absolute_difference': difference, 'exact_gate_tolerance': exact_tolerance, 'material_gate_tolerance': material_tolerance, 'exact_reproduction': bool(args.run_kind == 'full' and difference <= exact_tolerance), 'gate_passed': bool(args.run_kind == 'full' and difference <= material_tolerance), 'folds_run': folds_to_run}, path_by_name['baseline_reproduction.json'])
    manifest = build_manifest(args, [path for path in output_paths if path.name != 'run_manifest.json'], folds_to_run, protected_before)
    write_json_new(manifest, path_by_name['run_manifest.json'])
    print(f'output_dir={output_dir}')
    print(f"oof_predictions={path_by_name['oof_predictions.csv']}")
    print(f"summary={path_by_name['summary.csv']}")
    print(f"manifest={path_by_name['run_manifest.json']}")
if __name__ == '__main__':
    main()
