"""Priority-1 survival-model adequacy extension for Task 2.

The program refits the prespecified Cox models in the frozen five folds, first
verifies that their risk scores reproduce the validated OOF predictions, then
adds time-dependent discrimination, IPCW Brier score, calibration, proportional-
hazards diagnostics, and cancer-specific paired bootstrap intervals. Outputs are
published without replacing any existing file.
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
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests
PROJECT_ROOT = Path(str(_submission_path('project', '')))
EXPERIMENT_DIR = PROJECT_ROOT / 'experiments/task2_cancer_context'
V1_ROOT = PROJECT_ROOT / 'analysis/cancer_context'
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(V1_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(V1_ROOT / 'scripts'))
import run_task2_cancer_context_cox as original_cox
from task2_cancer_context_audit import grouped_concordance
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
HORIZONS = (6.0, 12.0, 24.0)
DATASET_PATH = PROJECT_ROOT / 'results/task2_cancer_context/task2_cancer_context_v1.csv'
FOLD_PATH = V1_ROOT / 'fold_assignments/task2_frozen_fold_assignments.csv'
REFERENCE_OOF_PATH = PROJECT_ROOT / 'results/task2_cancer_context/full_v2_fixed_bootstrap/oof_predictions.csv'
MODEL_NAMES = ('mutation_only_coxph_pca', 'cancer_type_only', 'mutation_pcs_clinical_tmb_cancer_type', 'mutation_pcs_clinical_tmb_stratified_cancer')
FULL_MODEL = 'mutation_pcs_clinical_tmb_cancer_type'
STRATIFIED_MODEL = 'mutation_pcs_clinical_tmb_stratified_cancer'
BASELINE_MODEL = 'mutation_only_coxph_pca'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'analysis/endpoint_controls')
    parser.add_argument('--bootstrap-reps', type=int, default=N_BOOTSTRAP)
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATASET_PATH)
    folds = pd.read_csv(FOLD_PATH)[['PATIENT_ID', 'fold']]
    data['PATIENT_ID'] = data['PATIENT_ID'].astype(str)
    folds['PATIENT_ID'] = folds['PATIENT_ID'].astype(str)
    data = data.merge(folds, on='PATIENT_ID', how='left', validate='one_to_one')
    if len(data) != 1610 or data['PATIENT_ID'].nunique() != 1610:
        raise AssertionError('Task 2 extension did not load 1,610 unique patients')
    if int(data['event'].sum()) != 802 or set(data['fold']) != {1, 2, 3, 4, 5}:
        raise AssertionError('Task 2 event or fold invariants changed')
    data.index = data['PATIENT_ID']
    return (data, folds)

def selected_model_specs(smoke: bool) -> tuple[original_cox.ModelSpec, ...]:
    wanted = (BASELINE_MODEL, FULL_MODEL, STRATIFIED_MODEL) if smoke else MODEL_NAMES
    by_name = {spec.name: spec for spec in original_cox.MODEL_SPECS}
    return tuple((by_name[name] for name in wanted))

def fit_fold_model(train: pd.DataFrame, test: pd.DataFrame, spec: original_cox.ModelSpec, mutation_train: pd.DataFrame, mutation_test: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_covariates, test_covariates, _ = original_cox.assemble_covariates(train, test, spec, mutation_train if spec.mutation_pcs else pd.DataFrame(index=train.index), mutation_test if spec.mutation_pcs else pd.DataFrame(index=test.index))
    fit_frame = train_covariates.copy()
    fit_frame['duration'] = train['duration'].to_numpy(dtype=float)
    fit_frame['event'] = train['event'].to_numpy(dtype=int)
    strata_column = None
    if spec.cancer_strata:
        strata_column = '__cancer_stratum'
        fit_frame[strata_column] = train[original_cox.MODEL_CANCER_COLUMN].astype(str).to_numpy()
    fitter = CoxPHFitter(penalizer=spec.penalizer)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fitter.fit(fit_frame, duration_col='duration', event_col='event', strata=strata_column)
    convergence = [item for item in captured if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f'Convergence warning for {spec.name}: {convergence}')
    predict_frame = test_covariates.copy()
    if strata_column:
        predict_frame[strata_column] = test[original_cox.MODEL_CANCER_COLUMN].astype(str).to_numpy()
    partial_hazard = fitter.predict_partial_hazard(predict_frame).to_numpy(dtype=float)
    risk = np.log(partial_hazard)
    signature = pd.util.hash_pandas_object(test_covariates, index=False)
    risk = pd.Series(risk, index=test_covariates.index).groupby(signature.to_numpy()).transform('mean').to_numpy(dtype=float)
    return (fitter, fit_frame, predict_frame, risk)

def censoring_km(train: pd.DataFrame) -> KaplanMeierFitter:
    km = KaplanMeierFitter()
    km.fit(train['duration'].to_numpy(dtype=float), event_observed=1 - train['event'].to_numpy(dtype=int))
    return km

def km_value(km: KaplanMeierFitter, time: float, *, left_limit: bool=False) -> float:
    query = np.nextafter(float(time), -np.inf) if left_limit else float(time)
    first_observed_time = float(np.min(km.timeline))
    if query < first_observed_time:
        return 1.0
    value = float(km.predict(query))
    if not np.isfinite(value):
        raise RuntimeError(f'Censoring KM returned a non-finite value at time {query}')
    return max(value, 1e-08)

def ipcw_components(duration: np.ndarray, event: np.ndarray, horizon: float, km: KaplanMeierFitter) -> dict[str, np.ndarray]:
    cases = (duration <= horizon) & (event == 1)
    controls = duration > horizon
    case_weights = np.zeros(len(duration), dtype=float)
    for index in np.flatnonzero(cases):
        case_weights[index] = 1.0 / km_value(km, duration[index], left_limit=True)
    control_weights = np.zeros(len(duration), dtype=float)
    control_weights[controls] = 1.0 / km_value(km, horizon)
    brier_weights = case_weights + control_weights
    known_outcome = cases | controls
    event_by_horizon = cases.astype(int)
    survival_at_horizon = controls.astype(int)
    return {'cases': cases, 'controls': controls, 'case_weights': case_weights, 'control_weights': control_weights, 'brier_weights': brier_weights, 'known_outcome': known_outcome, 'event_by_horizon': event_by_horizon, 'survival_at_horizon': survival_at_horizon}

def weighted_dynamic_auc(risk: np.ndarray, components: dict[str, np.ndarray]) -> tuple[float, float]:
    case_score = risk[components['cases']]
    control_score = risk[components['controls']]
    case_weight = components['case_weights'][components['cases']]
    control_weight = components['control_weights'][components['controls']]
    if len(case_score) == 0 or len(control_score) == 0:
        return (math.nan, 0.0)
    comparison = (case_score[:, None] > control_score[None, :]).astype(float)
    comparison += 0.5 * (case_score[:, None] == control_score[None, :])
    pair_weight = case_weight[:, None] * control_weight[None, :]
    total = float(pair_weight.sum())
    return (float((comparison * pair_weight).sum() / total), total)

def fold_time_metrics(test: pd.DataFrame, risk: np.ndarray, survival: np.ndarray, horizon: float, km: KaplanMeierFitter) -> dict[str, Any]:
    duration = test['duration'].to_numpy(dtype=float)
    event = test['event'].to_numpy(dtype=int)
    components = ipcw_components(duration, event, horizon, km)
    auc, pair_weight = weighted_dynamic_auc(risk, components)
    weights = components['brier_weights']
    truth_survival = components['survival_at_horizon']
    brier = float(np.sum(weights * (truth_survival - survival) ** 2) / len(test))
    return {'auc': auc, 'auc_pair_weight': pair_weight, 'brier': brier, 'n_cases_by_horizon': int(components['cases'].sum()), 'n_controls_at_horizon': int(components['controls'].sum()), 'n_censored_before_horizon': int((~components['known_outcome']).sum()), 'ipcw_weight_sum': float(weights.sum()), 'ipcw_event_by_horizon': components['event_by_horizon'], 'ipcw_weights': weights}

def run_cross_fitted_predictions(data: pd.DataFrame, specs: tuple[original_cox.ModelSpec, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mutation_columns = [column for column in data.columns if column not in original_cox.CONTEXT_COLUMNS + ['fold']]
    if len(mutation_columns) != 468:
        raise AssertionError(f'Expected 468 mutation columns, found {len(mutation_columns)}')
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        train = data[data['fold'] != fold].copy()
        test = data[data['fold'] == fold].copy()
        mutation_train, mutation_test, _ = original_cox.fit_mutation_transformer(train, test, mutation_columns)
        km = censoring_km(train)
        for spec in specs:
            fitter, _, predict_frame, risk = fit_fold_model(train, test, spec, mutation_train, mutation_test)
            survival_frame = fitter.predict_survival_function(predict_frame, times=list(HORIZONS))
            survival = survival_frame.T.reindex(test.index).to_numpy(dtype=float)
            if not np.isfinite(survival).all():
                raise RuntimeError(f'Incomplete survival predictions for {spec.name}, fold {fold}')
            for position, patient_id in enumerate(test.index):
                row = {'PATIENT_ID': patient_id, 'fold': fold, 'model': spec.name, 'duration': float(test.loc[patient_id, 'duration']), 'event': int(test.loc[patient_id, 'event']), 'CANCER_TYPE': test.loc[patient_id, 'CANCER_TYPE'], 'risk_score': float(risk[position])}
                for horizon_index, horizon in enumerate(HORIZONS):
                    row[f'survival_{int(horizon)}m'] = float(survival[position, horizon_index])
                prediction_rows.append(row)
            for horizon_index, horizon in enumerate(HORIZONS):
                horizon_risk = 1.0 - survival[:, horizon_index]
                metrics = fold_time_metrics(test, horizon_risk, survival[:, horizon_index], horizon, km)
                fold_metric_rows.append({'model': spec.name, 'fold': fold, 'horizon_months': horizon, **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)}})
                for position, patient_id in enumerate(test.index):
                    weight_rows.append({'PATIENT_ID': patient_id, 'fold': fold, 'model': spec.name, 'horizon_months': horizon, 'event_by_horizon': int(metrics['ipcw_event_by_horizon'][position]), 'ipcw_weight': float(metrics['ipcw_weights'][position])})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(fold_metric_rows), pd.DataFrame(weight_rows))

def verify_risk_reproduction(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_OOF_PATH)
    reference = reference[reference['model'].isin(predictions['model'].unique())][['PATIENT_ID', 'fold', 'model', 'risk_score']].rename(columns={'risk_score': 'reference_risk_score'})
    joined = predictions.merge(reference, on=['PATIENT_ID', 'fold', 'model'], how='left', validate='one_to_one')
    joined['absolute_difference'] = (joined['risk_score'] - joined['reference_risk_score']).abs()
    audit = joined.groupby('model', sort=False).agg(n=('PATIENT_ID', 'size'), max_absolute_difference=('absolute_difference', 'max'), mean_absolute_difference=('absolute_difference', 'mean')).reset_index()
    if (audit['n'] != 1610).any() or (audit['max_absolute_difference'] > 1e-10).any():
        raise AssertionError(f'Cross-fitted risk reproduction failed:\n{audit}')
    return audit

def pooled_time_metrics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for (model, horizon), metric_frame in fold_metrics.groupby(['model', 'horizon_months'], sort=False):
        pair_weights = metric_frame['auc_pair_weight'].to_numpy(dtype=float)
        aucs = metric_frame['auc'].to_numpy(dtype=float)
        summary_rows.append({'model': model, 'horizon_months': horizon, 'fold_weighted_dynamic_auc': float(np.average(aucs, weights=pair_weights)), 'fold_mean_dynamic_auc': float(np.mean(aucs)), 'fold_sd_dynamic_auc': float(np.std(aucs, ddof=1)), 'fold_mean_ipcw_brier': float(metric_frame['brier'].mean()), 'fold_sd_ipcw_brier': float(metric_frame['brier'].std(ddof=1)), 'n_cases_by_horizon': int(metric_frame['n_cases_by_horizon'].sum()), 'n_controls_at_horizon': int(metric_frame['n_controls_at_horizon'].sum()), 'n_censored_before_horizon': int(metric_frame['n_censored_before_horizon'].sum())})
        pred = predictions[predictions['model'] == model][['PATIENT_ID', f'survival_{int(horizon)}m']].copy()
        work = weights[(weights['model'] == model) & (weights['horizon_months'] == horizon)].merge(pred, on='PATIENT_ID', how='left', validate='one_to_one')
        work = work[work['ipcw_weight'] > 0].copy()
        predicted_risk = np.clip(1.0 - work[f'survival_{int(horizon)}m'].to_numpy(dtype=float), 1e-06, 1 - 1e-06)
        logit = np.log(predicted_risk / (1 - predicted_risk))
        outcome = work['event_by_horizon'].to_numpy(dtype=int)
        ipcw = work['ipcw_weight'].to_numpy(dtype=float)
        try:
            fit = sm.GLM(outcome, sm.add_constant(logit), family=sm.families.Binomial(), freq_weights=ipcw).fit()
            intercept = float(fit.params[0])
            slope = float(fit.params[1])
        except Exception:
            intercept = math.nan
            slope = math.nan
        calibration_rows.append({'model': model, 'horizon_months': horizon, 'n_known_outcomes': len(work), 'ipcw_observed_event_risk': float(np.average(outcome, weights=ipcw)), 'mean_predicted_event_risk': float(np.mean(predicted_risk)), 'ipcw_calibration_intercept': intercept, 'ipcw_calibration_slope': slope})
    return (pd.DataFrame(summary_rows), pd.DataFrame(calibration_rows))

def fit_full_model_for_ph(data: pd.DataFrame, spec: original_cox.ModelSpec) -> tuple[CoxPHFitter, pd.DataFrame]:
    mutation_columns = [column for column in data.columns if column not in original_cox.CONTEXT_COLUMNS + ['fold']]
    mutation_train, mutation_test, _ = original_cox.fit_mutation_transformer(data, data, mutation_columns)
    fitter, fit_frame, _, _ = fit_fold_model(data, data, spec, mutation_train, mutation_test)
    return (fitter, fit_frame)

def proportional_hazards_diagnostics(data: pd.DataFrame, specs: tuple[original_cox.ModelSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.name not in {FULL_MODEL, STRATIFIED_MODEL}:
            continue
        fitter, fit_frame = fit_full_model_for_ph(data, spec)
        result = proportional_hazard_test(fitter, fit_frame, time_transform='rank').summary
        p_values = result['p'].to_numpy(dtype=float)
        q_values = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]
        for (covariate, row), q_value in zip(result.iterrows(), q_values, strict=True):
            rows.append({'model': spec.name, 'covariate': covariate, 'test_statistic': float(row['test_statistic']), 'p_value': float(row['p']), 'fdr_bh_q_value_within_model': float(q_value), 'violates_q_lt_0_05': bool(q_value < 0.05), 'time_transform': 'rank', 'fit_scope': 'full_cohort_diagnostic_only'})
    return pd.DataFrame(rows)

def cancer_bootstrap(reps: int) -> pd.DataFrame:
    source = pd.read_csv(REFERENCE_OOF_PATH)
    source = source[source['model'].isin([BASELINE_MODEL, FULL_MODEL, STRATIFIED_MODEL])]
    wide = source.pivot(index=['PATIENT_ID', 'fold', 'duration', 'event', 'CANCER_TYPE'], columns='model', values='risk_score').reset_index()
    rows: list[dict[str, Any]] = []
    for cancer, frame in wide.groupby('CANCER_TYPE', sort=False):
        if len(frame) < 40 or int(frame['event'].sum()) < 20:
            continue
        point: dict[str, float] = {}
        for model in [BASELINE_MODEL, FULL_MODEL, STRATIFIED_MODEL]:
            work = frame.rename(columns={model: 'risk_score'})
            point[model] = grouped_concordance(work, ['fold']).c_index
        bootstrap_values = {model: [] for model in point}
        delta_values = {FULL_MODEL: [], STRATIFIED_MODEL: []}
        rng = np.random.default_rng(RANDOM_STATE)
        for _ in range(reps):
            sampled_parts: list[pd.DataFrame] = []
            for _, fold_frame in frame.groupby('fold', sort=True):
                positions = rng.choice(np.arange(len(fold_frame)), size=len(fold_frame), replace=True)
                sampled_parts.append(fold_frame.iloc[positions])
            sampled = pd.concat(sampled_parts, ignore_index=True)
            estimates: dict[str, float] = {}
            for model in point:
                work = sampled.rename(columns={model: 'risk_score'})
                estimates[model] = grouped_concordance(work, ['fold']).c_index
                bootstrap_values[model].append(estimates[model])
            for model in delta_values:
                delta_values[model].append(estimates[model] - estimates[BASELINE_MODEL])
        for model, estimate in point.items():
            values = np.asarray(bootstrap_values[model], dtype=float)
            row = {'cancer_type': cancer, 'n': len(frame), 'events': int(frame['event'].sum()), 'model': model, 'c_index': estimate, 'bootstrap_ci95_low': float(np.nanquantile(values, 0.025)), 'bootstrap_ci95_high': float(np.nanquantile(values, 0.975)), 'bootstrap_reps': reps, 'bootstrap_unit': 'patient', 'bootstrap_strata': 'frozen_fold_within_cancer', 'random_state': RANDOM_STATE, 'delta_vs_mutation_only': math.nan, 'delta_ci95_low': math.nan, 'delta_ci95_high': math.nan}
            if model in delta_values:
                deltas = np.asarray(delta_values[model], dtype=float)
                row['delta_vs_mutation_only'] = estimate - point[BASELINE_MODEL]
                row['delta_ci95_low'] = float(np.nanquantile(deltas, 0.025))
                row['delta_ci95_high'] = float(np.nanquantile(deltas, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)

def write_new(path: Path, writer: Callable[[Path], None]) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_csv_new(frame: pd.DataFrame, path: Path) -> None:
    write_new(path, lambda temporary: frame.to_csv(temporary, index=False))

def write_json_new(payload: Any, path: Path) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    write_new(path, lambda temporary: temporary.write_text(text, encoding='utf-8'))

def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    suffix = '_smoke_v3' if args.smoke else '_v3'
    targets = {'predictions': output / f'predictions/task2_time_dependent_oof_predictions{suffix}.csv', 'reproduction': output / f'metadata/task2_risk_reproduction_audit{suffix}.csv', 'fold_metrics': output / f'metrics/task2_time_dependent_fold_metrics{suffix}.csv', 'summary': output / f'metrics/task2_time_dependent_summary{suffix}.csv', 'calibration': output / f'metrics/task2_ipcw_calibration{suffix}.csv', 'ph': output / f'metrics/task2_proportional_hazards_diagnostics{suffix}.csv', 'cancer_bootstrap': output / f'metrics/task2_per_cancer_bootstrap{suffix}.csv', 'manifest': output / f'metadata/task2_priority1_manifest{suffix}.json'}
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f'Planned outputs already exist: {existing}')
    data, _ = load_data()
    specs = selected_model_specs(args.smoke)
    predictions, fold_metrics, weights = run_cross_fitted_predictions(data, specs)
    reproduction = verify_risk_reproduction(predictions)
    summary, calibration = pooled_time_metrics(predictions, fold_metrics, weights)
    ph = proportional_hazards_diagnostics(data, specs)
    cancer = pd.DataFrame() if args.smoke else cancer_bootstrap(args.bootstrap_reps)
    write_csv_new(predictions, targets['predictions'])
    write_csv_new(reproduction, targets['reproduction'])
    write_csv_new(fold_metrics, targets['fold_metrics'])
    write_csv_new(summary, targets['summary'])
    write_csv_new(calibration, targets['calibration'])
    write_csv_new(ph, targets['ph'])
    write_csv_new(cancer, targets['cancer_bootstrap'])
    manifest = {'analysis': 'Task 2 priority-1 survival-model adequacy extension', 'created_utc': datetime.now(timezone.utc).isoformat(), 'smoke': args.smoke, 'random_state': RANDOM_STATE, 'horizons_months': HORIZONS, 'models': [asdict(spec) for spec in specs], 'bootstrap_reps': 0 if args.smoke else args.bootstrap_reps, 'time_dependent_auc': 'fold-specific cumulative/dynamic IPCW AUC ranked by horizon-specific predicted event probability (1-S(t)); censoring KM fitted on each outer training fold', 'brier': 'fold-specific IPCW Brier score with censoring KM fitted on each outer training fold', 'calibration': 'pooled OOF IPCW logistic recalibration at each horizon', 'ph_test': 'Schoenfeld-residual proportional_hazard_test with rank transform and within-model BH FDR', 'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'lifelines', 'scikit-learn', 'statsmodels']}, 'inputs': {str(path): sha256(path) for path in [DATASET_PATH, FOLD_PATH, REFERENCE_OOF_PATH]}, 'outputs': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in targets.items() if key != 'manifest'}}
    write_json_new(manifest, targets['manifest'])
    print(json.dumps({'status': 'ok', 'outputs': {k: str(v) for k, v in targets.items()}}, indent=2))
if __name__ == '__main__':
    main()
