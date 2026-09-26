"""Evaluate fixed-fold out-of-fold Task 2 cancer-context Cox predictions."""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from lifelines.utils.concordance import _concordance_summary_statistics
from task2_context_common import DEFAULT_RESULTS_DIR, RANDOM_SEED, write_csv_new
BASELINE_MODEL = 'mutation_only_coxph_pca'
STRATIFIED_MODEL = 'mutation_pcs_clinical_tmb_stratified_cancer'
STRICT_MIN_N = 40
STRICT_MIN_EVENTS = 20
STRICT_MIN_PAIRS = 100
STRICT_MIN_FOLDS_WITH_PAIRS = 4
MODEL_DESCRIPTIONS = {BASELINE_MODEL: 'Mutation-only; train-fold StandardScaler + 10-PC PCA; CoxPH', 'cancer_type_only': 'Model cancer type one-hot covariates only; CoxPH', 'clinical_tmb': 'Sex, age group, drug type, and log1p canonical TMB; CoxPH', 'mutation_pcs_cancer_type': 'Mutation PCs plus model cancer type covariates; CoxPH', 'mutation_pcs_clinical_tmb_cancer_type': 'Mutation PCs plus sex, age group, drug type, log1p TMB, and cancer type; CoxPH', STRATIFIED_MODEL: 'Mutation PCs plus sex, age group, drug type, and log1p TMB; cancer-stratified Cox baseline hazards'}

@dataclass(frozen=True)
class ConcordanceResult:
    c_index: float
    permissible_pairs: int
    concordant_pairs: int
    tied_prediction_pairs: int

def concordance_with_pairs(duration: np.ndarray | pd.Series, event: np.ndarray | pd.Series, risk_score: np.ndarray | pd.Series) -> ConcordanceResult:
    """Harrell C for a risk score where larger means earlier event/higher risk."""
    duration_array = np.asarray(duration, dtype=float)
    event_array = np.asarray(event, dtype=int)
    risk_array = np.asarray(risk_score, dtype=float)
    if not len(duration_array) == len(event_array) == len(risk_array):
        raise ValueError('duration, event, and risk_score lengths differ')
    if len(duration_array) == 0:
        return ConcordanceResult(math.nan, 0, 0, 0)
    if not (np.isfinite(duration_array).all() and np.isfinite(event_array).all() and np.isfinite(risk_array).all()):
        raise ValueError('Concordance inputs contain non-finite values')
    if not set(np.unique(event_array).tolist()).issubset({0, 1}):
        raise ValueError('event must be binary')
    correct, tied, pairs = _concordance_summary_statistics(duration_array, -risk_array, event_array)
    pairs_int = int(pairs)
    if pairs_int == 0:
        score = math.nan
    else:
        score = float((float(correct) + 0.5 * float(tied)) / float(pairs))
    return ConcordanceResult(score, pairs_int, int(correct), int(tied))

def aggregate_concordance(results: list[ConcordanceResult]) -> ConcordanceResult:
    pairs = int(sum((result.permissible_pairs for result in results)))
    correct = int(sum((result.concordant_pairs for result in results)))
    tied = int(sum((result.tied_prediction_pairs for result in results)))
    score = math.nan if pairs == 0 else float((correct + 0.5 * tied) / pairs)
    return ConcordanceResult(score, pairs, correct, tied)

def crossfitted_concordance(frame: pd.DataFrame) -> ConcordanceResult:
    if 'fold' not in frame.columns:
        return concordance_with_pairs(frame['duration'], frame['event'], frame['risk_score'])
    results = [concordance_with_pairs(group['duration'], group['event'], group['risk_score']) for _, group in frame.groupby('fold', sort=True)]
    return aggregate_concordance(results)

def cancer_metrics(frame: pd.DataFrame, *, min_n: int, min_events: int, min_pairs: int, min_folds_with_pairs: int=1) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cancer, group in frame.groupby('CANCER_TYPE', sort=False, dropna=False):
        concatenated = concordance_with_pairs(group['duration'], group['event'], group['risk_score'])
        if 'fold' in group.columns:
            fold_components = [concordance_with_pairs(fold_group['duration'], fold_group['event'], fold_group['risk_score']) for _, fold_group in group.groupby('fold', sort=True)]
        else:
            fold_components = [concatenated]
        result = aggregate_concordance(fold_components)
        folds_with_pairs = int(sum((component.permissible_pairs > 0 for component in fold_components)))
        n = int(len(group))
        events = int(group['event'].sum())
        eligible = bool(n >= min_n and events >= min_events and (result.permissible_pairs >= min_pairs) and (folds_with_pairs >= min_folds_with_pairs) and np.isfinite(result.c_index))
        rows.append({'cancer_type': cancer, 'n': n, 'events': events, 'censored': int(n - events), 'event_rate': float(events / n) if n else math.nan, 'permissible_pairs': result.permissible_pairs, 'concordant_pairs': result.concordant_pairs, 'tied_prediction_pairs': result.tied_prediction_pairs, 'folds_with_permissible_pairs': folds_with_pairs, 'c_index': result.c_index if eligible else math.nan, 'crossfitted_pair_pooled_c_index': result.c_index, 'concatenated_oof_c_index': concatenated.c_index, 'concatenated_oof_permissible_pairs': concatenated.permissible_pairs, 'eligible': eligible, 'status': 'ok' if eligible else 'not_reportable', 'exclusion_reason': None if eligible else f'support below threshold: n={n}, events={events}, pairs={result.permissible_pairs}, folds_with_pairs={folds_with_pairs}', 'eligibility_rule': f'n>={min_n};events>={min_events};pairs>={min_pairs};folds_with_pairs>={min_folds_with_pairs}'})
    metrics = pd.DataFrame(rows)
    eligible = metrics[metrics['eligible']].copy()
    if eligible.empty:
        summary = {'sample_weighted_c_index': math.nan, 'pair_weighted_c_index': math.nan, 'eligible_cancers': 0, 'eligible_samples': 0, 'eligible_events': 0, 'eligible_pairs': 0}
    else:
        summary = {'sample_weighted_c_index': float(np.average(eligible['c_index'], weights=eligible['n'])), 'pair_weighted_c_index': float(np.average(eligible['c_index'], weights=eligible['permissible_pairs'])), 'eligible_cancers': int(len(eligible)), 'eligible_samples': int(eligible['n'].sum()), 'eligible_events': int(eligible['events'].sum()), 'eligible_pairs': int(eligible['permissible_pairs'].sum())}
    return (metrics, summary)

def validate_oof_predictions(oof: pd.DataFrame) -> None:
    required = {'PATIENT_ID', 'fold', 'model', 'risk_score', 'duration', 'event', 'CANCER_TYPE', 'CANCER_TYPE_MODEL'}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f'OOF predictions are missing columns: {missing}')
    if oof.duplicated(['PATIENT_ID', 'model']).any():
        raise ValueError('Each patient must have exactly one OOF prediction per model')
    expected_models = set(MODEL_DESCRIPTIONS)
    observed_models = set(oof['model'].astype(str))
    if observed_models != expected_models:
        raise ValueError(f'Expected the six prespecified models; missing={sorted(expected_models - observed_models)}, unexpected={sorted(observed_models - expected_models)}')
    model_counts = oof.groupby('model')['PATIENT_ID'].nunique()
    if model_counts.nunique() != 1:
        raise ValueError(f'Models do not cover the same OOF patients: {model_counts.to_dict()}')
    if not np.isfinite(oof['risk_score'].to_numpy(dtype=float)).all():
        raise ValueError('OOF risk scores contain non-finite values')
    baseline_ids = set(oof.loc[oof['model'] == BASELINE_MODEL, 'PATIENT_ID'].astype(str))
    for model, group in oof.groupby('model'):
        if set(group['PATIENT_ID'].astype(str)) != baseline_ids:
            raise ValueError(f'Patient set for {model} differs from the paired baseline')
    metadata_columns = ['duration', 'event', 'fold', 'CANCER_TYPE', 'CANCER_TYPE_MODEL']
    metadata_nunique = oof.groupby('PATIENT_ID')[metadata_columns].nunique(dropna=False)
    inconsistent = metadata_nunique.gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError(f'Cross-model OOF metadata differ for patients: {metadata_nunique.index[inconsistent].astype(str).tolist()[:5]}')
    fold_sets = oof.groupby('model')['fold'].apply(lambda values: tuple(sorted(set(values))))
    if fold_sets.nunique() != 1:
        raise ValueError(f'Models do not cover the same folds: {fold_sets.to_dict()}')

def build_fold_results(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, fold), group in oof.groupby(['model', 'fold'], sort=True):
        pooled = concordance_with_pairs(group['duration'], group['event'], group['risk_score'])
        _, within = cancer_metrics(group, min_n=2, min_events=1, min_pairs=1, min_folds_with_pairs=1)
        primary_scope = 'within_cancer_sample_weighted' if model == STRATIFIED_MODEL else 'pooled'
        primary_score = within['sample_weighted_c_index'] if model == STRATIFIED_MODEL else pooled.c_index
        rows.append({'model': model, 'fold': int(fold), 'n_test': int(len(group)), 'events_test': int(group['event'].sum()), 'pooled_c_index': pooled.c_index if model != STRATIFIED_MODEL else math.nan, 'pooled_c_index_appropriate': model != STRATIFIED_MODEL, 'pooled_permissible_pairs': pooled.permissible_pairs if model != STRATIFIED_MODEL else math.nan, 'within_cancer_sample_weighted_c_index': within['sample_weighted_c_index'], 'within_cancer_pair_weighted_c_index': within['pair_weighted_c_index'], 'within_cancer_eligible_cancers': within['eligible_cancers'], 'within_cancer_eligible_samples': within['eligible_samples'], 'within_cancer_eligible_events': within['eligible_events'], 'within_cancer_permissible_pairs': within['eligible_pairs'], 'primary_scope': primary_scope, 'primary_c_index': primary_score})
    return pd.DataFrame(rows).sort_values(['model', 'fold']).reset_index(drop=True)

def build_per_cancer_results(oof: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for model, group in oof.groupby('model', sort=True):
        metrics, _ = cancer_metrics(group, min_n=STRICT_MIN_N, min_events=STRICT_MIN_EVENTS, min_pairs=STRICT_MIN_PAIRS, min_folds_with_pairs=STRICT_MIN_FOLDS_WITH_PAIRS)
        metrics.insert(0, 'model', model)
        pieces.append(metrics)
    return pd.concat(pieces, ignore_index=True)

def build_summary(oof: pd.DataFrame, fold_results: pd.DataFrame, per_cancer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in oof.groupby('model', sort=True):
        folds = fold_results[fold_results['model'] == model]
        pooled = concordance_with_pairs(group['duration'], group['event'], group['risk_score'])
        crossfitted = crossfitted_concordance(group)
        _, within = cancer_metrics(group, min_n=STRICT_MIN_N, min_events=STRICT_MIN_EVENTS, min_pairs=STRICT_MIN_PAIRS, min_folds_with_pairs=STRICT_MIN_FOLDS_WITH_PAIRS)
        has_global_score = model != STRATIFIED_MODEL
        rows.append({'model': model, 'description': MODEL_DESCRIPTIONS.get(model, model), 'n_oof': int(len(group)), 'events_oof': int(group['event'].sum()), 'n_folds': int(folds['fold'].nunique()), 'fold_primary_scope': folds['primary_scope'].iloc[0], 'fold_mean_c_index': float(folds['primary_c_index'].mean()), 'fold_sd_c_index_ddof1': float(folds['primary_c_index'].std(ddof=1)), 'pooled_oof_c_index': pooled.c_index if has_global_score else math.nan, 'pooled_oof_appropriate': False, 'pooled_oof_status': 'descriptive_only_cross_fold_score_scale_sensitive' if has_global_score else 'not_applicable_cross_stratum_and_cross_fold', 'pooled_oof_permissible_pairs': pooled.permissible_pairs if has_global_score else math.nan, 'crossfitted_pair_pooled_c_index': crossfitted.c_index if has_global_score else math.nan, 'crossfitted_permissible_pairs': crossfitted.permissible_pairs if has_global_score else math.nan, 'crossfitted_pair_pooled_appropriate': has_global_score, 'cross_fold_score_scale_caveat': has_global_score, 'within_cancer_sample_weighted_oof_c_index': within['sample_weighted_c_index'], 'within_cancer_pair_weighted_oof_c_index': within['pair_weighted_c_index'], 'within_cancer_eligible_cancers': within['eligible_cancers'], 'within_cancer_eligible_samples': within['eligible_samples'], 'within_cancer_eligible_events': within['eligible_events'], 'within_cancer_permissible_pairs': within['eligible_pairs'], 'overall_primary_scope': 'within_cancer_sample_weighted' if model == STRATIFIED_MODEL else 'crossfitted_pair_pooled', 'overall_primary_c_index': within['sample_weighted_c_index'] if model == STRATIFIED_MODEL else crossfitted.c_index, 'cox_penalizer': 0.0})
    return pd.DataFrame(rows).sort_values('model').reset_index(drop=True)

def build_fold_paired_comparisons(fold_results: pd.DataFrame) -> pd.DataFrame:
    baseline = fold_results[fold_results['model'] == BASELINE_MODEL].set_index('fold')
    if baseline.empty:
        raise ValueError('Mutation-only baseline missing from fold results')
    rows: list[dict[str, Any]] = []
    for model in sorted(set(fold_results['model']).difference({BASELINE_MODEL})):
        variant = fold_results[fold_results['model'] == model].set_index('fold')
        common_folds = baseline.index.intersection(variant.index)
        scope = 'within_cancer_sample_weighted' if model == STRATIFIED_MODEL else 'pooled'
        score_column = 'within_cancer_sample_weighted_c_index' if scope == 'within_cancer_sample_weighted' else 'pooled_c_index'
        for fold in common_folds:
            baseline_score = float(baseline.loc[fold, score_column])
            model_score = float(variant.loc[fold, score_column])
            rows.append({'row_type': 'fold_delta', 'model': model, 'comparison_scope': scope, 'fold': int(fold), 'baseline_score': baseline_score, 'model_score': model_score, 'delta_vs_mutation_only': model_score - baseline_score, 'n_fold_deltas': math.nan, 'fold_delta_sd_ddof1': math.nan, 'positive_folds': math.nan, 'bootstrap_reps': math.nan, 'bootstrap_successful_reps': math.nan, 'bootstrap_mean_delta': math.nan, 'bootstrap_ci95_low': math.nan, 'bootstrap_ci95_high': math.nan, 'bootstrap_seed': math.nan, 'bootstrap_scheme': None})
        deltas = [row['delta_vs_mutation_only'] for row in rows if row['model'] == model and row['row_type'] == 'fold_delta']
        rows.append({'row_type': 'fold_delta_summary', 'model': model, 'comparison_scope': scope, 'fold': 'mean', 'baseline_score': float(baseline.loc[common_folds, score_column].mean()), 'model_score': float(variant.loc[common_folds, score_column].mean()), 'delta_vs_mutation_only': float(np.mean(deltas)), 'n_fold_deltas': int(len(deltas)), 'fold_delta_sd_ddof1': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else math.nan, 'positive_folds': int(np.sum(np.asarray(deltas) > 0)), 'bootstrap_reps': math.nan, 'bootstrap_successful_reps': math.nan, 'bootstrap_mean_delta': math.nan, 'bootstrap_ci95_low': math.nan, 'bootstrap_ci95_high': math.nan, 'bootstrap_seed': math.nan, 'bootstrap_scheme': None, 'bootstrap_valid_fraction': math.nan, 'bootstrap_status': None})
    return pd.DataFrame(rows)

def _wide_oof(oof: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = ['PATIENT_ID', 'duration', 'event', 'fold', 'CANCER_TYPE', 'CANCER_TYPE_MODEL']
    metadata = oof[metadata_columns].drop_duplicates().set_index('PATIENT_ID')
    if not metadata.index.is_unique:
        raise ValueError('Inconsistent OOF metadata after cross-model validation')
    risk = oof.pivot(index='PATIENT_ID', columns='model', values='risk_score')
    wide = metadata.join(risk, how='inner')
    if wide[risk.columns].isna().any().any():
        raise ValueError('Missing model risk score after OOF pivot')
    return wide.reset_index()

def _stratified_bootstrap_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for level in pd.unique(strata):
        indices = np.flatnonzero(strata == level)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)

def _within_score_from_wide(frame: pd.DataFrame, risk_column: str, *, eligible_cancers: tuple[str, ...] | None=None, cancer_weights: dict[str, float] | None=None) -> float:
    renamed = frame.rename(columns={risk_column: 'risk_score'})
    if eligible_cancers is not None:
        metrics, _ = cancer_metrics(renamed, min_n=1, min_events=0, min_pairs=1, min_folds_with_pairs=1)
        metrics = metrics.copy()
        metrics['__cancer_key'] = metrics['cancer_type'].astype(str)
        selected = metrics[metrics['__cancer_key'].isin(eligible_cancers)].copy()
        observed = set(selected['__cancer_key'])
        if observed != set(eligible_cancers):
            return math.nan
        selected = selected.set_index('__cancer_key').loc[list(eligible_cancers)]
        scores = selected['crossfitted_pair_pooled_c_index'].to_numpy(dtype=float)
        if cancer_weights is None:
            weights = selected['n'].to_numpy(dtype=float)
        else:
            if set(cancer_weights) != set(eligible_cancers):
                raise ValueError('Frozen cancer weights do not match the eligible cancer set')
            weights = np.asarray([cancer_weights[cancer] for cancer in eligible_cancers], dtype=float)
        if not np.isfinite(scores).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
            return math.nan
        return float(np.average(scores, weights=weights))
    _, summary = cancer_metrics(renamed, min_n=STRICT_MIN_N, min_events=STRICT_MIN_EVENTS, min_pairs=STRICT_MIN_PAIRS, min_folds_with_pairs=STRICT_MIN_FOLDS_WITH_PAIRS)
    return float(summary['sample_weighted_c_index'])

def _crossfitted_score_from_wide(frame: pd.DataFrame, risk_column: str) -> float:
    renamed = frame.rename(columns={risk_column: 'risk_score'})
    return float(crossfitted_concordance(renamed).c_index)

def bootstrap_oof_deltas(oof: pd.DataFrame, *, n_bootstrap: int, seed: int=RANDOM_SEED) -> pd.DataFrame:
    if n_bootstrap < 1:
        return pd.DataFrame()
    wide = _wide_oof(oof)
    model_columns = sorted(set(oof['model']))
    if BASELINE_MODEL not in model_columns:
        raise ValueError('Mutation-only baseline missing from OOF predictions')
    variants = [model for model in model_columns if model != BASELINE_MODEL]
    comparisons = [(model, scope) for model in variants for scope in ('pooled_crossfitted_pair_weighted', 'within_cancer_sample_weighted') if not (model == STRATIFIED_MODEL and scope == 'pooled_crossfitted_pair_weighted')]
    baseline_for_eligibility = wide.rename(columns={BASELINE_MODEL: 'risk_score'})
    eligibility_metrics, _ = cancer_metrics(baseline_for_eligibility, min_n=STRICT_MIN_N, min_events=STRICT_MIN_EVENTS, min_pairs=STRICT_MIN_PAIRS, min_folds_with_pairs=STRICT_MIN_FOLDS_WITH_PAIRS)
    frozen_eligible_cancers = tuple(sorted(eligibility_metrics.loc[eligibility_metrics['eligible'], 'cancer_type'].astype(str).tolist()))
    if not frozen_eligible_cancers:
        raise ValueError('No original-OOF cancer groups meet the reporting thresholds')
    eligible_original = eligibility_metrics[eligibility_metrics['eligible']].copy()
    eligible_original['__cancer_key'] = eligible_original['cancer_type'].astype(str)
    eligible_original = eligible_original.set_index('__cancer_key')
    frozen_eligible_n = {cancer: int(eligible_original.loc[cancer, 'n']) for cancer in frozen_eligible_cancers}
    frozen_eligible_total_n = int(sum(frozen_eligible_n.values()))
    frozen_cancer_weights = {cancer: n / frozen_eligible_total_n for cancer, n in frozen_eligible_n.items()}
    point_baseline_pooled = _crossfitted_score_from_wide(wide, BASELINE_MODEL)
    point_baseline_within = _within_score_from_wide(wide, BASELINE_MODEL, eligible_cancers=frozen_eligible_cancers, cancer_weights=frozen_cancer_weights)
    point_deltas: dict[tuple[str, str], float] = {}
    for model, scope in comparisons:
        if scope == 'pooled_crossfitted_pair_weighted':
            model_score = _crossfitted_score_from_wide(wide, model)
            point_deltas[model, scope] = model_score - point_baseline_pooled
        else:
            point_deltas[model, scope] = _within_score_from_wide(wide, model, eligible_cancers=frozen_eligible_cancers, cancer_weights=frozen_cancer_weights) - point_baseline_within
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str], list[float]] = {comparison: [] for comparison in comparisons}
    within_invalid_replicates = 0
    bootstrap_strata = (wide['CANCER_TYPE'].astype(str) + '::fold=' + wide['fold'].astype(str)).to_numpy()
    for _ in range(n_bootstrap):
        indices = _stratified_bootstrap_indices(bootstrap_strata, rng)
        sample = wide.iloc[indices]
        baseline_pooled = _crossfitted_score_from_wide(sample, BASELINE_MODEL)
        baseline_within = _within_score_from_wide(sample, BASELINE_MODEL, eligible_cancers=frozen_eligible_cancers, cancer_weights=frozen_cancer_weights)
        within_scores = {model: _within_score_from_wide(sample, model, eligible_cancers=frozen_eligible_cancers, cancer_weights=frozen_cancer_weights) for model in variants}
        within_replicate_valid = bool(np.isfinite(baseline_within) and all((np.isfinite(score) for score in within_scores.values())))
        if not within_replicate_valid:
            within_invalid_replicates += 1
        for model, scope in comparisons:
            if scope == 'pooled_crossfitted_pair_weighted':
                model_score = _crossfitted_score_from_wide(sample, model)
                delta = model_score - baseline_pooled
            else:
                delta = within_scores[model] - baseline_within if within_replicate_valid else math.nan
            if np.isfinite(delta):
                draws[model, scope].append(float(delta))
    rows: list[dict[str, Any]] = []
    for model, scope in comparisons:
        values = np.asarray(draws[model, scope], dtype=float)
        if len(values):
            low, high = np.percentile(values, [2.5, 97.5])
            mean = float(np.mean(values))
        else:
            low = high = mean = math.nan
        valid_fraction = float(len(values) / n_bootstrap)
        status = 'ok' if valid_fraction >= 0.95 else 'insufficient_valid_bootstrap_replicates'
        if status != 'ok':
            low = high = math.nan
        rows.append({'row_type': 'bootstrap_oof_delta', 'model': model, 'comparison_scope': scope, 'fold': 'all_oof', 'baseline_score': point_baseline_pooled if scope == 'pooled_crossfitted_pair_weighted' else point_baseline_within, 'model_score': point_deltas[model, scope] + (point_baseline_pooled if scope == 'pooled_crossfitted_pair_weighted' else point_baseline_within), 'delta_vs_mutation_only': point_deltas[model, scope], 'n_fold_deltas': math.nan, 'fold_delta_sd_ddof1': math.nan, 'positive_folds': math.nan, 'bootstrap_reps': int(n_bootstrap), 'bootstrap_successful_reps': int(len(values)), 'bootstrap_invalid_reps': int(n_bootstrap - len(values)), 'bootstrap_invalid_frozen_cancer_score_reps': int(within_invalid_replicates) if scope == 'within_cancer_sample_weighted' else 0, 'bootstrap_valid_fraction': valid_fraction, 'bootstrap_status': status, 'bootstrap_mean_delta': mean, 'bootstrap_ci95_low': float(low), 'bootstrap_ci95_high': float(high), 'bootstrap_seed': int(seed), 'bootstrap_scheme': 'paired patient bootstrap stratified by original broad cancer type and fold; within-cancer eligible set frozen from original OOF; percentile 95% CI', 'within_cancer_eligibility_mode': 'frozen_from_original_oof' if scope == 'within_cancer_sample_weighted' else 'not_applicable', 'within_cancer_frozen_cancer_count': len(frozen_eligible_cancers) if scope == 'within_cancer_sample_weighted' else math.nan, 'within_cancer_frozen_cancers': '|'.join(frozen_eligible_cancers) if scope == 'within_cancer_sample_weighted' else None, 'within_cancer_frozen_weights': json.dumps(frozen_cancer_weights, sort_keys=True) if scope == 'within_cancer_sample_weighted' else None})
    return pd.DataFrame(rows)

def evaluate_oof(oof: pd.DataFrame, *, bootstrap_reps: int, bootstrap_seed: int=RANDOM_SEED) -> dict[str, pd.DataFrame]:
    validate_oof_predictions(oof)
    fold_results = build_fold_results(oof)
    per_cancer = build_per_cancer_results(oof)
    summary = build_summary(oof, fold_results, per_cancer)
    fold_paired = build_fold_paired_comparisons(fold_results)
    bootstrap = bootstrap_oof_deltas(oof, n_bootstrap=bootstrap_reps, seed=bootstrap_seed)
    paired = pd.concat([fold_paired, bootstrap], ignore_index=True, sort=False)
    return {'fold_results': fold_results, 'summary': summary, 'per_cancer_results': per_cancer, 'paired_comparison': paired, 'bootstrap_comparison': bootstrap}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=RANDOM_SEED)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    oof = pd.read_csv(args.oof_predictions)
    outputs = evaluate_oof(oof, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed)
    for name, frame in outputs.items():
        path = args.output_dir / f'{name}.csv'
        write_csv_new(frame, path)
        print(f'{name}={path}')
if __name__ == '__main__':
    main()
