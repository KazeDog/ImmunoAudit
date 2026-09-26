"""Core utilities for the grouped Task 1/3/5 audit.

The functions in this module are deliberately independent of manuscript files.
They enforce patient-grouped partitions, fold-local preprocessing, deterministic
model fitting and cluster-aware uncertainty calculations.
"""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 42

class AuditInvariantError(RuntimeError):
    """Raised when an input or evaluation invariant is violated."""

@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    positive: int
    negative: int
    auroc: float
    auprc: float
    accuracy: float
    macro_f1: float

def set_deterministic_seed(seed: int=RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)

def canonical_task1_group(sample_id: str) -> str:
    """Map the 48 GSE120575 biopsy labels to 32 conservative subject tokens."""
    value = str(sample_id).strip()
    for prefix in ('Pre_', 'Post_'):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    for suffix in ('_T_enriched', '_myeloid_enriched', '_2'):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    if not value:
        raise AuditInvariantError(f'Could not derive Task 1 group from {sample_id!r}')
    return value

def validate_binary(values: Sequence[int], label: str='outcome') -> np.ndarray:
    result = np.asarray(values, dtype=int)
    if result.ndim != 1 or not set(np.unique(result)).issubset({0, 1}):
        raise AuditInvariantError(f'{label} must be a one-dimensional binary vector')
    if len(np.unique(result)) != 2:
        raise AuditInvariantError(f'{label} contains only one class')
    return result

def make_grouped_folds(outcomes: Sequence[int], groups: Sequence[str], *, n_splits: int=5, random_state: int=RANDOM_STATE) -> np.ndarray:
    """Create deterministic stratified group folds and verify zero group overlap."""
    y = validate_binary(outcomes)
    group_array = np.asarray([str(value) for value in groups], dtype=object)
    if len(y) != len(group_array):
        raise AuditInvariantError('Outcome and group lengths differ')
    if pd.isna(group_array).any() or any((not value.strip() for value in group_array)):
        raise AuditInvariantError('Groups contain missing or blank identifiers')
    if len(np.unique(group_array)) < n_splits:
        raise AuditInvariantError('Fewer unique groups than requested folds')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(y), dtype=int)
    dummy = np.zeros((len(y), 1), dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, y, group_array), start=1):
        train_groups = set(group_array[train_idx])
        test_groups = set(group_array[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise AuditInvariantError(f'Grouped split leaked groups: {sorted(overlap)[:5]}')
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            raise AuditInvariantError(f'Fold {fold} does not contain both classes')
        folds[test_idx] = fold
    if set(folds) != set(range(1, n_splits + 1)):
        raise AuditInvariantError('Fold assignment is incomplete')
    return folds

def classification_metrics(y_true: Sequence[int], score: Sequence[float], predicted: Sequence[int] | None=None) -> ClassificationMetrics:
    y = validate_binary(y_true)
    probability = np.asarray(score, dtype=float)
    if len(y) != len(probability) or not np.isfinite(probability).all():
        raise AuditInvariantError('Classification scores are incomplete or non-finite')
    labels = (probability >= 0.5).astype(int) if predicted is None else np.asarray(predicted, dtype=int)
    if len(labels) != len(y):
        raise AuditInvariantError('Predicted-label length differs from outcome length')
    return ClassificationMetrics(n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()), auroc=float(roc_auc_score(y, probability)), auprc=float(average_precision_score(y, probability)), accuracy=float(accuracy_score(y, labels)), macro_f1=float(f1_score(y, labels, average='macro')))

def aggregate_classification_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Average scores by patient when the response is patient-consistent."""
    required = {'group_id', 'outcome', 'score', 'fold'}
    if not required.issubset(frame.columns):
        raise AuditInvariantError(f'Missing columns for group aggregation: {sorted(required - set(frame.columns))}')
    consistency = frame.groupby('group_id')['outcome'].nunique()
    if (consistency > 1).any():
        raise AuditInvariantError('Cannot aggregate classification outcomes that differ within group')
    fold_consistency = frame.groupby('group_id')['fold'].nunique()
    if (fold_consistency > 1).any():
        raise AuditInvariantError('A patient appears in more than one fold')
    aggregated = frame.groupby('group_id', sort=False).agg(outcome=('outcome', 'first'), score=('score', 'mean'), fold=('fold', 'first')).reset_index()
    if 'predicted' in frame.columns:
        predicted = frame.groupby('group_id', sort=False)['predicted'].mean().ge(0.5).astype(int)
        aggregated['predicted'] = predicted.to_numpy()
    return aggregated

def fit_predict_classifier(model_factory: Callable[[], BaseEstimator], x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = model_factory()
    model.fit(x_train, y_train)
    if hasattr(model, 'predict_proba'):
        score = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    elif hasattr(model, 'decision_function'):
        margin = np.asarray(model.decision_function(x_test), dtype=float)
        score = 1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0)))
    else:
        score = np.asarray(model.predict(x_test), dtype=float)
    predicted = np.asarray(model.predict(x_test), dtype=int)
    return (score, predicted)

def classifier_oof_predictions(x: np.ndarray, y: Sequence[int], sample_ids: Sequence[str], group_ids: Sequence[str], folds: Sequence[int], model_factory: Callable[[], BaseEstimator], *, only_folds: Sequence[int] | None=None) -> pd.DataFrame:
    features = np.asarray(x, dtype=np.float32)
    outcome = validate_binary(y)
    samples = np.asarray([str(value) for value in sample_ids], dtype=object)
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    fold_array = np.asarray(folds, dtype=int)
    if not len(features) == len(outcome) == len(samples) == len(groups) == len(fold_array):
        raise AuditInvariantError('OOF input lengths differ')
    selected_folds = sorted(set(fold_array) if only_folds is None else {int(v) for v in only_folds})
    rows: list[pd.DataFrame] = []
    for fold in selected_folds:
        train = fold_array != fold
        test = fold_array == fold
        if set(groups[train]).intersection(groups[test]):
            raise AuditInvariantError(f'Group leakage detected before fitting fold {fold}')
        score, predicted = fit_predict_classifier(model_factory, features[train], outcome[train], features[test])
        rows.append(pd.DataFrame({'sample_id': samples[test], 'group_id': groups[test], 'outcome': outcome[test], 'fold': fold, 'score': score, 'predicted': predicted}))
    return pd.concat(rows, ignore_index=True)

def nested_select_from_outer_predictions(*, features: dict[str, np.ndarray], factories: dict[str, Callable[[], BaseEstimator]], y: Sequence[int], sample_ids: Sequence[str], group_ids: Sequence[str], outer_folds: Sequence[int], outer_predictions: pd.DataFrame, patient_level_metric: bool, only_folds: Sequence[int] | None=None, random_state: int=RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose a candidate inside each outer training set, then reuse its outer OOF score."""
    outcome = validate_binary(y)
    samples = np.asarray([str(v) for v in sample_ids], dtype=object)
    groups = np.asarray([str(v) for v in group_ids], dtype=object)
    folds = np.asarray(outer_folds, dtype=int)
    selected_folds = sorted(set(folds) if only_folds is None else {int(v) for v in only_folds})
    chosen_rows: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    for outer_fold in selected_folds:
        outer_train = folds != outer_fold
        train_y = outcome[outer_train]
        train_groups = groups[outer_train]
        inner_folds = None
        for inner_n_splits in range(min(4, len(np.unique(train_groups))), 1, -1):
            try:
                inner_folds = make_grouped_folds(train_y, train_groups, n_splits=inner_n_splits, random_state=random_state + outer_fold)
                break
            except AuditInvariantError:
                continue
        if inner_folds is None:
            raise AuditInvariantError(f'Could not construct valid inner grouped folds for outer fold {outer_fold}')
        candidate_scores: dict[str, float] = {}
        for name in sorted(factories):
            inner = classifier_oof_predictions(features[name][outer_train], train_y, samples[outer_train], train_groups, inner_folds, factories[name])
            evaluation = aggregate_classification_by_group(inner) if patient_level_metric else inner
            candidate_scores[name] = classification_metrics(evaluation['outcome'], evaluation['score'], evaluation['predicted']).auroc
        chosen = sorted(candidate_scores, key=lambda name: (-candidate_scores[name], name))[0]
        selection_rows.extend(({'outer_fold': outer_fold, 'candidate': name, 'inner_auroc': score, 'selected': name == chosen} for name, score in candidate_scores.items()))
        selected = outer_predictions[(outer_predictions['model'] == chosen) & (outer_predictions['fold'] == outer_fold)].copy()
        if selected.empty:
            raise AuditInvariantError(f'Missing outer prediction for selected model {chosen}, fold {outer_fold}')
        selected['model'] = 'nested_selected'
        selected['selected_candidate'] = chosen
        chosen_rows.append(selected)
    return (pd.concat(chosen_rows, ignore_index=True), pd.DataFrame(selection_rows))

def fit_cox_pca_predict_risk(x_train: np.ndarray, duration_train: np.ndarray, event_train: np.ndarray, x_test: np.ndarray, *, max_components: int=10, random_state: int=RANDOM_STATE) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=np.float32))
    test_scaled = scaler.transform(np.asarray(x_test, dtype=np.float32))
    n_components = min(max_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    if n_components < 1:
        raise AuditInvariantError('No PCA component can be fitted')
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    columns = [f'PC{i + 1}' for i in range(n_components)]
    train_frame = pd.DataFrame(train_pca, columns=columns)
    train_frame['duration'] = np.asarray(duration_train, dtype=float)
    train_frame['event'] = np.asarray(event_train, dtype=int)
    cox = CoxPHFitter()
    cox.fit(train_frame, duration_col='duration', event_col='event', show_progress=False)
    risk = cox.predict_partial_hazard(pd.DataFrame(test_pca, columns=columns)).to_numpy(dtype=float)
    return np.log(np.clip(risk, np.finfo(float).tiny, None))

def survival_oof_predictions(x: np.ndarray, duration: Sequence[float], event: Sequence[int], sample_ids: Sequence[str], group_ids: Sequence[str], folds: Sequence[int], *, only_folds: Sequence[int] | None=None, random_state: int=RANDOM_STATE) -> pd.DataFrame:
    features = np.asarray(x, dtype=np.float32)
    times = np.asarray(duration, dtype=float)
    events = np.asarray(event, dtype=int)
    samples = np.asarray([str(v) for v in sample_ids], dtype=object)
    groups = np.asarray([str(v) for v in group_ids], dtype=object)
    fold_array = np.asarray(folds, dtype=int)
    if not len(features) == len(times) == len(events) == len(samples) == len(groups) == len(fold_array):
        raise AuditInvariantError('Survival OOF input lengths differ')
    if not np.isfinite(times).all() or (times <= 0).any() or (not set(np.unique(events)).issubset({0, 1})):
        raise AuditInvariantError('Invalid survival outcomes')
    selected_folds = sorted(set(fold_array) if only_folds is None else {int(v) for v in only_folds})
    rows: list[pd.DataFrame] = []
    for fold in selected_folds:
        train = fold_array != fold
        test = fold_array == fold
        if set(groups[train]).intersection(groups[test]):
            raise AuditInvariantError(f'Group leakage detected before survival fold {fold}')
        risk = fit_cox_pca_predict_risk(features[train], times[train], events[train], features[test], random_state=random_state + fold)
        rows.append(pd.DataFrame({'sample_id': samples[test], 'group_id': groups[test], 'duration': times[test], 'event': events[test], 'fold': fold, 'risk': risk}))
    return pd.concat(rows, ignore_index=True)

def aggregate_survival_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'group_id', 'duration', 'event', 'risk', 'fold'}
    if not required.issubset(frame.columns):
        raise AuditInvariantError(f'Missing survival aggregation columns: {sorted(required - set(frame.columns))}')
    for outcome in ('duration', 'event', 'fold'):
        if (frame.groupby('group_id')[outcome].nunique() > 1).any():
            raise AuditInvariantError(f'{outcome} differs within survival patient')
    return frame.groupby('group_id', sort=False).agg(duration=('duration', 'first'), event=('event', 'first'), risk=('risk', 'mean'), fold=('fold', 'first')).reset_index()

def cindex_components(duration: Sequence[float], risk: Sequence[float], event: Sequence[int]) -> tuple[float, float]:
    times = np.asarray(duration, dtype=float)
    scores = np.asarray(risk, dtype=float)
    events = np.asarray(event, dtype=int)
    concordant = 0.0
    permissible = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] < times[j] and events[i] == 1:
                permissible += 1.0
                concordant += 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
            elif times[j] < times[i] and events[j] == 1:
                permissible += 1.0
                concordant += 1.0 if scores[j] > scores[i] else 0.5 if scores[i] == scores[j] else 0.0
    return (concordant, permissible)

def harrell_cindex(duration: Sequence[float], risk: Sequence[float], event: Sequence[int]) -> float:
    concordant, permissible = cindex_components(duration, risk, event)
    return float(concordant / permissible) if permissible else math.nan

def fold_restricted_cindex(frame: pd.DataFrame) -> float:
    concordant = 0.0
    permissible = 0.0
    for _, fold_frame in frame.groupby('fold', sort=True):
        fold_concordant, fold_permissible = cindex_components(fold_frame['duration'], fold_frame['risk'], fold_frame['event'])
        concordant += fold_concordant
        permissible += fold_permissible
    return float(concordant / permissible) if permissible else math.nan

def cluster_bootstrap_metric(frame: pd.DataFrame, metric: Callable[[pd.DataFrame], float], *, group_column: str='group_id', n_bootstrap: int=1000, random_state: int=RANDOM_STATE) -> np.ndarray:
    groups = pd.Index(frame[group_column].astype(str).unique())
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    indexed = {group: block.copy() for group, block in frame.groupby(group_column, sort=False)}
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups.to_numpy(), size=len(groups), replace=True)
        blocks: list[pd.DataFrame] = []
        for draw, group in enumerate(sampled):
            block = indexed[str(group)].copy()
            block[group_column] = f'draw{draw}:{group}'
            blocks.append(block)
        try:
            value = metric(pd.concat(blocks, ignore_index=True))
        except (AuditInvariantError, ValueError):
            continue
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < max(100, int(n_bootstrap * 0.8)):
        raise AuditInvariantError(f'Too few valid bootstrap replicates: {len(values)}/{n_bootstrap}')
    return np.asarray(values, dtype=float)

def percentile_interval(values: Sequence[float], alpha: float=0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return tuple((float(v) for v in np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])))
