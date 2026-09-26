"""Optional Task 2 Strategy 2 OOF rerun on the frozen five folds.

This script intentionally performs downstream modelling only.  It consumes the
canonical processed mutation matrix, the canonical survival labels, and a
previously frozen fold-assignment CSV.  It neither reprocesses omics data nor
downloads model weights.

The historical Strategy 2 models regress ``log1p(duration)`` with an ordinary
MSE-style regression loss.  The event indicator and censoring mechanism are not
used in that loss; censored observations are treated as ordinary duration
targets.  They are used only when calculating Harrell-style C-indices.  Both
the historical score orientation (predicted duration treated as if higher meant
higher risk) and the scientifically coherent risk orientation
(``risk = -predicted_duration``) are exported explicitly.
"""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 42
EXPECTED_N_SAMPLES = 1610
EXPECTED_FOLDS = (1, 2, 3, 4, 5)

class AuditInvariantError(RuntimeError):
    """Raised when a frozen-input or OOF invariant is violated."""

@dataclass(frozen=True)
class FTTransformerConfig:
    max_features: int = 512
    n_blocks: int = 3
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 1e-05
    patience: int = 10
    inner_validation_fraction: float = 0.1

@dataclass(frozen=True)
class TabPFNConfig:
    max_features: int = 2000
    checkpoint_name: str = 'tabpfn-v2.5-regressor-v2.5_default.ckpt'
    ignore_pretraining_limits: bool = False

@dataclass(frozen=True)
class Task2Inputs:
    features: pd.DataFrame
    labels: pd.DataFrame
    folds: pd.Series
    x_path: Path
    y_path: Path
    fold_path: Path

@dataclass(frozen=True)
class RunOutputPaths:
    predictions: Path
    fold_metrics: Path
    summary: Path
    selected_features: Path

    @classmethod
    def under(cls, output_root: Path, model_slug: str) -> 'RunOutputPaths':
        stem = f'task2_strategy2_{model_slug}'
        return cls(predictions=output_root / 'predictions' / f'{stem}_oof_predictions.csv', fold_metrics=output_root / 'metrics' / f'{stem}_fold_metrics.csv', summary=output_root / 'metrics' / f'{stem}_summary.json', selected_features=output_root / 'metadata' / f'{stem}_selected_features.csv')

    def all_paths(self) -> tuple[Path, ...]:
        return (self.predictions, self.fold_metrics, self.summary, self.selected_features)

def project_root_from_script() -> Path:
    return _submission_path('project', '')

def _first_existing(label: str, candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = '\n  - '.join((str(path) for path in candidates))
    raise FileNotFoundError(f'Could not locate {label}; checked:\n  - {rendered}')

def discover_canonical_paths(project_root: Path, *, x_path: Path | None=None, y_path: Path | None=None, fold_path: Path | None=None) -> tuple[Path, Path, Path]:
    """Resolve canonical Task 2 inputs without scanning the whole repository."""
    root = project_root.resolve()
    resolved_x = _first_existing('Task 2 X', [x_path]) if x_path is not None else _first_existing('Task 2 X', [root / 'processed_data_strategy1/task2_X_final.csv', root / 'final/shared_data/processed_data_strategy1/task2_X_final.csv'])
    resolved_y = _first_existing('Task 2 y', [y_path]) if y_path is not None else _first_existing('Task 2 y', [root / 'processed_data_strategy1/task2_y_final.csv', root / 'final/shared_data/processed_data_strategy1/task2_y_final.csv'])
    if fold_path is not None:
        resolved_fold = _first_existing('frozen Task 2 folds', [fold_path])
    else:
        audit_fold_dir = root / 'analysis/cancer_context/fold_assignments'
        named_candidates = [audit_fold_dir / 'task2_fold_assignments.csv', audit_fold_dir / 'task2_folds.csv', audit_fold_dir / 'fold_assignments.csv']
        discovered = sorted(audit_fold_dir.glob('*task2*.csv')) if audit_fold_dir.exists() else []
        resolved_fold = _first_existing('frozen Task 2 folds', [*named_candidates, *discovered, root / 'results/task2_cancer_context/fold_assignments.csv'])
    return (resolved_x, resolved_y, resolved_fold)

def _find_column(frame: pd.DataFrame, names: Iterable[str], label: str) -> str:
    by_lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    raise AuditInvariantError(f'Could not identify {label}; columns={list(frame.columns)!r}')

def _normalise_patient_ids(values: pd.Series, label: str) -> pd.Series:
    if values.isna().any():
        raise AuditInvariantError(f'{label} contains missing patient IDs')
    normalised = values.astype(str).str.strip()
    if normalised.eq('').any():
        raise AuditInvariantError(f'{label} contains blank patient IDs')
    return normalised

def load_frozen_fold_assignments(fold_path: Path, patient_ids: Sequence[str] | pd.Index, *, expected_folds: Sequence[int] | None=EXPECTED_FOLDS) -> pd.Series:
    """Load and align one immutable fold per patient."""
    frame = pd.read_csv(fold_path)
    patient_column = _find_column(frame, ('PATIENT_ID', 'patient_id', 'sample_id'), 'fold patient ID')
    fold_column = _find_column(frame, ('fold', 'Fold'), 'fold number')
    ids = _normalise_patient_ids(frame[patient_column], 'fold assignments')
    if ids.duplicated().any():
        duplicates = ids.loc[ids.duplicated(keep=False)].unique().tolist()
        raise AuditInvariantError(f'Frozen fold assignments contain duplicate patients: {duplicates[:10]}')
    numeric_folds = pd.to_numeric(frame[fold_column], errors='raise')
    if not np.all(np.equal(numeric_folds, np.floor(numeric_folds))):
        raise AuditInvariantError('Fold values must be integers')
    fold_series = pd.Series(numeric_folds.astype(int).to_numpy(), index=ids, name='fold')
    requested = pd.Index([str(value).strip() for value in patient_ids], name='patient_id')
    if not requested.is_unique:
        raise AuditInvariantError('Canonical Task 2 patient IDs are duplicated')
    missing = requested.difference(fold_series.index).tolist()
    extra = fold_series.index.difference(requested).tolist()
    if missing or extra:
        raise AuditInvariantError(f'Frozen folds and canonical Task 2 patients differ; missing={missing[:10]}, extra={extra[:10]}')
    aligned = fold_series.loc[requested]
    if expected_folds is not None and set(aligned.unique()) != set((int(value) for value in expected_folds)):
        raise AuditInvariantError(f'Unexpected frozen fold values: observed={sorted(aligned.unique())}, expected={list(expected_folds)}')
    if 'original_row_position_0based' in frame.columns:
        positions = pd.Series(pd.to_numeric(frame['original_row_position_0based'], errors='raise').astype(int).to_numpy(), index=ids).loc[requested]
        expected_positions = np.arange(len(requested), dtype=int)
        if not np.array_equal(positions.to_numpy(), expected_positions):
            raise AuditInvariantError('Frozen fold row positions do not match canonical Task 2 order')
    return aligned

def load_task2_inputs(x_path: Path, y_path: Path, fold_path: Path, *, expected_n: int | None=EXPECTED_N_SAMPLES, expected_folds: Sequence[int] | None=EXPECTED_FOLDS) -> Task2Inputs:
    """Load canonical features/labels and align the frozen fold assignment."""
    x_raw = pd.read_csv(x_path)
    y_raw = pd.read_csv(y_path)
    x_patient_column = _find_column(x_raw, ('PATIENT_ID', 'patient_id'), 'Task 2 X patient ID')
    y_patient_column = _find_column(y_raw, ('PATIENT_ID', 'patient_id'), 'Task 2 y patient ID')
    duration_column = _find_column(y_raw, ('duration',), 'survival duration')
    event_column = _find_column(y_raw, ('event',), 'survival event')
    x_ids = _normalise_patient_ids(x_raw[x_patient_column], 'Task 2 X')
    y_ids = _normalise_patient_ids(y_raw[y_patient_column], 'Task 2 y')
    if x_ids.duplicated().any() or y_ids.duplicated().any():
        raise AuditInvariantError('Task 2 X/y contain duplicate patient IDs')
    if set(x_ids) != set(y_ids):
        raise AuditInvariantError('Task 2 X/y patient sets differ')
    if expected_n is not None and (len(x_ids) != expected_n or len(y_ids) != expected_n):
        raise AuditInvariantError(f'Expected {expected_n} Task 2 patients; found X={len(x_ids)}, y={len(y_ids)}')
    feature_frame = x_raw.drop(columns=[x_patient_column]).copy()
    try:
        feature_frame = feature_frame.apply(pd.to_numeric, errors='raise').fillna(0.0)
    except (TypeError, ValueError) as exc:
        raise AuditInvariantError('Task 2 features must be numeric after removing PATIENT_ID') from exc
    feature_frame.index = pd.Index(x_ids, name='PATIENT_ID')
    labels = pd.DataFrame({'duration': pd.to_numeric(y_raw[duration_column], errors='raise').astype(float).to_numpy(), 'event': pd.to_numeric(y_raw[event_column], errors='raise').astype(int).to_numpy()}, index=pd.Index(y_ids, name='PATIENT_ID')).loc[feature_frame.index]
    if (labels['duration'] < 0).any() or not np.isfinite(labels['duration']).all():
        raise AuditInvariantError('Task 2 duration contains negative or non-finite values')
    if not set(labels['event'].unique()).issubset({0, 1}):
        raise AuditInvariantError('Task 2 event must be binary')
    folds = load_frozen_fold_assignments(fold_path, feature_frame.index, expected_folds=expected_folds)
    folds.index = feature_frame.index
    return Task2Inputs(features=feature_frame, labels=labels, folds=folds, x_path=x_path.resolve(), y_path=y_path.resolve(), fold_path=fold_path.resolve())

def select_features_by_training_variance(x_train: np.ndarray, x_test: np.ndarray, max_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select columns using variance computed solely from the outer training fold."""
    train = np.asarray(x_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError('x_train/x_test must be 2D arrays with the same number of features')
    if max_features <= 0:
        raise ValueError('max_features must be positive')
    training_variances = np.nanvar(train, axis=0)
    if train.shape[1] <= max_features:
        selected = np.arange(train.shape[1], dtype=int)
    else:
        selected = np.argsort(training_variances)[-max_features:]
        selected = np.sort(selected.astype(int))
    return (train[:, selected], test[:, selected], selected, training_variances[selected])

def harrell_c_index_from_higher_risk(durations: Sequence[float], higher_risk_scores: Sequence[float], events: Sequence[int]) -> float:
    """Historical pairwise C-index where a larger score means earlier event."""
    times = np.asarray(durations, dtype=float)
    scores = np.asarray(higher_risk_scores, dtype=float)
    observed = np.asarray(events, dtype=int)
    if not len(times) == len(scores) == len(observed):
        raise ValueError('durations, scores, and events must have equal length')
    if not (np.isfinite(times).all() and np.isfinite(scores).all()):
        raise ValueError('durations and scores must be finite')
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] < times[j] and observed[i] == 1:
                permissible += 1.0
                if scores[i] > scores[j]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    ties += 1.0
            elif times[j] < times[i] and observed[j] == 1:
                permissible += 1.0
                if scores[j] > scores[i]:
                    concordant += 1.0
                elif scores[i] == scores[j]:
                    ties += 1.0
    if permissible == 0:
        return float('nan')
    return float((concordant + 0.5 * ties) / permissible)

def compute_duration_orientation_metrics(durations: Sequence[float], predicted_durations: Sequence[float], events: Sequence[int]) -> dict[str, float]:
    predicted = np.asarray(predicted_durations, dtype=float)
    return {'legacy_c_index_predicted_duration_as_higher_risk': harrell_c_index_from_higher_risk(durations, predicted, events), 'corrected_c_index_negative_predicted_duration_as_higher_risk': harrell_c_index_from_higher_risk(durations, -predicted, events)}

def _set_seed(seed: int=RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _effective_ft_batch_size(n_features: int, requested: int) -> int:
    if n_features >= 1024:
        return min(requested, 8)
    if n_features >= 512:
        return min(requested, 16)
    return requested

def _inner_train_validation_indices(n_rows: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        raise AuditInvariantError('FT-Transformer requires at least two outer-training rows')
    n_validation = max(1, int(round(n_rows * fraction)))
    n_validation = min(n_validation, n_rows - 1)
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return (np.sort(permutation[n_validation:]), np.sort(permutation[:n_validation]))

def _run_fttransformer_fold(x_train: np.ndarray, target_log_duration: np.ndarray, x_test: np.ndarray, *, config: FTTransformerConfig, device_name: str, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    """Fit FT-Transformer without exposing the outer test outcome to early stopping."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        import rtdl_revisiting_models as rtdl
    except ImportError as exc:
        raise RuntimeError('FT-Transformer dependencies (torch and rtdl_revisiting_models) are required') from exc
    _set_seed(seed)
    device = torch.device(device_name)
    kwargs = rtdl.FTTransformer.get_default_kwargs(n_blocks=config.n_blocks)
    kwargs['d_out'] = 1
    model = rtdl.FTTransformer(n_cont_features=x_train.shape[1], cat_cardinalities=[], **kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    inner_train_idx, inner_validation_idx = _inner_train_validation_indices(len(x_train), config.inner_validation_fraction, seed)
    train_x = torch.tensor(x_train[inner_train_idx], dtype=torch.float32)
    train_y = torch.tensor(target_log_duration[inner_train_idx], dtype=torch.float32).view(-1, 1)
    validation_x = torch.tensor(x_train[inner_validation_idx], dtype=torch.float32, device=device)
    validation_y = torch.tensor(target_log_duration[inner_validation_idx], dtype=torch.float32, device=device).view(-1, 1)
    test_x = torch.tensor(x_test, dtype=torch.float32)
    effective_batch_size = _effective_ft_batch_size(x_train.shape[1], config.batch_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=min(effective_batch_size, len(train_x)), shuffle=True, generator=generator)

    def predict_tensor(values: Any) -> Any:
        outputs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(values), effective_batch_size):
                batch = values[start:start + effective_batch_size].to(device)
                outputs.append(model(batch, None).detach().cpu())
        return torch.cat(outputs, dim=0)
    best_state: dict[str, Any] | None = None
    best_validation_loss = float('inf')
    bad_epochs = 0
    for _ in range(config.epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x, None), batch_y)
            loss.backward()
            optimizer.step()
        validation_prediction = predict_tensor(validation_x.cpu()).to(device)
        validation_loss = float(loss_fn(validation_prediction, validation_y).item())
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = predict_tensor(test_x).squeeze(-1).numpy().astype(float)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (predictions, {'inner_train_n': int(len(inner_train_idx)), 'inner_validation_n': int(len(inner_validation_idx))})

def resolve_local_tabpfn_checkpoint(requested: Path | None, config: TabPFNConfig) -> Path:
    checkpoint = requested.expanduser() if requested is not None else _submission_path('models', 'tabpfn') / config.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Local TabPFN checkpoint not found: {checkpoint}. Automatic model download is disabled for this audit.')
    return checkpoint.resolve()

def _run_tabpfn_fold(x_train: np.ndarray, target_log_duration: np.ndarray, x_test: np.ndarray, *, config: TabPFNConfig, checkpoint: Path, device_name: str, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise RuntimeError('The local tabpfn package is required for the TabPFN rerun') from exc
    _set_seed(seed)
    model = TabPFNRegressor(device=device_name, model_path=str(checkpoint), random_state=seed, ignore_pretraining_limits=config.ignore_pretraining_limits)
    model.fit(x_train, target_log_duration)
    predictions = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    return (predictions, {'inner_train_n': int(len(x_train)), 'inner_validation_n': 0})

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def assert_outputs_absent(paths: RunOutputPaths) -> None:
    existing = [path for path in paths.all_paths() if path.exists()]
    if existing:
        raise FileExistsError('No-overwrite policy: refusing to run because output files already exist: ' + ', '.join((str(path) for path in existing)))

def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as handle:
        frame.to_csv(handle, index=False)

def write_outputs_no_overwrite(paths: RunOutputPaths, predictions: pd.DataFrame, fold_metrics: pd.DataFrame, summary: dict[str, Any], selected_features: pd.DataFrame) -> None:
    """Write all artifacts with exclusive creation; existing files are never replaced."""
    assert_outputs_absent(paths)
    _write_csv_exclusive(predictions, paths.predictions)
    _write_csv_exclusive(fold_metrics, paths.fold_metrics)
    _write_csv_exclusive(selected_features, paths.selected_features)
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

def _json_metric(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)

def run_oof(inputs: Task2Inputs, *, model_name: str, max_folds: int | None, device_name: str, tabpfn_checkpoint: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the selected downstream model on the saved outer folds."""
    model_slug = model_name.lower()
    if model_slug not in {'fttransformer', 'tabpfn'}:
        raise ValueError(f'Unsupported model: {model_name}')
    all_folds = sorted((int(value) for value in inputs.folds.unique()))
    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError('max_folds must be positive')
        folds_to_run = all_folds[:max_folds]
    else:
        folds_to_run = all_folds
    ft_config = FTTransformerConfig()
    tabpfn_config = TabPFNConfig()
    checkpoint = None
    if model_slug == 'tabpfn':
        checkpoint = resolve_local_tabpfn_checkpoint(tabpfn_checkpoint, tabpfn_config)
        max_features = tabpfn_config.max_features
    else:
        max_features = ft_config.max_features
    predictions_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    feature_names = np.asarray(inputs.features.columns.astype(str))
    for fold in folds_to_run:
        test_mask = inputs.folds.to_numpy() == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise AuditInvariantError(f'Fold {fold} has an empty train or test partition')
        x_train_raw = inputs.features.loc[train_mask].to_numpy(dtype=np.float32)
        x_test_raw = inputs.features.loc[test_mask].to_numpy(dtype=np.float32)
        train_duration = inputs.labels.loc[train_mask, 'duration'].to_numpy(dtype=np.float32)
        test_duration = inputs.labels.loc[test_mask, 'duration'].to_numpy(dtype=float)
        test_event = inputs.labels.loc[test_mask, 'event'].to_numpy(dtype=int)
        target_log_duration = np.log1p(train_duration)
        x_train, x_test, selected, training_variances = select_features_by_training_variance(x_train_raw, x_test_raw, max_features)
        if model_slug == 'fttransformer':
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train).astype(np.float32)
            x_test = scaler.transform(x_test).astype(np.float32)
            pred_log_duration, fit_metadata = _run_fttransformer_fold(x_train, target_log_duration, x_test, config=ft_config, device_name=device_name, seed=RANDOM_STATE)
        else:
            assert checkpoint is not None
            pred_log_duration, fit_metadata = _run_tabpfn_fold(x_train, target_log_duration, x_test, config=tabpfn_config, checkpoint=checkpoint, device_name=device_name, seed=RANDOM_STATE)
        pred_duration = np.expm1(np.asarray(pred_log_duration, dtype=float))
        if len(pred_duration) != int(test_mask.sum()) or not np.isfinite(pred_duration).all():
            raise AuditInvariantError(f'Fold {fold} produced invalid prediction values')
        fold_orientation = compute_duration_orientation_metrics(test_duration, pred_duration, test_event)
        test_ids = inputs.features.index[test_mask]
        for patient_id, duration, event, predicted_log, predicted_duration in zip(test_ids, test_duration, test_event, pred_log_duration, pred_duration):
            predictions_rows.append({'PATIENT_ID': patient_id, 'fold': fold, 'duration': float(duration), 'event': int(event), 'predicted_log1p_duration': float(predicted_log), 'predicted_duration': float(predicted_duration), 'legacy_orientation_score_predicted_duration': float(predicted_duration), 'corrected_risk_score_negative_predicted_duration': float(-predicted_duration), 'model': model_slug})
        metric_rows.append({'model': model_slug, 'fold': fold, 'n_train': int(train_mask.sum()), 'n_test': int(test_mask.sum()), 'n_events_test': int(test_event.sum()), 'n_features_input': int(x_train_raw.shape[1]), 'n_features_selected': int(len(selected)), **fit_metadata, **fold_orientation, 'loss_target': 'log1p_duration', 'event_indicator_used_in_loss': False, 'censoring_adjustment_used_in_loss': False})
        for feature_index, feature_variance in zip(selected, training_variances):
            selected_rows.append({'model': model_slug, 'fold': fold, 'feature_index_0based': int(feature_index), 'feature_name': str(feature_names[feature_index]), 'training_fold_variance': float(feature_variance)})
    predictions = pd.DataFrame(predictions_rows).sort_values(['fold', 'PATIENT_ID']).reset_index(drop=True)
    if predictions['PATIENT_ID'].duplicated().any():
        raise AuditInvariantError('OOF output contains duplicate patient predictions')
    expected_test_ids = set(inputs.features.index[inputs.folds.isin(folds_to_run)])
    if set(predictions['PATIENT_ID']) != expected_test_ids:
        raise AuditInvariantError('OOF output does not exactly cover the requested frozen test folds')
    fold_metrics = pd.DataFrame(metric_rows).sort_values('fold').reset_index(drop=True)
    selected_features = pd.DataFrame(selected_rows).sort_values(['fold', 'feature_index_0based']).reset_index(drop=True)
    pooled = compute_duration_orientation_metrics(predictions['duration'], predictions['predicted_duration'], predictions['event'])
    summary = {'analysis_name': 'Clinical and mutation feature survival prediction: optional Strategy 2 OOF rerun', 'model': model_slug, 'random_state': RANDOM_STATE, 'folds_completed': folds_to_run, 'all_frozen_folds': all_folds, 'is_complete_oof': folds_to_run == all_folds, 'n_predictions': int(len(predictions)), 'pooled_metrics': {key: _json_metric(value) for key, value in pooled.items()}, 'mean_fold_metrics': {'legacy_c_index_predicted_duration_as_higher_risk': _json_metric(float(fold_metrics['legacy_c_index_predicted_duration_as_higher_risk'].mean())), 'corrected_c_index_negative_predicted_duration_as_higher_risk': _json_metric(float(fold_metrics['corrected_c_index_negative_predicted_duration_as_higher_risk'].mean()))}, 'loss_contract': {'target': 'log1p(duration)', 'event_indicator_used_in_loss': False, 'censoring_adjustment_used_in_loss': False, 'warning': 'Censored observations are treated as ordinary duration-regression targets; event is used only for C-index evaluation. This is not a censoring-aware survival loss.'}, 'score_contract': {'legacy_orientation': 'predicted_duration passed to a higher-score-means-higher-risk C-index', 'scientific_orientation': 'corrected_risk = -predicted_duration'}, 'model_config': asdict(ft_config if model_slug == 'fttransformer' else tabpfn_config), 'tabpfn_checkpoint': str(checkpoint) if checkpoint is not None else None, 'inputs': {'x_path': str(inputs.x_path), 'x_sha256': sha256_file(inputs.x_path), 'y_path': str(inputs.y_path), 'y_sha256': sha256_file(inputs.y_path), 'fold_path': str(inputs.fold_path), 'fold_sha256': sha256_file(inputs.fold_path)}}
    return (predictions, fold_metrics, selected_features, summary)

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser(description='Optional Task 2 FT-Transformer/TabPFN OOF rerun on frozen folds.')
    parser.add_argument('--model', required=True, choices=('fttransformer', 'tabpfn'))
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--output-dir', type=Path, default=root / 'analysis/cancer_context', help='Audit output root; predictions/, metrics/, and metadata/ are created beneath it.')
    parser.add_argument('--x-path', type=Path, default=None)
    parser.add_argument('--y-path', type=Path, default=None)
    parser.add_argument('--fold-assignments', type=Path, default=None)
    parser.add_argument('--device', default=None, help="Model device. Defaults to CUDA/CPU for FT-Transformer and 'auto' for TabPFN.")
    parser.add_argument('--tabpfn-checkpoint', type=Path, default=None)
    parser.add_argument('--no-overwrite', action='store_true', default=True, help='Documented compatibility flag; no-overwrite is always enforced.')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = parse_args(argv)
    root = project_root_from_script()
    x_path, y_path, fold_path = discover_canonical_paths(root, x_path=args.x_path, y_path=args.y_path, fold_path=args.fold_assignments)
    inputs = load_task2_inputs(x_path, y_path, fold_path)
    output_paths = RunOutputPaths.under(args.output_dir.resolve(), args.model)
    assert_outputs_absent(output_paths)
    if args.device is None:
        if args.model == 'tabpfn':
            device_name = 'auto'
        else:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError('torch is required for FT-Transformer') from exc
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = str(args.device)
    print('LOSS CONTRACT: regress log1p(duration); event indicator and censoring mechanism do not enter the training loss. Event is used only for C-index evaluation.')
    predictions, fold_metrics, selected_features, summary = run_oof(inputs, model_name=args.model, max_folds=args.max_folds, device_name=device_name, tabpfn_checkpoint=args.tabpfn_checkpoint)
    summary['device'] = device_name
    summary['outputs'] = {key: str(value) for key, value in asdict(output_paths).items()}
    write_outputs_no_overwrite(output_paths, predictions, fold_metrics, summary, selected_features)
    for path in output_paths.all_paths():
        print(f'Wrote {path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
