"""Shared model factories and aligned matrix loaders for the current BCR pipeline.

The fitting entry point disables SVM probability calibration and uses margins.
Superseded analysis and evaluation entry points are not distributed.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
RANDOM_STATE = 42
N_SPLITS = 5
PRIMARY_LABEL_SOURCE = 'legacy_task4_y_final'
CORRECTED_LABEL_SOURCE = 'sensitivity_corrected_grade_parser'
EXPECTED_PATIENT_CANCER_COUNTS: dict[str, int] = {'Melanoma': 48, 'Non-small cell Lung Cancer': 39, 'Head and Neck Squamous Cell Cancer': 8, 'Other': 5, 'Renal Cell Carcinoma': 4, 'Small Cell Lung Cancer': 3, 'Control': 3, 'Adenocarcinoma of the Rectum': 1, 'Hepatocellular Carcinoma': 1, 'Mesothelioma': 1}
MAJOR_CANCER_SUBGROUPS = ('Melanoma', 'Non-small cell Lung Cancer')
METRIC_COLUMNS = ('auroc', 'auprc', 'f1', 'accuracy', 'precision', 'recall')

class AuditInvariantError(RuntimeError):
    """Raised when an invariant required by the audit contract is not met."""

@dataclass(frozen=True)
class ModelSpec:
    strategy: str
    feature: str
    model: str

    @property
    def model_key(self) -> str:
        return f'{self.strategy}|{self.feature}|{self.model}'

def load_feature_matrix(path: Path, sample_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise AuditInvariantError(f'Feature matrix has duplicate sample IDs: {path}')
    if set(frame.index) != set(sample_ids):
        raise AuditInvariantError(f'Feature matrix sample set differs from metadata: {path}')
    return frame.loc[list(sample_ids)].fillna(0)

def load_saved_embeddings(embedding_path: Path, samples_path: Path, sample_ids: Sequence[str]) -> np.ndarray:
    embeddings = np.load(embedding_path)
    sample_frame = pd.read_csv(samples_path)
    if 'sample_id' not in sample_frame.columns:
        raise AuditInvariantError(f'Embedding order file lacks sample_id: {samples_path}')
    order = sample_frame['sample_id'].astype(str).tolist()
    if embeddings.shape[0] != len(order):
        raise AuditInvariantError(f'Embedding rows/order rows differ for {embedding_path}')
    if len(order) != len(set(order)) or set(order) != set(sample_ids):
        raise AuditInvariantError(f'Embedding sample IDs differ from Task 4 metadata: {embedding_path}')
    positions = {sample_id: index for index, sample_id in enumerate(order)}
    return np.asarray(embeddings[[positions[sample_id] for sample_id in sample_ids]])

def build_estimator(spec: ModelSpec) -> BaseEstimator:
    """Reproduce existing classifier hyperparameters; preprocessing stays in pipelines."""
    if spec.strategy == 'strategy1':
        if spec.model == 'LogisticRegression':
            return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
        if spec.model == 'SVM':
            return Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=RANDOM_STATE))])
        if spec.model == 'RandomForest':
            return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        if spec.model == 'XGBoost':
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError('xgboost is required to reproduce Strategy 1 XGBoost') from exc
            return XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, use_label_encoder=False)
    if spec.strategy == 'strategy2':
        if spec.model == 'LR':
            classifier: BaseEstimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        elif spec.model == 'RF':
            classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        elif spec.model == 'SVM':
            classifier = SVC(probability=True, random_state=RANDOM_STATE)
        elif spec.model == 'MLP':
            classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE)
        elif spec.model == 'XGB':
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise RuntimeError('xgboost is required to reproduce Strategy 2 XGB') from exc
            classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        else:
            raise ValueError(spec)
        return Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
    if spec.strategy == 'metadata_only':
        return Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore')), ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    raise ValueError(f'Unknown model specification: {spec}')
