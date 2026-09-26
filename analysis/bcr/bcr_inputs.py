"""Current BCR input/endpoint helpers. No historical analysis runner is included."""
from __future__ import annotations
from submission_paths import path as _submission_path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
PROJECT_ROOT = Path(str(_submission_path('project', '')))
V1_ROOT = PROJECT_ROOT / 'analysis/cancer_context'
CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
from task4_grouped_core import load_saved_embeddings
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
THREADS = 8
CLINICAL_PATH = Path(str(_submission_path('data', '4-/task5_clinical_labels.csv')))
MAPPING_PATH = V1_ROOT / 'metadata/task4_sample_patient_cancer_mapping.csv'
FOLD_PATH = V1_ROOT / 'fold_assignments/task4_patient_grouped_folds.csv'
V1_PREDICTIONS_PATH = V1_ROOT / 'predictions/task4_all_oof_predictions.csv'
ESM2_PATH = PROJECT_ROOT / 'processed_data_strategy2/task4_esm2_X_emb.npy'
ESM2_SAMPLES_PATH = PROJECT_ROOT / 'processed_data_strategy2/task4_esm2_samples.csv'
ANTIBERTY_PATH = PROJECT_ROOT / 'processed_data_strategy2/task4_antiberty_X_emb.npy'
ANTIBERTY_SAMPLES_PATH = PROJECT_ROOT / 'processed_data_strategy2/task4_antiberty_samples.csv'
GRADE_DATE_PAIRS = (('irAE grade: Dermatitis', 'irAE date: Dermatitis'), ('irAE grade: Diarrhea/colitis', 'irAE date: Diarrhea/colitis'), ('irAE grade: Hypothyroidism', 'irAE date: Hypothyroidism'), ('irAE grade: Hyperthyroidism', 'irAE date: Hyperthyroidism'), ('irAE grade: Hypophysitis/hypopituitarism', 'irAE date: Hypophysitis/hypopituitarism'), ('irAE grade: Hepatitis', 'irAE date: Hepatitis'), ('irAE grade: Primary Adrenal Insufficiency', 'irAE date: AI'), ('irAE grade: Pneumonitis', 'irAE date: Pneumonitis'), ('irae_other_1_grade', 'irae_other_1_date'), ('irae_other_2_grade', 'irae_other_2_date'))

@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    baseline_only: bool
    exclude_control: bool
    label_column: str
ANALYSES = (AnalysisSpec('baseline_case_corrected_grade', True, True, 'corrected_grade_label'), AnalysisSpec('baseline_case_post_ici_incident', True, True, 'post_ici_incident_label'), AnalysisSpec('all_samples_case_corrected_grade', False, True, 'corrected_grade_label'), AnalysisSpec('all_samples_case_post_ici_incident', False, True, 'post_ici_incident_label'))

def numeric_tokens(value: Any, *, allow_negative: bool=False) -> list[float]:
    if pd.isna(value):
        return []
    pattern = '-?\\d+(?:\\.\\d+)?' if allow_negative else '\\d+(?:\\.\\d+)?'
    return [float(token) for token in re.findall(pattern, str(value))]

def aligned_grade_date_events(grade: Any, date: Any) -> list[tuple[float, float]]:
    grades = numeric_tokens(grade)
    dates = numeric_tokens(date, allow_negative=True)
    if not grades or not dates:
        return []
    if len(grades) == len(dates):
        return list(zip(grades, dates, strict=True))
    if len(grades) == 1:
        return [(grades[0], day) for day in dates]
    if len(dates) == 1:
        return [(max(grades), dates[0])]
    raise ValueError(f'Cannot align grade/date lists: grade={grade!r}, date={date!r}')

def derive_patient_label_audit(clinical: pd.DataFrame) -> pd.DataFrame:
    required = {'Internal Patient ID', *[item for pair in GRADE_DATE_PAIRS for item in pair]}
    missing = sorted(required.difference(clinical.columns))
    if missing:
        raise ValueError(f'Clinical table lacks required columns: {missing}')
    rows: list[dict[str, Any]] = []
    grade_columns = [grade for grade, _ in GRADE_DATE_PAIRS]
    for _, source in clinical.iterrows():
        events: list[dict[str, Any]] = []
        for grade_column, date_column in GRADE_DATE_PAIRS:
            for grade, day in aligned_grade_date_events(source[grade_column], source[date_column]):
                if grade > 0:
                    events.append({'grade_column': grade_column, 'date_column': date_column, 'grade': grade, 'day': day})
        all_grade_values = [number for column in grade_columns for number in numeric_tokens(source[column])]
        corrected = int(max(all_grade_values, default=0.0) > 0)
        legacy = int(pd.to_numeric(pd.Series([source[column] for column in grade_columns]), errors='coerce').fillna(0).max() > 0)
        positive_days = [event['day'] for event in events]
        post_days = [day for day in positive_days if day > 0]
        pre_or_start_days = [day for day in positive_days if day <= 0]
        rows.append({'patient_id': int(source['Internal Patient ID']), 'legacy_numeric_label': legacy, 'corrected_grade_label': corrected, 'post_ici_incident_label': int(bool(post_days)), 'has_pre_or_start_event': bool(pre_or_start_days), 'has_post_ici_event': bool(post_days), 'earliest_recorded_irae_day': min(positive_days) if positive_days else np.nan, 'earliest_post_ici_irae_day': min(post_days) if post_days else np.nan, 'max_recorded_grade': max(all_grade_values, default=0.0), 'n_positive_grade_date_events': len(events), 'event_detail_json': json.dumps(events, ensure_ascii=False, sort_keys=True)})
    result = pd.DataFrame(rows).sort_values('patient_id').reset_index(drop=True)
    if len(result) != 113 or result['patient_id'].duplicated().any():
        raise AssertionError('Label audit must contain 113 unique patients')
    if int(result['legacy_numeric_label'].sum()) != 80:
        raise AssertionError('Legacy label reproduction no longer yields 80 positive patients')
    if int(result['corrected_grade_label'].sum()) != 81:
        raise AssertionError('Corrected grade parser no longer yields 81 positive patients')
    if int(result['post_ici_incident_label'].sum()) != 79:
        raise AssertionError('Post-ICI incident parser no longer yields 79 positive patients')
    changed = result[result['legacy_numeric_label'] != result['corrected_grade_label']]
    if changed['patient_id'].tolist() != [254]:
        raise AssertionError(f"Unexpected legacy/corrected label changes: {changed['patient_id'].tolist()}")
    return result

def load_inputs(repaired_features: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | np.ndarray], pd.DataFrame]:
    mapping = pd.read_csv(MAPPING_PATH)
    folds = pd.read_csv(FOLD_PATH)[['sample_id', 'fold']]
    metadata = mapping.merge(folds, on='sample_id', how='left', validate='one_to_one')
    metadata['patient_id'] = metadata['patient_id'].astype(int)
    if len(metadata) != 256 or metadata['sample_id'].duplicated().any():
        raise AssertionError('Task 4 metadata must contain 256 unique samples')
    baseline = metadata['collection_time'].astype(str).str.lower().eq('bl')
    if int(baseline.sum()) != 113 or metadata.loc[baseline, 'patient_id'].nunique() != 113:
        raise AssertionError('Every patient must have exactly one BL sample')
    patient_folds = metadata.groupby('patient_id')['fold'].nunique()
    if not (patient_folds == 1).all():
        raise AssertionError('Frozen folds contain patient leakage')
    sample_ids = metadata['sample_id'].astype(str).tolist()
    corrected = np.load(repaired_features)
    if corrected.shape != (len(sample_ids), 8000) or not np.isfinite(corrected).all():
        raise AssertionError('Corrected repertoire matrix has invalid shape or values')
    np.testing.assert_allclose(corrected.sum(axis=1), 1)
    matrices: dict[str, pd.DataFrame | np.ndarray] = {'BCR_3mer': corrected, 'ESM2': load_saved_embeddings(ESM2_PATH, ESM2_SAMPLES_PATH, sample_ids), 'AntiBERTy': load_saved_embeddings(ANTIBERTY_PATH, ANTIBERTY_SAMPLES_PATH, sample_ids)}
    clinical = pd.read_csv(CLINICAL_PATH)
    labels = derive_patient_label_audit(clinical)
    metadata = metadata.merge(labels, on='patient_id', how='left', validate='many_to_one')
    if metadata[['corrected_grade_label', 'post_ici_incident_label']].isna().any().any():
        raise AssertionError('Patient labels did not cover every sample')
    if not np.array_equal(metadata['toxicity_corrected_grade_parser'].astype(int), metadata['corrected_grade_label'].astype(int)):
        raise AssertionError('New corrected parser differs from validated v1 corrected labels')
    return (metadata, matrices, labels)

def subset_matrix(X: pd.DataFrame | np.ndarray, mask: np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    return np.asarray(X)[mask]
