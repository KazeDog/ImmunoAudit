"""Shared constants and safe I/O for the Task 2 cancer-context experiment."""
from __future__ import annotations
from submission_paths import path as _submission_path
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
EXPERIMENT_VERSION = 'task2_cancer_context_v1'
RANDOM_SEED = 42
N_SPLITS = 5
BASELINE_TARGET_MEAN = 0.5541684629731131
PROJECT_ROOT = _submission_path('project', '')
RAW_DATA_ROOT = Path(str(_submission_path('data', '2-/tmb_mskcc_2018')))
CANONICAL_X_PATH = PROJECT_ROOT / 'processed_data_strategy1' / 'task2_X_final.csv'
CANONICAL_Y_PATH = PROJECT_ROOT / 'processed_data_strategy1' / 'task2_y_final.csv'
RAW_PATIENT_PATH = RAW_DATA_ROOT / 'data_clinical_patient.txt'
RAW_SAMPLE_PATH = RAW_DATA_ROOT / 'data_clinical_sample.txt'
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'results' / 'task2_cancer_context'
PROTECTED_PYTHON_SETTINGS = (PROJECT_ROOT / 'code_strategy3' / 'python_settings.py', PROJECT_ROOT / 'code_strategy3' / 'python_settings_local.py', PROJECT_ROOT / 'final' / 'strategy3' / 'code' / 'code_strategy3' / 'python_settings.py')
ORIGINAL_CANCER_COLUMN = 'CANCER_TYPE'
MODEL_CANCER_COLUMN = 'CANCER_TYPE_MODEL'
RARE_OR_UNKNOWN_LEVEL = 'Skin cancer (melanoma + non-melanoma)'
RARE_CANCER_RULE = {'Melanoma': RARE_OR_UNKNOWN_LEVEL, 'Skin Cancer, Non-Melanoma': RARE_OR_UNKNOWN_LEVEL}
CONTEXT_COLUMNS = ['PATIENT_ID', 'SAMPLE_ID', ORIGINAL_CANCER_COLUMN, MODEL_CANCER_COLUMN, 'CANCER_TYPE_DETAILED', 'ONCOTREE_CODE', 'PRIMARY_SITE', 'SEX', 'AGE_GROUP', 'DRUG_TYPE', 'TMB_NONSYNONYMOUS', 'duration', 'event']

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_lines(values: Iterable[Any]) -> str:
    payload = '\n'.join((str(value) for value in values)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def require_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)

def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    return Path(temporary_name)

def _commit_no_clobber(temporary: Path, path: Path) -> None:
    """Atomically publish a completed file without ever replacing an existing path."""
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f'Refusing to overwrite existing output: {path}') from exc

def write_csv_new(frame: pd.DataFrame, path: Path, *, index: bool=False) -> None:
    require_new_path(path)
    temporary = _atomic_path(path)
    try:
        frame.to_csv(temporary, index=index)
        _commit_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_text_new(text: str, path: Path) -> None:
    require_new_path(path)
    temporary = _atomic_path(path)
    try:
        temporary.write_text(text, encoding='utf-8')
        _commit_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def write_json_new(payload: dict[str, Any], path: Path) -> None:
    write_text_new(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', path)

def protected_file_hashes() -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in PROTECTED_PYTHON_SETTINGS}
