"""Build the immutable v1 Task 2 cancer-context dataset and audit report."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from task2_context_common import CANONICAL_X_PATH, CANONICAL_Y_PATH, CONTEXT_COLUMNS, DEFAULT_RESULTS_DIR, EXPERIMENT_VERSION, MODEL_CANCER_COLUMN, ORIGINAL_CANCER_COLUMN, RARE_CANCER_RULE, RARE_OR_UNKNOWN_LEVEL, RAW_PATIENT_PATH, RAW_SAMPLE_PATH, protected_file_hashes, sha256_file, write_csv_new, write_text_new
PATIENT_FIELDS = ['PATIENT_ID', 'SEX', 'AGE_GROUP', 'DRUG_TYPE', 'OS_MONTHS', 'OS_STATUS']
SAMPLE_FIELDS = ['PATIENT_ID', 'SAMPLE_ID', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'ONCOTREE_CODE', 'PRIMARY_SITE', 'TMB_NONSYNONYMOUS']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x', type=Path, default=CANONICAL_X_PATH)
    parser.add_argument('--y', type=Path, default=CANONICAL_Y_PATH)
    parser.add_argument('--patient-table', type=Path, default=RAW_PATIENT_PATH)
    parser.add_argument('--sample-table', type=Path, default=RAW_SAMPLE_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()

def load_cbioportal_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t', comment='#', dtype=str)
    if 'PATIENT_ID' not in frame.columns:
        raise ValueError(f'PATIENT_ID missing from {path}')
    return frame

def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')

def build_context_dataset(x: pd.DataFrame, y: pd.DataFrame, patient: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a row-order-preserving, one-patient-per-row combined dataset."""
    _require_columns(y, ['duration', 'event'], 'Task 2 outcomes')
    _require_columns(patient, PATIENT_FIELDS, 'patient table')
    _require_columns(sample, SAMPLE_FIELDS, 'sample table')
    x = x.copy()
    y = y.copy()
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    x.index.name = 'PATIENT_ID'
    y.index.name = 'PATIENT_ID'
    if not x.index.is_unique or not y.index.is_unique:
        raise ValueError('Canonical X/y patient IDs must be unique')
    if x.index.tolist() != y.index.tolist():
        raise ValueError('Canonical X/y patient IDs or row order differ')
    if x.columns.duplicated().any():
        raise ValueError('Canonical mutation feature names are not unique')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Canonical mutation matrix contains non-finite values')
    values = np.unique(x.to_numpy(dtype=float))
    if not set(values.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f'Mutation matrix must be binary; observed values={values.tolist()}')
    patient = patient[PATIENT_FIELDS].copy()
    sample = sample[SAMPLE_FIELDS].copy()
    patient['PATIENT_ID'] = patient['PATIENT_ID'].astype(str)
    sample['PATIENT_ID'] = sample['PATIENT_ID'].astype(str)
    analyzed = set(x.index)
    analyzed_patient = patient[patient['PATIENT_ID'].isin(analyzed)].copy()
    analyzed_sample = sample[sample['PATIENT_ID'].isin(analyzed)].copy()
    if analyzed_patient['PATIENT_ID'].duplicated().any():
        raise ValueError('Multiple patient-table rows found for an analyzed patient')
    if analyzed_sample['PATIENT_ID'].duplicated().any():
        duplicated = analyzed_sample.loc[analyzed_sample['PATIENT_ID'].duplicated(keep=False), 'PATIENT_ID'].tolist()
        raise ValueError(f'Multiple sample rows found for analyzed patients: {duplicated[:5]}')
    if len(analyzed_patient) != len(x) or len(analyzed_sample) != len(x):
        raise ValueError('Not all analyzed patients have exactly one patient and sample metadata row')
    metadata = pd.DataFrame({'PATIENT_ID': x.index}).merge(analyzed_patient, on='PATIENT_ID', how='left', validate='one_to_one').merge(analyzed_sample, on='PATIENT_ID', how='left', validate='one_to_one')
    metadata['TMB_NONSYNONYMOUS'] = pd.to_numeric(metadata['TMB_NONSYNONYMOUS'], errors='raise').astype(float)
    metadata['duration'] = pd.to_numeric(y['duration'], errors='raise').to_numpy(dtype=float)
    metadata['event'] = pd.to_numeric(y['event'], errors='raise').to_numpy(dtype=int)
    raw_duration = pd.to_numeric(metadata['OS_MONTHS'], errors='raise').to_numpy(dtype=float)
    raw_event = metadata['OS_STATUS'].astype(str).str.contains('DECEASED', case=False, na=False).astype(int).to_numpy()
    if not np.array_equal(raw_duration, metadata['duration'].to_numpy(dtype=float)):
        raise ValueError('Canonical duration differs from raw OS_MONTHS')
    if not np.array_equal(raw_event, metadata['event'].to_numpy(dtype=int)):
        raise ValueError('Canonical event differs from raw OS_STATUS')
    if not set(metadata['event'].unique()).issubset({0, 1}):
        raise ValueError('event must be binary')
    if (metadata['duration'] < 0).any():
        raise ValueError('duration must be non-negative')
    metadata[MODEL_CANCER_COLUMN] = metadata[ORIGINAL_CANCER_COLUMN].replace(RARE_CANCER_RULE)
    expected_collapsed_n = int(metadata[ORIGINAL_CANCER_COLUMN].isin(RARE_CANCER_RULE).sum())
    if int((metadata[MODEL_CANCER_COLUMN] == RARE_OR_UNKNOWN_LEVEL).sum()) != expected_collapsed_n:
        raise ValueError('Unexpected size after applying the documented rare-cancer rule')
    metadata = metadata.drop(columns=['OS_MONTHS', 'OS_STATUS'])
    metadata = metadata[CONTEXT_COLUMNS]
    combined = pd.concat([metadata.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    audit = {'experiment_version': EXPERIMENT_VERSION, 'n_analyzed': int(len(combined)), 'n_mutation_features': int(x.shape[1]), 'events': int(combined['event'].sum()), 'censored': int((1 - combined['event']).sum()), 'event_rate': float(combined['event'].mean()), 'raw_patient_rows': int(len(patient)), 'raw_sample_rows': int(len(sample)), 'analyzed_patient_rows': int(len(analyzed_patient)), 'analyzed_sample_rows': int(len(analyzed_sample)), 'multi_sample_analyzed_patients': 0, 'x_y_exact_order': True, 'rare_cancer_rule': RARE_CANCER_RULE}
    return (combined, audit)

def outcome_permissible_pairs(duration: np.ndarray, event: np.ndarray) -> int:
    pairs = 0
    for i in range(len(duration)):
        for j in range(i + 1, len(duration)):
            if duration[i] < duration[j] and event[i] == 1:
                pairs += 1
            elif duration[j] < duration[i] and event[j] == 1:
                pairs += 1
            elif duration[i] == duration[j] and event[i] != event[j]:
                pairs += 1
    return int(pairs)

def build_cancer_audit(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata_fields = [ORIGINAL_CANCER_COLUMN, 'CANCER_TYPE_DETAILED', 'ONCOTREE_CODE', 'PRIMARY_SITE', 'SEX', 'AGE_GROUP', 'DRUG_TYPE', 'TMB_NONSYNONYMOUS']
    for cancer, group in combined.groupby(ORIGINAL_CANCER_COLUMN, sort=False):
        duration = group['duration'].to_numpy(dtype=float)
        event = group['event'].to_numpy(dtype=int)
        rows.append({'cancer_type': cancer, 'model_cancer_type': group[MODEL_CANCER_COLUMN].iloc[0], 'n': int(len(group)), 'events': int(event.sum()), 'censored': int(len(group) - event.sum()), 'event_rate': float(event.mean()), 'duration_min': float(np.min(duration)), 'duration_median': float(np.median(duration)), 'duration_max': float(np.max(duration)), 'outcome_permissible_pairs': outcome_permissible_pairs(duration, event), 'metadata_missing_cells': int(group[metadata_fields].isna().sum().sum())})
    return pd.DataFrame(rows).sort_values('n', ascending=False).reset_index(drop=True)

def render_audit_markdown(combined: pd.DataFrame, cancer_audit: pd.DataFrame, audit: dict[str, Any], input_hashes: dict[str, str]) -> str:
    metadata_missing = combined[CONTEXT_COLUMNS].isna().sum()
    missing_lines = '\n'.join((f'- `{column}`: {int(count)}' for column, count in metadata_missing.items()))

    def format_cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f'{float(value):.4f}'
        return str(value).replace('|', '\\|')
    headers = cancer_audit.columns.tolist()
    table_lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    table_lines.extend(('| ' + ' | '.join((format_cell(value) for value in row)) + ' |' for row in cancer_audit.itertuples(index=False, name=None)))
    cancer_table = '\n'.join(table_lines)
    return f"# Task 2 cancer-context data audit\n\nExperiment version: `{EXPERIMENT_VERSION}`\n\n## Cohort and join\n\n- Analyzed patients: {audit['n_analyzed']}\n- Mutation features: {audit['n_mutation_features']}\n- Events / censored: {audit['events']} / {audit['censored']}\n- Event rate: {audit['event_rate']:.6f}\n- Raw patient / sample rows: {audit['raw_patient_rows']} / {audit['raw_sample_rows']}\n- Analyzed patient / sample rows: {audit['analyzed_patient_rows']} / {audit['analyzed_sample_rows']}\n- Multi-sample analyzed patients: {audit['multi_sample_analyzed_patients']}\n- Canonical X/y IDs are unique and in exact row order: yes\n- Raw OS duration/event exactly match canonical y: yes\n\nThe analyzed cohort is the existing 1,610-patient canonical cohort. The upstream code\nthat created the 468-gene matrix is unavailable; this experiment therefore freezes the\ncanonical matrix and does not claim to reconstruct its feature-selection provenance.\nThe available raw patient table has 1,661 rows; prior audit showed that the canonical\ncohort corresponds to the 1,610 patients with non-zero mutation/TMB representation,\nso 51 zero-mutation patients were excluded upstream before this experiment.\n\n## Model-only rare-level rule\n\nThe original broad cancer label is retained unchanged for audit and subgroup reporting.\nFor model encoding and Cox strata only, `Melanoma` (n=313) and the singleton\n`Skin Cancer, Non-Melanoma` (n=1) are mapped to the explicit anatomical composite\n`{RARE_OR_UNKNOWN_LEVEL}` (n=314). This prevents a one-patient model level without\nsilently relabelling the original cancer field. The raw broad and detailed labels remain\nunchanged for audit and subgroup reporting. The two diseases are biologically distinct;\nthe merge is a pragmatic singleton rule rather than a biological equivalence claim.\nThe singleton remains its own original subgroup and is ineligible for a cancer-specific\nC-index.\n\n## Metadata missingness\n\n{missing_lines}\n\n## Cancer-level audit\n\n{cancer_table}\n\n## Frozen input SHA-256\n\n" + '\n'.join((f'- `{path}`: `{digest}`' for path, digest in input_hashes.items())) + '\n'

def main() -> None:
    args = parse_args()
    for path in [args.x, args.y, args.patient_table, args.sample_table]:
        if not path.is_file():
            raise FileNotFoundError(path)
    protected_before = protected_file_hashes()
    x = pd.read_csv(args.x, index_col=0)
    y = pd.read_csv(args.y, index_col=0)
    patient = load_cbioportal_table(args.patient_table)
    sample = load_cbioportal_table(args.sample_table)
    combined, audit = build_context_dataset(x, y, patient, sample)
    cancer_audit = build_cancer_audit(combined)
    input_hashes = {str(args.x.resolve()): sha256_file(args.x), str(args.y.resolve()): sha256_file(args.y), str(args.patient_table.resolve()): sha256_file(args.patient_table), str(args.sample_table.resolve()): sha256_file(args.sample_table)}
    output_dir = args.output_dir.resolve()
    dataset_path = output_dir / f'{EXPERIMENT_VERSION}.csv'
    metadata_path = output_dir / f'{EXPERIMENT_VERSION}_metadata.csv'
    audit_csv_path = output_dir / 'data_audit.csv'
    audit_md_path = output_dir / 'data_audit.md'
    write_csv_new(combined, dataset_path)
    write_csv_new(combined[CONTEXT_COLUMNS], metadata_path)
    write_csv_new(cancer_audit, audit_csv_path)
    write_text_new(render_audit_markdown(combined, cancer_audit, audit, input_hashes), audit_md_path)
    protected_after = protected_file_hashes()
    if protected_before != protected_after:
        raise RuntimeError('A protected python_settings file changed during data build')
    print(f'dataset={dataset_path}')
    print(f'metadata={metadata_path}')
    print(f'audit_csv={audit_csv_path}')
    print(f'audit_md={audit_md_path}')
if __name__ == '__main__':
    main()
