# Required inputs (not distributed)

The package starts from curated study inputs and retained model outputs. It is
not an end-to-end raw-data download/preprocessing distribution. Obtain data under
the access conditions stated in the manuscript; retain patient identifiers only
inside an authorized analysis workspace. No example in this package contains an
actual patient record.

`PROJECT`, `DATA` and `MODELS` refer to the three launcher path parameters.
Existing canonical data filenames are preserved for alignment checks; old code
directories are not required. Selected path overrides are also available in each
script's `--help`. The list below describes required input classes and key paths,
not an exhaustive list of every cohort-specific file loaded by every optional
model adapter.

## Curated feature and prediction inputs

| Input | Location or requirement |
| --- | --- |
| Conventional features/endpoints | `PROJECT/processed_data_strategy1/`; paired `task*_X_final.csv`, `task*_y_final.csv` and cohort-specific matrices expected by each task loader |
| Saved representations | `PROJECT/processed_data_strategy2/`; matrices plus sample-order tables, including BCR ESM2/AntiBERTy `.npy` and `*_samples.csv` |
| Single-cell clinical metadata | `DATA/1-` or the study subdirectories referenced by the task adapters |
| Mutation/survival clinical metadata | `DATA/2-/tmb_mskcc_2018/`, including `data_clinical_patient.txt` and `data_clinical_sample.txt` |
| Bulk clinical metadata | `DATA/3-/` with cohort-specific clinical tables |
| BCR clinical table | `DATA/4-/task5_clinical_labels.csv` |
| Raw BCR CDR3 sequences | `DATA/4-/GSE296826_RAW/*BCR*.tsv*`; column `aaSeqImputedCDR3` |
| Fixed LLM outputs | `PROJECT/results/strategy3/` with task/model-specific prediction files expected by the adapters; requests and caches are not shipped |
| TabPFN checkpoint | `MODELS/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt`, or explicit `--tabpfn-checkpoint` |

Feature/label indices, sample-order tables and patient identifiers must agree.
Do not invent missing predictions, substitute zero features or reverse score
direction to obtain a desired AUROC. All transformation fitting must retain the
original training/test separation.

## Frozen evaluation artifacts

Install only the needed authorized artifacts in these **current submission
locations** (not archived code directories):

- `analysis/cancer_context/fold_assignments/task2_frozen_fold_assignments.csv`
- `analysis/cancer_context/metadata/task4_sample_patient_cancer_mapping.csv`
- `analysis/cancer_context/fold_assignments/task4_patient_grouped_folds.csv`
- `analysis/endpoint_controls/predictions/task4_priority1_oof_predictions.csv`
- `analysis/endpoint_controls/metadata/task4_patient_label_time_audit.csv`
- `analysis/transportability/metadata/task4_priority2_repeated_folds.csv`
- `analysis/transportability/predictions/task4_priority2_sample_predictions.csv`
- `results/task2_cancer_context/task2_cancer_context_v1.csv`
- `results/task2_cancer_context/full_v2_fixed_bootstrap/oof_predictions.csv`

The BCR entry point preserves folds and unaffected comparator predictions from
the retained analysis inputs. The filenames above identify those frozen input
tables, not historical executable code included in this release. No pre-repair
3-mer matrix is required: corrected frequencies are reconstructed from raw CDR3s.
Full BCR reproduction is blocked until these non-bundled inputs are available.

## Disagreement and case analysis

The engine expects:

- `analysis/grouped_tasks/full_v1/predictions/classification_predictions.csv`
- `analysis/grouped_tasks/full_v1/predictions/survival_predictions.csv`
- `analysis/bcr/bcr_case_input.csv` — **corrected** BCR predictions only.
- `analysis/endpoint_controls/metadata/task4_patient_label_time_audit.csv`

The BCR launcher writes `bcr_case_input.csv` in its chosen output directory after
`--mode repeats` or `--mode all`. Make that file available at the case engine's
configured location (or use its explicit input parameter). Do not substitute the
uncorrected comparator cache for this case-analysis input.

The fold-relative sensitivity accepts `--input` pointing to the resulting
`metadata/analysis_units_wide.csv`. It expects the original study's 286 survival
patients across three cohorts and five folds, and checks these invariants.
Its output is the primary survival disagreement scale in the manuscript; cohort-wide
ranks from the case engine remain sensitivity quantities, not calibrated risk.

The fixed cohort/patient counts are intentional safeguards for the reported
study. Adapting the code to another cohort is a new analysis, not exact replication.
