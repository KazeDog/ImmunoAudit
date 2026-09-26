# ImmunoAudit

This is a **code-only reviewer package for the ImmunoAudit analyses**, prepared
on 26 September 2026. It is not a raw-data release or a one-command reproduction
of every manuscript figure. No API credentials, patient tables, saved requests,
model weights, old releases, manuscript-build scripts or figure-revision chains
are included. The original research workspace is not modified by this export.

## Quick start: offline checks

Use Python 3.12. From the extracted package directory:

```bash
python -m pip install -r requirements-analysis.txt -r requirements-llm.txt
python scripts/verify_submission.py
python scripts/run_offline_tests.py
```

Alternatively, create the environment with `conda env create -f environment.yml`.
The requirements record versions observed in the existing analysis environment;
they are not a complete transitive lockfile. A clean-environment installation has
not been validated. Optional FT-Transformer/TabPFN fitting has additional packages
in `requirements-models-optional.txt` and requires separately obtained weights.

The offline suite uses synthetic fixtures and disables Python socket connections
in its process. One study-data integration test is explicitly deselected because
its patient table is not distributed. Passing these tests is not a claim that
the complete paper has been reproduced from this archive alone.

## Contents and current entry points

| Directory | Role |
| --- | --- |
| `analysis/grouped_tasks/` | Patient-grouped evaluation for Tasks 1, 3 and 5 |
| `experiments/task2_cancer_context/` | Task 2 data assembly and Cox models |
| `analysis/cancer_context/` | Task 2 concordance decomposition and model evaluation |
| `analysis/endpoint_controls/` | Survival horizons, IPCW and calibration |
| `analysis/transportability/` | Leave-one-cancer-type-out analysis |
| `analysis/temporal_summary/` | Pretreatment and matched-summary controls |
| `analysis/bcr/` | Current within-CDR3 features, SVM margins and within-fold AUROC |
| `analysis/disagreement/` | Quantitative disagreement and prespecified cases |
| `analysis/rank_sensitivity/` | Current fold-relative survival-rank sensitivity |
| `code_strategy3/` | LLM task adapters, prompts, decision rules and provider clients |

Only needed BCR input/model helpers are retained from shared research modules;
their superseded analysis entry points are removed. Original source hashes and
the export mapping are recorded in `source_export_manifest.json`, not as copies
of historical code. **Scientific figures are not included:** their original
renderers depend on the revision chain excluded at the author's request.

## Paths are parameters

Always run an analysis through the portable launcher:

```bash
python run.py \
  --project-root /path/to/reproduction_workspace \
  --data-root /path/to/raw_datasets \
  --models-root /path/to/external_models \
  --script analysis/grouped_tasks/scripts/run_task135_audit.py \
  -- --output-dir /path/to/new_grouped_run --mode smoke
```

Replace all example paths. `--project-root` defaults to the extracted package;
`--data-root` defaults to `PROJECT/data`, and `--models-root` to
`PROJECT/external_models`. A different project root must be a **working copy of
this package plus the input layout described in `INPUTS.md`**, not an arbitrary
empty directory or the author's historical workspace. Code paths, relative data
paths and inter-module imports use the reorganized submission layout.

Arguments before `--` configure shared paths; arguments after it belong to the
selected script. For scripts that also accept `--project-root` or `--raw-root`,
their defaults inherit the shared configuration. Do not specify contradictory
root arguments. An internal environment bridge propagates path parameters to
child processes; users need not configure path environment variables.

Use a fresh workspace and new output directories. Some modules have result
locations relative to their workspace; do not run experiments in an immutable
release copy. The integrity checker intentionally reports any subsequent edits
or newly generated files.

## Example analysis commands

These commands need the external inputs listed in `INPUTS.md`; **they were not
run on study data when preparing this submission package**. Starting from a
working copy with inputs installed:

```bash
# Show arguments without fitting models.
python run.py --script analysis/grouped_tasks/scripts/run_task135_audit.py -- --help

# Current corrected BCR protocol: smoke first, then --mode all for the full protocol.
python run.py --data-root /path/to/raw_datasets --script scripts/run_bcr.py \
  -- --output-dir /path/to/new_bcr_run --mode smoke --workers 8

# Current fold-relative survival ranks; no new model fits or LLM calls.
python run.py --script analysis/rank_sensitivity/analyse_rank_scales.py \
  -- --input /path/to/analysis_units_wide.csv --output-dir /path/to/new_rank_run
```

BCR uses up to 8 workers and up to 8 threads per fit. Its full protocol preserves
the original fixed folds, 20 repeat seeds and model settings. Repeated partitions
are not independent biological experiments. It reuses unaffected fixed-model
predictions; it does not call LLMs or regenerate embeddings. The current input
adapter reads the corrected triplet matrix directly, without loading a discarded
pre-correction feature matrix. Disagreement case analysis now points explicitly
to corrected BCR case inputs. See `EXPORT_NOTES.md` for these packaging changes.

## API credentials

Credentials are read **only from environment variables**:
`DASHSCOPE_API_KEY` or `QWEN_API_KEY` for the Qwen-compatible client, and
`GEMINI_API_KEY` for Gemini. `.env.example` contains empty fields and is not loaded
automatically. Private local Python settings are neither included nor imported.

Do not pass a key as a command-line argument, paste it in a notebook, or store it
in the source tree. Use a secret manager or a masked interactive shell prompt.
The launcher disables socket connections in its process by default; live API use
requires `--allow-network`. This is an accidental-use guard, not an OS network
sandbox. Provider clients and scripts run directly outside the launcher are not
covered by that guard. New inference can incur fees and send input text to an
external service; only use appropriately authorized, de-identified inputs.

Remote model availability can change. Re-querying a provider is not guaranteed to
reproduce frozen predictions, even with the same model name and temperature.
Provider identities/settings here document the study configuration, not a claim
that these endpoints remain available.

## Validation, data and licensing boundaries

See `VALIDATION.md` for executed checks and `SECURITY_REPORT.json` for scan scope.
No live API call, model download, full-data fit or figure rendering is part of
package validation. Known local credentials are checked in memory before export;
only pass/fail findings are reported. Pattern scanning is heuristic and cannot
guarantee detection of every possible secret.

Data accessions and access conditions remain in the manuscript's Data
Availability statement. This archive does not grant permission to redistribute
patient-level data, embeddings, commercial fonts or third-party weights. There
is no public repository identifier assigned by this local packaging operation.
See `LICENSE_NOTICE.md` before publicly releasing or licensing the code.
