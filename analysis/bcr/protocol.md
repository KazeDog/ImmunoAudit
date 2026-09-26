# Locked repair protocol — 2026-09-26

Written before viewing repaired results. This is a retrospective correction,
not a preregistration and not an independent validation study.

1. Preserve raw data, all historical experiment directories, and a pre-repair
   snapshot of the manuscript. Only publish corrected results after independent checks.
2. BCR features: fixed lexicographic vocabulary of 20^3 standard amino-acid
   triplets; count windows wholly within each original CDR3, never across clone
   boundaries or ambiguous characters. Retain the historical equal-clone-row
   counting rule (no new abundance weighting), then normalize each repertoire
   by its total valid triplet count. Fail on empty repertoires, do not impute.
3. Retain labels, exclusion rules, feature/embedding row alignment, hyperparameters,
   fixed patient folds and all 20 existing repeat folds. No seed or model selection
   based on repaired performance; no tuning to make AUROC exceed 0.5.
4. SVM: disable internal probability fitting; use signed decision_function for
   ranking and predict for class calls. Scores are explicitly not probabilities;
   do not compute Brier/calibration statistics on margins or post-hoc flip them.
5. Primary repaired BCR AUROC: sum concordant positive-negative pairs (ties half)
   within test folds, divided by all eligible same-fold pairs. Average repeated
   samples within patient first. All paired comparators use identical patients
   and pair eligibility. Mean-fold and pooled AUROC remain named sensitivities.
6. Patient bootstrap: 2000 draws, seed42, resample within fold × binary outcome;
   fixed predictions, paired resamples; percentile intervals, not model-refitting
   uncertainty. Repeated splits are descriptive, not independent experiments.
7. Refit only predictions affected by changes (3-mer, combined3-mer and SVM).
   Preserve and re-evaluate unaffected saved predictions, including fixed LLMs.
   No new embeddings, no new LLM calls. CPU cap: up to 8 workers × 8 threads.
8. Overfitting is a limitation, not an implementation error that can be guaranteed
   away. Preserve train/test diagnostics; any new tuning study needs separately
   locked training-only selection. Current repair will not silently add tuning.
9. Update the BCR narrative, displays, supplementary results and Source Data; keep
   old values explicitly historical rather than overwriting immutable experiments.
   Other classification analyses require separate impact tracing before changing
   their selection, uncertainty, disagreement analyses or displayed values.
