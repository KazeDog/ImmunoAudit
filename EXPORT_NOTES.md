# Export changes and scope

The submission package is a separate copy. Study source files, scientific
outputs, credentials and the manuscript are not edited by packaging.

1. Current analytical modules are reorganized under `analysis/`. Archived
   manuscript releases, publication scripts, old figure revisions and their
   renderer dependency chain are excluded. Shared BCR modules retain only the
   functions needed by the current implementation, not superseded runners.
2. Author-machine absolute paths and project-root discovery expressions are
   replaced by the parameter-driven `submission_paths` resolver. Python AST
   export normalizes formatting and removes comments from exported research
   files; statistical expressions are preserved except the explicit input/I/O
   changes below. Source and exported hashes are in the source manifest.
3. The private local credential module is excluded. Its import mechanism is
   replaced with an environment-only configuration containing public model
   identifiers but no keys. No local-settings fallback exists in this package.
4. BCR input loading now consumes the corrected, newly generated triplet matrix
   directly instead of reading a pre-correction matrix and immediately replacing
   it. Shape, finiteness and row-normalization checks are added. This changes
   input plumbing, not the features used for fitting. Full-data equality of this
   exported adapter has not been re-run in this packaging task.
5. The case engine is bound to corrected BCR case predictions. The BCR launcher
   assembles that input from the same 20 corrected repeat outputs and the same
   incident-endpoint XGBoost model as the current manuscript preparation.
6. The survival-rank entry point accepts explicit input/output paths. Seeds,
   bootstrap counts, rank conventions and within-fold aggregation are unchanged.
7. Added synthetic portability/security tests and an offline test launcher. One
   original study-data integration test is retained but deselected offline.

No new scientific experiments, model tuning, inference calls, embeddings,
provider queries or manuscript edits are performed. Complete numerical
equivalence of all exported entry points requires the external study inputs and
remains outside this package-level validation.
