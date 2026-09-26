# Validation scope

The final machine-readable check results are recorded in `SECURITY_REPORT.json`.

- Python source compilation without executing study analyses.
- Synthetic offline tests covering grouped splits, training-only processing,
  concordance direction, paired uncertainty, BCR triplet boundaries/ties,
  case selection, prompt profile configuration and CLI path propagation.
- Explicit deselection of the single test that needs the actual canonical
  Task 2 patient feature table.
- CLI help/import checks for current analysis entry points.
- Secret-pattern scanning, literal credential configuration checks, and exact
  in-memory matching against known locally configured credentials.
- Archive file-policy checks, per-file SHA-256 inventory and ZIP re-read checks.
- An extracted-copy offline test run verifies relocation away from the project.

Not performed: clean-environment dependency installation; complete-data
reproduction; independent scientific replication; API calls; model downloads;
GPU training; original figure rendering. This package is a curated code
submission, not a certification of numerical equivalence after reorganization.

Security scanning never prints a credential or a matching source line. Pattern
scanners can have false positives or miss unfamiliar secret formats; absence of
findings is not an absolute guarantee. The authors should still review the
archive, licensing and data-access requirements before external submission.
