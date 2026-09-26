"""Predefined fold-relative sensitivity of retained survival predictions; no fits."""
from submission_paths import path as _submission_path
from pathlib import Path
import hashlib, json, itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT = _submission_path('project', '')
H = Path(__file__).resolve().parent
SOURCE = ROOT / 'analysis/disagreement/metadata/analysis_units_wide.csv'
KEYS = ['conventional', 'representation', 'qwen']
PAIRS = [(0, 1), (0, 2), (1, 2)]

def ranks(x):
    return np.column_stack([(rankdata(x[:, j], method='average') - 0.5) / len(x) for j in range(3)])

def direct_ranks(x):
    return np.column_stack([((x[:, j, None] > x[None, :, j]).sum(1) + 0.5 * (x[:, j, None] == x[None, :, j]).sum(1)) / len(x) for j in range(3)])

def summarize(blocks):
    rr = [ranks(x) for x in blocks]
    with np.errstate(invalid='ignore', divide='ignore'):
        cor = np.array([[np.corrcoef(r[:, a], r[:, b])[0, 1] for a, b in PAIRS] for r in rr])
    weights = np.array([len(x) for x in blocks])
    weighted = (cor * weights[:, None]).sum(0) / weights.sum()
    d = np.concatenate([r.max(1) - r.min(1) for r in rr])
    return (weighted, np.quantile(d, [0.25, 0.5, 0.75]), cor, d, rr)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    all_units = pd.read_csv(source)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    surv = all_units[all_units.task == 'task5'].copy()
    assert len(surv) == 286
    corr_rows = []
    disp_rows = []
    fold_rows = []
    unit_rows = []
    bootstrap = []
    rng = np.random.default_rng(20260926)
    for ds, frame in surv.groupby('dataset', sort=True):
        frames = [g.copy() for _, g in frame.groupby('fold', sort=True)]
        assert len(frames) == 5 and frame.unit_id.is_unique
        blocks = [g[[f'score_{k}' for k in KEYS]].to_numpy(float) for g in frames]
        assert all((np.isfinite(x).all() for x in blocks))
        est, quant, folds, dispersion, rr = summarize(blocks)
        assert np.isfinite(est).all()
        for x, r in zip(blocks, rr):
            np.testing.assert_allclose(r, direct_ranks(x), atol=1e-14)
        transformed = [x * (i + 1.7) + (i - 2) * 100 for i, x in enumerate(blocks)]
        invariants = summarize(transformed)
        np.testing.assert_allclose(est, invariants[0], atol=1e-12)
        np.testing.assert_allclose(dispersion, invariants[3], atol=1e-12)
        draws = []
        for b in range(2000):
            sampled = [x[rng.integers(0, len(x), len(x))] for x in blocks]
            bc, bq, _, _, _ = summarize(sampled)
            draws.append([*bc, bq[1]])
        draws = np.array(draws)
        for j, (a, b) in enumerate(PAIRS):
            valid = np.isfinite(draws[:, j])
            assert valid.sum() >= 1900
            lo, hi = np.quantile(draws[valid, j], [0.025, 0.975])
            corr_rows.append(dict(dataset=ds, dataset_label=frame.dataset_label.iloc[0], strategy_a=KEYS[a], strategy_b=KEYS[b], n_patients=len(frame), estimate=est[j], ci_lower=lo, ci_upper=hi, valid_bootstrap=int(valid.sum()), original_cohort_spearman=float(np.corrcoef(ranks(np.vstack(blocks)).T)[a, b])))
        lo, hi = np.quantile(draws[:, 3], [0.025, 0.975])
        disp_rows.append(dict(dataset=ds, n_patients=len(frame), median=quant[1], q25=quant[0], q75=quant[2], ci_lower=lo, ci_upper=hi, original_cohort_median=frame.rank_dispersion.median()))
        for i, (g, r) in enumerate(zip(frames, rr)):
            fold = g.fold.iloc[0]
            for j, (a, b) in enumerate(PAIRS):
                fold_rows.append(dict(dataset=ds, fold=int(fold), n_patients=len(g), strategy_a=KEYS[a], strategy_b=KEYS[b], estimate=folds[i, j]))
            for n, (_, row) in enumerate(g.iterrows()):
                unit_rows.append(dict(task='task5', dataset=ds, dataset_label=row.dataset_label, unit_id=row.unit_id, cluster_id=row.cluster_id, endpoint_type='survival', n_samples=row.n_samples, fold=int(fold), rank_dispersion=r[n].max() - r[n].min(), **{f'percentile_{k}': r[n, j] for j, k in enumerate(KEYS)}))
        for i, draw in enumerate(draws):
            bootstrap.append(dict(dataset=ds, draw=i, conv_repr=draw[0], conv_qwen=draw[1], repr_qwen=draw[2], median_dispersion=draw[3]))
    for name, rows in [('correlations', corr_rows), ('dispersion_summary', disp_rows), ('fold_correlations', fold_rows), ('units', unit_rows), ('bootstrap', bootstrap)]:
        pd.DataFrame(rows).to_csv(out / f'survival_fold_relative_{name}.csv', index=False)
    report = dict(status='PASS', n_patients=286, n_cohorts=3, n_folds=15, bootstrap_per_cohort=2000, no_refitting=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), point_ranks_independently_verified=True, positive_fold_affine_invariance=True, definition='patient-count-weighted mean of within-fold Spearman; patient median of fold-relative dispersion', caveat='Fold-relative positions are not cohort-wide calibrated risk ranks. Conditional on retained predictions.')
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print(pd.DataFrame(disp_rows).to_string(index=False))
if __name__ == '__main__':
    main()
