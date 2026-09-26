"""Offline no-clobber BCR correction; resume only completed keyed checkpoints."""
from submission_paths import path as _submission_path
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, sys, json, hashlib, warnings, os, multiprocessing
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from repair_core import repertoire_frequencies, TRIPLETS, aucs, bootstrap_counts, bootstrap_auc
ROOT = Path(str(_submission_path('project', '')))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bcr_inputs as p1
from task4_grouped_core import ModelSpec, build_estimator
CONTEXT = 'metadata_only|cancer_type+ici_treatment|LogisticRegression'
XGB = 'strategy1|BCR_3mer|XGBoost'
COMBO = 'combined|BCR_3mer+cancer_type+ici_treatment|XGBoost'
REPEAT_KEYS = [CONTEXT, XGB, 'strategy2|ESM2|LR', 'strategy2|ESM2|MLP', COMBO, 'combined|ESM2+cancer_type+ici_treatment|LR']

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)

def js(obj, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')

def build_features():
    out = HERE / 'features'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        m = json.loads((out / 'manifest.json').read_text())
        assert digest(out / 'triplet_frequencies.npy') == m['matrix_sha256']
        assert all((digest(k) == v for k, v in m['input_hashes'].items()))
        return
    meta = pd.read_csv(p1.MAPPING_PATH)
    raw = Path(str(_submission_path('data', '4-/GSE296826_RAW')))
    lookup = {p.name.split('_')[0]: p for p in raw.glob('*BCR*.tsv*')}
    hashes = {str(p): digest(p) for p in lookup.values()}
    rows = []
    mat = []
    for sid in meta.sample_id:
        table = pd.read_csv(lookup[sid], sep='\t', usecols=['aaSeqImputedCDR3'], keep_default_na=False)
        a, q = repertoire_frequencies(table.aaSeqImputedCDR3)
        mat.append(a)
        rows.append(dict(sample_id=sid, **q))
    array = np.asarray(mat)
    np.testing.assert_allclose(array.sum(axis=1), 1)
    np.save(out / 'triplet_frequencies.npy', array)
    csv(pd.DataFrame(rows), out / 'sample_qc.csv')
    csv(pd.DataFrame({'triplet': TRIPLETS}), out / 'vocabulary.csv')
    assert all((digest(k) == v for k, v in hashes.items()))
    js(dict(status='PASS', shape=list(array.shape), matrix_sha256=digest(out / 'triplet_frequencies.npy'), input_hashes=hashes, definition='Equal row-count valid within-CDR3 triplets, normalized per repertoire; fixed 8000-word vocabulary'), out / 'manifest.json')

def patient_average(pred):
    assert pred.groupby(['analysis', 'model_key', 'patient_id']).true_label.nunique().max() == 1
    result = pred.groupby(['analysis', 'model_key', 'patient_id'], sort=False).agg(true_label=('true_label', 'first'), risk_score=('risk_score', 'mean'), fold=('fold', 'first'), cancer_type=('cancer_type', 'first'), ici_treatment=('ici_treatment', 'first'), n_samples=('sample_id', 'size')).reset_index()
    metadata = result.model_key.str.startswith('metadata_only|')
    result.loc[metadata, 'risk_score'] = result.loc[metadata, 'risk_score'].round(12)
    return result

def fit_one(meta, matrices, key):
    strategy, feature, model_name = key.split('|')
    y = meta.label.to_numpy(int)
    folds = meta.fold.to_numpy(int)
    if strategy == 'combined':
        X = np.asarray(matrices['BCR_3mer'])
    else:
        X = np.asarray(matrices[feature])
    score = np.full(len(y), np.nan)
    calls = np.full(len(y), -1, int)
    training = []
    for fold in sorted(set(folds)):
        tr = folds != fold
        te = ~tr
        assert not set(meta.loc[tr, 'patient_id']) & set(meta.loc[te, 'patient_id'])
        trainx = X[tr]
        testx = X[te]
        if strategy == 'combined':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cats = meta[['cancer_type', 'ici_treatment']].astype(str)
            trainx = np.hstack([trainx, encoder.fit_transform(cats.loc[tr])])
            testx = np.hstack([testx, encoder.transform(cats.loc[te])])
            model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=8)
        else:
            model = build_estimator(ModelSpec(strategy, feature, model_name))
            params = model.get_params(deep=True)
            for name in ['n_jobs', 'clf__n_jobs']:
                if name in params:
                    model.set_params(**{name: 1 if model_name in ['LogisticRegression', 'LR'] else 8})
            for name in ['use_label_encoder', 'clf__use_label_encoder']:
                if name in params:
                    model.set_params(**{name: None})
            if model_name == 'SVM':
                model.set_params(clf__probability=False)
        with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=8):
            warnings.simplefilter('always')
            model.fit(trainx, y[tr])
            scorer = model.decision_function if model_name == 'SVM' else lambda a: model.predict_proba(a)[:, 1]
            score[te] = scorer(testx)
            calls[te] = model.predict(testx)
            train_score = scorer(trainx)
        assert np.array_equal(model.classes_, [0, 1])
        training.append(dict(model_key=key, fold=int(fold), train_auc=float(roc_auc_score(y[tr], train_score)), test_auc=float(roc_auc_score(y[te], score[te])) if len(np.unique(y[te])) == 2 else np.nan, n_train=int(tr.sum()), n_test=int(te.sum()), warnings=' | '.join((str(w.message) for w in ws))))
    out = meta[['sample_id', 'patient_id', 'collection_time', 'cancer_type', 'ici_treatment', 'fold']].copy()
    out['model_key'] = key
    out['strategy'] = strategy
    out['feature'] = feature
    out['model'] = model_name
    out['true_label'] = y
    out['risk_score'] = score
    out['pred_label'] = calls
    out['score_kind'] = 'decision_margin' if model_name == 'SVM' else 'probability'
    return (out, pd.DataFrame(training))

def job(item):
    analysis, seed, mode = item
    out = HERE / mode / 'checkpoints' / f'{analysis}__{seed}'
    if (out / 'complete.json').exists():
        m = json.loads((out / 'complete.json').read_text())
        assert digest(out / 'sample_predictions.csv') == m['prediction_sha256']
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    meta, mats, _ = p1.load_inputs(HERE / 'features/triplet_frequencies.npy')
    mats['BCR_3mer'] = np.load(HERE / 'features/triplet_frequencies.npy')
    spec = next((s for s in p1.ANALYSES if s.name == analysis))
    mask = meta.cancer_type.ne('Control').to_numpy()
    if spec.baseline_only:
        mask &= meta.collection_time.astype(str).str.lower().eq('bl').to_numpy()
    matrices = {k: p1.subset_matrix(v, mask) for k, v in mats.items()}
    meta = meta.loc[mask].reset_index(drop=True)
    meta['label'] = meta[spec.label_column].astype(int)
    if seed != 'fixed':
        folds = pd.read_csv(ROOT / 'analysis/transportability/metadata/task4_priority2_repeated_folds.csv')
        folds = folds[folds.label_definition.eq(spec.label_column) & folds.repeat_seed.eq(int(seed))].set_index('patient_id').fold
        meta['fold'] = meta.patient_id.map(folds).astype(int)
        existing = pd.read_csv(ROOT / 'analysis/transportability/predictions/task4_priority2_sample_predictions.csv')
        existing = existing[existing.analysis.eq(analysis) & existing.repeat_seed.eq(int(seed))].copy()
        keys = [XGB, COMBO]
    else:
        existing = pd.read_csv(ROOT / 'analysis/endpoint_controls/predictions/task4_priority1_oof_predictions.csv')
        existing = existing[existing.analysis.eq(analysis)].copy()
        keys = [f'strategy1|BCR_3mer|{n}' for n in ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost']] + ['strategy2|ESM2|SVM', 'strategy2|AntiBERTy|SVM', COMBO]
    base = existing[~existing.model_key.isin(keys)].copy()
    base['score_kind'] = 'probability'
    chunks = [base]
    fits = []
    for key in keys:
        pred, train = fit_one(meta, matrices, key)
        pred['analysis'] = analysis
        train['analysis'] = analysis
        chunks.append(pred)
        fits.append(train)
    samples = pd.concat(chunks, ignore_index=True)
    samples['repeat_seed'] = str(seed)
    patient = patient_average(samples)
    patient['repeat_seed'] = str(seed)
    csv(samples, out / 'sample_predictions.csv')
    csv(patient, out / 'patient_predictions.csv')
    csv(pd.concat(fits), out / 'fit_diagnostics.csv')
    js(dict(status='PASS', prediction_sha256=digest(out / 'sample_predictions.csv'), models=patient.model_key.nunique(), patients=patient.patient_id.nunique()), out / 'complete.json')
    print(f'COMPLETE {analysis} {seed}', flush=True)
    return str(out)

def summarize(paths, mode):
    out = HERE / mode
    patients = pd.concat([pd.read_csv(Path(p) / 'patient_predictions.csv', dtype={'repeat_seed': str}) for p in paths], ignore_index=True)
    rows = []
    subgroup = []
    draws = []
    with threadpool_limits(limits=2):
        for (analysis, seed), block in patients.groupby(['analysis', 'repeat_seed'], sort=False):
            wide = block.pivot(index='patient_id', columns='model_key', values='risk_score')
            meta = block.drop_duplicates('patient_id').set_index('patient_id').loc[wide.index]
            y = meta.true_label.to_numpy(int)
            fold = meta.fold.to_numpy(int)
            counts = bootstrap_counts(y, fold) if seed == 'fixed' else None
            ref = aucs(y, wide[CONTEXT], fold)['auroc']
            refdraw = bootstrap_auc(y, wide[CONTEXT], fold, counts) if counts is not None else None
            for key in wide:
                a = aucs(y, wide[key], fold)
                row = dict(analysis=analysis, repeat_seed=seed, model_key=key, n_patients=len(y), n_positive=int(y.sum()), **a, context_auroc=ref, delta_auroc_vs_context=a['auroc'] - ref)
                if counts is not None:
                    vals = bootstrap_auc(y, wide[key], fold, counts)
                    delta = vals - refdraw
                    row.update(ci_low=float(np.quantile(vals, 0.025)), ci_high=float(np.quantile(vals, 0.975)), delta_ci_low=float(np.quantile(delta, 0.025)), delta_ci_high=float(np.quantile(delta, 0.975)), bootstrap_reps=2000)
                    draws.append(pd.DataFrame(dict(analysis=analysis, model_key=key, draw=np.arange(2000), auroc=vals, delta=delta)))
                rows.append(row)
                if seed == 'fixed':
                    for cancer in ['Melanoma', 'Non-small cell Lung Cancer']:
                        mask = meta.cancer_type.eq(cancer).to_numpy()
                        subgroup.append(dict(analysis=analysis, model_key=key, cancer_type=cancer, n_patients=int(mask.sum()), **aucs(y[mask], wide[key].to_numpy()[mask], fold[mask])))
    csv(patients, out / 'patient_predictions.csv')
    csv(pd.DataFrame(rows), out / 'metrics.csv')
    if subgroup:
        csv(pd.DataFrame(subgroup), out / 'subgroup_metrics.csv')
    if draws:
        csv(pd.concat(draws), out / 'bootstrap_draws.csv')
    allfits = pd.concat([pd.read_csv(Path(p) / 'fit_diagnostics.csv') for p in paths], ignore_index=True)
    csv(allfits, out / 'fit_diagnostics.csv')
    js(dict(status='PASS', analyses=patients.analysis.nunique(), predictions=len(patients), metric_rows=len(rows), n_new_fits=len(allfits), protocol_sha256=digest(HERE / 'protocol.md'), no_llm_calls=True, no_new_embeddings=True), out / 'run_complete.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'fixed', 'repeats'], required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    build_features()
    if (HERE / args.mode / 'run_complete.json').exists():
        raise FileExistsError('Already complete')
    if args.mode == 'smoke':
        jobs = [(p1.ANALYSES[0].name, 'fixed', 'smoke')]
    elif args.mode == 'fixed':
        jobs = [(a.name, 'fixed', 'fixed') for a in p1.ANALYSES]
    else:
        jobs = [(a.name, str(seed), 'repeats') for a in p1.ANALYSES for seed in range(42, 62)]
    if len(jobs) == 1 or all(((HERE / mode / 'checkpoints' / f'{a}__{s}' / 'complete.json').exists() for a, s, mode in jobs)):
        paths = [job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=multiprocessing.get_context('spawn')) as pool:
            paths = list(pool.map(job, jobs))
    summarize(paths, args.mode)
if __name__ == '__main__':
    main()
