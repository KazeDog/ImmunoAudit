"""Corrected deterministic feature and discrimination primitives."""
import itertools
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
TRIPLETS = tuple((''.join(t) for t in itertools.product(AMINO_ACIDS, repeat=3)))
VOCAB = {t: i for i, t in enumerate(TRIPLETS)}

def repertoire_frequencies(sequences):
    counts = np.zeros(len(TRIPLETS), dtype=np.int64)
    eligible = invalid = 0
    for seq in sequences:
        seq = str(seq).strip().upper()
        if len(seq) <= 3:
            continue
        eligible += 1
        for i in range(len(seq) - 2):
            token = seq[i:i + 3]
            if token in VOCAB:
                counts[VOCAB[token]] += 1
            else:
                invalid += 1
    total = int(counts.sum())
    if total == 0:
        raise ValueError('No valid within-clone triplets')
    return (counts / total, dict(n_eligible_clones=eligible, valid_windows=total, invalid_windows=invalid))

def pair_matrix(y, score, fold):
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    f = np.asarray(fold)
    if not np.isfinite(s).all():
        raise ValueError('Non-finite scores')
    valid = (y[:, None] == 1) & (y[None, :] == 0) & (f[:, None] == f[None, :])
    credit = (s[:, None] > s[None, :]).astype(float) + 0.5 * (s[:, None] == s[None, :])
    return (credit * valid, valid.astype(float))

def aucs(y, score, fold):
    y = np.asarray(y)
    s = np.asarray(score)
    f = np.asarray(fold)
    c, v = pair_matrix(y, s, f)
    per = [roc_auc_score(y[f == k], s[f == k]) for k in np.unique(f) if len(np.unique(y[f == k])) == 2]
    return dict(auroc=float(c.sum() / v.sum()) if v.sum() else np.nan, mean_fold_auroc=float(np.mean(per)) if per else np.nan, pooled_auroc=float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan, comparable_pairs=int(v.sum()))

def bootstrap_counts(y, fold, reps=2000, seed=42):
    y = np.asarray(y)
    fold = np.asarray(fold)
    rng = np.random.default_rng(seed)
    w = np.zeros((reps, len(y)), float)
    for f in np.unique(fold):
        for label in (0, 1):
            ix = np.flatnonzero((fold == f) & (y == label))
            if len(ix):
                w[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=reps)
    return w

def bootstrap_auc(y, score, fold, weights):
    credit, valid = pair_matrix(y, score, fold)
    numerator = np.sum(weights @ credit * weights, axis=1)
    denominator = np.sum(weights @ valid * weights, axis=1)
    if (denominator == 0).any():
        raise ValueError('No comparable pairs in bootstrap')
    return numerator / denominator
