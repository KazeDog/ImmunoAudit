import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT.parents[1] / 'analysis/grouped_tasks/scripts'))
from audit_core import request_payload, numeric_summary, metric_value, bootstrap, summary_oof, patient_stratified_folds
from task135_core import fold_restricted_cindex, harrell_cindex

def test_payload_suffix():
    assert request_payload({'user_prompt': 'Instruction\n{"a": 1}'}) == {'a': 1}
    with pytest.raises(ValueError):
        request_payload({'user_prompt': 'no payload'})

def test_identifier_exclusion():
    p = {'sample_id': 'secret', 'dataset_name': 'study', 'event': 1, 'evidence_balance': {'net': 0.4, 'n': 2}}
    f = numeric_summary(p)
    assert set(f) == {'evidence_balance.net', 'evidence_balance.net.__present', 'evidence_balance.n', 'evidence_balance.n.__present'}

def test_nested_leakage_rejected():
    with pytest.raises(ValueError):
        numeric_summary({'ecology_snapshot': {'event': 1}})

def test_program_identity_not_position():
    a = {'dominant_programs': [{'program': 'A', 'relative_activity_score': 0.5}, {'program': 'B', 'relative_activity_score': 0.2}]}
    b = {'dominant_programs': list(reversed(a['dominant_programs']))}
    assert numeric_summary(a) == numeric_summary(b)

@pytest.mark.parametrize('seed', range(8))
def test_cindex_matches_reference(seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({'duration': rng.integers(1, 9, 25), 'event': rng.integers(0, 2, 25), 'risk': rng.integers(0, 9, 25) / 10, 'fold': rng.integers(1, 6, 25)})
    assert metric_value(d, 'task5') == pytest.approx(fold_restricted_cindex(d))
    assert metric_value(d, 'task5', 'CINDEX_POOLED') == pytest.approx(harrell_cindex(d.duration, d.risk, d.event))

def test_paired_bootstrap_identical_zero():
    f = pd.DataFrame({'outcome': [0, 1] * 10, 'score': np.arange(20) / 20})
    lo, hi, n = bootstrap(f, 'task3', 'AUROC', 100, reference=f)
    assert lo == hi == 0 and n == 100

def test_train_only_vocabulary_and_scaler():
    features = [{'a': float(i)} for i in range(20)]
    labels = pd.DataFrame({'outcome': [0, 1] * 10})
    meta = pd.DataFrame({'sample_id': [str(i) for i in range(20)], 'group_id': [str(i) for i in range(20)]})
    folds = np.array([1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4)
    base = summary_oof(features, labels, meta, 'task3', folds, [1])
    for i in range(4):
        features[i]['test_only_feature'] = 1000000000.0
    mutated = summary_oof(features, labels, meta, 'task3', folds, [1])
    np.testing.assert_array_equal(base.score, mutated.score)

def test_group_overlap_rejected():
    meta = pd.DataFrame({'sample_id': ['a', 'b', 'c', 'd'], 'group_id': ['x', 'y', 'x', 'z']})
    with pytest.raises(ValueError, match='Patient leakage'):
        summary_oof([{'a': i} for i in range(4)], pd.DataFrame({'y': [0, 1, 0, 1]}), meta, 'task3', np.array([1, 1, 2, 2]))

def test_patient_stratification_has_both_classes_and_no_leakage():
    groups = np.repeat([str(i) for i in range(25)], 2)
    labels = np.repeat([0] * 20 + [1] * 5, 2)
    folds = patient_stratified_folds(labels, groups)
    for fold in range(1, 6):
        assert set(labels[folds == fold]) == {0, 1}
        assert not set(groups[folds == fold]) & set(groups[folds != fold])

def test_source_only_scores_are_tied_within_fold():
    source = np.array([0, 1] * 20)
    features = [{'Liu_source': float(s)} for s in source]
    meta = pd.DataFrame({'sample_id': [str(i) for i in range(40)], 'group_id': [str(i) for i in range(40)]})
    y = pd.DataFrame({'duration': np.arange(1, 41) * 10, 'event': [1, 1, 0, 1] * 10})
    folds = np.tile(np.arange(1, 6), 8)
    pred = summary_oof(features, y, meta, 'task5', folds)
    pred['risk'] = pred.risk.round(12)
    pred['source'] = [source[int(s)] for s in pred.sample_id]
    assert pred.groupby(['fold', 'source']).risk.nunique().max() == 1
