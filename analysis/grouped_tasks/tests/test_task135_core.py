from pathlib import Path
import sys
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from task135_core import aggregate_classification_by_group, aggregate_survival_by_group, canonical_task1_group, classification_metrics, cluster_bootstrap_metric, fold_restricted_cindex, harrell_cindex, make_grouped_folds

def test_task1_group_normalisation():
    assert canonical_task1_group('Pre_P1') == 'P1'
    assert canonical_task1_group('Post_P1') == 'P1'
    assert canonical_task1_group('Post_P1_2') == 'P1'

def test_grouped_folds_have_no_group_overlap():
    groups = np.repeat([f'P{i}' for i in range(20)], 2)
    y = np.tile([0, 1], 20)
    folds = make_grouped_folds(y, groups, n_splits=5)
    assert set(folds) == {1, 2, 3, 4, 5}
    for fold in range(1, 6):
        assert not set(groups[folds == fold]).intersection(groups[folds != fold])

def test_classification_patient_aggregation():
    frame = pd.DataFrame({'group_id': ['A', 'A', 'B', 'B'], 'outcome': [0, 0, 1, 1], 'score': [0.1, 0.3, 0.7, 0.9], 'predicted': [0, 0, 1, 1], 'fold': [1, 1, 2, 2]})
    aggregated = aggregate_classification_by_group(frame)
    assert np.allclose(aggregated['score'], [0.2, 0.8])
    assert classification_metrics(aggregated['outcome'], aggregated['score']).auroc == 1.0

def test_survival_orientation_and_fold_restriction():
    assert harrell_cindex([1, 2, 3], [3, 2, 1], [1, 1, 1]) == 1.0
    frame = pd.DataFrame({'group_id': ['A', 'B', 'C', 'D'], 'duration': [1, 2, 1, 2], 'event': [1, 1, 1, 1], 'risk': [0.9, 0.1, 0.8, 0.2], 'fold': [1, 1, 2, 2]})
    assert fold_restricted_cindex(frame) == 1.0
    assert len(aggregate_survival_by_group(frame)) == 4

def test_cluster_bootstrap_is_deterministic():
    frame = pd.DataFrame({'group_id': [f'P{i}' for i in range(20)], 'outcome': [0] * 10 + [1] * 10, 'score': np.linspace(0.0, 1.0, 20)})

    def metric(block):
        return classification_metrics(block['outcome'], block['score']).auroc
    first = cluster_bootstrap_metric(frame, metric, n_bootstrap=120, random_state=42)
    second = cluster_bootstrap_metric(frame, metric, n_bootstrap=120, random_state=42)
    assert np.array_equal(first, second)
