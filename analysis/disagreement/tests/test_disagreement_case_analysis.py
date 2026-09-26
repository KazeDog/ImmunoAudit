from __future__ import annotations
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_disagreement_case_analysis.py'
SPEC = importlib.util.spec_from_file_location('disagreement', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

class DisagreementAnalysisTests(unittest.TestCase):

    def test_midrank_percentile_with_ties(self):
        result = MODULE.midrank_percentile([1.0, 2.0, 2.0, 4.0])
        expected = np.array([0.125, 0.5, 0.5, 0.875])
        np.testing.assert_allclose(result, expected)

    def test_cluster_bootstrap_is_deterministic(self):
        values = np.array([0.0, 1.0, 2.0, 3.0])
        groups = np.array(['a', 'a', 'b', 'c'])
        first = MODULE.cluster_bootstrap((values,), groups, np.mean, 40, 19)
        second = MODULE.cluster_bootstrap((values,), groups, np.mean, 40, 19)
        np.testing.assert_array_equal(first, second)

    def test_generated_calls_use_majority_not_continuous_score(self):
        self.assertEqual(MODULE.majority_generated_call([0, 0, 1]), 0)
        self.assertEqual(MODULE.majority_generated_call([0, 1]), 1)

    def test_task3_case_rule_is_outcome_balanced_and_maximal(self):
        frame = pd.DataFrame({'task': ['task3'] * 6, 'dataset': ['A'] * 6, 'dataset_label': ['A'] * 6, 'endpoint_type': ['classification'] * 6, 'unit_id': ['p3', 'p1', 'p2', 'p6', 'p4', 'p5'], 'cluster_id': ['p3', 'p1', 'p2', 'p6', 'p4', 'p5'], 'outcome': [0, 0, 0, 1, 1, 1], 'rank_dispersion': [0.2, 0.8, 0.8, 0.1, 0.7, 0.4]})
        _, selected = MODULE.select_task3_cases(frame)
        self.assertEqual(set(selected['outcome']), {0, 1})
        self.assertEqual(selected[selected['outcome'] == 0]['unit_id'].iloc[0], 'p1')
        self.assertEqual(selected[selected['outcome'] == 1]['unit_id'].iloc[0], 'p4')

    def test_task4_maximum_and_lower_median_rule(self):
        frame = pd.DataFrame({'patient_id': ['p1', 'p2', 'p3', 'p4'], 'score_range': [0.1, 0.3, 0.2, 0.9]})
        chosen = dict(MODULE.choose_max_and_lower_median(frame))
        self.assertEqual(chosen['maximum range']['patient_id'], 'p4')
        self.assertEqual(chosen['median range']['patient_id'], 'p3')

    def test_timepoint_conversion(self):
        self.assertEqual(MODULE.timepoint_day('BL'), 0.0)
        self.assertEqual(MODULE.timepoint_day('6W'), 42.0)
        self.assertEqual(MODULE.timepoint_day('6-8W'), 49.0)
        self.assertAlmostEqual(MODULE.timepoint_day('3M'), 91.3125)
        self.assertEqual(MODULE.timepoint_day('TOX', 37.0), 37.0)
if __name__ == '__main__':
    unittest.main()
