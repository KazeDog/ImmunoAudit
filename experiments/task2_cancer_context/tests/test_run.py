from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from run_task2_cancer_context_cox import build_fold_assignments, fit_mutation_transformer, fit_tmb_block
from task2_context_common import CANONICAL_X_PATH, CANONICAL_Y_PATH, sha256_lines

class FoldAssignmentTests(unittest.TestCase):

    def test_exact_legacy_fold_membership_and_events(self):
        x_ids = pd.read_csv(CANONICAL_X_PATH, usecols=['PATIENT_ID'])['PATIENT_ID']
        y = pd.read_csv(CANONICAL_Y_PATH)
        assignments = build_fold_assignments(x_ids)
        self.assertEqual(assignments.groupby('fold').size().tolist(), [322] * 5)
        joined = assignments[['PATIENT_ID', 'fold']].merge(y, on='PATIENT_ID', validate='one_to_one')
        self.assertEqual(joined.groupby('fold')['event'].sum().astype(int).tolist(), [165, 155, 158, 170, 154])
        expected_hashes = ['c7d858c003f79456db9a82440a1a82ac9868c365fa049a43cec54ee2ef830d96', '4afc1d9943d92f6081b60351385b68f9a429bdd9417a41684d47a8a511a23428', '6bdfb7560de59405dfc9f069d4e8e65aa668d5afc51e24d4ecf34f343949c2de', '892ade3f578dc66c8004d05f7cb90ad8e7a6d4d0cd436b0312e4b49ae6835346', '10e60e45a0abbf1043675fe488b9da67fecc900055cee9e70331df18d3ae22d0']
        observed = [sha256_lines(assignments.loc[assignments.fold == fold, 'PATIENT_ID']) for fold in range(1, 6)]
        self.assertEqual(observed, expected_hashes)

class TrainOnlyPreprocessingTests(unittest.TestCase):

    def test_test_values_do_not_change_fitted_mutation_transform(self):
        rng = np.random.default_rng(42)
        columns = [f'G{i}' for i in range(12)]
        train = pd.DataFrame(rng.integers(0, 2, size=(30, 12)), columns=columns)
        test_a = pd.DataFrame(np.zeros((5, 12)), columns=columns)
        test_b = pd.DataFrame(np.ones((5, 12)), columns=columns)
        train_a, _, audit_a = fit_mutation_transformer(train, test_a, columns)
        train_b, _, audit_b = fit_mutation_transformer(train, test_b, columns)
        np.testing.assert_allclose(train_a, train_b)
        self.assertEqual(audit_a['mutation_pca_components_sha256'], audit_b['mutation_pca_components_sha256'])

    def test_tmb_imputation_and_scaling_are_train_only(self):
        train = pd.DataFrame({'TMB_NONSYNONYMOUS': [1.0, 3.0, np.nan]})
        test = pd.DataFrame({'TMB_NONSYNONYMOUS': [1000.0]})
        train_block, _, audit = fit_tmb_block(train, test)
        self.assertEqual(audit['tmb_train_median'], 2.0)
        self.assertAlmostEqual(float(train_block.mean().iloc[0]), 0.0, places=12)
if __name__ == '__main__':
    unittest.main()
