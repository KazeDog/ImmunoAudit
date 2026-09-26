from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from evaluate_task2_cancer_context import BASELINE_MODEL, MODEL_DESCRIPTIONS, _within_score_from_wide, aggregate_concordance, bootstrap_oof_deltas, cancer_metrics, concordance_with_pairs, crossfitted_concordance, validate_oof_predictions

class ConcordanceTests(unittest.TestCase):

    def test_risk_direction_and_public_lifelines_equivalence(self):
        duration = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 1])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        result = concordance_with_pairs(duration, event, risk)
        self.assertEqual(result.c_index, 1.0)
        self.assertEqual(result.permissible_pairs, 6)
        self.assertEqual(result.c_index, concordance_index(duration, -risk, event))

    def test_tied_event_censor_pair_matches_lifelines(self):
        duration = np.array([1.0, 1.0, 2.0, 3.0])
        event = np.array([1, 0, 1, 0])
        risk = np.array([3.0, 2.0, 1.5, 1.0])
        result = concordance_with_pairs(duration, event, risk)
        self.assertEqual(result.c_index, concordance_index(duration, -risk, event))
        self.assertEqual(result.permissible_pairs, 4)

    def test_constant_prediction_is_half(self):
        result = concordance_with_pairs(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 0]), np.zeros(3))
        self.assertEqual(result.c_index, 0.5)
        self.assertEqual(result.tied_prediction_pairs, result.permissible_pairs)

    def test_no_pairs_is_nan(self):
        result = concordance_with_pairs(np.array([1.0, 2.0]), np.array([0, 0]), np.array([1.0, 0.0]))
        self.assertTrue(math.isnan(result.c_index))
        self.assertEqual(result.permissible_pairs, 0)

    def test_pair_pooled_is_not_unweighted_fold_mean(self):
        first = concordance_with_pairs(np.array([1.0, 2.0]), np.array([1, 1]), np.array([2.0, 1.0]))
        second = concordance_with_pairs(np.arange(1.0, 5.0), np.ones(4), np.arange(1.0, 5.0))
        aggregated = aggregate_concordance([first, second])
        self.assertAlmostEqual(aggregated.c_index, 1.0 / 7.0)
        self.assertNotAlmostEqual(aggregated.c_index, (first.c_index + second.c_index) / 2)

    def test_crossfitted_cancer_only_sentinel(self):
        rows = []
        for fold in range(1, 6):
            for cancer, shift in [('A', 0.0), ('B', 10.0)]:
                for offset in range(10):
                    rows.append({'fold': fold, 'CANCER_TYPE': cancer, 'duration': float(offset + 1), 'event': 1, 'risk_score': shift + fold})
        frame = pd.DataFrame(rows)
        metrics, summary = cancer_metrics(frame, min_n=40, min_events=20, min_pairs=100, min_folds_with_pairs=4)
        self.assertTrue(np.allclose(metrics.loc[metrics.eligible, 'c_index'], 0.5))
        self.assertEqual(summary['sample_weighted_c_index'], 0.5)

class PairedBootstrapTests(unittest.TestCase):

    @staticmethod
    def make_identical_oof() -> pd.DataFrame:
        rows = []
        for model in [BASELINE_MODEL, 'cancer_type_only']:
            for index in range(100):
                fold = index % 5 + 1
                cancer = 'A' if index < 50 else 'B'
                duration = float(index % 50 + 1)
                event = 1 if index % 3 != 0 else 0
                risk = -duration
                rows.append({'PATIENT_ID': f'P{index:03d}', 'fold': fold, 'model': model, 'risk_score': risk, 'duration': duration, 'event': event, 'CANCER_TYPE': cancer, 'CANCER_TYPE_MODEL': cancer})
        return pd.DataFrame(rows)

    def test_identical_models_have_zero_bootstrap_delta(self):
        oof = self.make_identical_oof()
        first = bootstrap_oof_deltas(oof, n_bootstrap=25, seed=42)
        second = bootstrap_oof_deltas(oof, n_bootstrap=25, seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(np.allclose(first['delta_vs_mutation_only'], 0.0))
        self.assertTrue(np.allclose(first['bootstrap_ci95_low'], 0.0))
        self.assertTrue(np.allclose(first['bootstrap_ci95_high'], 0.0))
        within = first[first['comparison_scope'] == 'within_cancer_sample_weighted']
        self.assertTrue((within['within_cancer_eligibility_mode'] == 'frozen_from_original_oof').all())

    def test_fixed_cancer_set_does_not_reapply_reporting_thresholds(self):
        rows = []
        for cancer in ['A', 'B']:
            for offset in range(4):
                duration = float(offset + 1)
                rows.append({'PATIENT_ID': f'{cancer}{offset}', 'fold': 1, 'duration': duration, 'event': 1, 'CANCER_TYPE': cancer, 'CANCER_TYPE_MODEL': cancer, BASELINE_MODEL: -duration if cancer == 'A' else duration})
        wide = pd.DataFrame(rows)
        self.assertTrue(math.isnan(_within_score_from_wide(wide, BASELINE_MODEL)))
        self.assertEqual(_within_score_from_wide(wide, BASELINE_MODEL, eligible_cancers=('A', 'B'), cancer_weights={'A': 0.75, 'B': 0.25}), 0.75)
        no_pairs = wide.copy()
        no_pairs.loc[no_pairs['CANCER_TYPE'] == 'B', 'event'] = 0
        self.assertTrue(math.isnan(_within_score_from_wide(no_pairs, BASELINE_MODEL, eligible_cancers=('A', 'B'), cancer_weights={'A': 0.75, 'B': 0.25})))

    def test_all_within_comparisons_share_bootstrap_validity(self):
        template = self.make_identical_oof()
        baseline = template[template.model == BASELINE_MODEL].copy()
        frames = []
        for model in MODEL_DESCRIPTIONS:
            model_frame = baseline.copy()
            model_frame['model'] = model
            frames.append(model_frame)
        results = bootstrap_oof_deltas(pd.concat(frames, ignore_index=True), n_bootstrap=10, seed=42)
        within = results[results['comparison_scope'] == 'within_cancer_sample_weighted']
        self.assertEqual(within['bootstrap_successful_reps'].nunique(), 1)
        self.assertEqual(within['bootstrap_invalid_frozen_cancer_score_reps'].nunique(), 1)

    def test_duplicate_patient_prediction_is_rejected(self):
        oof = self.make_identical_oof()
        duplicate = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'exactly one OOF'):
            validate_oof_predictions(duplicate)

    def test_cross_model_metadata_mismatch_is_rejected(self):
        template = self.make_identical_oof()
        baseline = template[template.model == BASELINE_MODEL].copy()
        frames = []
        for model in MODEL_DESCRIPTIONS:
            model_frame = baseline.copy()
            model_frame['model'] = model
            frames.append(model_frame)
        oof = pd.concat(frames, ignore_index=True)
        mask = (oof['PATIENT_ID'] == 'P001') & (oof['model'] == 'clinical_tmb')
        oof.loc[mask, 'duration'] = 999.0
        with self.assertRaisesRegex(ValueError, 'metadata differ'):
            validate_oof_predictions(oof)
if __name__ == '__main__':
    unittest.main()
