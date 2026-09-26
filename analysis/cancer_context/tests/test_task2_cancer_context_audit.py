from __future__ import annotations
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from task2_cancer_context_audit import BASELINE_MODEL, DiscoveredInputs, _bh_adjust, _estimand_score, _model_scope, assert_output_targets_are_new, build_per_cancer_metrics, build_risk_distribution_metrics, concordance_with_pairs, discover_inputs, infer_fixed_prediction_identity, legacy_fixed_llm_c_index, load_and_validate_metadata, patient_bootstrap, write_text_new

class ConcordanceAndEstimandTests(unittest.TestCase):

    def test_higher_risk_means_earlier_event(self) -> None:
        result = concordance_with_pairs([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(result.c_index, 1.0)
        self.assertEqual(result.permissible_pairs, 6)

    def test_legacy_llm_reproduction_uses_historical_equal_time_rule(self) -> None:
        score = legacy_fixed_llm_c_index([1.0, 1.0, 2.0], [1, 0, 1], [3.0, 1.0, 2.0])
        self.assertEqual(score, 1.0)

    def test_cox_uses_same_fold_pairs_but_fixed_llm_uses_full_scale(self) -> None:
        frame = pd.DataFrame({'duration': [1.0, 2.0, 3.0, 4.0], 'event': [1, 1, 1, 1], 'fold': [1, 1, 2, 2], 'CANCER_TYPE': ['A'] * 4, 'risk_score': [2.0, 1.0, 12.0, 11.0]})
        cox_score = _estimand_score(frame, BASELINE_MODEL, 'global')
        fixed_score = _estimand_score(frame, 'fixed_llm__qwen', 'global')
        self.assertEqual(cox_score, 1.0)
        self.assertEqual(fixed_score, concordance_with_pairs(frame.duration, frame.event, frame.risk_score).c_index)
        self.assertNotEqual(cox_score, fixed_score)
        self.assertEqual(_model_scope(BASELINE_MODEL, 'global')[1], 'same_test_fold_pairs_only')
        self.assertIn('single_fixed', _model_scope('fixed_llm__qwen', 'global')[1])

    def test_within_cancer_excludes_between_cancer_signal(self) -> None:
        frame = pd.DataFrame({'duration': [1, 2, 3, 4, 10, 11, 12, 13], 'event': [1] * 8, 'fold': [1] * 8, 'CANCER_TYPE': ['A'] * 4 + ['B'] * 4, 'risk_score': [1.0] * 4 + [0.0] * 4})
        global_score = _estimand_score(frame, 'fixed_llm__qwen', 'global')
        within_score = _estimand_score(frame, 'fixed_llm__qwen', 'within_cancer')
        self.assertGreater(global_score, 0.5)
        self.assertEqual(within_score, 0.5)

    def test_stratified_cox_global_estimand_is_not_reported(self) -> None:
        _, scope, appropriate = _model_scope('mutation_pcs_clinical_tmb_stratified_cancer', 'global')
        self.assertFalse(appropriate)
        self.assertIn('not_applicable', scope)

class BootstrapTests(unittest.TestCase):

    @staticmethod
    def predictions() -> pd.DataFrame:
        rows = []
        for model in [BASELINE_MODEL, 'clinical_tmb']:
            for index in range(30):
                duration = float(index % 15 + 1)
                rows.append({'PATIENT_ID': f'P{index:03d}', 'duration': duration, 'event': 1 if index % 4 else 0, 'fold': index % 5 + 1, 'CANCER_TYPE': 'A' if index < 15 else 'B', 'model': model, 'risk_score': -duration})
        return pd.DataFrame(rows)

    def test_identical_cox_scores_have_zero_paired_delta(self) -> None:
        first_abs, first_pair = patient_bootstrap(self.predictions(), n_bootstrap=20, random_state=42)
        second_abs, second_pair = patient_bootstrap(self.predictions(), n_bootstrap=20, random_state=42)
        pd.testing.assert_frame_equal(first_abs, second_abs)
        pd.testing.assert_frame_equal(first_pair, second_pair)
        self.assertTrue(np.allclose(first_pair.point_delta, 0.0))
        self.assertTrue(np.allclose(first_pair.bootstrap_ci95_low, 0.0))
        self.assertTrue((first_pair.bootstrap_unit == 'patient').all())
        self.assertTrue((first_pair.model_score_scope == first_pair.baseline_score_scope).all())

    def test_fixed_llm_absolute_is_full_scale_but_paired_delta_uses_common_pairs(self) -> None:
        baseline = self.predictions()
        baseline = baseline[baseline.model == BASELINE_MODEL].copy()
        fixed = baseline.copy()
        fixed['model'] = 'fixed_llm__qwen__v2_1'
        absolute, paired = patient_bootstrap(pd.concat([baseline, fixed], ignore_index=True), n_bootstrap=5)
        fixed_global = absolute[(absolute.model == 'fixed_llm__qwen__v2_1') & (absolute.estimand == 'global')].iloc[0]
        self.assertIn('full_cohort', fixed_global.score_scope)
        comparison = paired[(paired.model == 'fixed_llm__qwen__v2_1') & (paired.estimand == 'global')].iloc[0]
        self.assertEqual(comparison.model_score_scope, 'common_same_test_fold_pairs_only')
        self.assertEqual(comparison.model_score_scope, comparison.baseline_score_scope)
        self.assertTrue(comparison.pair_restriction_identical)
        self.assertEqual(comparison.point_delta, 0.0)

class DistributionAndSupportTests(unittest.TestCase):

    def test_risk_distribution_has_requested_statistics_and_bh_fdr(self) -> None:
        rows = []
        for model, offset in [('m1', 0.0), ('m2', 1.0)]:
            for cancer, shift in [('A', 0.0), ('B', 3.0)]:
                for index in range(5):
                    rows.append({'model': model, 'CANCER_TYPE': cancer, 'risk_score': float(index + shift + offset)})
        summaries, tests = build_risk_distribution_metrics(pd.DataFrame(rows))
        for column in ('risk_mean', 'risk_sd_ddof1', 'risk_median', 'risk_iqr'):
            self.assertIn(column, summaries.columns)
        self.assertIn('h_statistic', tests.columns)
        self.assertIn('epsilon_squared', tests.columns)
        self.assertIn('p_value_bh_fdr_across_models', tests.columns)
        self.assertTrue((tests.p_value_bh_fdr_across_models >= tests.p_value - 1e-15).all())

    def test_bh_adjust_known_values(self) -> None:
        adjusted = _bh_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

    def test_tiny_cancer_is_na_not_overinterpreted(self) -> None:
        frame = pd.DataFrame({'model': [BASELINE_MODEL] * 5, 'CANCER_TYPE': ['rare'] * 5, 'duration': [1, 2, 3, 4, 5], 'event': [1, 1, 1, 0, 0], 'fold': [1, 2, 3, 4, 5], 'risk_score': [5, 4, 3, 2, 1]})
        result = build_per_cancer_metrics(frame).iloc[0]
        self.assertFalse(result.c_index_reportable)
        self.assertTrue(math.isnan(result.c_index))
        self.assertTrue(np.isfinite(result.raw_concatenated_oof_c_index))

class DiscoveryAndValidationTests(unittest.TestCase):

    def test_separate_output_root_still_discovers_canonical_strategy2_oof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / 'project'
            separate_output = Path(temporary) / 'smoke_output'
            required = [project / 'processed_data_strategy1/task2_X_final.csv', project / 'processed_data_strategy1/task2_y_final.csv', project / 'data_clinical_sample.txt', project / 'results/task2_cancer_context/fold_assignments.csv', project / 'results/task2_cancer_context/full_v2_fixed_bootstrap/oof_predictions.csv']
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n', encoding='utf-8')
            canonical_predictions = project / 'analysis/cancer_context/predictions'
            canonical_predictions.mkdir(parents=True)
            for model in ('fttransformer', 'tabpfn'):
                (canonical_predictions / f'task2_strategy2_{model}_oof_predictions.csv').write_text('PATIENT_ID,risk_score\nP1,0.5\n', encoding='utf-8')
            discovered = discover_inputs(project, separate_output)
            self.assertEqual({path.name for path in discovered.traditional_prediction_paths}, {'task2_strategy2_fttransformer_oof_predictions.csv', 'task2_strategy2_tabpfn_oof_predictions.csv'})

    def test_versioned_copied_prediction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / 'task2_kimi_k26_v21_summary.json'
            summary.write_text(json.dumps({'prompt_version': 'v2.1'}), encoding='utf-8')
            path = root / 'task2_kimi_k26_v21_predictions.csv'
            path.write_text('sample_id,risk_score\nP1,0.5\n', encoding='utf-8')
            identity = infer_fixed_prediction_identity(path)
            self.assertEqual(identity['provider'], 'kimi')
            self.assertEqual(identity['llm_model'], 'Kimi-K2.6')
            self.assertEqual(identity['prompt_version'], 'v2.1')
            self.assertIn('kimi_k2_6', identity['model_id'])

    def test_metadata_validation_accepts_exact_fixture_and_rejects_count_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [f'P{i}' for i in range(10)]
            pd.DataFrame({'PATIENT_ID': ids, 'x': range(10)}).to_csv(root / 'x.csv', index=False)
            pd.DataFrame({'PATIENT_ID': ids, 'duration': range(1, 11), 'event': [0, 1] * 5}).to_csv(root / 'y.csv', index=False)
            pd.DataFrame({'PATIENT_ID': ids, 'fold': [1, 2, 3, 4, 5] * 2, 'original_row_position_0based': range(10), 'n_splits': [5] * 10, 'random_state': [42] * 10}).to_csv(root / 'fold.csv', index=False)
            pd.DataFrame({'PATIENT_ID': ids, 'SAMPLE_ID': [f'S{i}' for i in range(10)], 'CANCER_TYPE': ['A'] * 10}).to_csv(root / 'clinical.txt', sep='\t', index=False)
            pd.DataFrame().to_csv(root / 'cox.csv', index=False)
            inputs = DiscoveredInputs(x_path=root / 'x.csv', y_path=root / 'y.csv', clinical_sample_path=root / 'clinical.txt', fold_path=root / 'fold.csv', cox_oof_path=root / 'cox.csv', fixed_prediction_paths=(), traditional_summary_paths=())
            metadata, _, _ = load_and_validate_metadata(inputs, expected_n=10, expected_counts={'A': 10})
            self.assertEqual(len(metadata), 10)
            with self.assertRaisesRegex(AssertionError, 'expected 11'):
                load_and_validate_metadata(inputs, expected_n=11, expected_counts={'A': 11})

class SafeOutputTests(unittest.TestCase):

    def test_writer_and_preflight_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'x.txt'
            write_text_new('first\n', path)
            with self.assertRaises(FileExistsError):
                write_text_new('second\n', path)
            output = root / 'audit'
            existing = output / 'metrics/task2_global_summary.csv'
            existing.parent.mkdir(parents=True)
            existing.write_text('already here\n', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                assert_output_targets_are_new(output)
if __name__ == '__main__':
    unittest.main()
