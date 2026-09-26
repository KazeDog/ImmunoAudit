from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts/task2_strategy2_oof.py'
SPEC = importlib.util.spec_from_file_location('task2_strategy2_oof', SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

class Task2Strategy2OOFTests(unittest.TestCase):

    def test_frozen_fold_loading_aligns_by_patient_and_checks_row_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x_path = root / 'task2_X_final.csv'
            y_path = root / 'task2_y_final.csv'
            fold_path = root / 'task2_fold_assignments.csv'
            pd.DataFrame({'PATIENT_ID': ['P3', 'P1', 'P2', 'P4'], 'f1': [3.0, 1.0, 2.0, 4.0], 'f2': [0.0, 1.0, 0.0, 1.0]}).to_csv(x_path, index=False)
            pd.DataFrame({'PATIENT_ID': ['P2', 'P4', 'P3', 'P1'], 'duration': [2.0, 4.0, 3.0, 1.0], 'event': [1, 0, 1, 1]}).to_csv(y_path, index=False)
            pd.DataFrame({'PATIENT_ID': ['P1', 'P2', 'P3', 'P4'], 'fold': [2, 1, 1, 2]}).to_csv(fold_path, index=False)
            loaded = MODULE.load_task2_inputs(x_path, y_path, fold_path, expected_n=4, expected_folds=(1, 2))
            self.assertEqual(loaded.features.index.tolist(), ['P3', 'P1', 'P2', 'P4'])
            self.assertEqual(loaded.labels['duration'].tolist(), [3.0, 1.0, 2.0, 4.0])
            self.assertEqual(loaded.folds.tolist(), [1, 2, 1, 2])

    def test_variance_selection_uses_training_fold_only(self) -> None:
        x_train = np.asarray([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float32)
        x_test = np.asarray([[5.0, -1000.0, 1.0], [6.0, 1000.0, 1.0]], dtype=np.float32)
        train_selected, test_selected, selected, variances = MODULE.select_features_by_training_variance(x_train, x_test, max_features=1)
        self.assertEqual(selected.tolist(), [0])
        self.assertEqual(train_selected.shape, (3, 1))
        self.assertEqual(test_selected[:, 0].tolist(), [5.0, 6.0])
        self.assertGreater(float(variances[0]), 0.0)

    def test_corrected_negative_duration_risk_reverses_legacy_orientation(self) -> None:
        durations = np.asarray([1.0, 2.0, 3.0])
        events = np.asarray([1, 1, 1])
        perfect_predicted_duration = np.asarray([1.0, 2.0, 3.0])
        metrics = MODULE.compute_duration_orientation_metrics(durations, perfect_predicted_duration, events)
        self.assertEqual(metrics['legacy_c_index_predicted_duration_as_higher_risk'], 0.0)
        self.assertEqual(metrics['corrected_c_index_negative_predicted_duration_as_higher_risk'], 1.0)

    def test_no_overwrite_is_enforced_for_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = MODULE.RunOutputPaths.under(Path(tmp), 'fixture')
            predictions = pd.DataFrame({'PATIENT_ID': ['P1'], 'fold': [1]})
            fold_metrics = pd.DataFrame({'fold': [1], 'legacy': [0.5]})
            selected = pd.DataFrame({'fold': [1], 'feature_index_0based': [0], 'feature_name': ['f1']})
            summary = {'ok': True}
            MODULE.write_outputs_no_overwrite(paths, predictions, fold_metrics, summary, selected)
            before = paths.predictions.read_text(encoding='utf-8')
            with self.assertRaises(FileExistsError):
                MODULE.write_outputs_no_overwrite(paths, predictions, fold_metrics, summary, selected)
            self.assertEqual(paths.predictions.read_text(encoding='utf-8'), before)
            self.assertEqual(json.loads(paths.summary.read_text(encoding='utf-8')), summary)
if __name__ == '__main__':
    unittest.main()
