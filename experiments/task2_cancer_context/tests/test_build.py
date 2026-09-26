from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from build_task2_cancer_context import build_context_dataset
from task2_context_common import MODEL_CANCER_COLUMN, RARE_OR_UNKNOWN_LEVEL, write_csv_new

class BuildDatasetTests(unittest.TestCase):

    def make_inputs(self):
        ids = ['P3', 'P1', 'P2', 'P4']
        x = pd.DataFrame({'G1': [1.0, 0.0, 1.0, 0.0], 'G2': [0.0, 1.0, 0.0, 1.0]}, index=pd.Index(ids, name='PATIENT_ID'))
        y = pd.DataFrame({'duration': [3.0, 1.0, 5.0, 2.0], 'event': [1, 1, 0, 1]}, index=pd.Index(ids, name='PATIENT_ID'))
        patient = pd.DataFrame({'PATIENT_ID': ids, 'SEX': ['F', 'M', 'F', 'M'], 'AGE_GROUP': ['31-50', '51-60', '61-70', '>71'], 'DRUG_TYPE': ['PD-1', 'Combo', 'PD-1', 'CTLA4'], 'OS_MONTHS': ['3', '1', '5', '2'], 'OS_STATUS': ['1:DECEASED', '1:DECEASED', '0:LIVING', '1:DECEASED']})
        sample = pd.DataFrame({'PATIENT_ID': ids, 'SAMPLE_ID': [f'{patient_id}-T' for patient_id in ids], 'CANCER_TYPE': ['Melanoma', 'Skin Cancer, Non-Melanoma', 'Bladder Cancer', 'Melanoma'], 'CANCER_TYPE_DETAILED': ['MEL', 'CSCC', 'BLCA', 'MEL'], 'ONCOTREE_CODE': ['MEL', 'CSCC', 'BLCA', 'MEL'], 'PRIMARY_SITE': ['Skin', 'Skin', 'Bladder', 'Skin'], 'TMB_NONSYNONYMOUS': ['5.0', '2.0', '3.0', '7.0']})
        return (x, y, patient, sample)

    def test_join_preserves_canonical_order_and_raw_labels(self):
        combined, audit = build_context_dataset(*self.make_inputs())
        self.assertEqual(combined['PATIENT_ID'].tolist(), ['P3', 'P1', 'P2', 'P4'])
        self.assertEqual(audit['n_analyzed'], 4)
        self.assertEqual(audit['events'], 3)
        skin = combined[combined['CANCER_TYPE'].isin(['Melanoma', 'Skin Cancer, Non-Melanoma'])]
        self.assertTrue((skin[MODEL_CANCER_COLUMN] == RARE_OR_UNKNOWN_LEVEL).all())
        self.assertEqual(combined.loc[1, 'CANCER_TYPE'], 'Skin Cancer, Non-Melanoma')
        np.testing.assert_array_equal(combined[['G1', 'G2']].to_numpy(), self.make_inputs()[0].to_numpy())

    def test_duplicate_sample_is_rejected(self):
        x, y, patient, sample = self.make_inputs()
        sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Multiple sample rows'):
            build_context_dataset(x, y, patient, sample)

    def test_x_y_reordering_is_rejected(self):
        x, y, patient, sample = self.make_inputs()
        with self.assertRaisesRegex(ValueError, 'row order differ'):
            build_context_dataset(x, y.iloc[::-1], patient, sample)

    def test_binary_mutation_contract(self):
        x, y, patient, sample = self.make_inputs()
        x.iloc[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, 'must be binary'):
            build_context_dataset(x, y, patient, sample)

    def test_safe_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'output.csv'
            write_csv_new(pd.DataFrame({'a': [1]}), path)
            with self.assertRaises(FileExistsError):
                write_csv_new(pd.DataFrame({'a': [2]}), path)
            self.assertEqual(pd.read_csv(path).loc[0, 'a'], 1)
if __name__ == '__main__':
    unittest.main()
