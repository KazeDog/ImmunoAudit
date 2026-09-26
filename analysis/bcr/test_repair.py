import unittest
import numpy as np
from repair_core import repertoire_frequencies, VOCAB, aucs, bootstrap_counts, bootstrap_auc

class RepairTests(unittest.TestCase):

    def test_boundaries(self):
        a, q = repertoire_frequencies(['ACDE', 'FGHI', 'AC*DE'])
        self.assertEqual(q['valid_windows'], 4)
        self.assertEqual(q['invalid_windows'], 3)
        self.assertEqual(np.count_nonzero(a), 4)
        for t in ['ACD', 'CDE', 'FGH', 'GHI']:
            self.assertEqual(a[VOCAB[t]], 0.25)
        self.assertEqual(a[VOCAB['EFG']], 0)

    def test_normalized(self):
        a, _ = repertoire_frequencies(['ACDEF'])
        b, _ = repertoire_frequencies(['ACDEF'] * 20)
        np.testing.assert_array_equal(a, b)

    def test_no_zero_imputation(self):
        with self.assertRaises(ValueError):
            repertoire_frequencies(['***', 'A*CD'])

    def test_fold_constants(self):
        y = [0, 1, 0, 1]
        f = [1, 1, 2, 2]
        s = [0.9, 0.9, 0.1, 0.1]
        self.assertEqual(aucs(y, s, f)['auroc'], 0.5)
        np.testing.assert_array_equal(bootstrap_auc(y, s, f, bootstrap_counts(y, f, 30)), np.full(30, 0.5))

    def test_monotone_by_fold(self):
        y = [0, 1, 0, 1]
        f = [1, 1, 2, 2]
        self.assertEqual(aucs(y, [0, 1, 0, 1], f)['auroc'], aucs(y, [100, 101, -1, 0], f)['auroc'])

    def test_single_class_fold(self):
        self.assertEqual(aucs([0, 1, 1], [0, 1, 2], [1, 1, 2])['comparable_pairs'], 1)
if __name__ == '__main__':
    unittest.main()
