
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace
import numpy as np
from experiments.common.recovery_evaluation_split import validation_sets
from experiments.studies.agnews_structure_comparison import IndependentPreparedDataset


class ValidationSplitTests(unittest.TestCase):
    def dataset(self, per_class=12):
        domains=np.tile(np.arange(4),per_class)
        rows=np.arange(len(domains))
        return SimpleNamespace(domain_names=['a','b','c','d'],arrays={
            'validation_domains':domains,'validation_tokens':rows[:,None],
            'validation_lengths':np.ones(len(rows),dtype=int)})

    def test_original_selection_and_disjoint_evaluation(self):
        dataset=self.dataset()
        for count in (8,9,16):
            a,b,meta=validation_sets(dataset,count)
            original=IndependentPreparedDataset.split(dataset,'validation',limit=count)
            for got,expected in zip(a,original):
                self.assertTrue(got.equal(expected))
            self.assertEqual(len(b[0]),count)
            self.assertFalse(set(meta['selection_row_ids']) & set(meta['recovery_row_ids']))
            self.assertEqual(np.bincount(a[2]).tolist(),np.bincount(b[2]).tolist())
            self.assertTrue(set(meta['selection_row_ids']).isdisjoint(meta['recovery_row_ids']))
            self.assertEqual(meta,validation_sets(dataset,count)[2])

    def test_no_silent_sample_reuse(self):
        with self.assertRaises(ValueError):validation_sets(self.dataset(3),8)
        with self.assertRaises(ValueError):validation_sets(self.dataset(),3)


if __name__=='__main__':unittest.main()
