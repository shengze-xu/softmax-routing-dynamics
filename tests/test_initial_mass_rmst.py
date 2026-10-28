"""Check the statistical unit of initial-probability recovery summaries."""
import unittest

import pandas as pd

from experiments.analysis.analyze_recovery_additions import BRANCHES, summarize_initial_masses


class InitialMassRMSTTests(unittest.TestCase):
    def test_seed_weighting_and_administrative_truncation(self):
        records = []
        # Different cell counts make an accidental pooled-cell mean detectable.
        for seed, control, projected in (
            (1, [128], [256]),
            (2, [256, 256, 1024], [512, 384, 1024]),
        ):
            for branch, times in zip(BRANCHES, (control, projected)):
                for time in times:
                    records.append(dict(cohort='no_drop', seed=seed, initial_mass=.01,
                                        branch=branch, threshold=.05,
                                        time=time, confirmed=time < 1024))
        # A different threshold must not enter the primary-threshold summary.
        records.append(dict(cohort='no_drop', seed=1, initial_mass=.01,
                            branch=BRANCHES[0], threshold=.08, time=1024, confirmed=False))
        summary = summarize_initial_masses(pd.DataFrame(records)).iloc[0]
        self.assertEqual(summary.seed_count, 2)
        self.assertEqual(summary.control_rmst, 320)
        self.assertEqual(summary.projection_rmst, 448)
        self.assertEqual(summary.difference, 128)
        self.assertAlmostEqual(summary.control_confirmed, 5 / 6)


if __name__ == '__main__':
    unittest.main()
