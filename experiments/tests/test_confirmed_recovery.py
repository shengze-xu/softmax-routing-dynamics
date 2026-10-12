"""Check administrative censoring and event-confirmation timing."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
import numpy as np
from experiments.analysis.analyze_confirmed_recovery import confirmed_rmst, analyze, verify_cells
class ConfirmedRecoveryTests(unittest.TestCase):
    def test_confirmation_at_last_evaluation(self):
        # The last confirmed onset is 960; confirmation and censoring both at 1024.
        self.assertEqual(confirmed_rmst(960,1),1024)
        self.assertEqual(confirmed_rmst(1088,0),1024)
    def test_mixed_cell_times_equal_survival_area(self):
        starts=np.array([64]*4+[256]*4+[960]*4+[1088]*4)
        event=starts<1088
        times=np.where(event,starts+64,1024)
        area=sum(np.mean(times>t)*64 for t in range(0,1024,64))
        self.assertEqual(confirmed_rmst(float(starts.mean()),float(event.mean())),area)
    def test_invalid_observation_summary(self):
        with self.assertRaises(ValueError):confirmed_rmst(1050,1)
        with self.assertRaises(ValueError):confirmed_rmst(900,.13)
if __name__=='__main__':unittest.main()
