
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from experiments.analysis.analyze_recovery_endpoint_sensitivity import recode_mean, sign_p


class EndpointSensitivityTests(unittest.TestCase):
    def test_summary_recoding_matches_individual_outcomes(self):
        # Two confirmed starts and two unconfirmed outcomes, including a late
        # first crossing whose persistence cannot be confirmed by step 1024.
        outcomes = [128, 960, None, None]
        mean_original = sum(x if x is not None else 1088 for x in outcomes) / 4
        for code in (1088, 1024, 960):
            expected = sum(x if x is not None else code for x in outcomes) / 4
            self.assertEqual(recode_mean(mean_original, 0.5, code), expected)
        self.assertEqual(recode_mean(500, 1, 960), 500)
        self.assertEqual(recode_mean(1088, 0, 960), 960)

    def test_sign_test_excludes_ties(self):
        self.assertEqual(sign_p([1] * 14 + [-1, 0]), 0.0009765625)
        self.assertEqual(sign_p([1] * 8), 0.0078125)
        self.assertEqual(sign_p([0, 0]), 1)

    def test_invalid_fractions_rejected(self):
        for fraction in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                recode_mean(500, fraction, 960)


if __name__ == "__main__":
    unittest.main()
