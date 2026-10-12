"""Independent checks for the controlled exponent experiment."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
import numpy as np
from experiments.studies.mobility_exponent_continuum import integrate, scaled_reference_time
from experiments.studies.multiexpert_face_mobility import h1, h2, normal_quantities


class MobilityExponentTests(unittest.TestCase):
    def test_full_matrix_field_has_prescribed_normal_coefficient(self):
        for n in (4, 8):
            for x in (1e-6, 0.001, 0.04):
                for alpha in (0, 0.25, 0.5, 0.75, 1, 1.5, 2):
                    p = np.r_[x, np.full(n-1, (1-x)/(n-1))]
                    g = np.r_[-x**(alpha-1)/(1+1/(n-1)), np.zeros(n-1)]
                    values = normal_quantities(p, g)
                    self.assertAlmostEqual(values['gamma']/x**alpha, 1.0, places=12)
                    self.assertAlmostEqual(values['actual_normal_velocity']/(x**(1+alpha)*(1-x)**2), 1.0, places=12)

    def test_quadrature_matches_closed_endpoints(self):
        for eps in (1e-9, 0.001):
            self.assertAlmostEqual(scaled_reference_time(eps, 0.05, 0)/h1(eps, 0.05), 1, places=11)
            self.assertAlmostEqual(scaled_reference_time(eps, 0.05, 1)/(eps*h2(eps, 0.05)), 1, places=11)

    def test_logit_event_time_matches_independent_quadrature(self):
        for n in (4, 8):
            for alpha in (0, 0.5, 1, 1.5, 2):
                result = integrate(n, 1e-9, alpha)
                self.assertLess(result['relative_time_error'], 1e-8)
                self.assertAlmostEqual(result['terminal_mass'], 0.05, places=11)

    def test_invalid_inputs(self):
        for n, eps, alpha in ((1, .001, .5), (4, .1, .5), (4, .001, -1)):
            with self.assertRaises(ValueError):
                integrate(n, eps, alpha)


if __name__ == '__main__':
    unittest.main()
