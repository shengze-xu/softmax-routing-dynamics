"""Check portable run plans and the recovery-analysis pipeline."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from experiments.analysis import analyze_recovery_results
from experiments.common.recovery_statistics import exact_two_sided_sign_p
from experiments.analysis.extract_recovery_results import run
from experiments.run import plan_commands


class PublicPackageTests(unittest.TestCase):
    def test_sign_test_excludes_ties(self):
        self.assertEqual(exact_two_sided_sign_p([1] * 14 + [-1, 0]), 0.0009765625)
        self.assertEqual(exact_two_sided_sign_p([1] * 8), 0.0078125)
        self.assertEqual(exact_two_sided_sign_p([0, 0]), 1)

    def test_analysis_accepts_results_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root / 'no-drop', 5201, 'no_drop')
            self.make_run(root / 'capacity', 5401, 'capacity_1_25')
            groups = {'no_drop_primary': {'directory': 'no-drop', 'seeds': [5201]},
                      'capacity_primary': {'directory': 'capacity', 'seeds': [5401]}}
            inputs, outputs = root / 'inputs', root / 'summary'
            inputs.mkdir()
            (inputs / 'extracted.json').write_text(json.dumps(run(root, groups)))
            command = ['analyze_recovery_results', '--input-dir', str(inputs),
                       '--output-dir', str(outputs)]
            with patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()):
                analyze_recovery_results.main()
            with (outputs / 'confirmed_recovery_rmst_summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(float(row['control']), 128)
                self.assertEqual(float(row['projection']), 256)
                self.assertEqual(float(row['difference']), 128)
            self.assertTrue((outputs / 'analysis.json').is_file())

    def make_run(self, folder, seed, cohort):
        branches = {}
        for branch in ['free_release', 'zero_advantage_gradient_projection_reference']:
            delay = 0 if branch == 'free_release' else 2
            branches[branch] = {
                'evaluation_steps': list(range(64, 1025, 64)),
                'evaluations': [{'raw_target_probability_by_layer': [.01] + [.01 if i < delay else .06] * 4 + [.01]} for i in range(16)]}
        result = {'seed': seed, 'cohort': cohort, 'cells': {str(m): {'branches': branches} for m in [.005, .01, .02, .04]}}
        destination = folder / str(seed)
        destination.mkdir(parents=True)
        (destination / 'result.json').write_text(json.dumps(result))

    def test_raw_results_use_second_evaluation_and_preserve_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root / 'no-drop', 5201, 'no_drop')
            self.make_run(root / 'capacity', 5401, 'capacity_1_25')
            groups = {'no_drop_primary': {'directory': 'no-drop', 'seeds': [5201]}, 'capacity_primary': {'directory': 'capacity', 'seeds': [5401]}}
            data = run(root, groups)
            selected = [r for r in data['threshold_cells'] if r[5] == .05]
            self.assertEqual(len(selected), 64)
            self.assertEqual({r[6] for r in selected if r[3] == 'free_release'}, {128})
            self.assertEqual({r[6] for r in selected if r[3] != 'free_release'}, {256})
            self.assertTrue(all(r[7] for r in selected))

    def test_run_plan_preserves_arguments_as_data(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'plan.json'
            value = 'output folder with spaces; literal text'
            file.write_text(json.dumps({'version': 1, 'runs': [{'experiment': 'mobility_exponent_continuum', 'args': ['--output-dir', value]}]}))
            commands = plan_commands(file, 'python3')
            self.assertEqual(commands[0][1][-1], value)
            file.write_text(json.dumps({'version': 1, 'runs': [{'experiment': 'unknown', 'args': []}]}))
            with self.assertRaises(ValueError):
                plan_commands(file, 'python3')

    def test_recovery_confirmation_at_the_observation_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root / 'no-drop', 5201, 'no_drop')
            path = root / 'no-drop/5201/result.json'
            result = json.loads(path.read_text())
            for cell in result['cells'].values():
                for branch in cell['branches'].values():
                    for index, evaluation in enumerate(branch['evaluations']):
                        evaluation['raw_target_probability_by_layer'] = [
                            .01, .01, .06 if index == 15 else .01,
                            .06 if index >= 14 else .01, .06, .01]
            path.write_text(json.dumps(result))
            data = run(root, {'no_drop_primary': {
                'directory': 'no-drop', 'seeds': [5201]}})
            for row in data['threshold_cells']:
                if row[5] == .05:
                    layer = row[4]
                    self.assertEqual(row[6], 128 if layer == 4 else 1024)
                    self.assertEqual(row[7], layer in (3, 4))


if __name__ == '__main__':
    unittest.main()
