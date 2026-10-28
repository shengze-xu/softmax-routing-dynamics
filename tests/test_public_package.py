"""Portable-input and endpoint checks without manuscript artifacts or datasets."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.analysis.analyze_confirmed_recovery import analyze
from experiments.analysis.analyze_recovery_endpoint_sensitivity import SPECS
from experiments.analysis.extract_recovery_additions import run
from experiments.run import plan_commands


class PublicPackageTests(unittest.TestCase):
    def test_rmst_summary_accepts_inputs_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for _, filename, seeds, free, projected, free_fraction, projected_fraction in SPECS:
                with (folder / filename).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['seed', free, projected, free_fraction, projected_fraction])
                    writer.writeheader()
                    writer.writerows({'seed': seed, free: 576, projected: 608, free_fraction: .5, projected_fraction: .5} for seed in seeds)
            seeds, summaries = analyze(folder)
            self.assertEqual(len(seeds), 24)
            self.assertEqual([s['seed_count'] for s in summaries], [16, 8])
            for row in summaries:
                self.assertEqual(row['control_rmst'], 576)
                self.assertEqual(row['projected_rmst'], 608)
                self.assertEqual(row['projected_minus_control_rmst'], 32)
                self.assertEqual(row['bootstrap_95_low'], 32)
                self.assertEqual(row['bootstrap_95_high'], 32)

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


if __name__ == '__main__':
    unittest.main()
