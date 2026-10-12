"""Check relocation, configurable audit names, and optional GPU identity."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.analysis.analyze_additional_recovery import load_run
from experiments.analysis.analyze_router_update_capacity import locate_seed
from experiments.audits import audit_router_update_capacity as capacity
from experiments.audits import audit_router_update_no_drop as no_drop
from experiments.data.paths import manifest_relative_path, resolve_tokenizer_directory


class PathPortabilityTests(unittest.TestCase):
    def test_dataset_and_tokenizer_survive_parent_move_and_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / 'original checkout'
            dataset = old / 'data/corpus'
            tokenizer = old / 'data/tokenizer'
            dataset.mkdir(parents=True)
            tokenizer.mkdir()
            (tokenizer / 'tokenizer.json').write_text('same tokenizer bytes')
            metadata = {'path': manifest_relative_path(tokenizer, dataset)}
            self.assertEqual(metadata['path'], '../tokenizer')
            (dataset / 'manifest.json').write_text(json.dumps({'tokenizer': metadata}))
            renamed = Path(temp) / '不同名称 relocated repo'
            shutil.move(old, renamed)
            moved_dataset = renamed / 'data/corpus'
            moved_metadata = json.loads((moved_dataset / 'manifest.json').read_text())['tokenizer']
            actual = resolve_tokenizer_directory(moved_metadata, moved_dataset)
            self.assertEqual((actual / 'tokenizer.json').read_text(), 'same tokenizer bytes')
            self.assertFalse(old.exists())

    def test_legacy_absolute_path_can_be_overridden_without_manifest_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / 'dataset'
            tokenizer = Path(temp) / 'moved tokenizer'
            dataset.mkdir()
            tokenizer.mkdir()
            legacy = {'path': str(Path(temp) / 'removed original tokenizer')}
            before = json.dumps(legacy)
            self.assertEqual(resolve_tokenizer_directory(legacy, dataset, override=tokenizer), tokenizer.resolve())
            self.assertEqual(json.dumps(legacy), before)
            self.assertEqual(resolve_tokenizer_directory({'path': str(tokenizer)}, dataset), tokenizer.resolve())
            with self.assertRaises(ValueError):
                resolve_tokenizer_directory({}, dataset)

    def test_default_and_custom_audit_names_locate_the_same_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'arbitrary cohort name'
            seed = 5601
            folder = root / str(seed)
            folder.mkdir(parents=True)
            result = {'seed': seed, 'role': 'additional', 'cohort': 'no_drop', 'protocol_id': 'test'}
            audit = {'status': 'passed', 'failures': [], 'probe_threshold_crossing_mismatches': 0}
            (folder / 'result.json').write_text(json.dumps(result))
            (folder / 'audit.json').write_text(json.dumps(audit))
            self.assertEqual(locate_seed([root], seed)[1], folder)
            self.assertEqual(load_run(root, seed=seed, role='additional', cohort='no_drop', protocol_id='test')[0], result)
            (folder / 'audit.json').rename(folder / 'custom-check.json')
            self.assertEqual(locate_seed([root], seed, audit_name='custom-check.json')[1], folder)
            self.assertEqual(load_run(root, seed=seed, role='additional', cohort='no_drop', protocol_id='test', audit_name='custom-check.json')[1], audit)

    def test_other_gpu_names_are_accepted_unless_exact_model_is_requested(self):
        for auditor in [no_drop, capacity]:
            with self.subTest(auditor=auditor.__name__):
                result = {'execution': {'device': 'cuda', 'gpu': 'NVIDIA A100', 'elapsed_seconds': 1.0}}
                ordinary = auditor.audit_result(result, expected_role='smoke')['failures']
                self.assertFalse(any('GPU identity' in error for error in ordinary))
                strict = auditor.audit_result(result, expected_role='smoke', expected_gpu='RTX 5090')['failures']
                self.assertIn('GPU identity does not match RTX 5090', strict)
                result['execution']['device'] = 'cpu'
                self.assertIn('CUDA execution', auditor.audit_result(result, expected_role='smoke')['failures'])


if __name__ == '__main__':
    unittest.main()
