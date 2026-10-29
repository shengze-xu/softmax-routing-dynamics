"""Check prepared-array loading and CPU training."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from experiments.models.agnews_autoregressive_moe import PreparedDataset
from experiments.models.agnews_gpt_moe import run_gpt
from experiments.studies.agnews_structure_comparison import IndependentPreparedDataset
from experiments.studies.multidomain_controlled_moe import LargeMultidomainPreparedDataset
from experiments.studies.multidomain_small_moe import SmallMultidomainPreparedDataset
from experiments.studies.uci_har_moe_validation import MODES, paired_statistics


class ExperimentInputTests(unittest.TestCase):
    def test_paired_statistics_accept_a_smaller_seed_set(self):
        rows = [
            {"mode": mode.name, "seed": seed, "specialization_score": value}
            for mode, values in ((MODES[0], [.1, .2]), (MODES[1], [.2, .4]))
            for seed, value in enumerate(values)
        ]
        summary = paired_statistics(rows)
        self.assertEqual(summary["pair_count"], 2)
        self.assertAlmostEqual(summary["paired_mean_difference"], .15)
        self.assertEqual(summary["favorable_wins"], 2)
        self.assertEqual(summary["exact_two_sided_sign_p"], .5)

    def make_corpus(self, root, status, length=4):
        arrays = {}
        rng = np.random.default_rng(41)
        for split in ("train", "validation"):
            for field, value in (
                ("tokens", rng.integers(0, 32, (8, length + 1), dtype=np.int64)),
                ("lengths", np.full(8, length + 1, dtype=np.int64)),
                ("domains", np.tile(np.arange(4, dtype=np.int64), 2)),
            ):
                name = f"{split}_{field}"
                np.save(root / f"{name}.npy", value)
                arrays[name] = {"file": f"{name}.npy", "shape": list(value.shape), "dtype": str(value.dtype)}
        manifest = {
            "schema_version": 1, "dataset_id": "multidomain-64m-2m-2m-v1",
            "status": status, "domain_names": ["a", "b", "c", "d"],
            "tokenizer": {"vocab_size": 32}, "selection": {"sequence_length": length},
            "arrays": arrays,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_all_prepared_loaders_accept_generated_arrays(self):
        for loader, status, length in (
            (PreparedDataset, "prepared_agnews_candidate", 4),
            (IndependentPreparedDataset, "prepared_agnews_candidate", 4),
            (SmallMultidomainPreparedDataset, "prepared_multidomain_small_candidate", 4),
            (LargeMultidomainPreparedDataset, "prepared_multidomain_large_candidate", 512),
        ):
            with self.subTest(loader=loader.__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_corpus(root, status, length)
                dataset = loader(root)
                tokens, lengths, domains = dataset.split("validation", limit=4)
                self.assertEqual(tuple(tokens.shape), (4, length + 1))
                self.assertEqual(lengths.tolist(), [length + 1] * 4)
                self.assertEqual(domains.tolist(), [0, 1, 2, 3])

    def test_cpu_training_reads_prepared_arrays_and_writes_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "corpus"
            dataset.mkdir()
            self.make_corpus(dataset, "prepared_agnews_candidate")
            args = SimpleNamespace(
                dataset_dir=dataset, output_dir=root / "run", device="cpu", precision="fp32",
                seed=41, epochs=1, batch_size=4, eval_batch_size=4, max_steps=2,
                limit_train=8, limit_validation=8, learning_rate=.001, weight_decay=.01,
                d_model=8, num_heads=2, num_layers=2, expert_hidden=16, num_experts=4, top_k=2,
                init_std=.02, router_init_std=.01, balance_lambda=.01,
                router_z_loss=.001, importance_gap_weight=1.,
                conditions=["shared_global_top2", "conditioned_top2"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_gpt(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["conditions"]), 2)
            self.assertTrue((root / "run/result.json").is_file())
            for condition in result["conditions"]:
                self.assertEqual(condition["optimizer_steps"], 2)
                self.assertTrue(np.isfinite(condition["final_validation"]["loss"]))


if __name__ == "__main__":
    unittest.main()
