"""Check that documented launch plans resolve to usable experiment entry points."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.run import ROOT, STUDIES, plan_commands


class EntryPointTests(unittest.TestCase):
    def test_agnews_release_entry_records_the_primary_role(self):
        module = importlib.import_module("experiments.studies.agnews_structure_comparison")
        command = ["agnews_structure_comparison", "--dataset-dir", "data/agnews_prepared",
                   "--output-dir", "outputs/structure", "--protocol-id", "agnews_structure_comparison_v1",
                   "--seed", "4301", "--config", str(ROOT / "configs/agnews_structure.json")]
        with patch.object(sys, "argv", command):
            with patch.object(module, "run") as run:
                module.main()
        self.assertEqual(run.call_args.args[0].role, "primary")

    def test_uci_har_default_design_reaches_dataset_loader(self):
        module = importlib.import_module("experiments.studies.uci_har_moe_validation")
        class DatasetReached(Exception):
            pass
        with patch.object(sys, "argv", ["uci_har_moe_validation", "--device", "cpu"]):
            with patch.object(module, "load_data", side_effect=DatasetReached) as loader:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(DatasetReached):
                        module.main()
                loader.assert_called_once()


    def test_every_registered_module_exposes_a_command_line_entry(self):
        for study in STUDIES:
            with self.subTest(study=study.name):
                module = importlib.import_module(f"experiments.studies.{study.name}")
                self.assertTrue(callable(module.main))
                with patch.object(sys, "argv", [study.name, "--help"]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as exit_context:
                            module.main()
                self.assertEqual(exit_context.exception.code, 0)

    def test_launch_plans_use_valid_arguments_and_bundled_designs(self):
        for path in sorted((ROOT / "configs").glob("*.json")):
            if "runs" not in json.loads(path.read_text()):
                continue
            for label, command in plan_commands(path, sys.executable):
                with self.subTest(plan=path.name, run=label):
                    module = importlib.import_module(command[2])
                    for option in ("--config", "--protocol", "--probe-protocol"):
                        if option in command:
                            referenced = Path(command[command.index(option) + 1])
                            self.assertFalse(referenced.is_absolute())
                            self.assertTrue((ROOT / referenced).is_file())
                    with patch.object(sys, "argv", [command[2], *command[3:]]):
                        with patch.object(module, "run") as run:
                            module.main()
                    run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
