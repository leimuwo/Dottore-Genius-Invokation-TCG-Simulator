import json
import tempfile
import unittest
from pathlib import Path

from trainers.cli import main


class TestCLI(unittest.TestCase):
    def test_smoke_run_writes_manifest_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(main(["--steps", "4", "--hidden-dim", "16", "--output", folder]), 0)
            self.assertTrue((Path(folder) / "checkpoint.pt").exists())
            manifest = json.loads((Path(folder) / "manifest.json").read_text())
            self.assertEqual(manifest["choice_steps"], 4)
            self.assertEqual(manifest["reward"]["shaping_lambda"], 1e-2)

    def test_sparse_reward_switch_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(main(["--steps", "4", "--no-shaping", "--output", folder]), 0)
            manifest = json.loads((Path(folder) / "manifest.json").read_text())
            self.assertFalse(manifest["reward"]["shaping"])

