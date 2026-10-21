import random
import tempfile
import unittest
from pathlib import Path

import torch

from trainers.checkpoint import load_checkpoint, save_checkpoint
from trainers.policy import PolicyValueNet
from trainers.ppo import PPOConfig, PPOTrainer


class TestCheckpoint(unittest.TestCase):
    def test_model_optimizer_and_rng_restore(self):
        random.seed(10); torch.manual_seed(10)
        model = PolicyValueNet(obs_dim=4, hidden_dim=16)
        trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, trainer, step=17, config={"tag": "test"})
            expected_py = random.random(); expected_torch = torch.rand(1)
            random.random(); torch.rand(1)
            info = load_checkpoint(path, trainer)
            self.assertEqual(info["step"], 17)
            self.assertEqual(info["config"]["tag"], "test")
            self.assertAlmostEqual(random.random(), expected_py, places=7)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))

