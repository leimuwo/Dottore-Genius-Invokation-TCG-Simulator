import unittest

import torch

from trainers.policy import PolicyValueNet
from trainers.ppo import PPOConfig, PPOItem, PPOTrainer


class TestPPO(unittest.TestCase):
    def test_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(2)
        model = PolicyValueNet(obs_dim=4, hidden_dim=32)
        items = []
        for i in range(4):
            items.append(PPOItem(
                torch.randn(4), torch.tensor(1.0), torch.tensor(0.0),
                torch.randn(3, 4), torch.tensor([0, 1, 4]), i % 3,
                -1.0, 0.5, 1.0))
        before = [x.detach().clone() for x in model.parameters()]
        metrics = PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=2)).update(items)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_empty_batch_rejected(self):
        model = PolicyValueNet(obs_dim=4, hidden_dim=16)
        with self.assertRaises(ValueError):
            PPOTrainer(model).update([])
