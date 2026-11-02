import unittest

import torch

from trainers.gae import compute_gae


class TestGAE(unittest.TestCase):
    def test_terminal_does_not_bootstrap(self):
        result = compute_gae([1.0], [10.0], [True], normalize=False,
                             next_value=999.0)
        self.assertTrue(torch.allclose(result.advantages, torch.tensor([-9.0])))
        self.assertTrue(torch.allclose(result.returns, torch.tensor([1.0])))

    def test_nonterminal_bootstraps_and_normalizes(self):
        result = compute_gae([1.0, 2.0], [0.0, 0.0], [False, False],
                             gamma=0.9, gae_lambda=1.0, next_value=3.0)
        self.assertTrue(torch.allclose(result.returns, torch.tensor([5.23, 4.7]), atol=1e-5))
        self.assertAlmostEqual(float(result.advantages.mean()), 0.0, places=6)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            compute_gae([float("nan")], [0.0], [True])

    def test_per_step_discount_can_disable_internal_choice_discount(self):
        result = compute_gae([0.0, 1.0], [0.0, 0.0], [False, True],
                             gamma=0.5, gammas=[1.0, 0.5], normalize=False)
        self.assertAlmostEqual(float(result.advantages[0]), 0.95, places=5)

