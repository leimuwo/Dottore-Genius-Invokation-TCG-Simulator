import unittest

import torch

from dgisim_ext.env_api import CardEnv
from trainers.policy import PolicyValueNet


class TestPolicyValueNet(unittest.TestCase):
    def test_outputs_only_current_candidates(self):
        env = CardEnv(shaping=False)
        obs = env.reset(seed=3)
        net = PolicyValueNet()
        logits, probs, value = net.distribution(obs)
        self.assertEqual(tuple(logits.shape), (len(obs.choices),))
        self.assertEqual(tuple(probs.shape), (len(obs.choices),))
        self.assertEqual(tuple(value.shape), ())
        self.assertAlmostEqual(float(probs.sum().detach()), 1.0, places=6)
        self.assertTrue(torch.isfinite(logits).all())

    def test_candidate_order_only_reorders_scores(self):
        env = CardEnv(shaping=False)
        obs = env.reset(seed=4)
        net = PolicyValueNet()
        if len(obs.choices) < 2:
            self.skipTest("seed did not produce multiple candidates")
        a = list(obs.choices)
        b = list(reversed(a))
        from dataclasses import replace
        logits_a, _, _ = net.distribution(obs)
        logits_b, _, _ = net.distribution(replace(obs, choices=tuple(b)))
        self.assertTrue(torch.allclose(logits_a, torch.flip(logits_b, (0,))))

    def test_act_is_legal_and_deterministic_when_requested(self):
        env = CardEnv(shaping=False)
        obs = env.reset(seed=5)
        net = PolicyValueNet()
        first = net.act(obs, deterministic=True)
        second = net.act(obs, deterministic=True)
        self.assertEqual(first[0], second[0])
        self.assertIn(first[0], range(len(obs.choices)))

