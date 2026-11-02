import unittest

import torch

from dgisim_ext.env_api import Obs, Choice
from src.dgisim.state.enums import Pid
from trainers.policy import PolicyValueNet


class TestPolicyEncoding(unittest.TestCase):
    def _obs(self, payment):
        return Obs(Pid.P1, (1, 2, 3),
                   (Choice("tuple", "CAST_SKILL", (1., 2., 3., 4.)),),
                   False, 1, 2,
                   {"choice_prefix": ("CAST_SKILL",),
                    "selection_prefix": payment})

    def test_structured_payment_prefix_changes_stage(self):
        a = PolicyValueNet.observation_stage(self._obs((1, 0)))
        b = PolicyValueNet.observation_stage(self._obs((0, 1)))
        self.assertNotEqual(a, b)

    def test_sample_and_recompute_log_prob_match_without_update(self):
        torch.manual_seed(7)
        model = PolicyValueNet(obs_dim=3, hidden_dim=16)
        obs = self._obs((1, 0))
        logits, probs, _ = model.distribution(obs)
        index = int(probs.argmax())
        old = torch.log(probs[index])
        enc, seat, stage, features, kinds = model.tensors_for(obs)
        new_logits, _ = model(enc, seat, stage, features, kinds)
        new = torch.log_softmax(new_logits, dim=-1)[index]
        self.assertAlmostEqual(float(stage), PolicyValueNet.observation_stage(obs), places=6)
        self.assertAlmostEqual(float(torch.exp(new - old).detach()), 1.0, places=6)
