import unittest
from types import SimpleNamespace

from trainers.evaluate import attack_first_decision
from dgisim_ext.env_api import Choice, Obs
from src.dgisim.state.enums import Pid


class TestAttackFirst(unittest.TestCase):
    def _obs(self, actions):
        choices = tuple(Choice("tuple", SimpleNamespace(name=action),
                               (1., 0., 0., 1.)) for action in actions)
        return Obs(Pid.P1, (0,), choices, False, 1, 0)

    def test_skill_precedes_end_round_and_swap(self):
        decision = attack_first_decision(
            self._obs(("SWAP_CHARACTER", "END_ROUND", "CAST_SKILL")))
        self.assertEqual(decision.action_index, 2)

    def test_no_skill_never_swaps(self):
        decision = attack_first_decision(
            self._obs(("SWAP_CHARACTER", "END_ROUND")))
        self.assertEqual(decision.action_index, 1)
