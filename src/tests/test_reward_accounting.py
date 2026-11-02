import unittest

from dgisim_ext.env_api import CardEnv
from trainers.rollout import collect_episode, random_decision
import random
from types import SimpleNamespace
from src.dgisim.state.enums import Pid
from trainers.loop import _decision_interval_rewards


class TestRewardAccounting(unittest.TestCase):
    def test_training_uses_actor_seat_reward_mapping(self):
        episode = collect_episode(CardEnv(), 3, random_decision(random.Random(4)), max_steps=8,
                                  allow_truncated=True)
        self.assertTrue(episode.transitions)
        for transition in episode.transitions:
            self.assertIn(transition.seat, transition.rewards)
            self.assertEqual(transition.reward, transition.rewards[transition.seat])

    def test_decision_interval_includes_opponent_reward_and_is_additive(self):
        p1 = SimpleNamespace(seat=Pid.P1, rewards={Pid.P1: 0.2, Pid.P2: -0.2})
        p2 = SimpleNamespace(seat=Pid.P2, rewards={Pid.P1: -0.3, Pid.P2: 0.3})
        p1_terminal = SimpleNamespace(seat=Pid.P1, rewards={Pid.P1: 1.0, Pid.P2: -1.0})
        episode = SimpleNamespace(transitions=[p1, p2, p1_terminal])
        rewards = _decision_interval_rewards(
            episode, {Pid.P1: [p1, p1_terminal], Pid.P2: [p2]})
        self.assertAlmostEqual(rewards[Pid.P1][0], -0.1)
        self.assertEqual(rewards[Pid.P1][1], 1.0)
        self.assertAlmostEqual(rewards[Pid.P2][0], -0.9)

