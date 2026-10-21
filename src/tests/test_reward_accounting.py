import unittest

from dgisim_ext.env_api import CardEnv
from trainers.rollout import collect_episode, random_decision
import random


class TestRewardAccounting(unittest.TestCase):
    def test_training_uses_actor_seat_reward_mapping(self):
        episode = collect_episode(CardEnv(), 3, random_decision(random.Random(4)), max_steps=8,
                                  allow_truncated=True)
        self.assertTrue(episode.transitions)
        for transition in episode.transitions:
            self.assertIn(transition.seat, transition.rewards)
            self.assertEqual(transition.reward, transition.rewards[transition.seat])

