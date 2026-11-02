import unittest

from dgisim_ext.env_api import CardEnv
from trainers.rollout import collect_episode, random_decision
import random
from types import SimpleNamespace
from src.dgisim.state.enums import Pid
from trainers.loop import _decision_interval_rewards


class TestRewardAccounting(unittest.TestCase):
    @staticmethod
    def _reference_interval(transitions, seat, gamma):
        """Independent hand implementation: discount each COMMIT boundary."""
        total = 0.0
        exponent = 0
        for transition in transitions:
            total += (gamma ** exponent) * transition.rewards[seat]
            if transition.action_label == "COMMIT":
                exponent += 1
        return total

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

    def test_opponent_commits_are_discounted_inside_interval(self):
        transitions = [
            SimpleNamespace(seat=Pid.P1, action_label="COMMIT",
                            rewards={Pid.P1: 0.0, Pid.P2: 0.0}),
            SimpleNamespace(seat=Pid.P2, action_label="COMMIT",
                            rewards={Pid.P1: 10.0, Pid.P2: -10.0}),
            SimpleNamespace(seat=Pid.P2, action_label="COMMIT",
                            rewards={Pid.P1: 20.0, Pid.P2: -20.0}),
            SimpleNamespace(seat=Pid.P1, action_label="COMMIT",
                            rewards={Pid.P1: 0.0, Pid.P2: 0.0}),
        ]
        episode = SimpleNamespace(transitions=transitions)
        rewards = _decision_interval_rewards(
            episode, {Pid.P1: [transitions[0], transitions[3]],
                      Pid.P2: [transitions[1], transitions[2]]}, gamma=0.5)
        self.assertAlmostEqual(rewards[Pid.P1][0], 0.5 * 10.0 + 0.25 * 20.0)
        self.assertAlmostEqual(rewards[Pid.P2][0], -10.0)
        self.assertAlmostEqual(
            rewards[Pid.P1][0], self._reference_interval(transitions[:3], Pid.P1, 0.5))

    def test_discounted_potential_shaping_telescopes(self):
        transitions = [
            SimpleNamespace(seat=Pid.P1, action_label="COMMIT",
                            rewards={Pid.P1: 5.0, Pid.P2: -5.0}),
            SimpleNamespace(seat=Pid.P2, action_label="COMMIT",
                            rewards={Pid.P1: -10.0, Pid.P2: 10.0}),
            SimpleNamespace(seat=Pid.P1, action_label="COMMIT",
                            rewards={Pid.P1: 0.0, Pid.P2: 0.0}),
        ]
        episode = SimpleNamespace(transitions=transitions)
        rewards = _decision_interval_rewards(
            episode, {Pid.P1: [transitions[0], transitions[2]],
                      Pid.P2: [transitions[1]]}, gamma=0.5)
        # Phi: 0 -> 10 -> 0 with gamma=.5 gives 5 + .5*(-10) = 0.
        self.assertAlmostEqual(rewards[Pid.P1][0], 0.0)

