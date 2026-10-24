import random
import unittest

from dgisim_ext.env_api import CardEnv
from trainers.rollout import collect_episode, random_decision


class TestRollout(unittest.TestCase):
    def test_random_rollout_records_layered_choices_and_terminal(self):
        episode = collect_episode(
            CardEnv(shaping=False), 12, random_decision(random.Random(99)))
        self.assertTrue(episode.valid)
        self.assertTrue(episode.transitions)
        self.assertTrue(episode.transitions[-1].done)
        self.assertIsNotNone(episode.terminal_reason)
        self.assertEqual(
            episode.transitions[-1].reward,
            episode.transitions[-1].rewards[episode.transitions[-1].seat],
        )

    def test_same_seed_and_policy_rng_reproduce_transition_summary(self):
        def run():
            return collect_episode(
                CardEnv(shaping=False), 8, random_decision(random.Random(123)))
        a, b = run(), run()
        summary = lambda ep: [
            (t.obs_fingerprint, t.action_index, t.reward, t.done)
            for t in ep.transitions
        ]
        self.assertEqual(summary(a), summary(b))

    def test_invalid_policy_is_excluded_with_diagnostic(self):
        def bad(_obs):
            from trainers.rollout import Decision
            return Decision(999999, 0.0, 0.0, 0.0)
        episode = collect_episode(CardEnv(shaping=False), 1, bad)
        self.assertFalse(episode.valid)
        self.assertEqual(episode.failure["category"], "illegal")
        self.assertEqual(episode.transitions, [])

    def test_commit_diagnostics_capture_state_delta(self):
        episode = collect_episode(
            CardEnv(shaping=False), 12, random_decision(random.Random(99)),
            max_steps=12, allow_truncated=True)
        self.assertTrue(episode.valid)
        self.assertTrue(episode.commit_diagnostics)
        row = episode.commit_diagnostics[0]
        self.assertIn(row["action_type"], {
            "skill", "card", "swap", "end_round", "DeathSwapAction",
            "CharacterSelectAction", "CardsSelectAction", "DiceSelectAction",
        })
        self.assertIn("damage_dealt", row)
        self.assertIn("shield_delta", row)
        self.assertIn("no_damage_streak", row)
        self.assertIn("P1", row["before"])
        self.assertIn("P2", row["after"])
