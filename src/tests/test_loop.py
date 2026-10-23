import unittest
from trainers.evaluate import load_decks
from dgisim_ext.env_api import CardEnv

from trainers.loop import train_steps
from trainers.policy import PolicyValueNet
from trainers.ppo import PPOConfig, PPOTrainer


class TestTrainingLoop(unittest.TestCase):
    def test_minimal_training_loop_updates_model(self):
        model = PolicyValueNet(hidden_dim=16)
        trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=8))
        result = train_steps(model, trainer, steps=4, seed=2, rollout_steps=2,
                             max_episode_steps=20)
        self.assertGreaterEqual(result.choice_steps, 4)
        self.assertEqual(result.failed_episodes, 0)
        self.assertEqual(result.updates, 1)
        self.assertEqual(len(result.batch_sizes), 1)
        self.assertIn("sampling", result.timings)

    def test_training_uses_explicit_training_decks(self):
        decks = load_decks("decks/train")
        seen = []

        class RecordingEnv(CardEnv):
            def reset(self, deck_p1=None, deck_p2=None, seed=None):
                seen.append((deck_p1, deck_p2))
                return super().reset(deck_p1, deck_p2, seed)

        model = PolicyValueNet(hidden_dim=16)
        trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=8))
        train_steps(model, trainer, steps=4, seed=2, rollout_steps=2,
                    max_episode_steps=20, env_factory=RecordingEnv,
                    training_decks=decks)
        self.assertTrue(seen)
        self.assertIsNotNone(seen[0][0])
        self.assertIsNotNone(seen[0][1])

