import unittest

from trainers.loop import train_steps
from trainers.policy import PolicyValueNet
from trainers.ppo import PPOConfig, PPOTrainer


class TestTrainingLoop(unittest.TestCase):
    def test_minimal_training_loop_updates_model(self):
        model = PolicyValueNet(hidden_dim=16)
        trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=8))
        result = train_steps(model, trainer, steps=4, seed=2)
        self.assertGreaterEqual(result.choice_steps, 4)
        self.assertEqual(result.failed_episodes, 0)
        self.assertEqual(result.updates, 1)

