import random
import unittest

from src.dgisim.agents import RandomAgent
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.env.linear_env import LinearEnv
from src.dgisim.state.enums import Pid


def _env_at_p2_turn(seed: int = 5, max_decisions: int = 80) -> LinearEnv:
    random.seed(seed)
    env = LinearEnv()
    agent = RandomAgent()
    for _ in range(max_decisions):
        state, enc, reward, turn, done = env.view()
        if done:
            break
        if turn == 2:
            return env
        env.step(agent.choose_action([env.full_view()], Pid(turn)))
    raise AssertionError("没有走到轮到 P2 的决策点")


class TestLinearEnvContract(unittest.TestCase):
    def test_view_encodes_in_acting_player_perspective(self):
        env = _env_at_p2_turn()
        _, enc, _, turn, _ = env.view()
        self.assertEqual(turn, 2)
        expected = env.full_view().perspective_view(Pid.P2).encoding(encoding_plan, Pid.P2)
        self.assertEqual(enc, expected, "view() 在 P2 行动时用了 P1 视角（B6）")

    def test_copy_does_not_advance_rng_and_keeps_flags(self):
        random.seed(2024)
        env = LinearEnv(fix_perspective=True)
        before = random.getstate()
        copied = env.copy()
        after = random.getstate()
        self.assertEqual(before, after, "copy() 推进了全局 RNG（B6/B2）")
        self.assertTrue(copied._fix_perspective, "copy() 丢了 fix_perspective（B6）")
