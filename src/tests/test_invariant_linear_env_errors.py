"""B8：非法动作不再"无声无息"。

引擎侧仍然把异常降级成罚分（保持上游行为），但必须：
1. 把失败原因记在 `env.last_error` 上，便于统计 exception_rate；
2. 在 `strict_error=True` 时直接抛出，便于 CI/调试中断言；
3. 一次正常决策后清空 `last_error`。
"""
import random
import unittest

from src.dgisim.action.action import ElementalTuningAction
from src.dgisim.agents import RandomAgent
from src.dgisim.element import Element
from src.dgisim.env.linear_env import LinearEnv
from src.dgisim.mode import DefaultMode
from src.dgisim.state.enums import Pid


def _env_in_action(seed: int = 7) -> LinearEnv:
    random.seed(seed)
    env = LinearEnv()
    agent = RandomAgent()
    for _ in range(200):
        state, enc, reward, turn, done = env.view()
        if done:
            break
        if type(env.full_view().phase).__name__ == "ActionPhase":
            return env
        env.step(agent.choose_action([env.full_view()], Pid(turn)))
    raise AssertionError("没有走到 ActionPhase")


def _illegal_action(env: LinearEnv) -> ElementalTuningAction:
    pid = env.full_view().waiting_for()
    assert pid is not None
    hand = set(env.full_view().get_player(pid).hand_cards.ordered_cards)
    pool = sorted(DefaultMode().all_cards(), key=lambda c: c.__name__)
    not_in_hand = next(c for c in pool if c not in hand)
    return ElementalTuningAction(card=not_in_hand, dice_elem=Element.OMNI)


class TestLinearEnvErrors(unittest.TestCase):
    def test_illegal_action_records_error_and_penalty(self):
        env = _env_in_action()
        _, _, reward, _, _ = env.step(_illegal_action(env))
        self.assertIsNotNone(env.last_error, "非法动作没有留下痕迹（B8）")
        self.assertTrue(env.last_error.startswith("ValueError"), env.last_error)
        self.assertNotEqual(reward, 0, "非法动作应当有罚分")

    def test_strict_error_raises(self):
        env = LinearEnv(strict_error=True)
        agent = RandomAgent()
        for _ in range(200):
            state, enc, reward, turn, done = env.view()
            if done:
                break
            if type(env.full_view().phase).__name__ == "ActionPhase":
                break
            env.step(agent.choose_action([env.full_view()], Pid(turn)))
        with self.assertRaises(ValueError):
            env.step(_illegal_action(env))

    def test_normal_step_clears_error(self):
        random.seed(11)
        env = LinearEnv()
        agent = RandomAgent()
        pid = env.full_view().waiting_for()
        env.step(agent.choose_action([env.full_view()], pid))
        self.assertIsNone(env.last_error)
