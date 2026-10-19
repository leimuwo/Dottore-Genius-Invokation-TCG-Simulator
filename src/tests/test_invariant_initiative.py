"""B5 复核：下回合先手归"先声明结束回合"的一方。

结论：**归档版实现是正确的**（133/133 回合，见 analysis/probe_initiative.py）。
本测试把这个行为锁死，防止以后有人"照着评测文档去修"从而把正确的实现改坏。

注意：`EndRoundAction` 在不同阶段含义不同——
- CardSelectPhase：放弃换牌
- RollPhase：不保留骰子
- ActionPhase：结束回合
只有第三种才是"结束回合"。
"""
import random
import unittest

from src.dgisim.action.action import EndRoundAction
from src.dgisim.agents import RandomAgent
from src.dgisim.state.game_state import GameState


def _initiative_pairs(seed: int, max_steps: int = 4000) -> list[tuple[int, str, str]]:
    random.seed(seed)
    agent = RandomAgent()
    gs = GameState.from_default()
    first_end: dict[int, str] = {}
    first_actor: dict[int, str] = {}
    steps = 0
    while not gs.game_end() and steps < max_steps:
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            steps += 1
            continue
        phase = type(gs.phase).__name__
        if phase == "ActionPhase":
            first_actor.setdefault(gs.round, w.name)
        action = agent.choose_action([gs], w)
        if isinstance(action, EndRoundAction) and phase == "ActionPhase":
            first_end.setdefault(gs.round, w.name)
        gs = gs.action_step(w, action)
        steps += 1
    return [
        (r, first_end[r], first_actor[r + 1])
        for r in sorted(first_end)
        if (r + 1) in first_actor
    ]


class TestInitiative(unittest.TestCase):
    def test_first_to_end_gets_next_round_initiative(self):
        total = 0
        ok = 0
        for seed in (0, 1, 2):
            for rnd, ender, actor in _initiative_pairs(seed):
                total += 1
                if ender == actor:
                    ok += 1
                else:  # pragma: no cover - 失败时给出定位信息
                    self.fail(f"seed={seed} round={rnd}: 先结束={ender} 但下回合先动={actor}")
        self.assertGreater(total, 20, "样本太少，测不出结论")
        self.assertEqual(ok, total)
