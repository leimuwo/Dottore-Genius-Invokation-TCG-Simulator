"""E1：GameState 的四个 checker 必须惰性构造。

现状（eager）：`GameState.__init__` 每次都构造 CardChecker / SwapChecker /
SkillChecker / ElementalTuningChecker 四个对象。而一次决策内部约 10 次状态迁移，
绝大多数状态根本不会被查询 checker —— 纯浪费。

修好后：不查询就不构造；查询两次拿到同一个对象。
"""
import unittest

from src.dgisim.state.game_state import GameState, CardChecker


class TestCheckerIsLazy(unittest.TestCase):
    def test_state_creation_does_not_build_checkers(self):
        created: list[int] = []
        original = CardChecker.__init__

        def spy(self, *args, **kwargs):
            created.append(1)
            original(self, *args, **kwargs)

        CardChecker.__init__ = spy
        try:
            for _ in range(50):
                GameState.from_default()
        finally:
            CardChecker.__init__ = original

        self.assertEqual(len(created), 0, "构造 GameState 时就 eager 建了 CardChecker（E1）")

    def test_checker_is_cached_per_state(self):
        state = GameState.from_default()
        self.assertIs(state.card_checker, state.card_checker, "checker 没有被缓存")
