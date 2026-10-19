import unittest

from src.dgisim.effect.effect import EffectsGroupEndEffect
from src.dgisim.effect.effect_stack import EffectStack
from src.dgisim.state.game_state import GameState


class TestStateIdentity(unittest.TestCase):
    def setUp(self):
        self._base = GameState.from_default()
        self._filler = EffectStack((EffectsGroupEndEffect(),))

    def _assert_differs(self, setter: str):
        a = getattr(self._base.factory(), setter)(EffectStack(())).build()
        b = getattr(self._base.factory(), setter)(self._filler).build()
        self.assertNotEqual(a, b, f"只有 {setter} 不同的两个状态被判相等")
        self.assertNotEqual(hash(a), hash(b), f"只有 {setter} 不同的两个状态 hash 相同")

    def test_dmg_stack(self):
        self._assert_differs("dmg_effect_stack")

    def test_common_stack(self):
        self._assert_differs("common_effect_stack")

    def test_lethal_stack(self):
        self._assert_differs("lethal_dmg_effect_stack")
