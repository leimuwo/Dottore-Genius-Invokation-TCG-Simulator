import random
import unittest

from src.dgisim.agents import RandomAgent
from src.dgisim.dice import ActualDice
from src.dgisim.element import Element
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.state.game_state import GameState
from src.dgisim.state.enums import Pid


def _state_with_dice(seed: int = 11) -> GameState:
    random.seed(seed)
    agent = RandomAgent()
    gs = GameState.from_default()
    for _ in range(500):
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            continue
        if gs.get_player(w).dice.num_dice() > 0:
            return gs
        gs = gs.action_step(w, agent.choose_action([gs], w))
    raise AssertionError("没有走到带骰子的决策点")


class TestEncodingIdempotent(unittest.TestCase):
    def test_dice_encoding_twice_is_equal(self):
        d = ActualDice({Element.PYRO: 2, Element.HYDRO: 1})
        first = d.encoding(encoding_plan)
        second = d.encoding(encoding_plan)
        self.assertNotEqual(first, [0] * encoding_plan.DICE_FIXED_LEN, "骰子编码全 0")
        self.assertEqual(first, second, "Dice.encoding() 两次结果不同")

    def test_state_encoding_twice_is_equal(self):
        gs = _state_with_dice()
        a = gs.encoding(encoding_plan, Pid.P1)
        b = gs.encoding(encoding_plan, Pid.P1)
        self.assertEqual(a, b, "GameState.encoding() 两次结果不同")
