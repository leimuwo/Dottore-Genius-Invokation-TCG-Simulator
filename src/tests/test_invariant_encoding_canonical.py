import random
import unittest

from src.dgisim.card.cards import Cards
from src.dgisim.dice import ActualDice
from src.dgisim.element import Element
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.mode import DefaultMode
from src.dgisim.state.enums import Pid
from src.dgisim.state.game_state import GameState


def _decision_state(seed: int = 3) -> GameState:
    random.seed(seed)
    gs = GameState.from_default()
    for _ in range(500):
        if gs.game_end():
            break
        if gs.waiting_for() is not None:
            return gs
        gs = gs.step()
    raise AssertionError("没有走到决策点")


class TestEncodingCanonical(unittest.TestCase):
    def test_cards_encoding_is_canonical(self):
        pool = sorted(DefaultMode().all_cards(), key=lambda c: c.__name__)
        a, b, c = pool[0], pool[1], pool[2]
        forward = Cards({a: 1, b: 1, c: 1}).encoding(encoding_plan)
        backward = Cards({c: 1, b: 1, a: 1}).encoding(encoding_plan)
        self.assertEqual(forward, backward, "Cards 编码依赖插入顺序（B13）")

    def test_dice_encoding_is_canonical(self):
        d1 = ActualDice({Element.PYRO: 1, Element.HYDRO: 2})
        d2 = ActualDice({Element.HYDRO: 2, Element.PYRO: 1})
        self.assertEqual(
            d1.encoding(encoding_plan),
            d2.encoding(encoding_plan),
            "Dice 编码依赖插入顺序（B13）",
        )

    def test_game_state_encoding_is_canonical(self):
        gs = _decision_state()
        hand_items: list[tuple[type, int]] = []
        for card in gs.player1.hand_cards.ordered_cards:
            if hand_items and hand_items[-1][0] is card:
                hand_items[-1] = (card, hand_items[-1][1] + 1)
            else:
                hand_items.append((card, 1))
        self.assertGreaterEqual(len(hand_items), 2, "手牌种类不足，测不出顺序差异")
        permuted = gs.factory().f_player1(
            lambda p: p.factory().hand_cards(Cards(dict(hand_items[::-1]))).build()
        ).build()
        self.assertEqual(
            gs.perspective_view(Pid.P1).encoding(encoding_plan, Pid.P1),
            permuted.perspective_view(Pid.P1).encoding(encoding_plan, Pid.P1),
            "GameState 编码依赖手牌插入顺序（B13）",
        )
