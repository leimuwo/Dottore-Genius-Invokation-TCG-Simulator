import random
import unittest

from src.dgisim.agents import RandomAgent
from src.dgisim.card.cards import OrderedCards
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.state.enums import Pid
from src.dgisim.state.game_state import GameState


def _decision_state(seed: int = 3) -> GameState:
    random.seed(seed)
    agent = RandomAgent()
    gs = GameState.from_default()
    for _ in range(500):
        if gs.game_end():
            break
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            continue
        if len(gs.player1.deck_cards.cards) >= 2 and len(gs.player2.deck_cards.cards) >= 2:
            return gs
        gs = gs.action_step(w, agent.choose_action([gs], w))
    raise AssertionError("没有走到牌库非空的决策点")


def _obs(gs: GameState) -> list[int]:
    return gs.perspective_view(Pid.P1).encoding(encoding_plan, Pid.P1)


class TestDeckOrderNotObserved(unittest.TestCase):
    def test_own_deck_order_not_encoded(self):
        gs = _decision_state()
        base = _obs(gs)
        reversed_deck = OrderedCards(list(gs.player1.deck_cards.cards)[::-1])
        perturbed = gs.factory().f_player1(
            lambda p: p.factory().deck_cards(reversed_deck).build()
        ).build()
        self.assertEqual(base, _obs(perturbed), "自己的牌库顺序进了观察（B12）")

    def test_opponent_deck_order_not_encoded(self):
        gs = _decision_state()
        base = _obs(gs)
        reversed_deck = OrderedCards(list(gs.player2.deck_cards.cards)[::-1])
        perturbed = gs.factory().f_player2(
            lambda p: p.factory().deck_cards(reversed_deck).build()
        ).build()
        self.assertEqual(base, _obs(perturbed), "对手牌库顺序进了观察（防护回归）")
