import random
import unittest

from src.dgisim.agents import RandomAgent
from src.dgisim.deck import MutableDeck
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.mode import DefaultMode
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
        return gs
        gs = gs.action_step(w, agent.choose_action([gs], w))
    raise AssertionError("没有走到决策点")


def _obs(gs: GameState) -> list[int]:
    return gs.perspective_view(Pid.P1).encoding(encoding_plan, Pid.P1)


class TestDecklistHidden(unittest.TestCase):
    def test_opponent_initial_deck_not_observed(self):
        gs = _decision_state()
        base = _obs(gs)
        pool = sorted(DefaultMode().all_cards(), key=lambda c: c.__name__)
        alt, other = pool[0], pool[1]
        perturbed = gs.factory().f_player2(
            lambda p: p.factory().initial_deck(
                MutableDeck(
                    chars=list(p.initial_deck.chars),
                    cards={alt: 15, other: 15},
                ).to_frozen()
            ).build()
        ).build()
        self.assertEqual(base, _obs(perturbed), "对手完整卡组表进了观察（B3）")

    def test_initial_deck_encoding_is_canonical(self):
        mode = DefaultMode()
        pool = sorted(mode.all_cards(), key=lambda c: c.__name__)
        chars = list(mode.all_chars())[:3]
        a, b = pool[0], pool[1]
        d1 = MutableDeck(chars=list(chars), cards={a: 2, b: 1}).to_frozen()
        d2 = MutableDeck(chars=list(chars), cards={b: 1, a: 2}).to_frozen()
        self.assertEqual(
            d1.encoding(encoding_plan),
            d2.encoding(encoding_plan),
            "Deck 编码依赖插入顺序（B13b）",
        )
