import io
import random
import unittest
from contextlib import redirect_stdout

from src.dgisim.action.action import CharacterSelectAction, ElementalTuningAction
from src.dgisim.agents import RandomAgent
from src.dgisim.element import Element
from src.dgisim.mode import DefaultMode
from src.dgisim.state.game_state import GameState


def _walk_to_phase(phase_name: str, seed: int = 7, max_steps: int = 400) -> GameState:
    random.seed(seed)
    agent = RandomAgent()
    gs = GameState.from_default()
    for _ in range(max_steps):
        if gs.game_end():
            break
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            continue
        if type(gs.phase).__name__ == phase_name:
            return gs
        gs = gs.action_step(w, agent.choose_action([gs], w))
    raise AssertionError(f"没有走到 {phase_name}")


class TestIllegalActionSemantics(unittest.TestCase):
    def test_invalid_char_id_raises_instead_of_silent_noop(self):
        gs = _walk_to_phase("StartingHandSelectPhase")
        pid = gs.waiting_for()
        with self.assertRaises(ValueError):
            gs.action_step(pid, CharacterSelectAction(char_id=99))

    def test_illegal_elemental_tuning_raises_and_prints_nothing(self):
        gs = _walk_to_phase("ActionPhase")
        pid = gs.waiting_for()
        hand = set(gs.get_player(pid).hand_cards.ordered_cards)
        pool = sorted(DefaultMode().all_cards(), key=lambda c: c.__name__)
        not_in_hand = next(c for c in pool if c not in hand)
        action = ElementalTuningAction(card=not_in_hand, dice_elem=Element.OMNI)
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(ValueError):
                gs.action_step(pid, action)
        self.assertEqual(buf.getvalue(), "", "非法调和把状态表打印到了 stdout（B9）")
