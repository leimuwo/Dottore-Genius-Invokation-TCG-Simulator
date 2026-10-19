import random
import unittest

from src.dgisim.mode import DefaultMode
from src.dgisim.state.player_state import PlayerState


class TestSamplingDeterminism(unittest.TestCase):
    def test_chars_come_from_name_sorted_pool(self):
        mode = DefaultMode()
        random.seed(11)
        expected = frozenset(
            c.__name__
            for c in random.sample(sorted(mode.all_chars(), key=lambda c: c.__name__), k=3)
        )
        random.seed(11)
        actual = frozenset(type(c).__name__ for c in PlayerState.example_player(mode).characters)
        self.assertEqual(expected, actual, "抽样结果依赖 frozenset 迭代顺序")
