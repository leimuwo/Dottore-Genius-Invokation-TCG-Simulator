import unittest

from src.dgisim.action.action import PlayerAction
from src.dgisim.env.linear_env import LinearEnv
from src.dgisim.state.game_state import GameState


class TestImportContract(unittest.TestCase):
    """A2：把导入契约钉死。

    本仓库现有两套导入约定：测试用 `src.dgisim.*`，安装后的包用 `dgisim.*`。
    这条测试只保证"仓库内开发路径"可用，并且公开面包含训练侧需要的名字。
    """

    def test_core_public_names(self):
        self.assertTrue(callable(getattr(LinearEnv, "step", None)))
        self.assertTrue(callable(getattr(LinearEnv, "view", None)))
        self.assertTrue(hasattr(GameState, "encoding"))
        self.assertTrue(hasattr(GameState, "waiting_for"))
        self.assertTrue(hasattr(PlayerAction, "decoding"))

    def test_env_bootstraps(self):
        env = LinearEnv()
        state, enc, reward, turn, done = env.view()
        self.assertIsInstance(enc, list)
        self.assertGreater(len(enc), 0)
        self.assertIn(turn, (0, 1, 2))
        self.assertIsInstance(done, bool)
