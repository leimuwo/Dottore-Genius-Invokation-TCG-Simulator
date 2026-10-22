"""公开状态势函数，用于 CardEnv 的奖励塑形。"""
from __future__ import annotations

from src.dgisim.state.enums import Pid
from src.dgisim.state.game_state import GameState

__all__ = ["public_potential"]


def _player_value(player) -> float:
    alive = player.characters.get_alive_characters()
    return 10.0 * len(alive) + sum(character.hp for character in alive)


def public_potential(state: GameState, pid: Pid) -> float:
    """公开信息势函数：存活角色数和总血量差。"""
    view = state.perspective_view(pid)
    me, opponent = ((view.player1, view.player2)
                    if pid is Pid.P1 else (view.player2, view.player1))
    return _player_value(me) - _player_value(opponent)
