"""B4 + C8：per-seat 收益矩阵。

规则（见 deviations.md D2）：
- 有人在回合上限内被打死：`{胜者:+1, 败者:-1}`
- 打到 15 回合无人阵亡：**双方都判负** → `{P1:−1, P2:−1}`

注意这与 `LinearEnv.real_reward` 的 P1 中心标量**不兼容**：
"双方都输"无法用单个标量表示，因此本模块返回的是二元组。
训练管线禁止再写 `r_p2 = -r_p1`。
"""
from __future__ import annotations

from enum import Enum

from src.dgisim.state.enums import Pid
from src.dgisim.state.game_state import GameState

__all__ = ["TerminalReason", "per_seat_payoff", "payoff_of", "classify_terminal"]


class TerminalReason(Enum):
    KILL = "kill"
    TIMEOUT = "timeout"


def classify_terminal(state: GameState) -> TerminalReason | None:
    """未结束返回 None。"""
    if not state.game_end():
        return None
    return TerminalReason.TIMEOUT if state.get_winner() is None else TerminalReason.KILL


def per_seat_payoff(
    reason: TerminalReason,
    winner: Pid | None,
    timeout_penalty: float = -1.0,
) -> dict[Pid, float]:
    if reason is TerminalReason.TIMEOUT:
        return {Pid.P1: timeout_penalty, Pid.P2: timeout_penalty}
    if winner is None:  # pragma: no cover - 防御
        raise ValueError("KILL 终局必须有胜者")
    loser = winner.other
    return {winner: 1.0, loser: -1.0}


def payoff_of(state: GameState, timeout_penalty: float = -1.0) -> dict[Pid, float]:
    reason = classify_terminal(state)
    if reason is None:
        raise ValueError("对局尚未结束")
    return per_seat_payoff(reason, state.get_winner(), timeout_penalty)
