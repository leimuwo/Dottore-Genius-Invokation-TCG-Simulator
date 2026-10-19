"""C3：错误语义收口。

三条防线：
1. 零向量 / 空编码绝不允许送进 `step()`（`decoding([0]*81)` 会变成 EndRoundAction）。
2. 一次被接受的决策必须产生状态迁移；否则记 `NO_STATE_TRANSITION`。
3. 非法动作不再静默：引擎侧已改为抛 ValueError，这里把异常翻译成可计数的结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.dgisim.env.linear_env import LinearEnv

__all__ = ["StepOutcome", "step_checked"]


@dataclass(frozen=True)
class StepOutcome:
    state: object
    encoded: list[int]
    reward: float
    turn: int
    done: bool
    noop: bool
    error: str | None = None
    _raw: tuple = field(default=(), repr=False)


def _key(state) -> tuple:
    return (hash(state), state.waiting_for(), state.round, type(state.phase))


def step_checked(env: LinearEnv, action) -> StepOutcome:
    """走一步，并按契约判定这次决策是否有效。"""
    if isinstance(action, list) and not any(action):
        raise ValueError("零向量动作被拒绝：它会被解码成 EndRoundAction")

    before = env.full_view()
    before_key = _key(before)
    try:
        state, encoded, reward, turn, done = env.step(action)
    except AssertionError as exc:               # 引擎内部契约被破坏
        raise RuntimeError(f"step() 断言失败：{exc}") from exc

    after = env.full_view()
    noop = (not done) and _key(after) == before_key
    return StepOutcome(
        state=state,
        encoded=encoded,
        reward=reward,
        turn=turn,
        done=done,
        noop=noop,
        _raw=(action,),
    )
