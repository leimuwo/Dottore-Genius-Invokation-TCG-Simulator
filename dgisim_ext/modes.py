"""C5：把"与归档版不同的规则口径"集中成一个 Mode 子类。

理由不是兼容上游（已 archive），而是**论文要能回答"结论是否依赖规则改动"**：
`UpstreamCompatMode` 必须能完整复现归档版行为，用于消融。
"""
from __future__ import annotations

from src.dgisim.mode import DefaultMode

__all__ = ["ForkMode", "UpstreamCompatMode"]


class ForkMode(DefaultMode):
    """本 fork 的默认规则口径。"""

    #: B4：15 回合耗尽 → 双方都判负，各自拿这个值。
    _TIMEOUT_PENALTY: float = -1.0
    #: D5：下回合先手 = 先结束回合的一方（契约位；引擎侧尚未接线）。
    _FIRST_TO_END_GETS_INITIATIVE: bool = True

    def timeout_penalty(self) -> float:
        return self._TIMEOUT_PENALTY

    def first_to_end_gets_initiative(self) -> bool:
        return self._FIRST_TO_END_GETS_INITIATIVE


class UpstreamCompatMode(ForkMode):
    """归档版行为：平局算 0，先手权保持归档版实现。用于消融与复现。"""

    _TIMEOUT_PENALTY = 0.0
    _FIRST_TO_END_GETS_INITIATIVE = False
