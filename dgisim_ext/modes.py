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

    def timeout_penalty(self) -> float:
        return self._TIMEOUT_PENALTY


class UpstreamCompatMode(ForkMode):
    """归档版行为：平局算 0。用于消融与复现。

    注：先手权（B5）经复核**归档版实现本来就是对的**（133/133 回合），
    因此这里没有对应的开关——那条"偏差"是评测时的测量错误。
    """

    _TIMEOUT_PENALTY = 0.0
