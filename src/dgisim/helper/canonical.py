"""把「无序容器」规范化成确定性顺序，供随机抽样使用（enginefix E7）。

为什么需要它
------------
`set` / `frozenset` 的迭代顺序取决于元素的 hash，而**枚举成员**与**类对象**的 hash 与
进程内 `id` 相关。于是 `random.choice(tuple(some_set))` 会把同一个随机数映射到不同元素：
同一个 seed 在两次运行里得到不同的骰子/元素，对局自此分叉，obs 也不再逐位相同。

规范化的做法是「先排序、再抽样」——抽样分布不变（仍是均匀），但随机数的映射固定了。
判别顺序用**元素自身的稳定属性**，绝不用 hash：
    * 元素（`Element` 枚举）→ `.value`
    * 类（卡牌 / 角色 / summon / support）→ `__name__`

注意：排序键只影响"随机数落在哪个元素上"，不影响任何游戏规则。
"""
from __future__ import annotations

from typing import Iterable, TypeVar

__all__ = ["canonical_elements", "canonical_types"]

_ElemT = TypeVar("_ElemT")
_TypeT = TypeVar("_TypeT")


def canonical_elements(elems: Iterable[_ElemT]) -> tuple[_ElemT, ...]:
    """按 `Element.value` 排序的 tuple（对任何带 `.value` 的枚举都适用）。"""
    return tuple(sorted(elems, key=lambda e: getattr(e, "value")))


def canonical_types(types: Iterable[_TypeT]) -> tuple[_TypeT, ...]:
    """按类名排序的 tuple（卡牌/角色/summon/support 都是类）。"""
    return tuple(sorted(types, key=lambda t: getattr(t, "__name__")))
