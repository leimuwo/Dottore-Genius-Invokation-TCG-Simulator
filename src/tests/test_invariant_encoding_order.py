"""E6：观察编码的排布必须由**声明顺序**决定，不能继承 frozenset 的迭代顺序。

实测症状：同一个 seed、同一个初始状态，两个独立进程算出的 `GameState.encoding()`
不相等（首个分歧落在每个角色段的 aura 子块，偏移 9/11/13/15/17）。
根因：`AURA_ELEMENTS` 是 `frozenset`，而 `Element` 是枚举 —— 枚举成员的 hash 与进程内
`id` 相关，于是 `ElementalAura.encoding()` 的元素顺序逐进程变化。

这条不是"风格问题"：T1 的验收判据是「同 seed 两进程 obs 逐位相同」，而观测直接来自
这个编码。编码不稳定 -> 观测不稳定 -> 训练出的策略与 checkpoint 不可复现。
"""
import unittest

from src.dgisim.element import AURA_ELEMENTS_ORDERED, Element, ElementalAura
from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.helper.hashable_dict import HashableDict


def _aura_with(*true_elems: Element, reverse: bool = False) -> ElementalAura:
    keys = list(AURA_ELEMENTS_ORDERED)
    if reverse:
        keys = list(reversed(keys))
    return ElementalAura(HashableDict((e, e in true_elems) for e in keys))


class TestAuraEncodingOrder(unittest.TestCase):
    def test_aura_encoding_follows_declared_order(self):
        plan = encoding_plan
        expected = []
        for elem in AURA_ELEMENTS_ORDERED:
            expected.extend([plan.encode_item(elem), 0])
        self.assertEqual(
            ElementalAura.from_default().encoding(plan),
            expected,
            "aura 编码没有按 AURA_ELEMENTS_ORDERED 排布（跨进程会给出不同的 obs）",
        )

    def test_aura_encoding_follows_declared_order_with_applied_elements(self):
        plan = encoding_plan
        aura = _aura_with(Element.PYRO, Element.CRYO)
        expected = []
        for elem in AURA_ELEMENTS_ORDERED:
            expected.extend([plan.encode_item(elem), 1 if elem in (Element.PYRO, Element.CRYO) else 0])
        self.assertEqual(aura.encoding(plan), expected)

    def test_aura_encoding_independent_of_insertion_order(self):
        plan = encoding_plan
        elems = (Element.HYDRO, Element.ELECTRO, Element.DENDRO)
        self.assertEqual(
            _aura_with(*elems).encoding(plan),
            _aura_with(*elems, reverse=True).encoding(plan),
            "aura 编码依赖 dict 插入顺序",
        )


if __name__ == "__main__":
    unittest.main()
