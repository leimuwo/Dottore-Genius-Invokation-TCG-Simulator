"""E7：从「无序容器」抽样的池子必须规范化，否则同 seed 的对局跨进程不可复现。

实测症状：`ActualDice.from_random()` 走的是
`random.choice(tuple(ActualDice._LEGAL_ELEMS - excepted_elems))`，
而 `_LEGAL_ELEMS` 是 `set` —— 迭代顺序实测为
`ANEMO, OMNI, PYRO, CRYO, DENDRO, ELECTRO, HYDRO, GEO`，
按 value 排序应为 `OMNI, PYRO, HYDRO, ANEMO, ...`。
于是同一个 seed 在两个进程里拿到不同的 8 颗骰子，对局分叉，obs 不再逐位相同
（`analysis/probe_obs_vs_t1.py` 的 X1b：首个分歧 seed=0 idx=4，差的正是骰子段）。

本文件钉两件事：
1. 抽样池的契约：`canonical_elements` / `canonical_types` 必须给出稳定顺序；
2. 代码里**不许再出现** `random.choice(tuple(...))` / `random.sample(tuple(...))`
   这种"直接对无序容器抽样"的写法（源码扫描，防止以后有人写回来）。
"""
import random
import unittest
from collections import Counter
from pathlib import Path

from src.dgisim.dice import ActualDice
from src.dgisim.element import PURE_ELEMENTS, Element
from src.dgisim.helper.canonical import canonical_elements, canonical_types
from src.dgisim.mode import DefaultMode

SRC = Path(__file__).resolve().parents[1]


def _nonzero(dice) -> dict:
    return {e: n for e, n in dice.readonly_dice_collection_ordered().items() if n}


class TestCanonicalPools(unittest.TestCase):
    def test_canonical_elements_sorted_by_value(self):
        pool = canonical_elements(ActualDice._LEGAL_ELEMS)
        self.assertEqual(pool, tuple(sorted(ActualDice._LEGAL_ELEMS, key=lambda e: e.value)))
        self.assertEqual([e.value for e in pool], sorted(e.value for e in pool))

    def test_canonical_types_sorted_by_name(self):
        pool = canonical_types(DefaultMode().all_cards())
        names = [c.__name__ for c in pool]
        self.assertEqual(names, sorted(names))

    def test_from_random_uses_canonical_pool(self):
        """端到端：同一个 seed 下，掷出的骰子必须等于「从规范池抽样」的结果。"""
        random.seed(0)
        got = _nonzero(ActualDice.from_random(8))
        random.seed(0)
        pool = canonical_elements(ActualDice._LEGAL_ELEMS)
        expected = dict(Counter(random.choice(pool) for _ in range(8)))
        self.assertEqual(got, expected, "from_random 没有从规范化的池子里抽样（E7）")

    def test_from_random_excludes_and_stays_canonical(self):
        random.seed(1)
        dice = ActualDice.from_random(8, excepted_elems={Element.PYRO, Element.CRYO})
        counts = _nonzero(dice)
        self.assertNotIn(Element.PYRO, counts)
        self.assertNotIn(Element.CRYO, counts)
        self.assertEqual(sum(counts.values()), 8)


class TestNoDrawFromUnorderedContainer(unittest.TestCase):
    """源码级护栏：这一类缺陷无法在同进程内从行为上测出来，只能禁掉写法本身。"""

    PATTERNS = ("random.choice(tuple(", "random.sample(tuple(",
                "random.choice(list(", "random.sample(list(")
    #: 这个文件自己的 docstring 要引用被禁的写法，跳过
    SKIP = {"dgisim/helper/canonical.py"}

    def test_source_has_no_tuple_of_unordered_container_draw(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel.startswith("tests/") or rel in self.SKIP:
                continue
            text = path.read_text(encoding="utf-8")
            for pat in self.PATTERNS:
                if pat in text:
                    offenders.append(f"{rel}: {pat}...)")
        self.assertEqual(
            offenders, [],
            "这些地方直接对无序容器抽样，会随进程得到不同结果（E7）：\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
