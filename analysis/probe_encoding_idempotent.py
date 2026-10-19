import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))
from dgisim.dice import ActualDice
from dgisim.element import Element
from dgisim.encoding.encoding_plan import encoding_plan as PLAN

d = ActualDice({Element.PYRO: 2, Element.HYDRO: 1, Element.ELECTRO: 3})
e1 = d.encoding(PLAN)
e2 = d.encoding(PLAN)
e3 = d.encoding(PLAN)
print("第一次编码:", e1[:10])
print("第二次编码:", e2[:10])
print("第三次编码:", e3[:10])
print("幂等吗:", e1 == e2 == e3)
print("ordered_dice 的类型:", type(ActualDice({Element.PYRO: 1}).ordered_dice).__name__)
dd = ActualDice({Element.PYRO: 1, Element.HYDRO: 1})
print("第一次 list(ordered_dice):", list(dd.ordered_dice))
print("第二次 list(ordered_dice):", list(dd.ordered_dice))
