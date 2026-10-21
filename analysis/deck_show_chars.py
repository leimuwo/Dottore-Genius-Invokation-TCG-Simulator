"""打印候选角色的元素/武器类型，用于选武器牌。"""
import sys

sys.path.insert(0, "src")

from dgisim.mode import DefaultMode

mode = DefaultMode()
by = {c.__name__: c for c in mode.all_chars()}
names = [
    "Venti", "RhodeiaOfLoch", "Fischl", "RaidenShogun", "Shenhe", "Chongyun",
    "Mona", "Nahida", "AratakiItto", "StonehideLawachurl", "Noelle", "Bennett",
    "FatuiPyroAgent", "Klee", "Albedo", "Wanderer", "Eula", "Keqing",
]
for name in names:
    c = by[name]
    elem = getattr(c, "ELEMENT", None)
    weapon = getattr(c, "WEAPON_TYPE", None)
    print(f"{name:22s} element={getattr(elem, 'name', elem)!s:10s} "
          f"weapon={getattr(weapon, 'name', weapon)}")
