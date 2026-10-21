"""替代方案验证：北斗/雷泽缺失时，超导物理队还能不能组。"""
import importlib.util
import sys

sys.path.insert(0, "src")

spec = importlib.util.spec_from_file_location("deck_smoke", "analysis/deck_smoke.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

CASES = [
    ("重云+优菈+菲谢尔（超导物理）", ["Chongyun", "Eula", "Fischl"]),
    ("重云+刻晴+菲谢尔（超导反应）", ["Chongyun", "Keqing", "Fischl"]),
    ("重云+雷神+菲谢尔（充能超导）", ["Chongyun", "RaidenShogun", "Fischl"]),
]

for name, chars in CASES:
    deck, info = ds.build_deck(chars)
    if deck is None:
        print(f"[不可用] {name}: {info}")
        continue
    st, ar = ds.run(deck, 15)
    print(f"[可用] {name}: 异常={st['errors']} 空转={st['noop']} "
          f"决策/局={st['decisions'] / max(1, st['games']):.0f} "
          f"平均回合={ar:.1f} 平局={st['draws']}/{st['games']}")
