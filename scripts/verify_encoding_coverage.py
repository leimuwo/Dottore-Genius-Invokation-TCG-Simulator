"""B10：编码映射覆盖校验。

检查 encoding/mappings.py 里的每个类都能被 encode_item 解析，且同一类别内码值唯一。
缺失/重复 → 退出码 1。

运行：python scripts/verify_encoding_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dgisim.encoding import mappings  # noqa: E402
from dgisim.encoding.encoding_plan import encoding_plan  # noqa: E402

CATEGORIES = ["MODE", "ENUM", "CHAR", "CARD", "EFFECT", "STT", "SUMM", "SUPP"]


def main() -> int:
    problems: list[str] = []
    for cat in CATEGORIES:
        mapping = getattr(mappings, f"{cat}_MAPPING", None)
        if mapping is None:
            problems.append(f"{cat}_MAPPING 不存在")
            continue
        values = list(mapping.values())
        if len(values) != len(set(values)):
            dup = sorted({v for v in values if values.count(v) > 1})
            problems.append(f"{cat} 码值重复：{dup[:8]}")
        bad = [item for item in mapping if encoding_plan.encode_item(item) != mapping[item]]
        if bad:
            problems.append(f"{cat} 有 {len(bad)} 个条目 encode_item 结果与映射表不一致")
        print(f"{cat:7s} {len(mapping):5d} 条，码值范围 [{min(values)}, {max(values)}]")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nPASS: 编码映射覆盖一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
