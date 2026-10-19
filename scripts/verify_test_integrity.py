"""D2：测试完整性。测试方法数与断言数只许增不许减。

首次运行会写出基线 scripts/test_baseline.json，之后按基线比较。

运行：python scripts/verify_test_integrity.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("test_baseline.json")
TESTS_DIR = REPO / "src" / "tests"

TEST_DEF = re.compile(r"^\s*def (test_\w+)\s*\(", re.M)
ASSERT = re.compile(r"\bself\.assert\w+\(|\bassertRaises\w*\(")


def measure() -> dict[str, int]:
    methods = asserts = files = 0
    for path in TESTS_DIR.rglob("*.py"):
        files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        methods += len(TEST_DEF.findall(text))
        asserts += len(ASSERT.findall(text))
    return {"files": files, "test_methods": methods, "assertions": asserts}


def main() -> int:
    current = measure()
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"已写入基线 {BASELINE.name}: {current}")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = [
        f"{key}: {current[key]} < 基线 {baseline[key]}"
        for key in ("test_methods", "assertions")
        if current[key] < baseline[key]
    ]
    if problems:
        for p in problems:
            print("FAIL", p)
        return 1
    print(f"PASS: {current}（基线 {baseline}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
