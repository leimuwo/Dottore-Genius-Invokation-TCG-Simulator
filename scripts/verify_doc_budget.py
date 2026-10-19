"""D5：文档预算。任何 *.md 超过上限即失败（白名单目录除外）。

上限 8 KB。dev_docs/ 与 analysis/ 属于上游/证据材料，列入白名单。

运行：python scripts/verify_doc_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIMIT = 8 * 1024
WHITELIST_DIRS = {"dev_docs", "analysis", ".git", "node_modules", "src"}
WHITELIST_FILES = {"CHANGELOG.md"}


def main() -> int:
    offenders = []
    for path in REPO.rglob("*.md"):
        rel = path.relative_to(REPO)
        if set(rel.parts[:-1]) & WHITELIST_DIRS:
            continue
        if rel.name in WHITELIST_FILES:
            continue
        size = path.stat().st_size
        if size > LIMIT:
            offenders.append((str(rel), size))
    for name, size in sorted(offenders, key=lambda x: -x[1]):
        print(f"FAIL {name}: {size} bytes > {LIMIT}")
    if offenders:
        return 1
    print(f"PASS: 所有受管 markdown 均 ≤ {LIMIT} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
