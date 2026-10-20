"""D3：引擎 diff 预算 + 提交前缀检查。

两条硬约束：
1. 相对基线 `fork-base-fd39ddc`，`src/dgisim/**` 的净改动行数不超过预算；
2. 任何碰过 `src/dgisim/**` 的 commit，message 必须以 `enginefix(` 或 `test(` 开头。

运行：python scripts/verify_engine_diff_budget.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = "fork-base-fd39ddc"
ENGINE_PATH = "src/dgisim"
LINE_BUDGET = 200          # 整个分支相对基线的净改动上限
ALLOWED_PREFIXES = ("enginefix(", "test(")


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def main() -> int:
    try:
        git("rev-parse", "--verify", BASE)
    except RuntimeError:
        print(f"SKIP: 找不到基线 tag {BASE}")
        return 0

    # 净改动（相对基线）
    numstat = git("diff", "--numstat", f"{BASE}..HEAD", "--", ENGINE_PATH)
    added = deleted = 0
    files: list[str] = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, d, path = parts[0], parts[1], parts[2]
        files.append(path)
        added += int(a) if a.isdigit() else 0
        deleted += int(d) if d.isdigit() else 0
    net = added + deleted

    # 每个碰过引擎的 commit 的前缀
    commits = [c for c in git("log", "--format=%H", f"{BASE}..HEAD", "--no-merges").splitlines() if c]
    bad: list[str] = []
    engine_commits = 0
    for c in commits:
        touched = git("show", "--name-only", "--format=", c, "--", ENGINE_PATH).strip()
        if not touched:
            continue
        engine_commits += 1
        subject = git("log", "-1", "--format=%s", c).strip()
        if not subject.startswith(ALLOWED_PREFIXES):
            bad.append(f"{c[:8]} {subject}")

    print(f"引擎文件 {len(files)} 个；净改动 {net} 行（+{added}/-{deleted}）；"
          f"预算 {LINE_BUDGET}；碰引擎的 commit {engine_commits} 个")
    problems = []
    if net > LINE_BUDGET:
        problems.append(f"引擎净改动 {net} 行超过预算 {LINE_BUDGET}")
    if bad:
        problems.append("以下 commit 改了引擎但前缀不是 enginefix(/test(：\n  " + "\n  ".join(bad))
    if problems:
        print("FAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("PASS: 引擎 diff 与提交前缀都在预算内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
