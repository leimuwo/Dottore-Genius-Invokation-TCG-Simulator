"""按验收契约机械判定一条改动（红 → 绿）。护栏在这里，不在 prompt 里。

用法：
  python scripts/verify_item.py B14 --phase red
  python scripts/verify_item.py B14 --phase green

退出码：0 = 该阶段契约满足；1 = 违约；2 = 用法错误。
设计期常量：ITEMS 里每条的引擎文件、测试文件、测试数、红期望、全量基线。
执行期只能调用它，不能改它（D4 CODEOWNERS 保护）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = r"C:\Users\wenxi\miniconda3\python.exe"
BUDGET_FILE = Path(__file__).resolve().parents[1] / ".verify_budget.json"
BASELINE_TESTS = 601   # B14(2) + B11(3) + B1(1) 之后的基线；只许增不许减
MAX_RUNS_PER_ITEM = 6          # 红+绿合计，防止"重试直到撞对"
CMD_TIMEOUT = 300

ITEMS: dict[str, dict] = {
    "B14": {
        "engine": ["src/dgisim/dice.py"],
        "test": "src/tests/test_invariant_encoding_idempotent.py",
        "module": "src.tests.test_invariant_encoding_idempotent",
        "n_tests": 2,
        "red_needle": "AssertionError",
    },
    "B11": {
        "engine": ["src/dgisim/state/game_state.py"],
        "test": "src/tests/test_invariant_state_identity.py",
        "module": "src.tests.test_invariant_state_identity",
        "n_tests": 3,
        "red_needle": "AssertionError",
    },
    "B1": {
        "engine": ["src/dgisim/state/player_state.py"],
        "test": "src/tests/test_invariant_sampling_determinism.py",
        "module": "src.tests.test_invariant_sampling_determinism",
        "n_tests": 1,
        "red_needle": "AssertionError",
    },
    "B12": {
        "engine": ["src/dgisim/card/cards.py"],
        "test": "src/tests/test_invariant_deck_order_not_observed.py",
        "module": "src.tests.test_invariant_deck_order_not_observed",
        "n_tests": 2,
        "red_needle": "AssertionError",
    },
    "B13": {
        "engine": ["src/dgisim/card/cards.py", "src/dgisim/dice.py"],
        "test": "src/tests/test_invariant_encoding_canonical.py",
        "module": "src.tests.test_invariant_encoding_canonical",
        "n_tests": 3,
        "red_needle": "AssertionError",
    },
    "E1": {
        "engine": ["src/dgisim/env/linear_env.py"],
        "test": "src/tests/test_invariant_linear_env_contract.py",
        "module": "src.tests.test_invariant_linear_env_contract",
        "n_tests": 2,
        "red_needle": "AssertionError",
    },
    "E2": {
        "engine": ["src/dgisim/phase/default/starting_hand_select_phase.py",
                   "src/dgisim/phase/default/action_phase.py"],
        "test": "src/tests/test_invariant_illegal_action_semantics.py",
        "module": "src.tests.test_invariant_illegal_action_semantics",
        "n_tests": 2,
        "red_needle": "AssertionError",
    },
    "E3": {
        "engine": ["src/dgisim/state/player_state.py"],
        "test": "src/tests/test_invariant_decklist_hidden.py",
        "module": "src.tests.test_invariant_decklist_hidden",
        "n_tests": 2,
        "red_needle": "AssertionError",
    },
}

ALL_TEST_FILES = {spec["test"] for spec in ITEMS.values()}

# 设计期声明的批次：同一批内的条目允许引擎改动同时在飞（仍然逐条提交、逐条验收）。
BATCHES: dict[str, list[str]] = {
    "ENGINE-A": ["E1", "E2", "E3"],
}


def run(args: list[str], timeout: int = CMD_TIMEOUT) -> tuple[int, str]:
    proc = subprocess.run(
        args, cwd=REPO, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def dirty_paths() -> set[str]:
    rc, out = run(["git", "status", "--porcelain"])
    if rc != 0:
        return {"<git-status-failed>"}
    paths = set()
    for line in out.splitlines():
        if len(line) > 3:
            paths.add(line[3:].strip().strip('"').replace("\\", "/"))
    return paths


def bump_budget(item: str, phase: str) -> int:
    data = {}
    if BUDGET_FILE.exists():
        try:
            data = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    key = f"{item}:{phase}"
    data[key] = data.get(key, 0) + 1
    BUDGET_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data[key]


def ran_count(out: str) -> int | None:
    m = re.search(r"Ran (\d+) tests?", out)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=sorted(ITEMS))
    ap.add_argument("--phase", choices=["red", "green"], required=True)
    ap.add_argument("--skip-suite", action="store_true", help="跳过全量回归（仅调试用，CI 禁止）")
    ap.add_argument("--batch", choices=sorted(BATCHES), default=None,
                    help="按设计期声明的批次放行引擎文件白名单（红阶段仍严格）")
    ns = ap.parse_args()

    spec = ITEMS[ns.item]
    verdict = {"item": ns.item, "phase": ns.phase, "checks": {}, "ok": False, "reason": ""}

    def fail(reason: str) -> int:
        verdict["reason"] = reason
        print(json.dumps(verdict, ensure_ascii=False))
        return 1

    runs = bump_budget(ns.item, ns.phase)
    verdict["runs_this_phase"] = runs
    if runs > MAX_RUNS_PER_ITEM:
        return fail(f"超出预算：{ns.item} {ns.phase} 已跑 {runs} 次（上限 {MAX_RUNS_PER_ITEM}）")

    # ── 护栏 1：改动范围白名单 ──────────────────────────────
    # 引擎文件按 item 严格限定；新增的不变量测试文件互相之间允许同时在飞
    # （它们只是设计期产物，不影响当前 item 的语义）。
    engine_allowed = set(spec["engine"])
    if ns.batch:
        for other in BATCHES[ns.batch]:
            engine_allowed |= set(ITEMS[other]["engine"])
    allowed = set(ALL_TEST_FILES) | (set() if ns.phase == "red" else engine_allowed)
    dirty = dirty_paths()
    extra = dirty - allowed
    verdict["checks"]["whitelist"] = {"dirty": sorted(dirty), "allowed": sorted(allowed)}
    if extra:
        return fail(f"改到了白名单外的文件：{sorted(extra)}")

    # ── 护栏 2：红阶段要求引擎文件未被修改 ───────────────────
    if ns.phase == "red" and set(spec["engine"]) & dirty & allowed:
        return fail(f"红阶段不允许已经改了引擎文件：{spec['engine']}")

    # ── 护栏 3：跑该条的唯一命令，按输出判定，不靠人读日志 ────
    rc, out = run([PY, "-m", "unittest", spec["module"], "-v"])
    n = ran_count(out)
    verdict["checks"]["target"] = {"rc": rc, "ran": n, "tail": out.strip().splitlines()[-3:]}

    if ns.phase == "red":
        if rc == 0:
            return fail("红阶段测试却通过了 → 这条测试无效（测不出该 bug）")
        if spec["red_needle"] not in out:
            return fail(f"红阶段失败但报错不含 {spec['red_needle']} → 是环境/导入错误，不是有效红")
        verdict["ok"] = True
        print(json.dumps(verdict, ensure_ascii=False))
        return 0

    # green
    if rc != 0 or "OK" not in out:
        return fail("绿阶段测试未通过")
    if n != spec["n_tests"]:
        return fail(f"绿阶段跑了 {n} 个测试，期望 {spec['n_tests']}")

    if not ns.skip_suite:
        # ── 护栏 4：全量回归，基线不许下降 ────────────────────
        env_args = ["-m", "unittest", "discover", "src/tests"]
        rc2, out2 = run([PY, *env_args])
        n2 = ran_count(out2)
        verdict["checks"]["suite"] = {"rc": rc2, "ran": n2, "tail": out2.strip().splitlines()[-2:]}
        if rc2 != 0 or "OK" not in out2:
            return fail("全量回归未通过")
        if n2 is None or n2 < BASELINE_TESTS:
            return fail(f"全量测试数 {n2} < 基线 {BASELINE_TESTS} → 有测试被删/被跳过")

    verdict["ok"] = True
    print(json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
