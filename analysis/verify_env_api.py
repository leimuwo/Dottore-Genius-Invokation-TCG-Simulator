#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 / T1 验收：`CardEnv` 的 A1–A4（D-A §4 预登记的四条）。

    A1 一局内 >=100 个 choice 阶段，每阶段 >=1 个候选；commit 必须真的推进状态
    A2 同 seed 两次运行（**两个独立进程**）的动作序列逐位一致
    A3 构造一次 `Obs` <= 1 ms
    A4 1e5 步 rollout <= 5 分钟（按实测步/秒外推）

策略用"按内容定序"的桩：对每个候选算 `sha1(label)`，取最大 —— 与候选在列表里的位置无关，
所以即使 `choices()` 顺序变了，动作也不会变（这正是 D-A 要求的"内容打分"的最小验证）。

    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' analysis\\verify_env_api.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgisim_ext.env_api import CardEnv, KIND_COMMIT  # noqa: E402
from dgisim_ext.modes import ForkMode                # noqa: E402

SEEDS = (0, 1, 2)
MIN_STAGES = 100
OBS_BUDGET_MS = 1.0
ROLLOUT_BUDGET_S = 300.0
ROLLOUT_STEPS = 100_000
TIMED_STEPS = 2_000
TIMED_OBS = 300


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def pick(obs) -> int:
    """内容定序的确定性桩：与候选位置无关。"""
    return max(range(len(obs.choices)), key=lambda i: _h(obs.choices[i].label))


def play(seed: int, env: CardEnv | None = None):
    env = env or CardEnv(ForkMode())
    obs = env.reset(seed=seed)
    stages = 0
    commits = 0
    while not env.done:
        assert len(obs.choices) >= 1, f"阶段没有候选 @seed={seed} stage={stages}"
        i = pick(obs)
        was_commit = obs.choices[i].kind == KIND_COMMIT
        before = hash(env.state)
        out = env.step(i)
        stages += 1
        if was_commit:
            commits += 1
            assert hash(env.state) != before, f"commit 没有推进状态 @seed={seed} stage={stages}"
        obs = out.obs  # type: ignore[assignment]
        if out.done:
            return stages, commits, out.info
    raise AssertionError("循环退出但未 done")


def dump_actions() -> list[dict]:
    rows = []
    for seed in SEEDS:
        env = CardEnv(ForkMode())
        env.reset(seed=seed)
        n = 0
        while not env.done:
            obs = env.obs()
            env.step(pick(obs))
            n += 1
            if n > 5000:
                break
        rows.append({"seed": seed, "log": list(env.action_log)})
    return rows


def child_rows() -> list[dict]:
    out = ROOT / "analysis" / "_m1_child_actions.json"
    err = ROOT / "analysis" / "_m1_child_err.txt"
    for p in (out, err):
        if p.exists():
            p.unlink()
    with open(err, "w", encoding="utf-8") as ef:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dump", str(out)],
            stdout=subprocess.DEVNULL, stderr=ef,
        )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"子进程失败 rc={proc.returncode}: "
                           f"{err.read_text(encoding='utf-8')[:800]}")
    rows = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    err.unlink()
    return rows


def main() -> int:
    if "--dump" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--dump") + 1])
        path.write_text(json.dumps(dump_actions(), ensure_ascii=False), encoding="utf-8")
        return 0

    print("=== 预登记判据 ===")
    print(f"  A1 stages >= {MIN_STAGES} / 局，每阶段 >=1 候选，commit 必须推进状态")
    print(f"  A2 同 seed 两个进程的动作序列逐位一致（seeds={SEEDS}）")
    print(f"  A3 构造 Obs <= {OBS_BUDGET_MS} ms")
    print(f"  A4 1e5 步 <= {ROLLOUT_BUDGET_S:.0f} s（按实测步/秒外推）")

    # ---------------- A1 ----------------
    print("\n=== A1 闭环 ===")
    a1 = True
    for seed in SEEDS:
        stages, commits, info = play(seed)
        ok = stages >= MIN_STAGES
        a1 &= ok
        print(f"  seed={seed} stages={stages} commits={commits} "
              f"reason={info.get('terminal_reason')} rewards={info.get('rewards')} "
              f"{'PASS' if ok else 'FAIL'}")

    # ---------------- A2 ----------------
    here = dump_actions()
    there = child_rows()
    a2 = here == there
    print("\n=== A2 跨进程动作序列 ===")
    for a, b in zip(here, there):
        same = a["log"] == b["log"]
        print(f"  seed={a['seed']} 步数={len(a['log'])} 子进程={len(b['log'])} "
              f"{'相同' if same else '不同'}")
        if not same:
            for i, (x, y) in enumerate(zip(a["log"], b["log"])):
                if x != y:
                    print(f"    首个分歧 idx={i}: {x} vs {y}")
                    break
    print(f"  {'PASS' if a2 else 'FAIL'}")

    # ---------------- A3 ----------------
    env = CardEnv(ForkMode())
    env.reset(seed=0)
    for _ in range(20):
        env.step(pick(env.obs()))
    t0 = time.perf_counter()
    for _ in range(TIMED_OBS):
        env.obs()
    per_obs_ms = (time.perf_counter() - t0) * 1000 / TIMED_OBS
    a3 = per_obs_ms <= OBS_BUDGET_MS
    print(f"\n=== A3 Obs 构造 ===")
    print(f"  {per_obs_ms:.3f} ms/次（{TIMED_OBS} 次） {'PASS' if a3 else 'FAIL'}")
    sample = env.obs()
    print(f"  Obs: enc_len={len(sample.enc)} choices={len(sample.choices)} "
          f"filled={sample.filled} labels={[c.label for c in sample.choices][:5]}")

    # ---------------- A4 ----------------
    env = CardEnv(ForkMode())
    env.reset(seed=0)
    steps = 0
    t0 = time.perf_counter()
    while steps < TIMED_STEPS:
        obs = env.obs()
        env.step(pick(obs))
        steps += 1
        if env.done:
            env.reset(seed=steps)
    dt = time.perf_counter() - t0
    rate = steps / dt
    est = ROLLOUT_STEPS / rate
    a4 = est <= ROLLOUT_BUDGET_S
    print(f"\n=== A4 rollout 吞吐 ===")
    print(f"  {rate:.0f} 步/秒 → 1e5 步 ≈ {est:.1f} s {'PASS' if a4 else 'FAIL'}")

    checks = {"A1 closed_loop": a1, "A2 cross_process_actions": a2,
              "A3 obs_budget": a3, "A4 rollout_budget": a4}
    print("\n=== 判据 ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\n" + json.dumps({"obs_ms": round(per_obs_ms, 3), "steps_per_s": round(rate, 1),
                             "est_1e5_s": round(est, 1), "checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
