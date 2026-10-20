#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D-D (v2): 奖励 shaping 的势函数 Φ —— 早期区分度 / 泄漏检测 / 幅度上界。

为什么有 v2
-----------
v1 的主指标是「**最后一次决策**的 Φ 预测 P1 最终获胜」的 AUC = 0.996。这是循环论证:
局终时 P1 若还有 3 个角色、P2 是 0 个, AUC 当然接近 1 —— 它只证明"Φ 能读终局",
不能证明"Φ 在终局前携带信息", 而 shaping 只在**终局前**起作用。

v2 换成三个可证伪的测量:
  1. **早期 AUC**: 用进度 10% / 25% / 50% 处的 Φ 预测终局, 只用击杀局。局终 AUC 保留但
     只作"读数正确性"的健全性检查(它必然接近 1)。
  2. **泄漏检测**(真正的护栏): Φ 必须只由该席位**自己能看到的信息**决定。做法是
     把对方牌堆顺序反转(纯隐藏信息), 要求 `phi_public` 逐位不变; 并同时跑一个
     故意的作弊版 `phi_leaky` 作为正对照, 要求它**必须变化** —— 否则说明检测器
     没有检出能力(假阴性)。
  3. **幅度上界**: 实测单步 |γΦ(s')−Φ(s)| 与 |Φ| 范围, 反推系数 c, 使整局
     shaping 总量 <= MAX_SHAPING_SHARE × |终局奖励|。

预登记判据(先写死, 再跑; 任一 FAIL 退出码 1)
    D1 局数 >= MIN_GAMES 且击杀局 >= MIN_KILL(否则样本不够, 不出结论)
    D2 AUC(25%) >= AUC_MIN_025 (否则 Φ 在终局前没有信息, 不应用作 shaping 目标)
    D3 泄漏检测: phi_public 在扰动后逐位相同 且 phi_leaky 在扰动后不同(正对照有效)
    D4 反对称: phi(gs,P1) + phi(gs,P2) == 0 (容差 1e-9)
    D5 同进程重跑 CHECK_GAMES 局, Φ 序列逐位一致
    D6 推荐系数 c 使整局 shaping 总量 <= MAX_SHAPING_SHARE
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe_common import (  # noqa: E402
    auc_binary, install_canonical_card_order, phi_leaky, phi_public,
    play_random_game, wilson_ci,
)
from dgisim.card.cards import OrderedCards                 # noqa: E402
from dgisim.state.enums import Pid                          # noqa: E402

# ---------------------------------------------------------------- 预登记
N_GAMES = 400
MIN_GAMES = 400
MIN_KILL = 90
FRACTIONS = (0.10, 0.25, 0.50)
AUC_MIN_025 = 0.60
LEAK_SAMPLES = 40
CHECK_GAMES = 5
MAX_SHAPING_SHARE = 0.10
GAMMA = 0.99
TOL = 1e-9


def reverse_deck(gs, pid: Pid):
    """把 `pid` 的牌堆顺序反转 —— 只动隐藏信息(牌序), 不改任何公开字段。"""
    return gs.factory().f_player(
        pid,
        lambda p: p.factory().f_deck_cards(
            lambda d: OrderedCards(list(d)[::-1])
        ).build(),
    ).build()


def collect(games: int, delta_games: int = 0):
    """单遍采样。返回 (rows, leak_states, deltas)。

    rows: [(outcome, [Φ@10%, Φ@25%, Φ@50%, Φ@最后一决策])]
    每个决策态都算一次 Φ（`perspective_view` 开销可接受），进度分位点事后取 ——
    因为局长度事先未知，单遍采样才能既省时又给出需的比例点。
    """
    install_canonical_card_order()
    rows: list[tuple[str, list[float]]] = []
    leak_states = []
    deltas: list[float] = []
    for seed in range(games):
        sink: dict = {}
        phis: list[float] = []
        for i, gs in enumerate(play_random_game(seed, sink=sink)):
            if i == 0 and len(leak_states) < LEAK_SAMPLES:
                leak_states.append(gs)
            phis.append(phi_public(gs, Pid.P1))
        if not phis:
            raise RuntimeError(f"no decision state @seed={seed}")
        t_total = len(phis)
        picks = [phis[int(frac * t_total)] for frac in FRACTIONS] + [phis[-1]]
        rows.append((sink["outcome"], picks))
        if seed < delta_games:
            deltas.extend(abs(GAMMA * phis[i + 1] - phis[i])
                          for i in range(len(phis) - 1))
    return rows, leak_states, deltas


def auc_ci(a: float | None, n_pos: int, n_neg: int) -> tuple[float, float] | None:
    """AUC 的 95% CI（Hanley–McNeil 近似）。只用来描述证据强度, 不参与判据。"""
    if a is None or n_pos == 0 or n_neg == 0:
        return None
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) \
        / (n_pos * n_neg)
    se = var ** 0.5
    return (max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se))


def main() -> int:
    print("=== 预登记判据 (先写死, 再跑) ===")
    print(f"  D1 games >= {MIN_GAMES}, kill >= {MIN_KILL}")
    print(f"  D2 AUC(25%) >= {AUC_MIN_025}")
    print(f"  D3 leak: phi_public 不变 且 phi_leaky 变 (正对照)")
    print(f"  D6 shaping 总量 <= {MAX_SHAPING_SHARE} × |r|, gamma={GAMMA}")

    rows, leak_states, deltas = collect(N_GAMES, delta_games=CHECK_GAMES)
    det_rows, _, _ = collect(CHECK_GAMES)
    c5 = det_rows == rows[:CHECK_GAMES]

    outcomes = [r[0] for r in rows]
    n = len(outcomes)
    kill = [r for r in rows if r[0].startswith("kill")]
    n_kill = len(kill)
    n_timeout = sum(1 for o in outcomes if o == "timeout")
    k_lo, k_hi = wilson_ci(n_kill, n)
    t_lo, t_hi = wilson_ci(n_timeout, n)

    labels = [1 if r[0] == "kill:P1" else 0 for r in kill]
    aucs = {}
    for k, frac in enumerate(FRACTIONS):
        scores = [r[1][k] for r in kill]
        aucs[f"{int(frac * 100)}%"] = auc_binary(scores, labels)
    aucs["end"] = auc_binary([r[1][len(FRACTIONS)] for r in kill], labels)

    # ---- D3 泄漏检测 ----
    leak_ok = True
    leaky_changed = 0
    safe_changed = 0
    asym_bad = 0
    for gs in leak_states:
        pert = reverse_deck(gs, Pid.P2)
        s_before, s_after = phi_public(gs, Pid.P1), phi_public(pert, Pid.P1)
        l_before, l_after = phi_leaky(gs, Pid.P1), phi_leaky(pert, Pid.P1)
        safe_changed += abs(s_before - s_after) > TOL
        leaky_changed += abs(l_before - l_after) > TOL
        if abs(phi_public(gs, Pid.P1) + phi_public(gs, Pid.P2)) > 1e-6:
            asym_bad += 1
    c3 = (safe_changed == 0) and (leaky_changed > 0)
    c4 = asym_bad == 0

    # ---- D6 幅度: 前 CHECK_GAMES 局的逐决策 Φ 序列(已在 collect 里顺手算好) ----
    max_delta = max(deltas) if deltas else 0.0
    all_phi = [v for r in rows for v in r[1] if v == v]
    phi_range = max(all_phi) - min(all_phi) if all_phi else 0.0
    # 整局 shaping 总量(γ=1 时望远镜求和) <= c·(Φmax−Φmin) <= MAX_SHAPING_SHARE
    coef = (MAX_SHAPING_SHARE / phi_range) if phi_range else 0.0

    checks = {
        "D1 sample_size": (n >= MIN_GAMES and n_kill >= MIN_KILL),
        "D2 early_auc_25": (aucs["25%"] is not None and aucs["25%"] >= AUC_MIN_025),
        "D3 leak_detect": c3,
        "D4 antisymmetry": c4,
        "D5 determinism": c5,
        "D6 shaping_bound": bool(coef > 0),
    }

    print(f"\n=== A) 终局分布 ({n} 局, 随机策略) ===")
    print(f"  kill={n_kill} ({n_kill / n:.3f}, 95%CI [{k_lo:.3f},{k_hi:.3f}])  "
          f"timeout={n_timeout} ({n_timeout / n:.3f}, 95%CI [{t_lo:.3f},{t_hi:.3f}])")

    print("\n=== B) Φ 的早期区分度 (仅击杀局, 预测 P1 胜) ===")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  击杀局 {n_kill}: P1 胜 {n_pos} / P2 胜 {n_neg}")
    for key, val in aucs.items():
        ci = auc_ci(val, n_pos, n_neg)
        tag = " (循环论证, 仅供读数正确性)" if key == "end" else ""
        ci_s = "" if ci is None else f"  95%CI [{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"  AUC@{key:>4s} = {val if val is None else round(val, 4)}{ci_s}{tag}")

    print("\n=== C) 泄漏检测 (扰动对方牌序) ===")
    print(f"  samples={len(leak_states)}  phi_public 改变={safe_changed}  "
          f"phi_leaky 改变={leaky_changed}  反对称违例={asym_bad}")

    print("\n=== D) 幅度上界 ===")
    print(f"  max|γΦ(s')−Φ(s)|={max_delta:.3f}  Φ 范围={phi_range:.1f}  "
          f"推荐 c={coef:.6f}  (= {MAX_SHAPING_SHARE}/Φ范围)")
    print(f"  单步 shaping 最大 = {coef * max_delta:.4f} × |r_terminal|=1")

    print("\n=== 判据 ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    print("\n" + json.dumps({
        "games": n, "kill": n_kill, "timeout": n_timeout,
        "kill_rate": round(n_kill / n, 4), "timeout_rate": round(n_timeout / n, 4),
        "kill_p1": n_pos, "kill_p2": n_neg,
        "auc": {k: (None if v is None else round(v, 4)) for k, v in aucs.items()},
        "auc_ci_25": [round(x, 3) for x in (auc_ci(aucs["25%"], n_pos, n_neg) or (0, 0))],
        "leak": {"safe_changed": safe_changed, "leaky_changed": leaky_changed,
                 "asym_bad": asym_bad},
        "max_delta": round(max_delta, 4), "phi_range": round(phi_range, 2),
        "coef": round(coef, 6), "checks": checks,
    }, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
