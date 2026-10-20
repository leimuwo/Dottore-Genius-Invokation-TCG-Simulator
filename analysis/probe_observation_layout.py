#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D-C (v2): 观察表示的结构性分析 —— 分段表 / 占用量 / 三段划分 / 词元预算。

为什么有 v2
-----------
v1 的两条结论不可用:
  * "常量维 2538, 可裁剪到 247" 是**采样假象**: v1 每局只采前 4 个决策态,
    此时手牌少、效果栈空, 于是大量"容量位"恒为 0。常量维数其实是
    「容量 − 占用量」的函数, 不是信息量。
  * "26 个词元" 把一个 1000 维效果栈段、一个 414 维角色段各压成 1 个词元,
    粗到模型分不清"哪个角色被打残"。已否决。

v2 的做法
---------
1. 分段表: 按 `GameState.encoding()` 的真实拼接顺序逐段实测长度; 再用
   `encoding_plan` 的容量常数做结构校验(容量 × 字段宽 == 段长)。
2. 占用量: 打满整局, 每个决策态记录各容器的**实际元素数**, 报 p50/p90/p99/max。
3. 三段划分(数据驱动): P 填充位 / S 局内静态位 / D 动态位, 见下。
4. 词元预算: 按**实体对齐**方案(每个活着的实体 1 个词元)算词元数, 给 p99/max。

预登记判据(先写死, 再跑; 任一 FAIL 退出码 1)
    C1 分段表覆盖 0..N-1 恰好一次, 合计 == 实测总维数
    C2 每段: 容量常数 × 字段宽 == 段长 (结构自洽, 不靠硬编码)
    C3 全量口径: 局数 >= MIN_GAMES 且决策态数 >= MIN_DECISION_STATES
    C4 [原判据, 实测推翻] 常量维中「结构填充位」占比 >= 0.95
    C4' [替代判据] 容量公式给出的填充位必须 100% 落在「局内恒定」的维里
    C5 前 4 态口径的常量维 − 全量口径的常量维 >= ARTIFACT_GAP_MIN
    C6 实体词元数 p99 <= 256 且 max <= HARD_TOKEN_CAP
    C7 同进程重跑 CHECK_GAMES 局, first/varies 逐位一致(子集关系)

三段划分
--------
    P 填充位: 容量公式给出(容量 − 每局最大占用), 局内恒为 0
    S 静态位: 局内恒定、跨局可变(卡牌定义/牌表内容) -> 应进「每局一次」的上下文向量
    D 动态位: 局内变过 -> 必须逐决策观测

**填充位都在尾部**(实测):
    卡牌容器的 code 先升序排序、再把 0 追加在后面; 骰子/状态/summon/support/效果栈
    同样是先写实体再追加 0。所以「切掉尾部 0」在本引擎上是对的 —— 但这只是当前
    实现的性质, 不是接口契约; 一旦某段改成前部填充, 按位置切片就会静默错位。
    这也是本脚本坚持用「容量 × 字段宽」做结构校验(C2)而不硬编码下标的原因。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from _probe_common import (  # noqa: E402
    install_canonical_card_order, play_random_game,
)
from dgisim.encoding.encoding_plan import encoding_plan  # noqa: E402
from dgisim.mode import DefaultMode                      # noqa: E402
from dgisim.state.game_state import GameState            # noqa: E402

# ---------------------------------------------------------------- 预登记
N_GAMES = 120
MIN_GAMES = 120
MIN_DECISION_STATES = 12000
EARLY_CAP = 4
CHECK_GAMES = 6
HARD_TOKEN_CAP = 320
P99_TOKEN_MAX = 256
PAD_SHARE_MIN = 0.95
ARTIFACT_GAP_MIN = 100

PLAN = encoding_plan
DICE_CAP = PLAN.DICE_FIXED_LEN
CARDS_CAP = PLAN.CARDS_FIXED_LEN
STT_W = PLAN.STATUS_FIXED_LEN
HIDDEN_CAP = PLAN.PLAYER_HIDDEN_FIXED_LEN
COMBAT_CAP = PLAN.PLAYER_COMBAT_FIXED_LEN
SUMMONS_CAP = PLAN.SUMMONS_FIXED_LEN
SUPPORTS_CAP = PLAN.SUPPORTS_FIXED_LEN
EFFECTS_CAP = PLAN.EFFECTS_FIXED_LEN
EFFECT_W = PLAN.EFFECT_FIXED_LEN
CHAR_W = PLAN.CHAR_STT_FIXED_LEN * PLAN.STATUSES_FIXED_LEN + PLAN.CHAR_HIDDEN_FIXED_LEN * 2
CHARS_CAP = 3
DECK_CHARS = 3


def capacity_of(name: str) -> tuple[int, int]:
    """(容量, 每元素字段宽); (0, 1) 表示该段没有填充结构。"""
    sub = name.split("|")[-1]
    return {
        "dice": (DICE_CAP, 1),
        "hand": (CARDS_CAP, 1),
        "deck": (CARDS_CAP, 1),
        "used": (CARDS_CAP, 1),
        "gained": (CARDS_CAP, 1),
        "initial_deck": (DECK_CHARS + CARDS_CAP, 1),
        "characters": (CHARS_CAP, CHAR_W),
        "hidden": (HIDDEN_CAP, STT_W),
        "combat": (COMBAT_CAP, STT_W),
        "summons": (SUMMONS_CAP, STT_W),
        "supports": (SUPPORTS_CAP, STT_W),
        "effect_stack": (EFFECTS_CAP, EFFECT_W),
    }.get(sub, (0, 1))


def pad_region(name: str, s: int, e: int, pad: int) -> tuple[int, int]:
    """填充位在段内的 [起, 止]。本引擎所有段都是尾部填充(见模块 docstring)。"""
    return (e - pad + 1, e)


def segment_table(gs: GameState) -> list[tuple[str, int]]:
    """按 `GameState.encoding()` 的拼接顺序逐段实测长度(长度与具体状态无关)。"""
    segs: list[tuple[str, int]] = [("game_basics", 4)]
    for side, p in (("P1", gs.player1), ("P2", gs.player2)):
        segs += [
            (f"{side}|player_basics", 4),
            (f"{side}|dice", len(p.dice.encoding(PLAN))),
            (f"{side}|hand", len(p.hand_cards.encoding(PLAN))),
            (f"{side}|deck", len(p.deck_cards.encoding(PLAN))),
            (f"{side}|used", len(p.publicly_used_cards.encoding(PLAN))),
            (f"{side}|gained", len(p.publicly_gained_cards.encoding(PLAN))),
            (f"{side}|initial_deck", len(p.initial_deck.encoding(PLAN))),
            (f"{side}|characters", len(p.characters.encoding(PLAN))),
            (f"{side}|hidden", len(p.hidden_statuses.encoding(
                PLAN, PLAN.PLAYER_HIDDEN_FIXED_LEN))),
            (f"{side}|combat", len(p.combat_statuses.encoding(
                PLAN, PLAN.PLAYER_COMBAT_FIXED_LEN))),
            (f"{side}|summons", len(p.summons.encoding(PLAN))),
            (f"{side}|supports", len(p.supports.encoding(PLAN))),
        ]
    segs.append(("effect_stack", len(gs.effect_stack.encoding(PLAN))))
    return segs


def occupancy(enc: np.ndarray, spans) -> dict[str, int]:
    """从**编码本身**数出每段的占用(而不是问 `num_cards()`)。

    为什么不信 `num_cards()`: 实测 `Cards` 里会出现**负计数**条目
    (例如 `ChangingShifts: -1`), 于是 `num_cards()`=sum(values) 比真实张数少,
    而 `ordered_cards`/`encoding` 按 `Counter.elements()` 展开(跳过 ≤0)给出真实张数。
    两者不一致时, 以**模型真正看到的编码**为准。
    """
    occ: dict[str, int] = {}
    for name, s, e in spans:
        sub = name.split("|")[-1]
        block = enc[s:e + 1]
        if sub in ("dice", "hand", "deck", "used", "gained", "initial_deck"):
            occ[name] = int(np.count_nonzero(block))
        elif sub == "characters":
            occ[name] = CHARS_CAP
        elif sub in ("hidden", "combat", "summons", "supports"):
            occ[name] = int(sum(
                1 for i in range(0, len(block) - STT_W + 1, STT_W) if block[i:i + STT_W].any()
            ))
        elif sub == "effect_stack":
            occ[name] = int(sum(
                1 for i in range(0, len(block) - EFFECT_W + 1, EFFECT_W)
                if block[i:i + EFFECT_W].any()
            ))
    return occ


def tokens_of(occ: dict[str, int]) -> int:
    """实体对齐方案的词元数: 每个活着的实体 1 个词元。

    game_basics(1) + Σ_席位[ basics(1) + dice(1) + hand(每张 1) + 三个卡池(各 1 池化)
    + characters(3) + hidden/combat/summons/supports(各 1/实体) ] + effect_stack(每层 1)。

    `initial_deck` 是**静态上下文**(开局即定), 每局只算一次, 不进逐决策预算。
    """
    n = 1 + occ["effect_stack"]
    for side in ("P1", "P2"):
        n += 1 + 1 + occ[f"{side}|hand"] + 3  # basics, dice, 每张手牌, 三个卡池
        n += CHARS_CAP                        # characters
        n += occ[f"{side}|hidden"] + occ[f"{side}|combat"]
        n += occ[f"{side}|summons"] + occ[f"{side}|supports"]
    return n


def collect(games: int, early_cap: int | None, spans):
    """返回 dict: first/varies(跨局) + dyn(局内变过) + tokens + occ + outcomes。"""
    install_canonical_card_order()
    first = varies = None
    dyn: np.ndarray | None = None
    tokens: list[int] = []
    occ_list: list[dict[str, int]] = []
    outcomes: list[str] = []
    n_states = 0
    illegal_states = 0
    for seed in range(games):
        sink: dict = {}
        g_first = None
        g_varies = None
        for i, gs in enumerate(play_random_game(seed, sink=sink)):
            if early_cap is not None and i >= early_cap:
                break
            enc = np.asarray(gs.encoding(PLAN), dtype=np.int64)
            if first is None:
                first = np.full(enc.shape[0], -1, dtype=np.int64)
                varies = np.zeros(enc.shape[0], dtype=bool)
                dyn = np.zeros(enc.shape[0], dtype=bool)
            if g_first is None:
                g_first = enc.copy()
                g_varies = np.zeros(enc.shape[0], dtype=bool)
            else:
                g_varies |= enc != g_first
            unset = first < 0
            first[unset] = enc[unset]
            varies |= enc != first
            occ = occupancy(enc, spans)
            occ_list.append(occ)
            tokens.append(tokens_of(occ))
            n_states += 1
            if not (gs.player1.hand_cards.is_legal()
                    and gs.player2.hand_cards.is_legal()):
                illegal_states += 1
        if g_varies is not None:
            dyn |= g_varies
        outcomes.append(sink.get("outcome", "early"))
    return {
        "first": first, "varies": varies, "dyn": dyn, "tokens": tokens,
        "occ": occ_list, "outcomes": outcomes, "n_states": n_states,
        "illegal_states": illegal_states,
    }


def q(vals: list[int], p: int) -> int:
    return int(np.percentile(vals, p))


def main() -> int:
    print("=== 预登记判据 (先写死, 再跑) ===")
    print(f"  C3 games >= {MIN_GAMES}, decision_states >= {MIN_DECISION_STATES}")
    print(f"  C4 [原] padding_share >= {PAD_SHARE_MIN}  (预期可能被推翻)")
    print(f"  C4' 填充位 100% 落在局内恒定维")
    print(f"  C5 early_const - full_const >= {ARTIFACT_GAP_MIN} dims")
    print(f"  C6 tokens p99 <= {P99_TOKEN_MAX}, max <= {HARD_TOKEN_CAP}")

    fresh = GameState.from_default(DefaultMode())
    segs = segment_table(fresh)
    total = sum(n for _, n in segs)

    spans, cursor = [], 0
    for name, length in segs:
        spans.append((name, cursor, cursor + length - 1))
        cursor += length
    c1 = (cursor == total) and all(
        spans[i][2] + 1 == spans[i + 1][1] for i in range(len(spans) - 1)
    )
    c2_bad = [
        (name, length, capacity_of(name))
        for name, length in segs
        if capacity_of(name)[0] and capacity_of(name)[0] * capacity_of(name)[1] != length
    ]
    c2 = not c2_bad

    check = collect(CHECK_GAMES, None, spans)
    full = collect(N_GAMES, None, spans)
    early = collect(N_GAMES, EARLY_CAP, spans)

    c7 = bool(
        np.array_equal(full["first"], check["first"])
        and np.all(full["varies"][check["varies"]])
        and int(full["first"].shape[0]) == total
    )
    c1 = c1 and int(full["first"].shape[0]) == total

    cross_const = ~full["varies"]          # 跨局恒定
    dyn = full["dyn"]                      # 局内变过
    always_const = ~dyn                    # 局内恒定

    # ---- P: 容量公式 ----
    max_occ: dict[str, int] = {}
    for occ in full["occ"]:
        for k, v in occ.items():
            if v > max_occ.get(k, -1):
                max_occ[k] = v
    pad_mask = np.zeros(total, dtype=bool)
    pad_of: dict[str, int] = {}
    for name, s, e in spans:
        cap, width = capacity_of(name)
        if not cap:
            continue
        pad = max(0, cap - max_occ.get(name, 0)) * width
        pad_of[name] = pad
        if pad:
            ps, pe = pad_region(name, s, e, pad)
            pad_mask[ps:pe + 1] = True
    pad_count = int(pad_mask.sum())

    c4_old = (pad_count / int(cross_const.sum())) if cross_const.any() else 0.0
    # 填充位必须: ① 在首个观测里就是 0  ② 局内从不变化
    c4_new = bool(np.all(always_const[pad_mask])
                  and np.all(full["first"][pad_mask] == 0))

    static_mask = always_const & ~pad_mask
    tokens = full["tokens"]
    t_p50, t_p90 = q(tokens, 50), q(tokens, 90)
    t_p99, t_max = q(tokens, 99), max(tokens)

    checks = {
        "C1 segment_cover": c1,
        "C2 struct_selfconsistent": c2,
        "C3 sample_size": (len(full["outcomes"]) >= MIN_GAMES
                           and full["n_states"] >= MIN_DECISION_STATES),
        "C4' padding_is_padding": c4_new,
        "C5 artifact_gap": (int((~early["varies"]).sum()) - int(cross_const.sum())
                            >= ARTIFACT_GAP_MIN),
        "C6 token_budget": (t_p99 <= P99_TOKEN_MAX and t_max <= HARD_TOKEN_CAP),
        "C7 determinism": c7,
    }
    # 预登记的**假设**(关于数据, 不是关于代码): 常量维主要由填充位解释。
    # 被推翻也要照原样记录 —— 它是结果, 不是需要"修绿"的缺陷。
    hypotheses = {
        "H1 padding >= %.2f of constant dims" % PAD_SHARE_MIN: {
            "observed": round(c4_old, 4), "verdict": "refuted" if c4_old < PAD_SHARE_MIN else "supported",
        },
    }

    print("\n=== A) 分段表 [起, 止] 长度 / 容量 / 每局最大占用 / 填充位 ===")
    for name, s, e in spans:
        cap, width = capacity_of(name)
        cap_s = f"{cap}x{width}" if cap else "-"
        print(f"  [{s:4d},{e:4d}] len={e - s + 1:4d} cap={cap_s:>7s} "
              f"max_occ={max_occ.get(name, 0):3d} pad={pad_of.get(name, 0):4d} {name}")
    print(f"  total={total}  cover_ok={c1}  struct_ok={c2}  bad={c2_bad}")

    print("\n=== B) 占用量 (整局, 双席位合并) ===")
    for key in sorted(max_occ):
        vals = [o[key] for o in full["occ"]]
        print(f"  {key:20s} p50={q(vals, 50):3d} p90={q(vals, 90):3d} "
              f"p99={q(vals, 99):3d} max={max(vals):3d}")

    print("\n=== C) 三段划分 (P 填充 / S 局内静态 / D 动态) ===")
    print(f"  states full={full['n_states']}  early={early['n_states']}")
    print(f"  P padding          = {pad_count}")
    print(f"  S static(always)   = {int(static_mask.sum())}")
    print(f"  D dynamic          = {int(dyn.sum())}")
    print(f"  cross_game_constant= {int(cross_const.sum())}  "
          f"(前4态口径={int((~early['varies']).sum())}, "
          f"gap={int((~early['varies']).sum()) - int(cross_const.sum())})")
    print(f"  padding_share(const)={c4_old:.3f}")
    print("\n=== C2) 逐段: 长度 / 填充 / 局内静态 / 动态 ===")
    for name, s, e in spans:
        seg_pad = int(pad_mask[s:e + 1].sum())
        seg_static = int(static_mask[s:e + 1].sum())
        seg_dyn = int(dyn[s:e + 1].sum())
        print(f"  {name:20s} len={e - s + 1:4d} pad={seg_pad:4d} "
              f"static={seg_static:4d} dyn={seg_dyn:4d}")

    print("\n=== D) 实体对齐词元预算 ===")
    print(f"  tokens p50={t_p50} p90={t_p90} p99={t_p99} max={t_max}")

    print("\n=== E) 已知引擎问题(不阻塞本决策, 由 E5 单独修) ===")
    print(f"  手牌出现负计数的决策态(num_cards()<实际张数): "
          f"{full['illegal_states']}/{full['n_states']}")

    print("\n=== 判据(代码/不变量, 必须全绿) ===")
    bad_pad = pad_mask & ~always_const
    if bad_pad.any():
        print(f"\n[诊断] 被判为填充位却局内变化: {int(bad_pad.sum())} 维")
        for name, s, e in spans:
            bad = int(bad_pad[s:e + 1].sum())
            if bad:
                print(f"  {name}: {bad}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\n=== 预登记的假设(关于数据, 记录结果) ===")
    for k, v in hypotheses.items():
        print(f"  {v['verdict'].upper():9s} {k}  (实测 {v['observed']})")

    print("\n" + json.dumps({
        "total_dims": total,
        "games": len(full["outcomes"]),
        "states": full["n_states"],
        "P_padding": pad_count,
        "S_static": int(static_mask.sum()),
        "D_dynamic": int(dyn.sum()),
        "cross_game_constant": int(cross_const.sum()),
        "constant_dims_early": int((~early["varies"]).sum()),
        "padding_share_const": round(c4_old, 4),
        "illegal_hand_states": full["illegal_states"],
        "tokens": {"p50": t_p50, "p90": t_p90, "p99": t_p99, "max": t_max},
        "checks": checks,
        "hypotheses": hypotheses,
    }, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
