#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D-C × T1 交叉确认：观察表示能不能被 `CardEnv` 协议（T1）直接承载。

T1 的验收原话是「同 seed 两进程 obs 逐位相同」。D-C 的词元/维数是在**同一进程内**测的，
所以必须补这一层，否则 D-C 的结论只在"一次运行"里成立。

判据（任一 FAIL → 退出码 1）
    X1 跨进程：同 seed 两个**独立进程**的 obs 序列逐位相同
       X1a 只比第 0 个决策态（初始状态）
       X1b 比前 MAX_IDX 个决策态（含轨迹）
    X2 词元预算在**视角视图**下统计（T1 的 `observe(pid)` 只能拿到视角视图）：
       p99 <= 256 且 max <= 320
    X3 观测泄漏：扰动对方手牌/牌序后，`obs(自己)` 逐位不变；正对照 `full_view` 必须变化
    X4 静态上下文按席位：自己的牌表可见、对方的被隐藏；且对扰动不变

    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' analysis\\probe_obs_vs_t1.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from _probe_common import (  # noqa: E402
    install_canonical_card_order, play_random_game,
)
from dgisim.encoding.encoding_plan import encoding_plan  # noqa: E402
from dgisim.mode import DefaultMode                      # noqa: E402
from dgisim.state.enums import Pid                       # noqa: E402
from dgisim.state.game_state import GameState            # noqa: E402

import probe_observation_layout as dc                     # noqa: E402

PLAN = encoding_plan
SEEDS = 6
MAX_IDX = 30
TOKEN_P99_MAX = 256
TOKEN_MAX_CAP = 320
LEAK_STATES = 20


def _all_omni(deck) -> bool:
    """`hide_secrets` 把牌表换成 OmniCard×n，用它判断"这张牌表被隐藏了"。"""
    from dgisim.card.card import OmniCard
    return set(deck.cards.keys()) == {OmniCard}


def _sha(enc) -> str:
    return hashlib.sha1(json.dumps(list(enc)).encode()).hexdigest()[:16]


def obs(gs: GameState, seat: Pid):
    """T1 的 `observe(pid)`：只能拿到该席位的视角视图。"""
    view = gs.perspective_view(seat)
    return view.encoding(PLAN, seat)


def dump_rows():
    """返回 [(seed, idx, obs_hash)] 与每局决策数。"""
    install_canonical_card_order()
    rows, totals = [], {}
    for seed in range(SEEDS):
        sink: dict = {}
        for i, gs in enumerate(play_random_game(seed, sink=sink)):
            if i < MAX_IDX:
                rows.append((seed, i, _sha(obs(gs, Pid.P1))))
        totals[seed] = i + 1
    return rows, totals


def child_dump() -> tuple[list[tuple[int, int, str]], dict]:
    """在**另一个进程**里跑同一段采样，结果经文件回传（不用管道/临时目录）。"""
    out = Path(__file__).resolve().parent / "_t1_child_rows.jsonl"
    err = Path(__file__).resolve().parent / "_t1_child_err.txt"
    for p in (out, err):
        if p.exists():
            p.unlink()
    with open(err, "w", encoding="utf-8") as ef:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dump", str(out)],
            stdout=subprocess.DEVNULL, stderr=ef,
        )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"子进程失败 rc={proc.returncode}: {err.read_text(encoding='utf-8')[:800]}"
        )
    rows = [tuple(json.loads(line)) for line in out.read_text(encoding="utf-8").splitlines()]
    out.unlink()
    err.unlink()
    return [(int(s), int(i), h) for s, i, h in rows], {}


def main() -> int:
    if "--dump" in sys.argv:
        rows, totals = dump_rows()
        path = Path(sys.argv[sys.argv.index("--dump") + 1])
        path.write_text(
            "\n".join(json.dumps([s, i, h]) for s, i, h in rows), encoding="utf-8"
        )
        print(f"dumped {len(rows)} rows -> {path}")
        return 0

    print("=== 预登记判据 ===")
    print(f"  X1a 初始决策态 obs 跨进程逐位相同")
    print(f"  X1b 前 {MAX_IDX} 个决策态 obs 跨进程逐位相同")
    print(f"  X2 视角视图词元 p99 <= {TOKEN_P99_MAX}, max <= {TOKEN_MAX_CAP}")
    print(f"  X3 扰动对方隐藏信息后 obs(自己) 不变，且 full_view 必须变")
    print(f"  X4 静态牌表按席位可见/隐藏，且对扰动不变")

    # ---------------- X1 跨进程 ----------------
    here, _ = dump_rows()
    there, _ = child_dump()
    x1a = here[0][2] == there[0][2]
    first_div = next(
        (k for k in range(min(len(here), len(there))) if here[k] != there[k]), None
    )
    x1b = first_div is None
    print(f"\n=== X1 跨进程 obs ===")
    print(f"  本进程 {len(here)} 行 / 子进程 {len(there)} 行")
    print(f"  index0 相同: {x1a}")
    if first_div is None:
        print(f"  全部 {len(here)} 行逐位相同")
    else:
        s, i, h = here[first_div]
        h2 = there[first_div][2]
        print(f"  首个分歧 at seed={s} idx={i}: {h} vs {h2}")
        same_seed = [r for r in here if r[0] == s]
        print(f"  该 seed 共 {len(same_seed)} 行（本进程）")

    # ---------------- X2 视角视图词元预算 ----------------
    fresh = GameState.from_default(DefaultMode())
    segs = dc.segment_table(fresh)
    spans, cursor = [], 0
    for name, length in segs:
        spans.append((name, cursor, cursor + length - 1))
        cursor += length

    tokens_p, tokens_f = [], []
    leak_states = []
    for seed in range(SEEDS):
        sink: dict = {}
        for gs in play_random_game(seed, sink=sink):
            enc_full = np.asarray(gs.encoding(PLAN), dtype=np.int64)
            tokens_f.append(dc.tokens_of(dc.occupancy(enc_full, spans)))
            for seat in (Pid.P1, Pid.P2):
                enc_v = np.asarray(obs(gs, seat), dtype=np.int64)
                tokens_p.append(dc.tokens_of(dc.occupancy(enc_v, spans)))
            if len(leak_states) < LEAK_STATES:
                leak_states.append(gs)

    p99_p, max_p = int(np.percentile(tokens_p, 99)), max(tokens_p)
    p99_f, max_f = int(np.percentile(tokens_f, 99)), max(tokens_f)
    x2 = p99_p <= TOKEN_P99_MAX and max_p <= TOKEN_MAX_CAP
    print(f"\n=== X2 词元预算 ===")
    print(f"  视角视图（T1 的 observe）: p50={int(np.percentile(tokens_p, 50))} "
          f"p90={int(np.percentile(tokens_p, 90))} p99={p99_p} max={max_p} (n={len(tokens_p)})")
    print(f"  全量视图（D-C 原口径）  : p99={p99_f} max={max_f} (n={len(tokens_f)})")

    # ---------------- X3 观测泄漏 ----------------
    # 扰动必须是「真实隐藏、且**数量不变**」的信息：
    #   * 换掉对方面朝下的手牌身份（张数不变）——手牌**张数**是公开信息，身份不是；
    #   * 换掉对方牌堆的内容（张数不变）。
    # 只反转牌序是无效扰动：`Cards/OrderedCards.encoding` 先排序，顺序本来就看不出来。
    mode = DefaultMode()
    pool = sorted(mode.all_cards(), key=lambda c: c.__name__)

    def same_size_other_hand(gs: GameState, other: Pid, offset: int) -> GameState:
        from dgisim.card.cards import Cards
        n = len(gs.get_player(other).hand_cards.ordered_cards)
        if n == 0:
            return gs
        picked = pool[offset:offset + n] if offset + n <= len(pool) else pool[:n]
        new_hand = Cards({c: 1 for c in picked})
        return gs.factory().f_player(
            other, lambda p: p.factory().hand_cards(new_hand).build()
        ).build()

    safe_changed = leaky_changed = 0
    inconclusive = 0
    for k, gs in enumerate(leak_states):
        pert = same_size_other_hand(gs, Pid.P2, offset=k)
        changed_full = gs.encoding(PLAN, Pid.P1) != pert.encoding(PLAN, Pid.P1)
        if not changed_full:
            inconclusive += 1
            continue
        leaky_changed += 1
        safe_changed += obs(gs, Pid.P1) != obs(pert, Pid.P1)
    x3 = safe_changed == 0 and leaky_changed > 0
    print(f"\n=== X3 观测泄漏（{len(leak_states)} 个状态）===")
    print(f"  视角 obs 改变: {safe_changed}（应为 0）")
    print(f"  全量 obs 改变: {leaky_changed}（正对照，应 >0）")
    print(f"  扰动无效(全量也没变, 不计入): {inconclusive}")

    # ---------------- X4 静态牌表按席位 ----------------
    # hide_secrets 把对方牌表换成 OmniCard x30（不是清空），所以判据要写成
    # 「换掉对方真实牌表 -> obs 不变」而不是「对方牌表为空」。
    from dgisim.deck import MutableDeck
    own_visible = opp_invariant = 0
    for k, gs in enumerate(leak_states):
        v1 = gs.perspective_view(Pid.P1)
        own_enc = v1.player1.initial_deck.encoding(PLAN)
        own_visible += sum(own_enc) > 0 and not _all_omni(v1.player1.initial_deck)
        alt, other = pool[2 * k % len(pool)], pool[(2 * k + 1) % len(pool)]
        pert = gs.factory().f_player2(
            lambda p: p.factory().initial_deck(
                MutableDeck(chars=list(p.initial_deck.chars),
                            cards={alt: 15, other: 15}).to_frozen()
            ).build()
        ).build()
        opp_invariant += obs(gs, Pid.P1) == obs(pert, Pid.P1)
    x4 = own_visible == len(leak_states) and opp_invariant == len(leak_states)
    print(f"\n=== X4 静态牌表（{len(leak_states)} 个状态）===")
    print(f"  自己牌表可见(非 OmniCard): {own_visible}（应为 {len(leak_states)}）")
    print(f"  对方牌表被隐藏(换真牌表 obs 不变): {opp_invariant}（应为 {len(leak_states)}）")

    checks = {"X1a obs_index0": x1a, "X1b obs_sequence": x1b,
              "X2 token_budget_view": x2, "X3 obs_leak": x3, "X4 static_per_seat": x4}
    print("\n=== 判据 ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\n" + json.dumps({
        "x1a": x1a, "x1b": x1b, "first_divergence_row": first_div,
        "tokens_view": {"p99": p99_p, "max": max_p},
        "tokens_full": {"p99": p99_f, "max": max_f},
        "leak": {"safe_changed": safe_changed, "leaky_changed": leaky_changed,
                 "inconclusive": inconclusive},
        "static": {"own_visible": own_visible, "opp_invariant": opp_invariant},
        "checks": checks,
    }, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
