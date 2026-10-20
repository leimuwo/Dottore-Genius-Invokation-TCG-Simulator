#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D-B 卡组池校验器：`decks/train/*.json` + `decks/test/*.json`。

为什么要有它
------------
牌表是**设计决策**（"像不像一套能打的牌"只能人来判断），本脚本只管**机器能判的部分**：
角色/牌数/同名张数/名字是否真在本仓库实现集里/训练集与测试集是否重叠。
外网不可用时人凭记忆写牌名会幻觉 —— 这一层就是兜它的。

判据（任一 FAIL → 退出码 1）
    S1  JSON 能解析，且**无重复键**（`json.load` 会静默去重，必须用 hook 查）
    S2  字段齐全：`chars` 为 3 个字符串，`cards` 为非空 {名字: 正整数}
    S3  角色不重复
    S4  牌总数 == `DefaultMode().deck_cards_requirement()`
    S5  同名 1..2 张
    S6  每个名字都存在于 `mode.all_chars()` / `mode.all_cards()`
    S7  `FrozenDeck.from_json` 能加载（训练侧直接吃文件走这条路径）
    S8  `mode.valid_deck(deck)` 为真（引擎自己的合法性判据，含每张牌的入组条件）
    S9  牌表逐套不相同（同集合内 + 跨集合）
    S10 数量：train >= 6 套，test >= 3 套

    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' scripts\\verify_decks.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dgisim.deck import FrozenDeck, MutableDeck   # noqa: E402
from dgisim.mode import DefaultMode              # noqa: E402

MIN_TRAIN, MIN_TEST = 6, 3


def _no_dup_hook(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f"重复键 {k}")
        seen[k] = v
    return seen


def canon_key(chars: list[str], cards: dict[str, int]) -> str:
    """规范化后的牌表内容，用于「是否同一套牌」的比较（与指纹同源）。"""
    return json.dumps(
        {"chars": sorted(chars), "cards": dict(sorted(cards.items()))},
        ensure_ascii=False, sort_keys=True,
    )


def check_one(path: Path, mode) -> tuple[list[str], str | None, int, str | None]:
    """返回 (problems, 指纹, 牌种数, 规范内容)。"""
    try:
        raw = path.read_text(encoding="utf-8")
        spec = json.loads(raw, object_pairs_hook=_no_dup_hook)
    except Exception as e:  # noqa: BLE001
        return [f"S1 解析失败: {type(e).__name__}: {e}"], None, 0, None

    chars, cards = spec.get("chars"), spec.get("cards")
    problems: list[str] = []
    if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars) or len(chars) != 3:
        problems.append("S2 chars 必须是 3 个字符串")
    if not isinstance(cards, dict) or not cards or not all(
        isinstance(v, int) and v > 0 for v in cards.values()
    ):
        problems.append("S2 cards 必须是非空 {名字: 正整数}")
    if problems:
        return problems, None, 0, None

    if len(set(chars)) != 3:
        problems.append(f"S3 角色重复: {chars}")

    required = mode.deck_cards_requirement()
    total = sum(cards.values())
    if total != required:
        problems.append(f"S4 牌数 {total} != {required}")

    over = {k: v for k, v in cards.items() if not 1 <= v <= 2}
    if over:
        problems.append(f"S5 同名张数越界: {over}")

    char_pool = {c.__name__: c for c in mode.all_chars()}
    card_pool = {c.__name__: c for c in mode.all_cards()}
    unknown_chars = [c for c in chars if c not in char_pool]
    unknown_cards = [c for c in cards if c not in card_pool]
    if unknown_chars:
        problems.append(f"S6 角色未实现: {unknown_chars}")
    if unknown_cards:
        problems.append(f"S6 牌未实现: {unknown_cards}")
    fp = fingerprint(chars, cards) if not (unknown_chars or unknown_cards) else None
    if problems:
        return problems, fp, len(cards), None

    try:
        deck = FrozenDeck.from_json(raw)
    except Exception as e:  # noqa: BLE001
        problems.append(f"S7 FrozenDeck.from_json 失败: {type(e).__name__}: {e}")
        return problems, fp, len(cards), None

    if not mode.valid_deck(deck):
        for cname in cards:
            probe = MutableDeck(chars=[char_pool[c] for c in chars], cards={card_pool[cname]: 1})
            if not card_pool[cname].valid_in_deck(probe):
                problems.append(f"S8 {cname} 不满足入组条件")
        if not problems:
            problems.append("S8 valid_deck(deck) 为假，但逐张检查都通过（需人工看）")

    return problems, fp, len(cards), (None if problems else canon_key(chars, cards))


def fingerprint(chars: list[str], cards: dict[str, int]) -> str:
    return hashlib.sha1(canon_key(chars, cards).encode("utf-8")).hexdigest()[:12]


def main() -> int:
    mode = DefaultMode()
    folders = {"train": REPO / "decks" / "train", "test": REPO / "decks" / "test"}
    all_ok = True
    rows: dict[str, list[tuple[str, str, str]]] = {"train": [], "test": []}
    content: dict[str, dict[str, str]] = {"train": {}, "test": {}}

    for split, folder in folders.items():
        files = sorted(folder.glob("*.json"))
        if not files:
            print(f"FAIL {split}: 目录里没有 json: {folder}")
            all_ok = False
            continue
        for path in files:
            problems, fp, n_kinds, canon = check_one(path, mode)
            if problems:
                all_ok = False
                rows[split].append((path.stem, "FAIL", "; ".join(problems)))
            else:
                content[split][path.stem] = canon  # type: ignore[assignment]
                rows[split].append((path.stem, "PASS", f"{mode.deck_cards_requirement()} 张 / "
                                                       f"{n_kinds} 种 / {fp}"))

    for split, need in (("train", MIN_TRAIN), ("test", MIN_TEST)):
        n_ok = sum(1 for _, st, _ in rows[split] if st == "PASS")
        if n_ok < need:
            print(f"FAIL S10 {split}: 合格 {n_ok} 套 < {need}")
            all_ok = False

    seen: dict[str, str] = {}
    for split in ("train", "test"):
        for name, canon in content[split].items():
            if canon in seen:
                print(f"FAIL S9 {split}/{name} 与 {seen[canon]} 是同一套牌")
                all_ok = False
            else:
                seen[canon] = f"{split}/{name}"

    for split in ("train", "test"):
        print(f"\n=== {split} ===")
        for name, status, detail in rows[split]:
            print(f"  {status:4s} {name:32s} {detail}")

    print(f"\n训练 {len(content['train'])} 套 / 测试 {len(content['test'])} 套")
    print("PASS: 全部判据通过" if all_ok else "FAIL: 见上面的原因")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
