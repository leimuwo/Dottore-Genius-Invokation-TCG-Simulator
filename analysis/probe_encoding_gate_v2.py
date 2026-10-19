"""两部分：
① B14：Dice.encoding() 非幂等（cached_property 缓存了 iterator）在一局真实对局里的影响面
② 修正后的观察通道检查（每次观测前重建 dice，避免被 B14 污染）

运行：python analysis/probe_encoding_gate_v2.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402
from dgisim.encoding.encoding_plan import encoding_plan as PLAN  # noqa: E402
from dgisim.card.cards import Cards, OrderedCards  # noqa: E402
from dgisim.dice import ActualDice  # noqa: E402
from dgisim.element import Element  # noqa: E402
from dgisim.deck import MutableDeck  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402

DICE_LEN = PLAN.DICE_FIXED_LEN


def fresh_dice_state(gs):
    """把双方的 dice 换成新对象，隔离 B14 的"共享 dice 被编码烧坏"效应。"""
    def reb(ps):
        return ps.factory().dice(ActualDice(ps.dice.to_dict())).build()
    return gs.factory().f_player1(reb).f_player2(reb).build()


def obs(gs):
    """P1 视角观察向量；先重建 dice 再编码。"""
    return fresh_dice_state(gs).perspective_view(Pid.P1).encoding(PLAN, Pid.P1)


print("=" * 84)
print("① B14 影响面：Dice.encoding() 不幂等")
print("=" * 84)
random.seed(3)
gs = GameState.from_default()
agent = RandomAgent()
seen_dice = []          # 持有引用，避免 id() 被回收复用
decisions = corrupted = 0
for _ in range(3000):
    if gs.game_end():
        break
    w = gs.waiting_for()
    if w is None:
        gs = gs.step()
        continue
    decisions += 1
    pid = w
    dice = gs.get_player(pid).dice
    n = dice.num_dice()
    if n > 0:
        if any(dice is old for old in seen_dice):
            corrupted += 1
        seen_dice.append(dice)
    gs = gs.action_step(w, agent.choose_action([gs], w))
print(f"  一局共 {decisions} 个决策点；其中 {corrupted} 个决策点的 dice 对象"
      f"此前已被编码过 → 这些点上骰子会被编码成全 0（{corrupted / max(decisions, 1):.0%}）")

# 直接示范：同一 state 连续编码两次
random.seed(3)
gs2 = GameState.from_default()
agent2 = RandomAgent()
while type(gs2.phase).__name__ != "ActionPhase" or gs2.waiting_for() is None:
    w = gs2.waiting_for()
    gs2 = gs2.step() if w is None else gs2.action_step(w, agent2.choose_action([gs2], w))
    if gs2.game_end():
        break
v1 = gs2.encoding(PLAN, Pid.P1)
v2 = gs2.encoding(PLAN, Pid.P1)
d1 = v1[8:8 + DICE_LEN]
d2 = v2[8:8 + DICE_LEN]
print(f"  同一个 state 编码两次：骰子段第一次 {d1[:8]}…")
print(f"                        第二次 {d2[:8]}…  → 相同吗 {d1 == d2}")

print()
print("=" * 84)
print("② 修正后的观察通道检查（每次先重建 dice）")
print("=" * 84)


def get_state(seed=3):
    random.seed(seed)
    g = GameState.from_default()
    a = RandomAgent()
    for _ in range(500):
        if g.game_end():
            break
        w = g.waiting_for()
        if w is None:
            g = g.step()
            continue
        if type(g.phase).__name__ == "ActionPhase":
            return g
        g = g.action_step(w, a.choose_action([g], w))
    return g


gs = get_state()
p1, p2 = gs.player1, gs.player2
base = obs(gs)
all_cards = sorted(gs.mode.all_cards(), key=lambda c: c.__name__)
alt = [c for c in all_cards if c not in set(p1.hand_cards.ordered_cards)
       and c not in set(p2.hand_cards.ordered_cards)][0]

rows = []


def check(name, s, should_change):
    n = sum(1 for x, y in zip(base, obs(s)) if x != y)
    ok = (n > 0) == should_change
    rows.append((name, n, "应不变" if not should_change else "应变化", "PASS" if ok else "FAIL"))


check("对手手牌内容（应不可见）",
      gs.factory().f_player2(lambda p: p.factory().hand_cards(
          Cards({alt: p.hand_cards.num_cards()})).build()).build(), False)
check("对手牌库顺序",
      gs.factory().f_player2(lambda p: p.factory().deck_cards(
          OrderedCards(list(p.deck_cards.cards)[::-1])).build()).build(), False)
check("对手骰子面（应只剩个数）",
      gs.factory().f_player2(lambda p: p.factory().dice(
          ActualDice.from_all(p.dice.num_dice(), Element.PYRO)).build()).build(), False)
check("对手 initial_deck（B3）",
      gs.factory().f_player2(lambda p: p.factory().initial_deck(
          MutableDeck(chars=list(p.initial_deck.chars),
                      cards={all_cards[0]: 15, all_cards[1]: 15}).to_frozen()).build()).build(), False)
check("自己牌库顺序（B12）",
      gs.factory().f_player1(lambda p: p.factory().deck_cards(
          OrderedCards(list(p.deck_cards.cards)[::-1])).build()).build(), False)

items = []
for c in p1.hand_cards.ordered_cards:
    if items and items[-1][0] is c:
        items[-1] = (c, items[-1][1] + 1)
    else:
        items.append((c, 1))
check("自己手牌插入顺序（B13）",
      gs.factory().f_player1(lambda p: p.factory().hand_cards(
          Cards(dict(items[::-1]))).build()).build(), False)
check("自己骰子插入顺序（Dice 顺序是否进观察）",
      gs.factory().f_player1(lambda p: p.factory().dice(
          ActualDice(Counter(list(p.dice.ordered_dice)[::-1]))).build()).build(), False)
check("【哨兵】自己手牌内容（本就该可见）",
      gs.factory().f_player1(lambda p: p.factory().hand_cards(
          Cards({alt: p.hand_cards.num_cards()})).build()).build(), True)

print(f"{'通道':44s} {'变化位':>5s}  {'期望':6s} {'结果'}")
print("-" * 84)
for name, n, expect, res in rows:
    print(f"{name:44s} {n:5d}  {expect:6s} {res}")
