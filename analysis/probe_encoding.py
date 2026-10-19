"""核验四条新主张：
B12  自己的牌库顺序（= 未来抽牌）进观察
B13  Cards / Dice 的编码依赖插入顺序（同一状态两种向量）
C2b  choices() 的顺序未规范化
Gate 合并建议：用一个通用扰动测试能否同时管住以上

运行：python analysis/probe_encoding.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402
from dgisim.encoding.encoding_plan import encoding_plan as PLAN  # noqa: E402
from dgisim.card.cards import Cards, OrderedCards  # noqa: E402
from dgisim.card.card import OmniCard  # noqa: E402
from dgisim.dice import ActualDice  # noqa: E402
from dgisim.action.enums import ActionType  # noqa: E402


def get_state(seed=3):
    random.seed(seed)
    gs = GameState.from_default()
    while gs.waiting_for() is None and not gs.game_end():
        gs = gs.step()
    return gs


def find_subseq(seq, sub):
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


def diff_idx(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


print("=" * 78)
print("B12：自己的牌库顺序是否进了观察")
print("=" * 78)
gs = get_state()
p1 = gs.player1
deck = list(p1.deck_cards.cards)          # 底 -> 顶
codes = [PLAN.encode_item(c) for c in deck]
enc = p1.encoding(PLAN)
start = find_subseq(enc, codes)
print(f"  牌库 {len(deck)} 张；在 PlayerState 编码中的连续起点 = {start}"
      f"（整条观察向量里的下标 = {4 + start if start >= 0 else 'N/A'}，因为前面有 4 个 basics）")
print(f"  编码里是按'底→顶'存的：OrderedCards.pick() 取的是 _cards[-n:]（从顶部抽）"
      f" → 抽牌顺序 = 编码顺序的逆序，但顺序信息 100% 保留")

# 交换牌库最后两张（= 下两张要抽的牌）
d = deck[:]
d[-1], d[-2] = d[-2], d[-1]
gs_swapped = gs.factory().f_player1(
    lambda p: p.factory().deck_cards(OrderedCards(d)).build()
).build()
enc2 = gs_swapped.player1.encoding(PLAN)
print(f"  交换'接下来要抽的两张' → 编码变化的位置: {diff_idx(enc, enc2)}  (共 {len(diff_idx(enc, enc2))} 位)")

# 对手牌库是否被隐藏
opp = gs.perspective_view(Pid.P1).player2
opp_codes = [PLAN.encode_item(c) for c in opp.deck_cards.cards]
print(f"  对手牌库在自己视角下：{len(set(opp_codes))} 种取值（1 = 全部 OmniCard，已隐藏 ✅）")

print()
print("=" * 78)
print("B13：Cards / Dice 的编码依赖插入顺序")
print("=" * 78)
hand = list(p1.hand_cards.ordered_cards)
A, B, C = hand[0], hand[1], hand[2]
c1 = Cards({A: 1, B: 1, C: 1})
c2 = Cards({C: 1, B: 1, A: 1})
e1, e2 = c1.encoding(PLAN), c2.encoding(PLAN)
print(f"  Cards 同多重集不同插入序：== {c1 == c2} | hash 相等 {hash(c1) == hash(c2)} "
      f"| 编码相等 {e1 == e2} | 差异位 {diff_idx(e1, e2)}")
print(f"    前 3 位: {e1[:3]}  vs  {e2[:3]}")

# 整个 GameState
gs_a = gs.factory().f_player1(
    lambda p: p.factory().hand_cards(Cards({A: 1, B: 1, C: 1})).build()
).build()
gs_b = gs.factory().f_player1(
    lambda p: p.factory().hand_cards(Cards({C: 1, B: 1, A: 1})).build()
).build()
ea, eb = gs_a.encoding(PLAN, Pid.P1), gs_b.encoding(PLAN, Pid.P1)
print(f"  整个 GameState：== {gs_a == gs_b} | hash 相等 {hash(gs_a) == hash(gs_b)} "
      f"| 编码相等 {ea == eb} | 差异位 {diff_idx(ea, eb)} (共 {len(diff_idx(ea, eb))} 位)")

# Dice 是否同病
dice = p1.dice
elems = list(dice.ordered_dice)
if len(elems) >= 2:
    from collections import Counter
    perm = elems[:]
    perm[0], perm[1] = perm[1], perm[0]
    d1 = ActualDice(Counter(elems))
    d2 = ActualDice(Counter(perm))
    print(f"  Dice 同多重集不同插入序：编码相等 {d1.encoding(PLAN) == d2.encoding(PLAN)} "
          f"| 差异位 {diff_idx(d1.encoding(PLAN), d2.encoding(PLAN))}")

print()
print("=" * 78)
print("C2b：choices() 的顺序是否规范化")
print("=" * 78)
# 找一个 ActionPhase 的决策点
random.seed(5)
g = GameState.from_default()
agent_choices = None
for _ in range(400):
    if g.game_end():
        break
    w = g.waiting_for()
    if w is None:
        g = g.step()
        continue
    gen = g.action_generator(w)
    if gen is not None and gen.choices() and ActionType.PLAY_CARD in tuple(gen.choices()):
        agent_choices = (g, w, gen)
        break
    from dgisim.agents import RandomAgent
    g = g.action_step(w, RandomAgent().choose_action([g], w))

if agent_choices is None:
    print("  没找到 ActionPhase 的 PLAY_CARD 决策点，跳过")
else:
    g, w, gen = agent_choices
    gen2 = gen.choose(ActionType.PLAY_CARD)
    ch = gen2.choices()
    print(f"  一级 choices 类型={type(gen.choices()).__name__} 顺序={[a.name for a in gen.choices()]}")
    print(f"  二级 choices（选哪张牌）类型={type(ch).__name__}")
    cards_ordered = list(ch.cards) if hasattr(ch, "cards") else list(ch)
    print(f"  二级顺序 #1: {[c.__name__ for c in cards_ordered]}")

    # 把手牌按不同插入顺序重建，再看同一个二级 choices 的顺序
    hand_dict = {}
    for c in g.get_player(w).hand_cards.ordered_cards:
        hand_dict[c] = hand_dict.get(c, 0) + 1
    items = list(hand_dict.items())[::-1]
    g_perm = g.factory().f_player(
        w, lambda p: p.factory().hand_cards(Cards(dict(items))).build()
    ).build()
    gen_perm = g_perm.action_generator(w).choose(ActionType.PLAY_CARD)
    ch_perm = gen_perm.choices()
    cards_perm = list(ch_perm.cards) if hasattr(ch_perm, "cards") else list(ch_perm)
    print(f"  二级顺序 #2（手牌插入序反转后）: {[c.__name__ for c in cards_perm]}")
    print(f"  同一多重集，顺序相同吗：{cards_ordered == cards_perm}")
