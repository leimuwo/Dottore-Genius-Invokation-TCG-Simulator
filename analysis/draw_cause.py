"""平局的成因拆解：规则 / 策略 / 构筑，各占多少。

对比三种策略在**同一副牌**上的平局率，以及终局时场上的实际战况：
  - RandomAgent      ：完全随机（会随机结束回合、随机支付骰子、随机选目标）
  - AttackAgent      ：能打就打（优先技能 > 打牌 > 切人 > 调和 > 结束回合）
  - LazyAgent        ：两边都只结束回合（下界）

同时记录终局形态：死了几个角色、双方剩余总血量。
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import random  # noqa: E402

from dgisim.action.action import PlayerAction  # noqa: E402
from dgisim.action.enums import ActionType  # noqa: E402
from dgisim.action.action_generator import ActionGenerator  # noqa: E402
from dgisim.agents import LazyAgent, RandomAgent  # noqa: E402
from dgisim.card.cards import Cards  # noqa: E402
from dgisim.dice import ActualDice, AbstractDice  # noqa: E402
from dgisim.deck import MutableDeck  # noqa: E402
from dgisim.env.linear_env import LinearEnv  # noqa: E402
from dgisim.mode import DefaultMode  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402

PREFERENCE = [ActionType.CAST_SKILL, ActionType.PLAY_CARD,
              ActionType.ELEMENTAL_TUNING, ActionType.SWAP_CHARACTER,
              ActionType.END_ROUND]


class AttackAgent:
    """能打就打；其余一律取第一个合法项。"""

    def choose_action(self, history, pid):
        gs = history[-1]
        ag = gs.action_generator(pid)
        if ag is None:
            return RandomAgent().choose_action(history, pid)
        first = True
        while not ag.filled():
            choices = ag.choices()
            if isinstance(choices, tuple):
                pick = None
                if first:
                    for want in PREFERENCE:
                        if want in choices:
                            pick = want
                            break
                    first = False
                ag = ag.choose(pick if pick is not None else choices[0])
            elif isinstance(choices, AbstractDice):
                sel = ag.dice_available().smart_selection(
                    choices, ag.game_state.get_player(ag.pid).characters)
                if sel is None:
                    return RandomAgent().choose_action(history, pid)
                ag = ag.choose(sel)
            elif isinstance(choices, Cards):
                _, choice = choices.pick_random(0)
                ag = ag.choose(choice)
            elif isinstance(choices, ActualDice):
                _, choice = choices.pick_random_dice(0)
                ag = ag.choose(choice)
            else:  # pragma: no cover
                return RandomAgent().choose_action(history, pid)
        return ag.generate_action()


FILLER = [
    "Paimon", "Liben", "Vanarana", "Timmie", "Wagner", "Timaeus", "ChefMao",
    "ChangTheNinth", "ParametricTransformer", "SeedDispensary", "NRE",
    "GamblersEarrings", "Strategize", "Starsigns", "LeaveItToMe", "SendOff",
    "ChangingShifts", "QuickKnit", "TossUp", "WindAndFreedom",
    "TeyvatFriedEgg", "LotusFlowerCrisp", "MushroomPizza", "MintyMeatRolls",
    "NorthernSmokedChicken", "MagicGuide", "WhiteIronGreatsword", "RavenBow",
]


def build_deck(char_names, card_names=None):
    mode = DefaultMode()
    chars = {c.__name__: c for c in mode.all_chars()}
    cards = {c.__name__: c for c in mode.all_cards()}
    deck_cards = Counter()
    for name in (card_names or FILLER):
        card = cards.get(name)
        if card is None:
            continue
        if deck_cards[card] >= mode.deck_card_limit_per_kind():
            continue
        deck_cards[card] += 2
        if sum(deck_cards.values()) >= mode.deck_cards_requirement():
            break
    deck = MutableDeck(chars=[chars[c] for c in char_names],
                       cards=dict(deck_cards)).to_frozen()
    assert mode.valid_deck(deck), "牌组不合法"
    return deck


def play(p1_agent, p2_agent, deck, games, seed=7):
    random.seed(seed)
    env = LinearEnv()
    out = {"games": 0, "draws": 0, "p1_wins": 0, "p2_wins": 0,
           "rounds": [], "dead": [], "total_hp": [], "errors": 0}
    for _ in range(games):
        try:
            env.reset_with_decks(deck, deck)
            gs, _, _, turn, done = env.view()
            n = 0
            while not done and n < 4000:
                n += 1
                w = gs.waiting_for()
                agent = p1_agent if w is Pid.P1 else p2_agent
                gs, _, _, turn, done = env.step(agent.choose_action([gs], w))
            full = env.full_view()
            w = full.get_winner()
            out["games"] += 1
            if w is None:
                out["draws"] += 1
            elif w is Pid.P1:
                out["p1_wins"] += 1
            else:
                out["p2_wins"] += 1
            out["rounds"].append(full.round)
            dead = 0
            hp = 0
            for pid in (Pid.P1, Pid.P2):
                for c in full.get_player(pid).characters:
                    # alive 在部分版本是 property、部分是方法，两种都兼容
                    alive = c.alive() if callable(c.alive) else c.alive
                    if alive:
                        hp += c.hp
                    else:
                        dead += 1  # 阵亡角色的 hp 是哨兵值，不能计入总血量
            out["dead"].append(dead)
            out["total_hp"].append(hp)
        except Exception:
            out["errors"] += 1
    return out


def summarize(tag, s):
    n = max(1, s["games"])
    print(f"{tag:26s} 局={s['games']:3d} 平局={s['draws'] / n:5.0%} "
          f"P1胜={s['p1_wins'] / n:4.0%} P2胜={s['p2_wins'] / n:4.0%} "
          f"平均回合={sum(s['rounds']) / n:4.1f} "
          f"平均阵亡={sum(s['dead']) / n:3.1f}/6 "
          f"终局总血={sum(s['total_hp']) / n:5.1f}/60 异常={s['errors']}")


def main():
    games = 40
    for arg in sys.argv:
        if arg.isdigit():
            games = int(arg)
    deck = build_deck(["Venti", "RhodeiaOfLoch", "Fischl"])
    print(f"同一副牌（温迪/纯水/皇女 + 通用填充），每格 {games} 局，seed=7\n")
    summarize("双方都随机", play(RandomAgent(), RandomAgent(), deck, games))
    summarize("双方都只结束回合", play(LazyAgent(), LazyAgent(), deck, games))
    summarize("双方都只进攻", play(AttackAgent(), AttackAgent(), deck, games))
    summarize("P1进攻 vs P2随机", play(AttackAgent(), RandomAgent(), deck, games))


if __name__ == "__main__":
    main()
