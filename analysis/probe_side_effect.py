"""T5 的前提检验：侧别效应到底有多大？

设计：**同一套牌对同一套牌**（唯一差别 = 谁先手），随机策略，测击杀局的座位胜率。
这样卡组对位、卡组强度全部抵消，剩下的就是纯侧别效应。

运行：python analysis/probe_side_effect.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgisim.action.action import EndRoundAction  # noqa: F401,E402
from dgisim.agents import RandomAgent  # noqa: E402
from dgisim.card.card import ArcaneLegendCard, Card  # noqa: E402
from dgisim.deck import MutableDeck  # noqa: E402
from dgisim.env.linear_env import LinearEnv  # noqa: E402
from dgisim.mode import DefaultMode  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402


def build_deck(mode: DefaultMode):
    """确定性地造一套合法卡组（3 角色 / 30 张牌）。"""
    chars = sorted(mode.all_chars(), key=lambda c: c.__name__)[:3]
    pool: list[type[Card]] = []
    tmp = MutableDeck(chars=list(chars), cards={})
    for card in sorted(mode.all_cards(), key=lambda c: c.__name__):
        if not card.valid_in_deck(tmp):
            continue
        pool.extend([card] if issubclass(card, ArcaneLegendCard) else [card] * 2)
    cards: Counter = Counter()
    for card in pool:
        if sum(cards.values()) >= 30:
            break
        if cards[card] >= 2:
            continue
        cards[card] += 1
    assert sum(cards.values()) == 30, sum(cards.values())
    return MutableDeck(chars=list(chars), cards=dict(cards)).to_frozen()


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    mode = DefaultMode()
    deck = build_deck(mode)
    agent = RandomAgent()
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    p1_kills = p2_kills = timeouts = 0
    for seed in range(games):
        random.seed(seed)
        env = LinearEnv(mode=mode)
        env.reset_with_decks(deck, deck)
        gs = env.full_view()
        steps = 0
        while not gs.game_end() and steps < 4000:
            w = gs.waiting_for()
            if w is None:
                gs = gs.step()
                steps += 1
                continue
            gs = gs.action_step(w, agent.choose_action([gs], w))
            steps += 1
        if not gs.game_end():
            continue
        winner = gs.get_winner()
        if winner is Pid.P1:
            p1_kills += 1
        elif winner is Pid.P2:
            p2_kills += 1
        else:
            timeouts += 1

    decided = p1_kills + p2_kills
    lo, hi = wilson(p1_kills, decided)
    print(f"games={games} 击杀局={decided} 超时={timeouts} ({timeouts / max(games, 1):.1%})")
    print(f"击杀局里 P1(先手) 胜率 = {p1_kills}/{decided} = {p1_kills / max(decided, 1):.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"侧别效应（P1 相对 50%）= {(p1_kills / max(decided, 1) - 0.5) * 100:+.1f} 个百分点")
    print(f"要检出 5% 的侧别差（power=0.8），击杀局需要 ≈1570×2 局 → 总对局 ≈"
          f" {int(2 * 1570 / max(decided / max(games, 1), 0.01))} 局")


if __name__ == "__main__":
    main()
