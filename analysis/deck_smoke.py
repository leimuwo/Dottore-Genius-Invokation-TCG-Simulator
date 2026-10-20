"""卡组可行性冒烟：给定 3 个角色，自动凑一副合法 30 张牌组，跑 N 局随机对局。

用途：验证"这套阵容在 dgisim 里能不能真的跑起来"，而不是只看角色是否实现。

    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' analysis\\deck_smoke.py --games 30
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import random  # noqa: E402

from dgisim.deck import MutableDeck  # noqa: E402
from dgisim.mode import DefaultMode  # noqa: E402
from dgisim.env.linear_env import LinearEnv  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402

# 一套"万能填充"：无元素条件、无同名限制冲突的支援/事件/食物/武器
FILLER = [
    "Paimon", "Liben", "Vanarana", "Timmie", "Wagner", "Timaeus", "ChefMao",
    "ChangTheNinth", "ParametricTransformer", "SeedDispensary", "NRE",
    "GamblersEarrings", "Strategize", "Starsigns", "LeaveItToMe", "SendOff",
    "ChangingShifts", "QuickKnit", "TossUp", "WindAndFreedom",
    "TeyvatFriedEgg", "LotusFlowerCrisp", "MushroomPizza", "MintyMeatRolls",
    "NorthernSmokedChicken", "MagicGuide", "WhiteIronGreatsword", "RavenBow",
]

DECKS = [
    ("1 温水皇", ["Venti", "RhodeiaOfLoch", "Fischl"]),
    ("2 雷鹤(皇女版)", ["RaidenShogun", "Shenhe", "Fischl"]),
    ("4 莫皇草", ["Mona", "Fischl", "Nahida"]),
    ("5 一斗双岩", ["AratakiItto", "Mona", "StonehideLawachurl"]),
    ("6 双火秒杀(讨债人版)", ["Mona", "Bennett", "FatuiPyroAgent"]),
    ("7 水债皇", ["RhodeiaOfLoch", "Fischl", "FatuiPyroAgent"]),
    ("8 双岩一斗(莫娜版)", ["AratakiItto", "Mona", "Noelle"]),
    ("9 Klee/Albedo/Wanderer", ["Klee", "Albedo", "Wanderer"]),
    # 对照组：角色缺失的原始阵容，应当构建失败
    ("3 北重雷(原始)", ["Beidou", "Chongyun", "Razor"]),
]


def build_deck(char_names):
    mode = DefaultMode()
    by_name = {c.__name__: c for c in mode.all_chars()}
    cards_by_name = {c.__name__: c for c in mode.all_cards()}
    missing = [c for c in char_names if c not in by_name]
    if missing:
        return None, f"角色缺失: {missing}"
    chars = [by_name[c] for c in char_names]
    cards = Counter()
    for name in FILLER:
        card = cards_by_name.get(name)
        if card is None:
            continue
        if cards[card] >= mode.deck_card_limit_per_kind():
            continue
        cards[card] += 2
        if sum(cards.values()) >= mode.deck_cards_requirement():
            break
    deck = MutableDeck(chars=chars, cards=dict(cards))
    frozen = deck.to_frozen()
    if not mode.valid_deck(frozen):
        return None, f"牌组不合法（共 {sum(cards.values())} 张）"
    return frozen, f"{len(chars)} 角色 / {sum(cards.values())} 张牌 / {len(cards)} 种"


def run(deck, games, seed=7):
    modes = DefaultMode()
    random.seed(seed)
    env = LinearEnv()
    env.reset_with_decks(deck, deck)
    agent = RandomAgent()
    stats = {"games": 0, "errors": 0, "noop": 0, "decisions": 0,
             "draws": 0, "wins": 0, "rounds": [], "first_error": ""}
    for _ in range(games):
        try:
            env.reset_with_decks(deck, deck)
            gs, _, _, turn, done = env.view()
            n = 0
            while not done and n < 3000:
                n += 1
                prev = gs
                gs, _, _, turn, done = env.step(agent.choose_action([gs], Pid(turn)))
                stats["decisions"] += 1
                if gs == prev:
                    stats["noop"] += 1
            stats["games"] += 1
            stats["rounds"].append(env.full_view().round)
            if env.full_view().get_winner() is None:
                stats["draws"] += 1
            else:
                stats["wins"] += 1
        except Exception as exc:
            stats["errors"] += 1
            if not stats["first_error"]:
                stats["first_error"] = f"{type(exc).__name__}: {exc}"[:160]
    avg_rounds = sum(stats["rounds"]) / len(stats["rounds"]) if stats["rounds"] else 0
    return stats, avg_rounds


def main():
    games = 30
    for arg in sys.argv:
        if arg.isdigit():
            games = int(arg)
    print(f"每个卡组随机对局 {games} 局（双方使用同一副牌，seed=7）\n")
    for name, chars in DECKS:
        deck, info = build_deck(chars)
        if deck is None:
            print(f"[不可用] {name:26s} {info}")
            continue
        stats, avg_rounds = run(deck, games)
        verdict = "可用  " if stats["errors"] == 0 and stats["noop"] == 0 else "有问题"
        print(f"[{verdict}] {name:26s} {info}")
        print(f"          局数={stats['games']} 异常={stats['errors']} "
              f"空转={stats['noop']} 决策/局={stats['decisions'] / max(1, stats['games']):.0f} "
              f"平均回合={avg_rounds:.1f} 平局={stats['draws']}/{stats['games']}")
        if stats["first_error"]:
            print(f"          首个异常：{stats['first_error']}")


if __name__ == "__main__":
    main()
