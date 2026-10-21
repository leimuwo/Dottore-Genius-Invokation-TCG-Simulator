"""6 训练 + 3 测试卡组的"实现集构筑"，并做合法性校验 + 攻击性实测。

只使用 dgisim v0.6.0 实现集里的牌（202 类卡，含 OmniCard 等内部类）。
每套 3 角色 + 30 张，同名 <= 2。

    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' analysis\\deck_pool.py            # 校验
    & 'C:\\Users\\wenxi\\miniconda3\\python.exe' analysis\\deck_pool.py 40         # 校验 + 每套跑 40 局
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

from dgisim.action.enums import ActionType  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402
from dgisim.card.cards import Cards  # noqa: E402
from dgisim.deck import MutableDeck  # noqa: E402
from dgisim.dice import AbstractDice, ActualDice  # noqa: E402
from dgisim.env.linear_env import LinearEnv  # noqa: E402
from dgisim.mode import DefaultMode  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402

# ---------------------------------------------------------------- 卡组定义
DECKS = {
    "T1 温水皇（慢速控制/召唤）": {
        "chars": ["Venti", "RhodeiaOfLoch", "Fischl"],
        "cards": {
            "Paimon": 2, "Liben": 2, "Vanarana": 2, "ChefMao": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
            "ElegyForTheEnd": 2, "SacrificialBow": 2,
            "ViridescentVenerer": 2, "ThunderingFury": 2,
        },
    },
    "T2 雷鹤（中速充能爆发）": {
        "chars": ["RaidenShogun", "Shenhe", "Fischl"],
        "cards": {
            "Paimon": 2, "Liben": 2, "Tenshukaku": 2, "Vanarana": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2,
            "EngulfingLightning": 2, "Moonpiercer": 2,
            "EmblemOfSeveredFate": 2, "BlizzardStrayer": 2, "ThunderingFury": 2,
        },
    },
    "T3 超导快攻（重云/雷神/皇女）": {
        "chars": ["Chongyun", "RaidenShogun", "Fischl"],
        "cards": {
            "ElementalResonanceHighVoltage": 2, "ElementalResonanceWovenThunder": 2,
            "Paimon": 2, "Liben": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2,
            "WolfsGravestone": 2, "EngulfingLightning": 2,
            "ThunderingFury": 2, "BlizzardStrayer": 2,
            "SacrificialBow": 2, "RavenBow": 2,
        },
    },
    "T4 莫皇草（快攻激化）": {
        "chars": ["Mona", "Fischl", "Nahida"],
        "cards": {
            "AThousandFloatingDreams": 2, "SacrificialFragments": 2, "ElegyForTheEnd": 2,
            "DeepwoodMemories": 2, "GildedDreams": 2,
            "Paimon": 2, "Liben": 2, "Vanarana": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
    "T5 一斗双岩（中速 OTK）": {
        "chars": ["AratakiItto", "Mona", "StonehideLawachurl"],
        "cards": {
            "WolfsGravestone": 2, "TheBell": 2, "SacrificialFragments": 2,
            "TenacityOfTheMillelith": 2, "ArchaicPetra": 2,
            "Paimon": 2, "Liben": 2, "JadeChamber": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
    "T6 双火单点（莫娜/班尼特/讨债人）": {
        "chars": ["Mona", "Bennett", "FatuiPyroAgent"],
        "cards": {
            "AquilaFavonia": 2, "FavoniusSword": 2, "AThousandFloatingDreams": 2,
            "CrimsonWitchOfFlames": 2, "GamblersEarrings": 2,
            "Paimon": 2, "Liben": 2, "GoldenHouse": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
    "V7 水债皇（测试：与 T1 同体系换轴）": {
        "chars": ["RhodeiaOfLoch", "Fischl", "FatuiPyroAgent"],
        "cards": {
            "ElegyForTheEnd": 2, "SacrificialBow": 2,
            "OceanHuedClam": 2, "ThunderingFury": 2, "GamblersEarrings": 2,
            "Paimon": 2, "Liben": 2, "Vanarana": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
    "V8 双岩一斗（测试：与 T5 同核心换辅助）": {
        "chars": ["AratakiItto", "Mona", "Noelle"],
        "cards": {
            "WolfsGravestone": 2, "WhiteIronGreatsword": 2, "SacrificialFragments": 2,
            "TenacityOfTheMillelith": 2, "ArchaicPetra": 2,
            "Paimon": 2, "Liben": 2, "KnightsOfFavoniusLibrary": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
    "V9 Klee/Albedo/Wanderer（测试：分布外）": {
        "chars": ["Klee", "Albedo", "Wanderer"],
        "cards": {
            "AThousandFloatingDreams": 2, "SacrificialFragments": 2,
            "TravelersHandySword": 2,
            "CrimsonWitchOfFlames": 2, "ArchaicPetra": 2,
            "Paimon": 2, "Liben": 2, "Vanarana": 2,
            "AdeptusTemptation": 2, "LotusFlowerCrisp": 2, "MushroomPizza": 2,
            "Starsigns": 2, "LeaveItToMe": 2, "SendOff": 2, "Strategize": 2,
        },
    },
}


# ---------------------------------------------------------------- 校验
def build(name, spec):
    mode = DefaultMode()
    chars = {c.__name__: c for c in mode.all_chars()}
    cards = {c.__name__: c for c in mode.all_cards()}
    problems = []
    if len(spec["chars"]) != 3:
        problems.append(f"角色数 {len(spec['chars'])} != 3")
    for c in spec["chars"]:
        if c not in chars:
            problems.append(f"角色未实现: {c}")
    total = sum(spec["cards"].values())
    if total != 30:
        problems.append(f"牌数 {total} != 30")
    for cname, n in spec["cards"].items():
        if cname not in cards:
            problems.append(f"牌未实现: {cname}")
        if n > 2:
            problems.append(f"{cname} 超过 2 张")
    if problems:
        return None, problems
    deck = MutableDeck(
        chars=[chars[c] for c in spec["chars"]],
        cards={cards[c]: n for c, n in spec["cards"].items()},
    ).to_frozen()
    if not mode.valid_deck(deck):
        # 逐条定位是哪张牌不合法
        for cname in spec["cards"]:
            probe = MutableDeck(
                chars=[chars[c] for c in spec["chars"]],
                cards={cards[cname]: 1},
            )
            if not cards[cname].valid_in_deck(probe):
                problems.append(f"{cname} 不满足自身的入组条件")
    return (deck if not problems else None), problems


# ---------------------------------------------------------------- 攻击型代理
PREFERENCE = [ActionType.CAST_SKILL, ActionType.PLAY_CARD,
              ActionType.ELEMENTAL_TUNING, ActionType.SWAP_CHARACTER,
              ActionType.END_ROUND]


class AttackAgent:
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


def match(deck, games, agent_cls=AttackAgent, seed=7):
    random.seed(seed)
    env = LinearEnv()
    out = {"games": 0, "draws": 0, "rounds": [], "decisions": 0, "errors": 0}
    for _ in range(games):
        try:
            env.reset_with_decks(deck, deck)
            gs, _, _, turn, done = env.view()
            a1, a2 = agent_cls(), agent_cls()
            n = 0
            while not done and n < 3000:
                n += 1
                w = gs.waiting_for()
                agent = a1 if w is Pid.P1 else a2
                gs, _, _, turn, done = env.step(agent.choose_action([gs], w))
            out["games"] += 1
            out["decisions"] += n
            out["rounds"].append(env.full_view().round)
            if env.full_view().get_winner() is None:
                out["draws"] += 1
        except Exception:
            out["errors"] += 1
    return out


def main():
    games = 0
    export = "--export" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            games = int(arg)
    built = {}
    print("=" * 96)
    for name, spec in DECKS.items():
        deck, problems = build(name, spec)
        if deck is None:
            print(f"[✗] {name}\n     " + "\n     ".join(problems))
            continue
        built[name] = deck
        kinds = len(spec["cards"])
        print(f"[✓] {name:38s} 3 角色 / 30 张 / {kinds} 种")
    print("=" * 96)

    if export:
        import json
        from dgisim.deck import FrozenDeck
        base = ROOT / "decks"
        written = 0
        for name, spec in DECKS.items():
            if name not in built:
                continue
            folder = base / ("test" if name.startswith("V") else "train")
            folder.mkdir(parents=True, exist_ok=True)
            slug = name.split("（")[0].replace(" ", "_").replace("/", "_")
            path = folder / f"{slug}.json"
            payload = {"chars": spec["chars"], "cards": spec["cards"]}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            # 读回校验：确保训练侧能直接 FrozenDeck.from_json 加载
            loaded = FrozenDeck.from_json(path.read_text(encoding="utf-8"))
            assert loaded is not None and DefaultMode().valid_deck(loaded), path
            written += 1
        print(f"已导出 {written} 份卡组到 decks/train/ 与 decks/test/（读回校验通过）")

    if not games:
        print("校验完成。加一个数字参数即可实测，例如：deck_pool.py 40")
        return
    print(f"\n攻击策略（优先技能）内战，每套 {games} 局，seed=7\n")
    for name, deck in built.items():
        s = match(deck, games)
        n = max(1, s["games"])
        print(f"{name:38s} 局={s['games']:3d} "
              f"平局={s['draws'] / n:5.0%} 平均回合={sum(s['rounds']) / n:4.1f} "
              f"决策/局={s['decisions'] / n:4.0f} 异常={s['errors']}")


if __name__ == "__main__":
    main()
