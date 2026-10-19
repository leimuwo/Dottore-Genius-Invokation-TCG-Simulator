"""把一张 GameState 摊开成"12 个格子"的面板，并在游戏过程中只打印变化的格子。

运行：python analysis/inspect_state.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402


def _len(obj):
    try:
        return len(obj)
    except TypeError:
        return sum(1 for _ in obj)


def panel(gs):
    """把一张快照摊平成 20 个格子（键 -> 可读字符串）。"""
    out = {}
    out["回合 round"] = str(gs.round)
    out["全局阶段 GameState.phase"] = type(gs.phase).__name__
    out["先手权 active_player_id"] = gs.active_player_id.name
    out["效果栈 effect_stack"] = f"{len(gs.effect_stack._effects)} 件"
    out["栈顶要结算的事"] = "-" if gs.effect_stack.is_empty() else type(gs.effect_stack.peek()).__name__
    out["现在等谁动手 waiting_for"] = "-" if gs.waiting_for() is None else gs.waiting_for().name
    for tag, ps in (("P1", gs.player1), ("P2", gs.player2)):
        chars = "  ".join(
            f"{type(c).__name__}({c.hp}/{c.max_hp},能量{c.energy})"
            for c in ps.characters
        )
        out[f"{tag} 玩家状态 player.phase"] = ps.phase.name
        out[f"{tag} 骰子 dice"] = f"{ps.dice.num_dice()} 个"
        out[f"{tag} 手牌 hand_cards"] = f"{ps.hand_cards.num_cards()} 张"
        out[f"{tag} 牌库 deck_cards"] = f"{ps.deck_cards.num_cards()} 张"
        out[f"{tag} 出战角色"] = str(ps.characters.get_active_character_id())
        out[f"{tag} 三个角色"] = chars
        out[f"{tag} 状态/召唤/支援"] = f"{_len(ps.hidden_statuses)} / {_len(ps.summons)} / {_len(ps.supports)}"
    return out


def show(title, gs):
    print(f"\n===== {title} =====")
    for k, v in panel(gs).items():
        print(f"  {k:26s} {v}")


def show_diff(k, kind, before, after, mark=""):
    changed = [(key, before[key], after[key]) for key in before if before[key] != after[key]]
    head = f"#{k:04d}{mark} {kind}"
    if not changed:
        print(f"{head:52s} （没有任何格子变化）")
        return
    print(f"{head}")
    for key, a, b in changed:
        print(f"        {key:26s} {a:>18s}  →  {b}")


def main():
    random.seed(11)
    gs = GameState.from_default()
    agent = RandomAgent()
    show("快照 0：刚创建（from_default）", gs)

    prev = panel(gs)
    k = 0
    first_step_shown = False
    first_action_shown = False
    skill_shown = False
    while not gs.game_end() and k < 400:
        w = gs.waiting_for()
        if w is not None:
            action = agent.choose_action([gs], w)
            nxt = gs.action_step(w, action)
            kind = f"action_step({type(action).__name__}) by {w.name}"
        else:
            nxt = gs.step()
            kind = "step()"
        k += 1
        cur = panel(nxt)
        jump_to_real = cur["全局阶段 GameState.phase"] == "ActionPhase" and prev["全局阶段 GameState.phase"] != "ActionPhase"
        show_diff(k, kind, prev, cur, "◆" if cur["现在等谁动手 waiting_for"] != "-" else " ")
        if not first_step_shown and k == 1:
            show("快照 1：第一次 step()（发牌）之后", nxt)
            first_step_shown = True
        if jump_to_real:
            show("快照：刚进入 ActionPhase（行动阶段）", nxt)
        if (not skill_shown) and kind.startswith("action_step(SkillAction)"):
            show("快照：刚放完一个技能（这一瞬间技能只是被'登记'了，还没结算）", nxt)
            skill_shown = True
        gs, prev = nxt, cur


if __name__ == "__main__":
    main()
