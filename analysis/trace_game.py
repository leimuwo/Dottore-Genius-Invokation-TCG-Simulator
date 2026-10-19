"""把一局真实对局从开局到终局逐次转移打印出来，用来讲解"状态是怎么一步步被改动的"。

输出：analysis/trace_game.log
运行：python analysis/trace_game.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402


def snap(gs):
    def side(ps):
        return {
            "phase": ps.phase.name,
            "hand": ps.hand_cards.num_cards(),
            "deck": ps.deck_cards.num_cards(),
            "dice": ps.dice.num_dice(),
            "active": ps.characters.get_active_character_id(),
            "hp": sum(c.hp for c in ps.characters),
            "stt": len(ps.hidden_statuses),
        }

    return {
        "round": gs.round,
        "active": gs.active_player_id.name,
        "phase": type(gs.phase).__name__,
        "wait": "-" if gs.waiting_for() is None else gs.waiting_for().name,
        "stack": len(gs.effect_stack._effects),
        "top": "-" if gs.effect_stack.is_empty() else type(gs.effect_stack.peek()).__name__,
        "p1": side(gs.player1),
        "p2": side(gs.player2),
    }


def flat(s):
    out = {}
    for key in ("round", "active", "phase", "wait", "stack", "top"):
        out[key] = s[key]
    for pid in ("p1", "p2"):
        for k, v in s[pid].items():
            out[f"{pid}.{k}"] = v
    return out


def diff(a, b):
    fa, fb = flat(a), flat(b)
    keys = [k for k in fa if fa[k] != fb[k]]
    if not keys:
        return ""
    return " | ".join(f"{k}:{fa[k]}→{fb[k]}" for k in keys[:6])


def main():
    random.seed(11)
    gs = GameState.from_default()
    agent = RandomAgent()
    rows = []
    prev = snap(gs)
    rows.append(f"#{0:04d} {'[初始状态 from_default]':38s} -> {prev['phase']:24s} "
                f"round={prev['round']} wait={prev['wait']} stack={prev['stack']} top={prev['top']}")

    k = 0
    while not gs.game_end() and k < 5000:
        w = gs.waiting_for()
        if w is not None:
            action = agent.choose_action([gs], w)
            nxt = gs.action_step(w, action)
            kind = f"action_step({type(action).__name__}) by {w.name}"
        else:
            nxt = gs.step()
            kind = "step()"
        k += 1
        cur = snap(nxt)
        mark = "◆" if cur["wait"] != "-" else " "
        rows.append(f"#{k:04d}{mark}{kind:38s} -> {cur['phase']:24s} "
                    f"round={cur['round']} wait={cur['wait']:2s} stack={cur['stack']} "
                    f"top={cur['top']:28s} | {diff(prev, cur)}")
        gs, prev = nxt, cur

    with open(Path(__file__).with_name("trace_game.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"共 {k} 次转移，写入 analysis/trace_game.log")
    print("决策点数量:", sum(1 for r in rows if "◆" in r[:6]))


if __name__ == "__main__":
    main()
