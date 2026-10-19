"""C) 是否存在"静默空转"（同一状态返回、无异常、无罚分）的错误路径
D) RollPhase 上非法骰子选择的语义

运行：python analysis/probe_noop.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.action.action import (  # noqa: E402
    CharacterSelectAction, DiceSelectAction, EndRoundAction,
)
from dgisim.dice import ActualDice  # noqa: E402
from dgisim.env.linear_env import LinearEnv  # noqa: E402


def desc(gs):
    w = gs.waiting_for()
    return (f"phase={type(gs.phase).__name__} p1={gs.player1.phase.name} "
            f"p2={gs.player2.phase.name} wait={w.name if w else None}")


def walk_to(phase_name, seed=7, max_steps=400):
    """用一个笨驱动走到目标阶段的决策点。"""
    random.seed(seed)
    gs = GameState.from_default()
    for _ in range(max_steps):
        if gs.game_end():
            return None
        if type(gs.phase).__name__ == phase_name and gs.waiting_for() is not None:
            return gs
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            continue
        try:
            if type(gs.phase).__name__ == "StartingHandSelectPhase":
                chars = gs.get_player(w).characters
                cid = [c.id for c in chars][0]
                gs = gs.action_step(w, CharacterSelectAction(char_id=cid))
            else:
                gs = gs.action_step(w, EndRoundAction())
        except Exception as e:  # noqa: BLE001
            print(f"  驱动在 {type(gs.phase).__name__} 卡住: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}")
            return None
    return None


def run(gs, label, action):
    pid = gs.waiting_for()
    try:
        new = gs.action_step(pid, action)
    except Exception as e:  # noqa: BLE001
        print(f"  - {label:34s} 抛 {type(e).__name__}: {str(e).splitlines()[0][:44]}")
        return
    if new is None:
        print(f"  - {label:34s} 返回 None（LinearEnv 会 assert 失败）")
        return
    print(f"  - {label:34s} 接受 | 同一对象={new is gs} hash相同={hash(new) == hash(gs)}"
          f" | {desc(new)}")


print("=== C) StartingHandSelectPhase：无效 char_id 是否静默空转 ===")
gs = walk_to("StartingHandSelectPhase")
if gs is None:
    print("  未找到该阶段决策点")
else:
    print(f"  初始: {desc(gs)}")
    run(gs, "CharacterSelectAction(99)", CharacterSelectAction(char_id=99))
    run(gs, "CharacterSelectAction(-1)", CharacterSelectAction(char_id=-1))
    run(gs, "EndRoundAction（错类型）", EndRoundAction())
    env = LinearEnv()
    env._curr_state = gs
    for i in range(2):
        try:
            _, _, r, turn, done = env.step(CharacterSelectAction(char_id=99))
            print(f"  LinearEnv.step(非法 char_id) 第{i + 1}次 -> reward={r} turn={turn} done={done}")
        except Exception as e:  # noqa: BLE001
            print(f"  LinearEnv.step 抛 {type(e).__name__}: {str(e).splitlines()[0][:60]}")

print("\n=== D) RollPhase：非法骰子选择 ===")
gs = walk_to("RollPhase")
if gs is None:
    print("  未找到该阶段决策点")
else:
    pid = gs.waiting_for()
    print(f"  初始: {desc(gs)}")
    print(f"  该玩家骰子: {gs.get_player(pid).dice}")
    run(gs, "DiceSelectAction(空骰)", DiceSelectAction(selected_dice=ActualDice({})))
    run(gs, "EndRoundAction（错类型）", EndRoundAction())
