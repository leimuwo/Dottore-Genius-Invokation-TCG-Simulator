"""验证用户指出的两处 LinearEnv / 引擎行为：
A) copy() 的第二个副作用：__init__ 里的 reset() 会抽一局新牌库，推进全局 RNG
B) 非法/错类型动作在 CardSelectPhase 的错误语义（三条不一致路径）

运行：python analysis/probe_actions.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dgisim.env.linear_env import LinearEnv  # noqa: E402
from dgisim.state.game_state import GameState  # noqa: E402
from dgisim.state.enums import Pid  # noqa: E402
from dgisim.encoding.encoding_plan import encoding_plan  # noqa: E402
from dgisim.action.action import (  # noqa: E402
    CardsSelectAction, CharacterSelectAction, DiceSelectAction, EndRoundAction,
    PlayerAction, SwapAction, DiceOnlyInstruction,
)
from dgisim.card.cards import Cards  # noqa: E402
from dgisim.dice import ActualDice  # noqa: E402


def part_a():
    print("=== A) copy() 的两个副作用 ===")

    def stream(n=4):
        return [round(random.random(), 12) for _ in range(n)]

    random.seed(2024)
    LinearEnv()
    base = stream()

    random.seed(2024)
    env = LinearEnv()
    env.copy()
    after_copy = stream()

    print(f"  baseline stream : {base}")
    print(f"  after .copy()   : {after_copy}")
    print(f"  RNG 流是否被推进 : {base != after_copy}")

    # 只看 copy() 自身消耗了多少随机数（用比较法而非计数法，避免依赖实现）
    random.seed(7)
    LinearEnv()
    a = stream(1)[0]
    random.seed(7)
    LinearEnv()
    LinearEnv()          # 一次完整 reset ≈ 一局新牌库
    b = stream(1)[0]
    print(f"  一次完整 reset 后的首个数 : {b}  (对比无 reset: {a})  -> 等价: {a == b}")

    e2 = LinearEnv(fix_perspective=True)
    print(f"  fix_perspective=True -> copy()._fix_perspective = {e2.copy()._fix_perspective}")


def find_card_select_state(seed=7, max_steps=50):
    random.seed(seed)
    gs = GameState.from_default()
    for _ in range(max_steps):
        if type(gs.phase).__name__ == "CardSelectPhase" and gs.waiting_for() is not None:
            return gs
        if gs.game_end():
            break
        gs = gs.step()
    return None


def describe(gs):
    return {
        "phase": type(gs.phase).__name__,
        "p1": gs.player1.phase.name,
        "p2": gs.player2.phase.name,
        "waiting_for": None if gs.waiting_for() is None else gs.waiting_for().name,
    }


def part_b():
    print("\n=== B) CardSelectPhase 的动作错误语义 ===")
    gs = find_card_select_state()
    if gs is None:
        print("  未找到 CardSelectPhase 决策点，跳过")
        return
    pid = gs.waiting_for()
    print(f"  初始: {describe(gs)}  (pid={pid.name})")

    candidates = [
        ("EndRoundAction", EndRoundAction()),
        ("all-zero 编码(→EndRoundAction)", PlayerAction.decoding([0] * 81, encoding_plan)),
        ("CardsSelectAction(空选择)", CardsSelectAction(selected_cards=Cards({}))),
        ("CharacterSelectAction(0)", CharacterSelectAction(char_id=0)),
        ("DiceSelectAction(空骰)", DiceSelectAction(selected_dice=ActualDice({}))),
        ("SwapAction(0)", SwapAction(char_id=0, instruction=DiceOnlyInstruction(dice=ActualDice({})))),
    ]

    for name, action in candidates:
        if action is None:
            print(f"  - {name:32s} decoding 返回 None")
            continue
        try:
            new_gs = gs.action_step(pid, action)
        except Exception as e:  # noqa: BLE001
            print(f"  - {name:32s} 抛异常 {type(e).__name__}: {str(e).splitlines()[0][:60]}")
            continue
        if new_gs is None:
            print(f"  - {name:32s} 返回 None  (LinearEnv 里会 assert 失败)")
            continue
        changed = hash(new_gs) != hash(gs)
        print(f"  - {name:32s} 被接受 | 状态改变={changed} | {describe(new_gs)}")

    # LinearEnv 层面的三条路径表现
    print("\n  --- LinearEnv.step 对同一状态的三条路径 ---")
    env = LinearEnv()
    env._curr_state = gs
    for name, action in candidates[:1] + candidates[5:6]:
        try:
            _, _, reward, turn, done = env.step(action)
            print(f"  - {name:32s} reward={reward} turn={turn} done={done}")
        except AssertionError as e:
            print(f"  - {name:32s} AssertionError (未处理)")
        except Exception as e:  # noqa: BLE001
            print(f"  - {name:32s} {type(e).__name__}: {str(e).splitlines()[0][:50]}")


if __name__ == "__main__":
    part_a()
    part_b()
