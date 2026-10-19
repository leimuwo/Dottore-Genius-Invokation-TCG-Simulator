"""量 B5：下回合的"先手"到底归谁。

定义（按官方规则）：先手 = 下回合 ActionPhase 里第一个行动的人。
先把"谁先声明结束回合"和"下回合谁先动"都记录下来，再比对。

运行：python analysis/probe_initiative.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgisim.action.action import EndRoundAction  # noqa: E402
from dgisim.agents import RandomAgent  # noqa: E402
from dgisim.state.game_state import GameState  # noqa: E402


def one_game(seed: int) -> tuple[int, int]:
    random.seed(seed)
    agent = RandomAgent()
    gs = GameState.from_default()
    first_end: dict[int, str] = {}
    first_actor: dict[int, str] = {}
    steps = 0
    while not gs.game_end() and steps < 6000:
        w = gs.waiting_for()
        if w is None:
            gs = gs.step()
            steps += 1
            continue
        rnd = gs.round
        phase = type(gs.phase).__name__
        if phase == "ActionPhase":
            first_actor.setdefault(rnd, w.name)
        action = agent.choose_action([gs], w)
        # 注意：EndRoundAction 在 CardSelectPhase 里表示"放弃换牌"，在 RollPhase 里
        # 表示"不保留骰子"，只有 ActionPhase 里的才是"结束回合"。
        if isinstance(action, EndRoundAction) and phase == "ActionPhase":
            first_end.setdefault(rnd, w.name)
        gs = gs.action_step(w, action)
        steps += 1

    pairs = [(r, first_end[r], first_actor.get(r + 1)) for r in sorted(first_end) if r + 1 in first_actor]
    same = sum(1 for _, e, a in pairs if e == a)
    return same, len(pairs)


def main() -> None:
    total_same = total = 0
    for seed in range(10):
        same, n = one_game(seed)
        total_same += same
        total += n
        print(f"seed {seed}: {same}/{n}")
    print(f"\n合计：下回合先手 == 先结束者 : {total_same}/{total}")


if __name__ == "__main__":
    main()
