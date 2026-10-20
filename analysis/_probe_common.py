"""analysis/ 探针共用的测量装配。

只做三件事, 都不改变引擎语义:
1. `install_canonical_card_order()`: 把 `Cards.pick_random*` 的候选池顺序规范化为
   按类名排序。抽样分布不变(均匀), 但让"同 seed 两次运行"逐位可比。
   这是测量侧装配, 不修改 `src/`; 见 docs/decisions/D-D-reward-shaping.md。
2. `play_random_game()`: 固定 seed 打一整局, 产出「决策态」序列与终局分类。
3. 统计小工具: `wilson_ci` / `auc_binary` / `phi_public`。
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dgisim.agents import RandomAgent                      # noqa: E402
from dgisim.card.cards import Cards                       # noqa: E402
from dgisim.mode import DefaultMode, Mode                 # noqa: E402
from dgisim.state.enums import Pid                        # noqa: E402
from dgisim.state.game_state import GameState             # noqa: E402

MAX_STEPS = 60000  # 单局 step 上界, 只为防死循环; 正常对局远低于此


# ---------------------------------------------------------------- 装配
def install_canonical_card_order() -> None:
    """把 `Cards.pick_random*` 的候选池顺序按类名排序（均匀抽样, 分布不变）。"""
    def _keys(cards: Cards) -> list:
        return sorted(cards._cards.keys(), key=lambda c: c.__name__)

    def _pick_random(self: Cards, num: int):
        num = min(self.num_cards(), num)
        if num == 0:
            return (self, Cards.from_empty())
        picked = dict(Counter(
            random.sample(_keys(self), counts=self._cards.values(), k=num)
        ))
        return Cards(self._cards - picked), Cards(picked)

    def _pick_random_of_type(self: Cards, num: int, card_type):
        qualified = {c: n for c, n in self._cards.items() if issubclass(c, card_type)}
        num = min(sum(qualified.values()), num)
        if num == 0:
            return (self, Cards.from_empty())
        picked = dict(Counter(random.sample(
            sorted(qualified.keys(), key=lambda c: c.__name__),
            counts=qualified.values(), k=num,
        )))
        return Cards(self._cards - picked), Cards(picked)

    Cards.pick_random = _pick_random
    Cards.pick_random_of_type = _pick_random_of_type


# ---------------------------------------------------------------- 对局
def play_random_game(seed: int, mode: Mode | None = None,
                     agent: RandomAgent | None = None,
                     sink: dict | None = None) -> Iterator[GameState]:
    """固定 seed 打一整局, 产出每个「决策态」(waiting_for() is not None)。

    调用方用完即弃, 不保留全部状态 → 内存 O(1)。
    终局信息写进 `sink`（`outcome` = 'kill:P1'/'kill:P2'/'timeout'，`steps` = step 数），
    因为生成器无法把返回值交给调用方。
    """
    mode = mode or DefaultMode()
    agent = agent or RandomAgent()
    random.seed(seed)
    gs = GameState.from_default(mode)
    history: list[GameState] = []
    steps = 0
    while not gs.game_end() and steps < MAX_STEPS:
        steps += 1
        who = gs.waiting_for()
        if who is None:
            gs = gs.step()
            history = []
            continue
        yield gs
        action = agent.choose_action(history + [gs], who)
        nxt = gs.action_step(who, action)
        if nxt is None:  # 非法动作: 本局丢弃, 由调用方计数
            raise RuntimeError(f"illegal action @seed={seed}")
        gs = nxt
        history.append(gs)
    if not gs.game_end():
        raise RuntimeError(f"未终止 @seed={seed}")
    if sink is not None:
        sink["outcome"] = terminal_kind(gs)
        sink["steps"] = steps


def terminal_kind(gs: GameState) -> str:
    """'kill:P1' / 'kill:P2' / 'timeout'。超时=双方判负(与 deviations.md D2 一致)。"""
    winner = gs.get_winner()
    return "timeout" if winner is None else f"kill:{winner.name}"


# ---------------------------------------------------------------- 统计
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 区间（小样本/极端比例比正态近似稳）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def auc_binary(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U 形式的 AUC（并列记 0.5），不引入 sklearn。"""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    sum_rank_pos = 0.0
    i, n = 0, len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        group_len = j - i + 1
        n_pos_in_group = sum(1 for _, la in pairs[i:j + 1] if la == 1)
        neg_before = sum(1 for _, la in pairs[:i] if la == 0)
        for _, la in pairs[i:j + 1]:
            if la == 1:
                sum_rank_pos += neg_before + (group_len - n_pos_in_group) * 0.5
        i = j + 1
    return sum_rank_pos / float(n_pos * n_neg)


# ---------------------------------------------------------------- Φ
def _public_value(player) -> float:
    """只用**公开可见**的字段: 存活角色数 + 总血量。"""
    return 10.0 * len(player.characters.get_alive_characters()) + sum(
        c.hp for c in player.characters
    )


def phi_public(gs: GameState, pid: Pid) -> float:
    """席位 `pid` 视角下的势函数 Φ（经 `perspective_view` 取, 天然看不到对方手牌/牌序）。

    Φ = 10·(自己存活角色数 − 对方) + (自己总血 − 对方总血)，对席位反对称。
    """
    view = gs.perspective_view(pid)
    me, opp = ((view.player1, view.player2) if pid is Pid.P1
               else (view.player2, view.player1))
    return _public_value(me) - _public_value(opp)


def phi_leaky(gs: GameState, pid: Pid) -> float:
    """**故意的作弊版 Φ**（对照用，永不进训练）: 额外偷看对方牌堆顶 3 张及其顺序。

    存在的唯一目的是给"泄漏检测"提供正对照: 如果扰动对方牌序后它不变,
    说明检测器没有检出能力（假阴性）。
    """
    opp = gs.player2 if pid is Pid.P1 else gs.player1
    top = list(opp.deck_cards)[:3]
    return phi_public(gs, pid) + 0.001 * sum(
        (i + 1) * len(c.__name__) for i, c in enumerate(top)
    )
