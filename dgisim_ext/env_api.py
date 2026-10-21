"""M1 / T1：`CardEnv` —— D-A §2 冻结的 4 函数接口。

只使用引擎公开面（`src.dgisim.*`），不碰内核。

为什么不是 `LinearEnv`
----------------------
`LinearEnv` 只吃"已经拼好的 `PlayerAction`"，而本引擎的动作是**分层**的：
一次落子要经过若干次 `ActionGenerator.choose()` 才 `filled()`，且 `choices()` 返回
四种异质类型（`tuple` / `ActualDice` / `AbstractDice` / `Cards`，见
`action_generator.py:77`）。所以这里暴露的是"推进一层 choice"的接口，
让策略头可以按需要打分，而不是先把动作压平成定长离散。

四个函数
--------
`reset(deck_p1, deck_p2, seed) -> Obs`
`obs(seat) -> Obs`                 该席位视角下的观测（含当前层候选项）
`step(choice_index) -> StepOut`    推进一层 choice；`filled` 时落子
`fork() -> CardEnv`                共享同一个不可变 state 的分叉（rollout / 搜索）

奖励口径
--------
按 `deviations.md` D2：击杀 `{胜者:+1, 败者:-1}`；**超时双方都 −1**。
所以 `info["rewards"]` 是二元组，训练侧禁止写 `r_p2 = -r_p1`。
"""
from __future__ import annotations

import random
import time
from contextlib import contextmanager
from enum import Enum
from threading import RLock
from dataclasses import dataclass, field
from typing import Any

from src.dgisim.action.action import PlayerAction
from src.dgisim.action.action_generator import ActionGenerator
from src.dgisim.card.cards import Cards
from src.dgisim.deck import Deck
from src.dgisim.dice import AbstractDice, ActualDice
from src.dgisim.element import Element
from src.dgisim.encoding.encoding_plan import EncodingPlan
from src.dgisim.encoding.mappings import (
    CARD_MAPPING, CHAR_MAPPING, EFFECT_MAPPING, ENUM_MAPPING, MODE_MAPPING,
    STT_MAPPING, SUMM_MAPPING, SUPP_MAPPING,
)
from src.dgisim.helper.hashable_dict import HashableDict
from src.dgisim.mode import Mode
from src.dgisim.state.enums import Pid
from src.dgisim.state.game_state import GameState

from .modes import ForkMode, UpstreamCompatMode
from .payoff import payoff_of
from .selection import Selection
from .encoding import FastEncodingPlan

_RNG_LOCK = RLock()


class CardEnvError(RuntimeError):
    """A poisoned episode: discard its samples, keep its diagnostic record."""
    def __init__(self, category: str, message: str):
        super().__init__(f'{category}: {message}')
        self.category = category


class InvalidChoiceError(CardEnvError, ValueError):
    pass

__all__ = ["Choice", "Obs", "StepOut", "CardEnv", "CardEnvError", "build_plan", "DEFAULT_PLAN"]

# ---------------------------------------------------------------- 编码计划
# `ForkMode` / `UpstreamCompatMode` 是我们在 `dgisim_ext` 里新加的 Mode 子类，
# 引擎自带的 `encoding_plan` 里没有它们的码。为了不动引擎（`src/**` 有 diff 预算），
# 这里用同一批 mapping 造一个**只多两个 mode 码**的计划：维度与布局不变。
_ALL_MAPPINGS = (ENUM_MAPPING, CARD_MAPPING, CHAR_MAPPING, EFFECT_MAPPING,
                 MODE_MAPPING, STT_MAPPING, SUMM_MAPPING, SUPP_MAPPING)
_NEXT_CODE = max(max(m.values()) for m in _ALL_MAPPINGS) + 1


def build_plan() -> EncodingPlan:
    """引擎布局 + 两个 fork 私有 mode 码（码值接在全部现有码之后，保证不撞）。"""
    return FastEncodingPlan(
        mode_mapping=HashableDict({
            **MODE_MAPPING,
            ForkMode: _NEXT_CODE,
            UpstreamCompatMode: _NEXT_CODE + 1,
        }),
        card_mapping=CARD_MAPPING,
        enum_mapping=ENUM_MAPPING,
        char_mapping=CHAR_MAPPING,
        effect_mapping=EFFECT_MAPPING,
        status_mapping=STT_MAPPING,
        summon_mapping=SUMM_MAPPING,
        support_mapping=SUPP_MAPPING,
    )


#: 本模块默认使用的编码计划（与 `LinearEnv` 的 `encoding_plan` 只差 mode 码）
DEFAULT_PLAN = build_plan()

#: `Choice.kind` 的取值
KIND_TUPLE = "tuple"        # 从 tuple 里挑一个（ActionType 等）
KIND_CARD = "card"          # 从 `Cards` 里挑一张
KIND_DICE = "dice"          # 从 `ActualDice` 里挑一个子集
KIND_COST = "cost"          # `AbstractDice` 费用（逐元素选择合法支付数量）
KIND_COMMIT = "commit"      # 已 filled，选它就是落子


@dataclass(frozen=True)
class Choice:
    """一个候选项：**内容 + 内容特征**，不含 index 语义（顺序不可依赖）。"""

    kind: str
    payload: Any                    # 引擎选项，或适配层 (种类, 数量)
    feature: tuple[float, ...]      # 顺序无关的内容特征（策略头据此打分）

    @property
    def label(self) -> str:
        if self.kind.endswith('_count'):
            item, count = self.payload
            return f'{self.kind}:{getattr(item, "__name__", getattr(item, "name", str(item)))}={count}'
        if self.kind == KIND_COMMIT:
            return "COMMIT"
        if self.kind == KIND_TUPLE:
            return getattr(self.payload, "__name__", getattr(self.payload, "name", str(self.payload)))
        if self.kind == KIND_CARD:
            (card,) = self.payload.ordered_cards[:1]
            return card.__name__
        if self.kind == KIND_DICE:
            return "dice:" + ",".join(
                f"{e.name}{n}" for e, n in sorted(
                    self.payload.readonly_dice_collection_ordered().items(),
                    key=lambda kv: kv[0].value,
                ) if n
            )
        return "cost"


@dataclass(frozen=True)
class Obs:
    """一个决策点（或一个中间 choice 层）的观测。"""

    seat: Pid
    enc: tuple[int, ...]              # 视角视图编码（2778 维，见 D-C）
    choices: tuple[Choice, ...]
    filled: bool
    decision_index: int               # 本局第几个决策态
    stage_index: int                  # 本决策内第几层 choice
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.enc)


@dataclass(frozen=True)
class StepOut:
    obs: Obs | None
    reward: float                     # 该席位本步奖励（非终局恒为 0）
    done: bool
    info: dict

    @property
    def terminal_reason(self) -> str | None:
        return self.info.get("terminal_reason")


class CardEnv:
    """分层动作的两席位对局环境。引擎随机源是进程级 `random`，见 `reset()`。"""

    def __init__(
            self,
            mode: Mode | None = None,
            plan: EncodingPlan = DEFAULT_PLAN,
            timeout_penalty: float = -1.0,
    ) -> None:
        self._mode = mode or ForkMode()
        self._plan = plan
        self._timeout_penalty = timeout_penalty
        self._rng_state = random.Random().getstate()
        with self._engine_rng():
            self._gs: GameState = GameState.from_default(self._mode)
        self._gen: ActionGenerator | None = None
        self._seat: Pid = Pid.P1
        self._decision_index = 0
        self._stage_index = 0
        self._log: list[str] = []
        self._selection: Selection | None = None
        self._choices_cache: tuple[Choice, ...] | None = None
        self._enc_cache: dict[Pid, tuple[int, ...]] = {}
        self._prefix: tuple[str, ...] = ()
        self._failed = False
        self._stats = self._empty_stats()

    @staticmethod
    def _empty_stats():
        return dict(attempts=0, commits=0, illegal=0, exception=0, noop=0,
                    encodes=0, encode_seconds=0.0, engine_seconds=0.0)

    @contextmanager
    def _engine_rng(self):
        # Engine uses process-global random. Swap only at the public boundary.
        with _RNG_LOCK:
            external = random.getstate()
            random.setstate(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = random.getstate()
                random.setstate(external)

    def _fail(self, category, message):
        self._stats[category] += 1
        self._failed = True
        error = InvalidChoiceError if category == 'illegal' else CardEnvError
        raise error(category, message)

    def _stage_changed(self):
        self._choices_cache = None
        self._selection = None

    # ------------------------------------------------------------ reset
    def reset(self, deck_p1: Deck | None = None, deck_p2: Deck | None = None,
              seed: int | None = None) -> Obs:
        """开一局并推进到第一个决策点。

        `seed` 同时决定引擎抽牌/掷骰（进程级 `random`）。**策略采样不要用它**：
        策略必须自己持有 `random.Random(...)`，否则 CRN 与"同套件配对评估"都做不了。
        """
        if (deck_p1 is None) != (deck_p2 is None):
            raise ValueError('provide both decks or neither')
        if seed is not None:
            self._rng_state = random.Random(seed).getstate()
        self._log = []
        self._decision_index = 0
        self._stage_index = 0
        self._stats = self._empty_stats()
        self._failed = False
        self._prefix = ()
        self._enc_cache = {}
        self._stage_changed()
        try:
            with self._engine_rng():
                if deck_p1 is None:
                    self._gs = GameState.from_default(self._mode)
                else:
                    if not all(self._mode.valid_deck(d) for d in (deck_p1, deck_p2)):
                        self._fail('illegal', 'invalid deck')
                    self._gs = GameState.from_decks(self._mode, deck_p1, deck_p2)
                self._advance()
            return self.obs()
        except CardEnvError:
            raise
        except Exception as exc:
            self._fail('exception', f'reset: {type(exc).__name__}: {exc}')

    # ------------------------------------------------------------ 内部
    def _advance(self) -> None:
        """引擎自身推进到下一个决策点（或终局）。"""
        guard = 0
        while self._gs.waiting_for() is None and not self._gs.game_end():
            self._gs = self._gs.step()
            guard += 1
            if guard > 100000:      # pragma: no cover - 防御
                raise RuntimeError("引擎没有推进到决策点")
        if self._gs.game_end():
            self._gen = None
            return
        self._seat = self._gs.waiting_for()  # type: ignore[assignment]
        self._gen = self._gs.action_generator(self._seat)
        assert self._gen is not None, "决策态却拿不到 ActionGenerator"
        self._decision_index += 1
        self._stage_index = 0
        self._prefix = ()
        self._stage_changed()

    def _feature_of(self, kind: str, payload: Any) -> tuple[float, ...]:
        """内容特征：只依赖候选项本身，不依赖它在列表里的位置。"""
        if kind == KIND_TUPLE:
            if isinstance(payload, (type, Enum)):
                return (float(self._plan.encode_item(payload)), 1.0)
            if isinstance(payload, int):
                return (float(payload), 1.0)
            # StaticTarget and similar engine dataclasses encode their contents.
            return tuple(float(v) for v in payload.encoding(self._plan)) + (1.0,)
        if kind == KIND_CARD:
            (card,) = payload.ordered_cards[:1]
            return (float(self._plan.encode_item(card)), 2.0)
        if kind == KIND_DICE:
            counts = payload.readonly_dice_collection_ordered()
            return (float(sum(counts.values())),
                    float(sum(e.value * n for e, n in counts.items())), 3.0)
        if kind == KIND_COST:
            return (0.0, 0.0, 4.0)
        return (0.0, 0.0, 5.0)

    def _choices(self) -> tuple[Choice, ...]:
        assert self._gen is not None
        if self._choices_cache is not None:
            return self._choices_cache
        if self._gen.filled():
            self._choices_cache = (Choice(KIND_COMMIT, None, (0.0, 0.0, 5.0)),)
            return self._choices_cache
        if self._selection is not None:
            return self._selection_choices()
        raw = self._gen.choices()
        out: list[Choice] = []
        if isinstance(raw, tuple):
            for item in sorted(set(raw), key=lambda x: self._feature_of(KIND_TUPLE, x)):
                out.append(Choice(KIND_TUPLE, item, self._feature_of(KIND_TUPLE, item)))
        elif isinstance(raw, Cards):
            self._selection = Selection.cards(raw)
        elif isinstance(raw, AbstractDice):
            available = self._gs.get_player(self._seat).dice
            self._selection = Selection.payment(available, raw)
        elif isinstance(raw, ActualDice):
            self._selection = Selection.dice(raw)
        else:  # pragma: no cover - 防御
            raise NotImplementedError(f"未知 choices 类型: {type(raw)}")
        if self._selection is not None:
            return self._selection_choices()
        if not out:
            self._fail('illegal', 'empty candidate set')
        self._choices_cache = tuple(out)
        return self._choices_cache

    def _selection_choices(self):
        selection = self._selection
        if selection.done:
            self._gen = self._gen.choose(selection.result())
            self._stage_changed()
            return self._choices()
        item = selection.items[len(selection.prefix)]
        code = self._plan.encode_item(item)
        kind_code = {'dice': 3.0, 'card': 2.0, 'cost': 4.0}[selection.kind]
        self._choices_cache = tuple(
            Choice(selection.kind + '_count', (item, n), (float(code), float(n), kind_code))
            for n in selection.counts())
        if not self._choices_cache:
            self._fail('illegal', 'no legal selection completion')
        return self._choices_cache

    # ------------------------------------------------------------ obs
    def obs(self, seat: Pid | None = None) -> Obs:
        """`seat` 视角下的观测。终局时返回空候选集。"""
        seat = seat or self._seat
        if self._failed:
            raise CardEnvError('failed', 'reset required')
        if seat not in self._enc_cache:
            start = time.perf_counter()
            view = self._gs.perspective_view(seat)
            self._enc_cache[seat] = tuple(self._plan.encode(view, seat))
            self._stats['encodes'] += 1
            self._stats['encode_seconds'] += time.perf_counter() - start
        enc = self._enc_cache[seat]
        if self._gen is None:
            choices: tuple[Choice, ...] = ()
        elif seat is self._seat:
            choices = self._choices()
        else:
            choices = ()       # 不是该席位行动，没有候选项
        return Obs(
            seat=seat,
            enc=enc,
            choices=choices,
            filled=bool(seat is self._seat and self._gen and self._gen.filled()),
            decision_index=self._decision_index,
            stage_index=self._stage_index if seat is self._seat else 0,
            meta={"waiting_for": self._seat.name if self._gen else None,
                  "choice_prefix": self._prefix if seat is self._seat else (),
                  "selection_prefix": self._selection.prefix
                      if seat is self._seat and self._selection else ()},
        )

    # ------------------------------------------------------------ step
    def step(self, choice_index: int) -> StepOut:
        """推进一层 choice。

        - `filled` 时落子（推进真实状态），奖励按 D2 口径结算；
        - 否则只是把 choice 喂给 `ActionGenerator.choose()`，状态不变、奖励为 0。
        """
        if self._failed:
            raise CardEnvError('failed', 'reset required')
        if self._gen is None:
            raise CardEnvError('ended', 'reset required')
        self._stats['attempts'] += 1
        try:
            return self._step(choice_index)
        except CardEnvError:
            raise
        except Exception as exc:
            self._fail('exception', f'{type(exc).__name__}: {exc}')

    def _step(self, choice_index):
        choices = self._choices()
        if type(choice_index) is not int or not 0 <= choice_index < len(choices):
            self._fail('illegal', f'invalid choice index: {choice_index!r}')
        choice = choices[choice_index]
        actor = self._seat
        self._stage_index += 1
        self._log.append(f"{self._decision_index}.{self._stage_index}:{choice.label}")
        self._prefix += (choice.label,)
        if self._stage_index > 100:
            self._fail('noop', 'choice stage limit exceeded')

        if choice.kind != KIND_COMMIT:
            if choice.kind.endswith('_count'):
                self._selection = self._selection.choose(choice.payload[1])
                self._choices_cache = None
                if self._selection.done:
                    self._gen = self._gen.choose(self._selection.result())
                    self._stage_changed()
            else:
                self._gen = self._gen.choose(choice.payload)
                self._stage_changed()
            return StepOut(self.obs(), 0.0, False,
                           {"filled": self._gen.filled(), "actor": actor.name,
                            "rewards": {Pid.P1: 0.0, Pid.P2: 0.0}, "committed": False})

        action: PlayerAction = self._gen.generate_action()
        before = (hash(self._gs), self._gs.waiting_for())
        start = time.perf_counter()
        with self._engine_rng():
            state = self._gs.action_step(actor, action)
            if state is None:
                self._fail('illegal', f'rejected {type(action).__name__}')
            if (hash(state), state.waiting_for()) == before:
                self._fail('noop', f'no state transition: {type(action).__name__}')
            self._gs = state
            self._advance()
        self._stats['engine_seconds'] += time.perf_counter() - start
        self._stats['commits'] += 1
        self._enc_cache = {}
        self._stage_changed()
        if self._gs.game_end():
            rewards = payoff_of(self._gs, self._timeout_penalty)
            return StepOut(
                None,
                rewards[actor],
                True,
                {"rewards": rewards, "terminal_reason": _reason(self._gs),
                 "decision_index": self._decision_index, "actor": actor.name,
                 "committed": True},
            )
        return StepOut(self.obs(), 0.0, False,
                       {"filled": False, "actor": actor.name, "committed": True,
                        "rewards": {Pid.P1: 0.0, Pid.P2: 0.0}})

    def fork(self) -> "CardEnv":
        """共享同一个不可变 `GameState` 的分叉；choice 层状态也一并复制。"""
        new = object.__new__(type(self))
        new._mode = self._mode
        new._plan = self._plan
        new._timeout_penalty = self._timeout_penalty
        new._gs = self._gs
        new._gen = self._gen
        new._seat = self._seat
        new._decision_index = self._decision_index
        new._stage_index = self._stage_index
        new._log = list(self._log)
        new._rng_state = self._rng_state
        new._selection = self._selection
        new._choices_cache = self._choices_cache
        new._enc_cache = dict(self._enc_cache)
        new._prefix = self._prefix
        new._failed = self._failed
        new._stats = dict(self._stats)
        return new

    @property
    def stats(self):
        return dict(self._stats)

    @property
    def failed(self):
        return self._failed

    # ------------------------------------------------------------ 只读属性
    @property
    def state(self) -> GameState:
        return self._gs

    @property
    def seat(self) -> Pid:
        return self._seat

    @property
    def action_log(self) -> tuple[str, ...]:
        return tuple(self._log)

    @property
    def done(self) -> bool:
        return self._gs.game_end()


def _reason(state: GameState) -> str:
    return "timeout" if state.get_winner() is None else f"kill:{state.get_winner().name}"
