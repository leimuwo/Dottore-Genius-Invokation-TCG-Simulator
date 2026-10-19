# deviations.md — 与归档版 dgisim 的差异登记

> 上游 `Jarvis-Yu/Dottore-Genius-Invokation-TCG-Simulator` 已 **archive**，基线 commit `fd39ddc`（v0.6.0 未发布）。
> 本表登记本 fork 相对基线的所有**行为差异**。每一条都必须有：一个 `enginefix(<ID>)` commit、一个 `tests/invariants` 用例、一个可关闭的开关（如果它改变博弈语义）。
> 论文附录直接引用本表回答"结论是否依赖规则改动"。

## 已修复的缺陷（不改变官方规则，只是把实现改对）

| ID | 内容 | 是否改变博弈语义 | 对应测试 |
|---|---|---|---|
| B14 | `Dice.encoding()` 不幂等（`cached_property` 缓存了迭代器，第二次编码骰子全 0） | 否（只影响观察） | `test_invariant_encoding_idempotent.py` |
| B11 | `GameState._all_unique_data()` 漏三个 effect stack，导致不同状态 hash 相等 | 否 | `test_invariant_state_identity.py` |
| B1 | `example_player` 采样依赖 `frozenset` 迭代顺序 → 跨进程同 seed 不可复现 | 否 | `test_invariant_sampling_determinism.py` + `scripts/verify_cross_process_sampling.py` |
| B12 | 自己的牌库顺序进入观察（泄漏未来抽牌） | 否（只影响观察） | `test_invariant_deck_order_not_observed.py` |
| B13 | `Cards` / `Dice` 编码依赖插入顺序（同一信息集两种向量） | 否（只影响观察） | `test_invariant_encoding_canonical.py` |
| B6 | `LinearEnv.view()` 在 P2 行动时用 P1 视角；`copy()` 丢 `fix_perspective` | 否 | `test_invariant_linear_env_contract.py` |
| B2 | `LinearEnv.copy()` 经 `__init__` 触发 `reset()`，白烧一整局随机数 | 否 | `test_invariant_linear_env_contract.py` |
| B7 | `StartingHandSelectPhase` 对无效 `char_id` 返回同一状态（无罚分无限空转） | 否（原先是不确定行为） | `test_invariant_illegal_action_semantics.py` |
| B9 | 非法元素调和 `print` 整张状态表 + `assert False`（`-O` 下退化为 `return None`） | 否 | `test_invariant_illegal_action_semantics.py` |
| B3 | `hide_secrets()` 未隐藏 `initial_deck`，对手卡组表泄漏 | 否（只影响观察） | `test_invariant_decklist_hidden.py` |

## 待实施的规则差异（会改变博弈语义，必须带开关）

| ID | 内容 | 开关 | 状态 |
|---|---|---|---|
| D2 | 15 回合耗尽 → **双方都判负**（各自 −1），不做"存活角色/血量"代偿判定 | `ForkMode.timeout_penalty` | open |
| D5 | 下回合先手 = **先结束回合的一方**（现状相反，实测 528/528） | `ForkMode.first_to_end_gets_initiative` | open |

## 记录规则

1. 新增一条差异前，先在 `tests/invariants` 写测试（它此刻必须是红的）。
2. 差异必须**可关闭**：`UpstreamCompatMode` 要能复现归档版行为，用于论文的"规则依赖"消融。
3. 不允许"顺手改"：一条差异一个 commit，message 用 `enginefix(<ID>): ...`。
