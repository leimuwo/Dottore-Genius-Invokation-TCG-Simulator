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
| E6 | `ElementalAura.encoding()` 迭代 `frozenset` 而非声明顺序 → 同一状态**两个进程**算出的 obs 不同（首个分歧在每角色段偏移 9/11/13/15/17） | 否（只影响观察） | `test_invariant_encoding_order.py` + `analysis/probe_obs_vs_t1.py`（X1a） |
| E7 | 从 `set`/`dict` 抽样（`dice.py` `_LEGAL_ELEMS`/`PURE_ELEMENTS`/`all_cards()`/`summons`），以及 `Cards.__iter__`、`Dice.elems/__iter__/ordered_*` 继承 dict 插入顺序 → 同 seed 的**对局与 obs 在两个进程里都不同** | 否（只改随机数到候选的映射，抽样分布不变） | `test_invariant_sampling_order.py` + `analysis/probe_obs_vs_t1.py`（X1b） |
| E8 | `Status.encoding()` 用 `set` 遍历 dataclass 字段 → 同 seed、同轨迹的 `Support` 编码跨进程漂移 | 否（只影响观察） | `test_invariant_status_field_order.py` + `analysis/cardenv_acceptance.py --determinism` |

> E6/E7 的发现方式值得记一笔：这两条**不可能**在同一进程内测出来（同进程内集合/字典顺序固定），
> 只有"D-C × T1 交叉确认"里那条跨进程比对才把它们逼出来。E7 修完后 6 个 seed × 30 个决策态的
> obs 在两个独立进程里逐位相同。

## 待实施的规则差异（会改变博弈语义，必须带开关）

| ID | 内容 | 开关 | 状态 |
|---|---|---|---|
| D2 | 15 回合耗尽 → **双方都判负**（各自 −1），不做"存活角色/血量"代偿判定 | `ForkMode.timeout_penalty` | 已实现（`dgisim_ext/payoff.py` + `test_ext_contract.py`） |

## 待修复的缺陷（已知、**未修**，所以还不在上表）

| ID | 内容 | 谁卡在它上面 | 判据 |
|---|---|---|---|
| E5 | `Cards` 允许**负计数**（实测 `ChangingShifts: -1`）→ `num_cards()=sum(values)` 比真实张数少，而 `encoding` 按 `Counter.elements()` 给真实张数，两者不一致 | D-C（已改为从编码数占用，绕开了它）；任何用 `num_cards()` 的手牌数特征 | 实测 57/17 998 决策态（0.32%）触发 |

## 复核结论：D5 先手权——**归档版实现是正确的，不是偏差**

`dgisim-RL环境评估.md` 曾记录"下回合先手权反了（528/528）"。复核结论：**该结论来自测量错误**。

- 错误原因：`EndRoundAction` 在不同阶段含义不同——`CardSelectPhase` 是"放弃换牌"、`RollPhase` 是"不保留骰子"、只有 `ActionPhase` 才是"结束回合"。原探针把前两者也算成了"结束回合"，于是"先结束者"被记成了后一次换牌/换骰的人。
- 正确测量（`analysis/probe_initiative.py`）：**133/133 个回合**，下回合 ActionPhase 的第一个行动者 == 本回合先声明结束的人。
- 因此：**不需要任何规则修正**，也不再保留 `first_to_end_gets_initiative` 开关（曾经的契约位已删除）。
- 行为已用 `src/tests/test_invariant_initiative.py` 锁定，防止以后有人"照着旧文档去修"从而把正确实现改坏。

## 记录规则

1. 新增一条差异前，先在 `tests/invariants` 写测试（它此刻必须是红的）。
2. 差异必须**可关闭**：`UpstreamCompatMode` 要能复现归档版行为，用于论文的"规则依赖"消融。
3. 不允许"顺手改"：一条差异一个 commit，message 用 `enginefix(<ID>): ...`。
