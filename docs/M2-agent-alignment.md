# M2 Agent 对齐：M1 约束与交接契约

日期：2026-10-21。先读本文件与 [AGENTS.md](../AGENTS.md)。

## 1. 当前交付与边界

M2训练栈已定torch自写PPO，模型及训练/评估闭环尚未实现。
当前 enc 为2778维，没有 dense/tokens；需M2适配。Choice含 kind/payload/feature/label。


## 2. 工程硬约束

- 使用本 fork 的 `src.dgisim.*` 公开面；禁止 `src.dgisim._core`，不以 PyPI 上游替换本仓库。
- 引擎 `src/dgisim/**` 每条修改需 `enginefix(<ID>):` 独立提交、deviations 登记与不变量测试；改变博弈语义必须可关闭。
- 不删除、跳过或削弱既有测试断言。每步先检查状态，修改后跑全量 unittest，通过后提交。
- `.github/workflows/**`、`pyproject.toml`、`scripts/verify_*.py` 需人工批准后才能修改；不为过门禁改基线。
- 不新增卡/角色；固定 v0.6.0 子集。不拆 card.py/status.py，不改包映射或 EffectStack 语义。
- 单份受管 Markdown ≤8192 bytes；诊断走 logging，不向 stdout 打印状态表。
- 不用 `LinearEnv.copy()` 或 `GameStateMachine` 代替 CardEnv；不向 step 传零向量、补零动作或完整动作编码。

## 3. 正式接口：以实际字段为准


| 接口 | 契约 |
|---|---|
| `reset(deck_p1, deck_p2, seed) -> Obs` | 显式提供双方牌组和种子；牌组用 FrozenDeck.from_json 加载 JSON 文本 |
| `obs(seat=None) -> Obs` | 默认当前行动者；可读另一席位视图，其候选为空 |
| `step(choice_index: int) -> StepOut` | 当前候选数组内的整数索引，每次只推进一层选择 |
| `fork() -> CardEnv` | 共享不可变游戏状态，复制局内选择、缓存及随机流；两环境后续互不干扰 |

`Obs` 实际字段为 `seat, enc, choices, filled, decision_index, stage_index, meta`。

- 按候选内容打分，索引只用于本次选择；不学习固定 index 的动作语义。批处理 padding 必须掩码。
- 候选特征长度随类型不同；适配层保留类型及字段语义。重掷/换牌/付款按元素或牌种选择数量，
  完整覆盖子集；付款仅保留存在合法完成方案的数量分支。不要回退到少量预设骰子组合。
- 模型输入同时保留 `meta.choice_prefix`、`meta.selection_prefix` 与阶段信息；同一 enc 下存在不同中间决策。
- 非 commit 的 choice 只推进动作构造，游戏状态不变且奖励为零；`info['committed']` 区分真实落子。
  noop 检查针对 commit：hash 与 waiting_for 都未变则失败。不能把每层选择都强制判为游戏状态迁移。
- 合法终局 `done=True, obs=None`；用 `out.terminal_reason` 区分 `kill:P1`、`kill:P2`、`timeout`。

## 4. 观察与信息隔离

模型只消费席位视角观察，禁止从 `env.state`、完整状态或对手牌库补特征。
自己手牌和牌表可见；自己未来抽牌顺序不可见；对方手牌身份、真实牌表及牌库顺序不可见。

[D-C](decisions/D-C-observation.md) 已接受实体对齐词元、动态标量、自己的局内静态上下文；
token 预算 256。校验schema，不能按旧样本删字段。
旧词元p99≈50；M2重测，不静默截断。
已知 E5：Cards 可出现负计数，num_cards() 与编码占用不同；占用从编码校验，异常另记，不能默默修规则。

## 5. 双方奖励、轨迹与随机性

- 默认击杀：胜者 +1、败者 -1；15 回合超时双方 -1。超时双负使总收益不严格零和。
- `out.reward` 属于 `info['actor']` 指定的执行席位，不属于 out.obs 的下一行动者。
  `info['rewards']` 是以 `Pid.P1/P2` 为键的字典；每步都有双方值，非终局均为 0。
  禁止 `r_P2 = -r_P1`；对手触发终局也需归回双方收益。
- GAE 不能把两个席位的交替动作直接当作同一玩家连续轨迹。先明确按席位的时间线、bootstrap 和终止掩码；
  分层 choice 与完整落子采用何种折扣时间单位是 M2 待定义项，必须写入配置并测试。
- CardEnv 保存独立引擎 RNG，在引擎边界交换并恢复进程 random 状态；策略用独立 RNG，
  不直接调用引擎推进绕开边界。固定 seed 的 fork、交错运行和跨进程摘要均有回归覆盖。
- Python、NumPy、torch、CUDA 的随机源由 M2 自行固定并记录。M1 复现通过不等于 GPU 训练位级确定。
- 同 seed 配对只提供共同随机条件；策略分支不同会消耗不同数量随机数，不保证后续抽牌逐张相同。
- `CardEnvError.category` 区分 illegal/exception/noop，失败局 `failed=True`，要求 reset；丢弃该局样本，保存seed/动作/traceback，不伪装timeout。

## 6. 牌组、评估与 shaping

训练只用 `decks/train` 六套；`decks/test` 三套只做评估，不用测试回报调参或选 checkpoint。
M1 压测用全部九套仅用于环境验证，没有训练。牌表固定，不凭记忆新增卡名。
留出是卡组级：V7/V8 与训练集共享角色，只有 V9 角色完全不重叠，不能声称完整角色泛化。

固定评估需记录牌组、种子、双方策略与先后手，独立评估训练前/后模型。
按卡组/先后手报告胜率、超时率、payoff、CI，从账本重算。

[D-E](decisions/D-E-side-and-pairing.md) 仍为 proposed：弱侧 55%/60% 配额、侧别画像和 opening_score
定义未定，不自动改牌表 schema 或启用弱侧过采样。先后手均衡是路线图评估要求。
评估配对、两张表及起手质量记账的详细协议来自 D-E，实施前在 M2 spec 中写清采用范围。

[D-D](decisions/D-D-reward-shaping.md)：默认 λ=0。候选 Φ=10×存活数差+血量差，只看公开信息；
实验形式 λ(γΦ′−Φ)，γ=.99、候选 λ=.001。启用/保留需当前策略 AUC@25%≥.60、
泄漏检测通过、单步幅度≤终局奖励5%，以及同 seed/超参的 λ=.001 对 λ=0 对照：
前10万交互超时率降低≥5个百分点、Wilson 95%CI不重叠。未满足则关闭，试验预登记，不反复调λ。历史随机策略 AUC 不能代替当前策略验证。

## 7. M1 证据与已知限制

| 项目 | 验收结果 |
|---|---|
| 完整冷 obs() 三轮平均 | .789 / .786 / .796 ms；p95 1.36–1.45 ms |
| 串行实际10万 choice | 77.107 s；并非10万完整落子 |
| 8进程压力测试 | 10,000局，8,436,504 choice，1,531,303 commit；735.010 s |
| 三率与吞吐 | illegal/exception/noop 均0；11,478 choice/s |
| 终局 | 击杀1,807局；超时8,193局（81.93%） |
| 复现与回归 | 18条完整轨迹跨进程一致；全量651 tests通过；独立账本审计通过 |

≤1ms是平均门槛；随机高超时率不是训练效果。
E6/E7/E8 已修，旧 D-C/D-D 中“E7未修/跨进程漂移”是历史记录，不是当前状态。
E5 仍未修；零失败压测不证明所有规则无缺陷；不得“修正”已核实正确的先手权。
验收详见 [报告](M1-acceptance-2026-10-21.md) 与 [摘要目录](../analysis/evidence/cardenv-m1-20261021/)。

## 8. M2 开工与完成标准

先写 M2 spec，明确 tokenizer、候选/前缀表示、按席位回报与GAE、折扣单位、对手、
采样/更新预算、评估种子/局数/CI、checkpoint恢复语义及依赖版本；
先验证采样→更新→保存/恢复→评估，无NaN、奖励与终止掩码正确。
路线图 Gate 2 另要求独立固定测试、先后手均衡下对 Random≥80%胜率并优于训练前模型；这是目标，尚未达成。

从仓库根目录运行（Windows 用 AGENTS.md 的绝对解释器）：

```bash
python -B -m unittest discover src/tests
python -B scripts/verify_encoding_coverage.py
python -B scripts/verify_test_integrity.py
python -B scripts/verify_decks.py
python -B scripts/verify_doc_budget.py
python -B scripts/verify_engine_diff_budget.py
python -B analysis/cardenv_acceptance.py --games 162 --workers 4 --determinism --out analysis/m2-env-smoke
```

输出目录需全新；环境改动重验性能/复现/压测，模型改动跑回归及训练验证。
