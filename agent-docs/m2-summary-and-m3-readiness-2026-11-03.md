# M2 总结与 M3 就绪结论（2026-11-03）

## 结论

可以推进 M3，但必须冻结当前 M2 checkpoint、评估协议和代码版本，把 M2 作为可复现基线。M2 的工程闭环已经完成；当前固定测试协议下，100k checkpoint 在 Random 上达到并超过 80% 目标，且明显优于对应未训练模型。历史报告中的 M2-C 未通过仍需保留，因为它使用的是更早的 checkpoint、较小样本和旧评估结果，不能被删除或改写成已通过。

## M2 遇到的问题与解决方法

### 1. 奖励归属错误

最初按执行席位拆轨迹，遗漏了对手行动期间产生的本席位 shaping；终局奖励还可能覆盖已有 shaping。

解决：按“本席位一次决策到下一次决策”建立奖励账本，分别累计两席位 reward，终局 reward 与已有 reward 相加。提交 `23c85df`。

### 2. choice 与 COMMIT 时间尺度不一致

内部 choice 不推进游戏时间，真实 COMMIT 才推进一步；原实现对区间奖励和 bootstrap 使用了不同 COMMIT 计数，且漏算行动者自己的 COMMIT。

解决：内部 choice 使用 `gamma=1`，每个真实 COMMIT 后推进所有已开始席位区间的时间；区间奖励按 `r0 + gamma*r1 + gamma^2*r2...` 累计，下一状态使用相同 COMMIT 数计算的 `gamma^n`。加入独立手算测试和 `Phi: 0 -> 10 -> 0` 抵消测试。提交 `02f91e3`。

### 3. 终止和截断 bootstrap

终局状态不应 bootstrap，普通截断要使用边界状态；总预算裁剪不能把后半段裁掉后清空 bootstrap。

解决：终止状态 `Phi=0`，终局 value bootstrap 为零；普通截断保存双方观察；总预算按 episode 边界分配，新增 budget 与 `max_episode_steps` 截断统计。提交 `adcb3cc`、`668d9e6`。

### 4. 观察编码和前缀不一致

采样、PPO 重算和 bootstrap 曾使用不同 stage；付款前缀用字符求和压成一个小数，存在碰撞和信息损失。

解决：统一观察 stage 编码函数，三条路径共用；rollout 保存编码后的输入；目标 ID 已进入候选特征；付款数量和动作前缀使用结构化序列后再生成稳定摘要。参数未更新时概率比约为 1。提交 `c5c3332`。

当前仍有表示限制：前缀摘要不是独立 embedding，后续可作为 M3 之外的单独表示实验，不能与 self-play 改动混合。

### 5. 指标解释和评估吞吐

旧 `skill_rate/swap_rate` 分母混入 layered choice，容易误读；评估器没有逐局续跑和耗时记录。

解决：动作比例改按 COMMIT 统计；评估支持任务命名空间、逐局 JSONL、断点续跑，记录耗时、choice、COMMIT、推理和环境时间。性能诊断显示 1 个 PyTorch 线程最快，4 workers 约 2.17 倍于单 worker。提交 `2a068e0`。

### 6. 对手和评估协议

只用 Random 容易高估能力；无条件“不能攻击就换人”会制造换人循环。

解决：加入保守 Attack-first：有技能才攻击，无明确后续攻击时不换人，直接结束回合；使用固定牌组、固定 seed、交换席位、关闭 shaping，并保留 Wilson 区间和逐局终局原因。提交 `9a36369`。

## 当前验收证据

- 全量测试：`684 tests OK`（最新代码提交时）；
- 训练：固定 `decks/train`，100k choice × 3 seeds，中途 checkpoint 已保存；
- 当前 100k checkpoint 初筛：Random/Attack-first、P1/P2、3 seeds、45 局/组，共 540 局，模型胜 `538/540`；
- 新 seed `5..14` 复核：每个 checkpoint/对手/席位 90 局，共 1080 局，胜率仍约 96.7%–100%；
- 对应未训练模型在同一 540 局任务上明显更差，尤其 seed2 为 0 胜且大量 timeout；
- Attack-first 抽查确实执行技能并造成伤害，不是只结束回合的空对手；
- seed2 的少量失败集中在 Random/P1，没有复现训练期大面积 timeout。

原始记录位于 `runs/m2-baseline-eval/`、`runs/m2-540-eval/`、`runs/m2-newseed-eval/`；评估报告见 `agent-docs/m2-evaluation-validation-2026-11-03.md`。

## 仍需保留的限制

1. 历史 `docs/m2-c-acceptance-2026-10-23.md` 的未通过结论是当时真实结果，不能删除；当前结果应标为新 checkpoint/新协议复核。
2. 当前高胜率只证明对固定测试牌组和两类弱基线的能力，不能宣称通用竞技强度。
3. 评估比较 checkpoint 时应继续使用同一任务键和配对分析；不要把相关任务当成完全独立样本。
4. M2 尚未包含历史策略池、population self-play、跨卡组新测试集或研究方法，这些属于 M3/M4。

## M3 开始条件

开始 M3 前冻结：Git commit、100k checkpoint、训练/测试牌组、奖励与折扣定义、PPO 配置、评估任务清单和结果账本。M3 第一阶段只新增历史策略池与对手采样机制，保持环境、奖励、网络和基础评估协议不变；每项改动单独 commit、单独消融，并继续报告 P1/P2、Random、Attack-first、timeout 和配对差值。
