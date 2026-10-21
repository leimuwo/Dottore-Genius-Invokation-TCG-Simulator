# M2 基线训练规格

状态：draft，作为 M2 实现与验收的单一规格来源。
前置：M1 CardEnv 已通过接口、复现和随机稳定性验收。M2 不新增角色或卡牌，不修改
`src/dgisim/**` 的规则语义。

## 1. 目标与边界

M2 交付一个可复现的 Policy-Value PPO 基线，能够：

1. 使用 `CardEnv.reset / obs / step / fork` 完成分层动作 rollout；
2. 在六套训练牌组上训练，在三套测试牌组上固定评估；
3. 保存并恢复模型、优化器、环境配置、随机状态和训练进度；
4. 输出可离线重算的训练与评估账本；
5. 用短程实验验证奖励塑形是否比纯稀疏奖励有效。

M2 不包含历史对手池、population self-play、自适应采样或论文方法。

## 2. 固定口径

### 环境

- 只通过公开的 `src.dgisim.*` 面使用引擎。
- 训练牌组：`decks/train/` 六套；评估牌组：`decks/test/` 三套。
- 训练 rollout 不配对；先后手按 50/50 配额，具体配额写入 run manifest。
- 评估使用固定 `(deck_p1, deck_p2, seed)` 套件，并交换席位后再跑一局。
- 引擎 RNG 与策略 RNG 分离；每局记录环境版本、配置、seed、牌组和席位。
- 非法动作、异常、空转和缺失状态迁移使 episode 失效；样本丢弃并记录诊断，不计为胜负或超时。
- 终局从 `info["rewards"]` 读取双方奖励。击杀为胜者 +1、败者 -1；超时双方 -1。
  不得写 `r_p2 = -r_p1`。

### 奖励

默认开启公开势函数塑形：

```
r_t(seat) = terminal_reward(seat)
          + lambda * (gamma * Phi(s') - Phi(s))
```

- `gamma=0.99`；
- `shaping_lambda=1e-2`，可配置；
- `shaping=True`，可用 `shaping=False` 或 `lambda=0` 做消融；
- `Phi` 只使用该席位 perspective view 的存活角色数和总血量：
  `10 * (alive_me - alive_opp) + (hp_me - hp_opp)`；
- 塑形在每次真实 commit 状态迁移上计算，choice 中间层不重复计奖励；
- rollout 同时保存 terminal reward、shaping reward 和合计 reward，便于审计。

首次短程比较固定 `lambda=0` 与 `lambda=1e-2`，各运行 100,000 个 choice steps。
若信号仍弱，再测 `3e-2` 或 `0.1`；若出现为刷 Phi 放弃关键击杀的行为，测 `3e-3`。
调参不能改变其他配置或评估套件。

## 3. 模型与动作头

实现 `PolicyValueNet`（PyTorch）：

- 输入：D-C 定义的实体对齐 observation、当前席位、choice 内容 feature 和 choice kind；
- 共享编码器；
- policy head 对当前候选逐项打分，只在候选集合内 softmax；
- value head 输出当前席位价值；
- 不使用候选 index 的固定语义；候选顺序变化时，内容相同的候选得分应保持一致；
- 记录 logits、合法候选 mask、采样 index、log-prob、value、entropy；
- 推理模式支持确定性 argmax 和带独立策略 RNG 的采样。

第一版允许使用稳定的 MLP/实体池化编码器，不要求 Transformer。模型结构和参数量必须写入
manifest，后续结构变化视为新实验版本。

## 4. Rollout 与 PPO

### Rollout

每个 episode 从固定牌组和 seed reset。每个 choice step 保存：

`obs fingerprint, seat, candidate features, action index, log_prob, value,
reward, done, terminal_reason`。

只在 commit 或终局处更新环境状态；所有 choice 层都进入轨迹，不能把一次完整动作误算成一个
无记录的黑盒步骤。失败 episode 单独保存诊断并从 PPO batch 排除。

### GAE

按单席位轨迹计算：

```
delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
A_t = delta_t + gamma * gae_lambda * (1-done) * A_{t+1}
```

- `gae_lambda` 是独立配置，默认 `0.95`；
- advantage 标准化只在有效 batch 内进行；
- 双方轨迹分别按执行席位收集，不能用零和假设替换 P2 奖励；
- 终局 bootstrap 为 0；超时双方奖励各自为 -1。

### PPO 更新

默认配置必须集中在 config/manifest：

- clip range `0.2`；
- entropy coefficient、value coefficient、learning rate、batch size、epochs、grad clip；
- Adam 优化器；
- 检查 NaN/Inf、梯度范数、有效样本数和 clip fraction；
- 出现数值异常时停止该 run 并保存诊断，不继续写“正常结果”。

## 5. 训练对手与评估

第一版提供：

- Random policy：每个 choice 层在合法候选中均匀采样；
- 当前 policy；
- 可选固定 checkpoint policy。

最小闭环先跑 current policy vs Random，再跑 self-play；不能把 Random 胜率当作训练完成证明。

固定评估至少报告：

1. 三套测试牌组的总胜率、击杀率、超时率和双方平均 payoff；
2. 先手/后手分层结果及 95% Wilson CI；
3. 交换席位后的配对差分；
4. 训练前 checkpoint 与训练后 checkpoint 的同套件比较；
5. shaping 与 `lambda=0` 的短程对照。

## 6. 可复现与产物

每次 run 写入独立目录：

```
runs/<run_id>/
  manifest.json
  checkpoints/
  train_metrics.jsonl
  events.jsonl
  eval.json
  failures.jsonl
```

manifest 至少包含 Git HEAD/dirty 状态、源码和牌组指纹、Python/PyTorch/CUDA 版本、
模型/优化器/config、所有 seed、训练步数、有效/失败 episode 数和奖励口径。

checkpoint 必须包含 model state、optimizer state、训练步、所有 RNG state（Python、NumPy、
PyTorch、CUDA）以及配置指纹。恢复后继续训练的第一批 rollout 必须能复现。

## 7. 分阶段验收

### Gate M2-A：闭环

- 单次 rollout 至少 1,000 个 choice steps；
- 采样、GAE、PPO update、checkpoint save/load、评估全部可运行；
- 无 NaN/Inf；失败 episode 不进入 batch；
- 恢复运行的轨迹摘要与不中断运行一致。

### Gate M2-B：短程奖励实验

- `lambda=0` 与 `1e-2` 各完成 100,000 choice steps；
- 同一配置、同一评估套件、至少 3 个 seed；
- 报告训练曲线、timeout rate、击杀率、平均 payoff 和 95% CI；
- 检查是否出现刷 Phi：提高势函数却降低关键击杀率或延长对局；
- 只有证据支持时才调整 lambda，并保留原始 run。

### Gate M2-C：基线能力

- 固定三套测试牌组、先后手均衡；
- 训练后模型胜率目标为 Random 至少 80%，且高于训练前模型；
- 结果须在至少 3 个 seed 上方向一致；
- 80% 是验收目标，不得在未达到前宣称 M2 完成。

## 8. 测试与门禁

新增测试覆盖：

- choice 内容打分与候选 mask；
- 两席位奖励和超时双负；
- potential 的公开信息、不泄漏和反对称；
- GAE 的终局 bootstrap、超时双方奖励和失败样本过滤；
- PPO 更新数值稳定；
- checkpoint 恢复 RNG 与参数；
- 固定评估的席位交换和 CI 输入。

每个实现步骤后运行：

```
python -B -m unittest discover src/tests
python scripts/verify_decks.py
python scripts/verify_test_integrity.py
```

M2 spec 本身不引入训练效果结论；所有效果结论必须来自 run 账本和固定评估产物。
