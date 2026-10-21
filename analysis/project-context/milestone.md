我建议把项目规划成一个 16 周的研究型 Milestone Roadmap，目标不是单纯训练出一个强大的七圣召唤 Agent，而是同时实现两个成果：

1. 工程目标：基于 dgisim，训练一个具有较强竞技能力、可稳定复现的卡牌决策模型。
2. 科研目标：围绕 Self-play 的样本效率、策略泛化或不完全信息决策，形成一篇具备 NeurIPS / ICML / ICLR / AAAI 主会投稿潜力的论文。

整体遵循一个原则：先建立可靠的实验环境和强基线，再寻找真正的研究创新点。

# 16 周项目 Milestone Roadmap

项目周期

# 16 周

计算资源

## 2 × H200

M1 · 环境验证

M2 · 基线模型

M3 · Self-play

M4 · 研究创新

M5 · 实验与论文

W1

W4

W8

W12

W16

## M1 — 环境验证与训练接口（Week 1–2）

目标：确认 dgisim 可以作为可靠、可扩展的强化学习环境。

| 工作项                    | 交付成果                     |
| ------------------------- | ---------------------------- |
| Fork dgisim，固定版本     | 可复现的项目仓库             |
| 验证 GameState、LinearEnv | 完整随机对局                 |
| 检查玩家视角观察          | 无隐藏信息泄漏               |
| 合法动作枚举与编码        | Observation / Action Adapter |
| 单核、多核性能测试        | 环境吞吐 Benchmark           |
| 固定随机种子              | 确定性回归测试               |

验收标准： 至少完成 10,000 局随机对战，无未处理异常；规则与动作合法性测试通过；记录 games/s、decisions/s、CPU 占用与状态编码耗时。

10,000 局只是稳定性压力测试，不能替代规则正确性测试。

Gate 1：能够稳定生成合法、可复现的训练轨迹。

## M2 — 神经网络与基础 RL（Week 3–5）

目标：训练出第一个真正会玩游戏的 Agent。

建议采用 6 层 Transformer、hidden size 256，配合候选合法动作评分和 Value Head。

| 工作项                   | 交付成果           |
| ------------------------ | ------------------ |
| Random / Heuristic Agent | 两类基础对手       |
| Policy-Value Network     | PyTorch 模型       |
| 轨迹收集与 GAE           | PPO 训练数据管线   |
| PPO Learner              | 可运行训练脚本     |
| 固定对手评估             | 初始胜率曲线       |
| Checkpoint 管理          | 模型恢复与版本比较 |

验收标准： PPO 训练过程数值稳定，在预先固定的测试卡组、先后手均衡设置下，对 Random Agent 达到至少 80% 胜率，并在独立评估中优于训练前模型。

这里 80% 是项目目标值，不是已有实验结果。还需要检查 Agent 是否只是利用了 Random Agent 的明显弱点。

Gate 2：获得可重复训练、可量化提升的神经网络 Agent。

## M3 — 强化 Self-play 与对手池（Week 6–8）

目标：让 Agent 从击败弱对手，转向学习真正的对抗策略。

实现 Historical Policy Pool、Population Self-play 和独立的 Tournament Evaluator。

训练过程中使用多个历史版本作为对手，定期保存策略快照，避免只和最新策略对战。

| 工作项                 | 交付成果                 |
| ---------------------- | ------------------------ |
| Historical Policy Pool | 历史模型管理             |
| 对手采样机制           | Uniform / Latest / Mixed |
| 自我对弈训练           | Population PPO           |
| Elo 或 TrueSkill       | 相对实力跟踪             |
| 交叉对战矩阵           | 策略克制分析             |
| 跨卡组评估             | 未见卡组测试集           |

验收标准： 最新 Agent 在固定评估池中稳定超过基础 PPO，且提升在至少 3 个随机种子中方向一致；保存所有训练曲线、对战结果和配置。

Gate 3：建立可信的强基线，开始具备科研实验条件。

## M4 — 研究创新与方法验证（Week 9–12）

目标：提出并验证一个真正有论文价值的算法改进。

这一阶段是整个项目的核心。我建议默认研究方向为：

Adaptive Self-play Sampling for Sample-Efficient Card Game Agents

核心假设是：不同对手、不同游戏局面对策略学习的贡献不同。通过自适应地选择训练对手和局面，可能减少无效自我对弈，提高训练效率。

我们可以先比较三种方案：

| 方法       | 说明                                     |
| ---------- | ---------------------------------------- |
| Baseline A | PPO + Uniform Self-play                  |
| Baseline B | PPO + Historical Population              |
| Proposed   | PPO + Adaptive Opponent / State Sampling |

对于 Proposed 方法，可以研究使用策略分歧、局面不确定性、对手多样性等指标控制采样分布。

但这里有一个重要问题：如果改变中间状态的采样分布，可能引入隐藏信息偏差。因此必须保留一个只改变对手采样、不改变初始局面分布的版本作为对照。

Week 9：问题诊断

分析训练样本的分布、不同对手的策略差异、重复局面比例，以及 Agent 在不同阶段的胜率。

Week 10：算法实现

实现自适应采样机制，建立必要的消融开关。

Week 11：初步实验

使用相同的环境交互预算和近似相同的计算预算，与两个基线比较。

Week 12：方法选择

检查改进是否稳定，是否真正提升样本效率，以及是否仅仅增加了训练计算量。

Gate 4： 如果方法在至少 3 个随机种子上呈现一致的正向结果，且有合理的机制解释，进入完整实验。否则缩小研究假设或更换方法，不建议直接扩展模型规模。

## M5 — 多环境验证、消融实验与论文（Week 13–16）

目标：从一个强 Agent 项目，转化为一篇有证据支持的研究论文。

| 周次    | 主要任务                   | 交付成果                     |
| ------- | -------------------------- | ---------------------------- |
| Week 13 | 多随机种子、跨卡组测试     | 主实验结果                   |
| Week 14 | LOCM / 简化博弈环境验证    | 泛化实验                     |
| Week 15 | 消融、显著性与算力成本分析 | 完整实验图表                 |
| Week 16 | 论文撰写、代码整理与复现   | 论文初稿 + Release Candidate |

最终论文至少应该包含：

- Learning Curves：胜率随环境交互数变化。
- Sample Efficiency：达到相同实力所需的训练样本量。
- Cross-Deck Generalization：未见卡组上的表现。
- Ablation Study：各算法组件的独立贡献。
- Compute Cost：GPU 小时、CPU 小时、环境吞吐。
- Cross-Play Matrix：不同版本和不同策略之间的胜负关系。

如果 Week 12 的研究结果不理想，Week 13–16 应优先用于分析失败原因、补强基线，而不是为了投稿强行包装创新点。

# 六、建议的算力与实验管理

在整个项目中，我建议先按一张 H200 能够完成基础训练来设计，第二张用于并行实验、推理服务或搜索，而不是默认两张 GPU 始终满载。

初始 GPU 预算（规划值）

总预算

# 640 GPU·h

双卡同时运行折算

# 320 小时

环境与基线

80 GPU·h

Self-play 实验

180 GPU·h

新方法探索

220 GPU·h

消融与复现

160 GPU·h

这是用于管理实验的初始预算，不是实测训练成本。正式预算应在 Week 2 完成模拟器吞吐测试后修订。CPU 资源、存储与评估对战成本需单独核算。

# 七、建议建立的项目仓库结构

```
card-game-agent/
├── envs/
│   ├── dgisim_adapter.py
│   ├── observation.py
│   └── action_encoder.py
├── agents/
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   └── policy_value_net.py
├── training/
│   ├── ppo.py
│   ├── rollout_worker.py
│   ├── self_play.py
│   └── opponent_pool.py
├── research/
│   ├── adaptive_sampling.py
│   └── ablations/
├── evaluation/
│   ├── tournament.py
│   ├── cross_play.py
│   └── benchmark.py
├── configs/
├── tests/
├── scripts/
└── paper/
```

从第一周开始，就应该记录 Git commit、随机种子、模型配置、环境版本和训练数据来源，保证所有结果可追溯。

# 八、项目 Milestone 验收清单

你可以把下面作为项目进度看板。

研究进度

0 / 5 完成

M1 · 环境验证

Week 1–2：dgisim 接口、规则测试、性能测试

M2 · 基础 RL

Week 3–5：Policy-Value Network、PPO、初始胜率

M3 · Population Self-play

Week 6–8：历史对手池、强基线、跨卡组评估

M4 · 算法创新

Week 9–12：自适应采样方法与初步消融

M5 · 论文实验

Week 13–16：完整对照、泛化、论文初稿

 复制进度清单

## 我认为最重要的三个决策节点

Week 2：dgisim 是否足够快、足够可靠？ 如果不行，应优先优化环境，而不是继续训练模型。

Week 8：基础 Agent 是否真正学会了博弈？ 如果它只能击败 Random Agent，说明需要继续加强训练基线。

Week 12：是否找到了可重复的研究增益？ 这才是决定能否冲击 A 会主会的核心节点。

最终，16 周的合理目标是：一套可复现的强卡牌决策 Agent、一套严格的对战 Benchmark，以及一个经过初步验证的算法创新和论文初稿。 这不等于保证 16 周就能达到 A 会录用标准，但能让项目尽早形成清晰的科研成果。