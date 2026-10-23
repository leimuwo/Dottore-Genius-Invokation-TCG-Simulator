# 100k 固定训练牌组实验（2026-10-23）

训练现已使用 `decks/train/` 的 6 套预设牌组，按 seed 可复现选择 P1/P2 牌组。每个 seed 训练 100,000 choice steps，CUDA，shaping lambda `1e-2`。

## 训练期指标

| seed | episodes | timeout rate | kill rate | swap rate | skill rate | 平均 choice length |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120 | 80.0% | 18.3% | 13.45% | 8.61% | 833 |
| 1 | 121 | 76.0% | 23.1% | 13.52% | 8.90% | 826 |
| 2 | 123 | 76.4% | 17.9% | 13.02% | 8.80% | 813 |

与此前默认随机牌组训练相比，timeout 下降，kill rate 上升；说明固定训练牌组使训练分布更集中，也提高了训练期有效终局比例。

## 固定测试牌组评估

评估使用 `decks/test` 的 3 套牌组全部 3×3 配对，评估 seeds `0,1,2`，模型分别执 P1/P2，Random 对手，关闭 shaping，deterministic policy。Wilson 区间为 95%。

原始数据：[`runs/m2-c-100k-trainpool-evaluation.json`](../runs/m2-c-100k-trainpool-evaluation.json)。

| checkpoint | 席位 | 胜/局 | 胜率 | Wilson 95% CI | 击杀 | timeout | model payoff |
|---|---:|---:|---:|---:|---:|---:|---:|
| trainpool seed0 | P1 | 0/27 | 0.0% | 0.0–12.5% | 0 | 27 | -1.000 |
| trainpool seed0 | P2 | 0/27 | 0.0% | 0.0–12.5% | 0 | 27 | -1.000 |
| trainpool seed1 | P1 | 23/27 | 85.2% | 67.5–94.1% | 23 | 4 | 0.704 |
| trainpool seed1 | P2 | 25/27 | 92.6% | 76.6–97.9% | 25 | 2 | 0.852 |
| trainpool seed2 | P1 | 0/27 | 0.0% | 0.0–12.5% | 2 | 25 | -1.000 |
| trainpool seed2 | P2 | 0/27 | 0.0% | 0.0–12.5% | 1 | 26 | -1.000 |

## 结论

固定 6 套训练牌组改善了训练期 timeout/kill 比例，但没有解决 seed 方差：只有 seed1 在固定测试牌组上达到 80% 目标，seed0/2 完全失败。因此 M2-C 仍未通过，不能把单个 seed1 的高胜率当作稳定能力。

这说明训练牌组分布是重要变量，但当前 PPO/动作表示仍存在强烈的初始化或轨迹依赖。下一步应保留这组结果作为正式基线，继续检查训练中期 checkpoint 和牌组配对覆盖，再决定是否拆分 policy head。
