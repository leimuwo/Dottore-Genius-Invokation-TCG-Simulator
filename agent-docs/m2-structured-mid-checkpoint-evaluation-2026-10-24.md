# 结构化候选特征中途 checkpoint 评估（2026-10-24）

## 配置

使用固定 `decks/train` 六套牌组、结构化候选特征、choice 级 GAE、CUDA、4 个环境 worker、100k choice steps × 3 seeds；每 20k 左右保存 checkpoint。固定评估使用 `decks/test` 3×3 配对、3 个评估 seed、P1/P2 分别执子、Random 对手、关闭 shaping、deterministic policy。

原始评估：[`runs/m2-c-structured-mid-evaluation.json`](../runs/m2-c-structured-mid-evaluation.json)。

## 评估结果（P1/P2 胜数，均为 27 局）

| seed | ~20k | ~40k | ~60k | ~80k | 100k |
|---:|---:|---:|---:|---:|---:|
| 0 P1 | 21 | 15 | 21 | 16 | 11 |
| 0 P2 | 19 | 8 | 18 | 18 | 18 |
| 1 P1 | 21 | 22 | 16 | 14 | 11 |
| 1 P2 | 19 | 20 | 19 | 9 | 11 |
| 2 P1 | 16 | 18 | 14 | 13 | 12 |
| 2 P2 | 13 | 19 | 13 | 15 | 13 |

Wilson 95% CI 与完整击杀/timeout/payoff 字段保存在原始 JSON 中。最高点分别出现在约 20k–40k，100k 最终 checkpoint 没有稳定优于中途 checkpoint。

## 训练期指标

| seed | timeout rate | kill rate | swap rate | skill rate |
|---:|---:|---:|---:|---:|
| 0 | 7.4% | 53.7% | 14.7% | 10.9% |
| 1 | 8.1% | 51.4% | 14.8% | 10.7% |
| 2 | 14.4% | 53.2% | 14.7% | 10.4% |

## 结论

1. 固定训练牌组和结构化候选特征显著改善了训练期推进：timeout 降至 7.4%–14.4%，kill rate 超过 51%。
2. 策略能力在约 20k–40k 已出现，之后发生明显波动；继续训练到 100k 可能产生策略漂移或过拟合。
3. 不能只保存最终 checkpoint。后续训练应保留中途 checkpoint，并用独立评估选择最佳 checkpoint；选择标准必须只使用验证牌组/固定评估协议，不能用训练期 reward 代替。
4. 本次中途评估仍未满足“三 seed 稳定达到 80%”：最高单点约 22/27 或 21/27，Wilson 区间下界仍低于 80%。
