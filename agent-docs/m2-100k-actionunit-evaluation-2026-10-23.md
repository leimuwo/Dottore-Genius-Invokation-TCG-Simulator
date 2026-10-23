# 100k PlayerAction 级 GAE 实验（2026-10-23）

## 改动

把一个 PlayerAction 中从首个 choice 到 `COMMIT` 的所有 choice 视为一个决策单元：按动作块计算 GAE，并把该动作块的 advantage/return 复制给其中每个 choice。训练使用 `decks/train` 六套牌组、lambda `1e-2`、CUDA、4 个独立采样 worker，100,000 choice steps × 3 seeds。

实现提交：`92e9146 feat(m2): assign advantages per player action`。

## 训练期指标

| seed | episodes | timeout rate | kill rate | swap rate | skill rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 123 | 76.4% | 22.8% | 13.1% | 8.9% |
| 1 | 123 | 71.5% | 27.6% | 13.2% | 8.9% |
| 2 | 124 | 74.2% | 25.0% | 13.2% | 8.8% |

## 固定测试评估

协议与既有 M2-C 相同：3 套测试牌组 3×3 配对，评估 seeds `0,1,2`，P1/P2 分别执子，Random 对手，关闭 shaping，deterministic policy。

原始数据：[`runs/m2-c-100k-actionunit-evaluation.json`](../runs/m2-c-100k-actionunit-evaluation.json)。

| seed | 席位 | 胜/局 | Wilson 95% CI | 击杀 | timeout | 平均 payoff |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | P1 | 0/27 | 0.0–12.5% | 2 | 25 | -1.000 |
| 0 | P2 | 0/27 | 0.0–12.5% | 1 | 26 | -1.000 |
| 1 | P1 | 0/27 | 0.0–12.5% | 1 | 26 | -1.000 |
| 1 | P2 | 0/27 | 0.0–12.5% | 3 | 24 | -1.000 |
| 2 | P1 | 4/27 | 5.9–32.5% | 6 | 21 | -0.704 |
| 2 | P2 | 6/27 | 10.6–40.8% | 7 | 20 | -0.556 |

## 结论

本次动作级 GAE 没有改善评估表现，反而明显低于先前 choice 级 GAE 的固定训练牌组 100k 结果（seed1 P1/P2 为 23/27、25/27）。训练期 kill rate 看起来尚可，但固定测试牌组泛化失败，说明动作级回报复制到块内所有 choice 可能把同一终局 credit 过度扩散到动作构造中的每个子选择；或现有 value baseline 与动作级折扣尺度不匹配。

因此不应直接用这组 checkpoint 替代 choice 级基线，也不建议据此继续加训练预算。下一步应先做受控对照：同 seed、同牌组、同 rollout/update 配置，只改变 GAE 单位；检查 action block 的 reward/discount 定义，并考虑对动作内部 choice 使用按层级折扣或只训练动作类型/首个 choice。

验证：全量 `674` 个测试通过。
