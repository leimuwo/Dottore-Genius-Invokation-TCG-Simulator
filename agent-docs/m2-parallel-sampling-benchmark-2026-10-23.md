# 并行环境采样 benchmark（2026-10-23）

配置：10,000 choice steps，seed `0`，CUDA，`torch-threads=2`，固定 `decks/train` 牌组池。

| env workers | sampling | GAE | PPO update | 合计 |
|---:|---:|---:|---:|---:|
| 1 | 49.327s | 3.386s | 8.336s | 61.048s |
| 2 | 26.854s | 3.468s | 8.116s | 38.439s |
| 4 | 21.300s | 3.705s | 8.289s | 33.294s |

相对单 worker：

- 2 workers：总耗时下降约 37%，sampling 下降约 46%；
- 4 workers：总耗时下降约 45%，sampling 下降约 57%。

三个配置均为 0 failed episodes，PPO 阶段耗时基本不变，说明收益来自环境采样并行化。4 workers 在本机优于 2 workers，但未达到线性加速；继续增加 worker 需要重新 benchmark 进程通信和 CPU 争用。
