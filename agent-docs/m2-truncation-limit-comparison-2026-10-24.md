# M2 截断上限对照（10k choice steps）

本次只改变 `max_episode_steps`，其余参数固定：seed 0、固定 `decks/train`、CUDA、4 个环境 worker、10,000 choice steps、rollout batch 2,048。原始结果保存在 `runs/m2-truncation-10k-{1000,2000,4000}/manifest.json`。

| 上限 | 有效局数 | truncation rate | engine timeout rate | kill rate | 平均 choice/局 | 平均 COMMIT/局 | 截断时平均存活角色 | 截断时平均总 HP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 13 | 15.4% | 53.8% | 30.8% | 769.2 | 140.9 | 4.5 | 38.5 |
| 2000 | 13 | 7.7% | 38.5% | 53.8% | 769.2 | 137.9 | 6.0 | 53.0 |
| 4000 | 12 | 0.0% | 75.0% | 25.0% | 833.3 | 150.1 | — | — |

## 截断尾部

1000 上限产生 2 个截断局，2000 上限产生 1 个截断局；每局最多保存最后 20 个 COMMIT。记录格式为 `seat:stage_index/block_length`，例如 `P1:8/9` 表示 P1 的 COMMIT，前一动作块包含 9 个 choice。尾部均出现连续高 stage（6–9）与短块（2–4）交替，说明截断前仍在正常完成回合提交，没有卡在单个未完成 choice。

4000 上限没有截断局，因此没有尾部动作或截断时 HP/alive 数据。

## 解读

这次每个条件只有 12–13 局，比例步长约 7.7 个百分点，不能据此断言 2000 优于 1000。上限提高确实减少了 rollout truncation，但 engine timeout 仍然存在，并且 4000 条件的 timeout 反而较高；因此当前主要问题是策略/引擎达到 timeout，而不是单纯的 rollout 上限太小。后续应在固定策略 checkpoint 上做独立评估，并增加对局数后再决定上限。
