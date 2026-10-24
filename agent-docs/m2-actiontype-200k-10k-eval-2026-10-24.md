# 动作类型拆分与 200k 上限训练验证

## 配置

- checkpoint：新训练 `runs/m2-actiontype-10k-seed0/checkpoint.pt`
- 训练：10,000 choice steps、seed 0、CUDA、4 workers、固定 `decks/train`
- rollout 上限：`max_episode_steps=200000`
- 评估：固定 `decks/test`，Random 对手，模型分别执 P1/P2，各 3 局

策略实现为候选的复合 embedding：每个候选同时编码原有 choice kind 和粗粒度 action type（skill/card/swap/end_round），动作类型使用独立参数块，避免四类动作只共享一个 kind 表示。

## 训练结果

训练没有 rollout truncation：`truncated_rate=0`。引擎终局统计为 timeout 41.7%、kill 58.3%。完整 COMMIT 差分：

| 类型 | 次数 | 造成伤害次数 | 总 HP 损失 |
|---|---:|---:|---:|
| skill | 249 | 229 | 558 |
| card | 578 | 0 | 0 |
| swap | 371 | 0 | 0 |
| end_round | 329 | 33 | 36 |

## 固定测试牌组 + Random

| 模型席位 | 局数 | kill | engine timeout | skill | skill 伤害率 | card | swap | end_round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 3 | 1 | 2 | 57 | 89.5% | 143 | 73 | 89 |
| P2 | 3 | 0 | 3 | 77 | 84.4% | 146 | 93 | 92 |

评估中技能一旦使用仍大多造成伤害，但完整动作中卡牌、换人和结束回合数量明显更高；P2 席位 3 局全部 timeout。样本量仍小，不能宣称胜率改善，但已确认动作类型偏好模式在训练和固定测试中一致存在。

## 结论

200k rollout 上限解决了人为截断问题，但没有解决引擎 timeout。动作类型拆分没有破坏技能执行，下一步应在此基础上增加技能/攻击候选特征，或对连续无伤害的结束回合和换人行为增加小幅辅助信号。
