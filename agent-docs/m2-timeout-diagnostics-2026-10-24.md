# Timeout 最后 20 个 COMMIT 诊断（2026-10-24）

## 诊断协议

使用 `m2-100k-trainpool-seed0/1/2` 的 choice 级 checkpoint，固定测试牌组 3×3 配对，模型 deterministic、对手 Random、关闭 shaping。每个模型席位共 27 局，只统计终止原因为 `timeout` 的对局，并记录终局前最后 20 个已执行 PlayerAction。

原始数据：[`runs/m2-timeout-diagnostics.json`](../runs/m2-timeout-diagnostics.json)。

## 动作统计

统计单位是完整 PlayerAction；`SELECT_DICE`、`ELEMENTAL_TUNING` 是动作构造过程中的主动作层标签，`COMMIT` 本身不计入下表。

| 模型 | 席位 | timeout 局数 | SWAP | END_ROUND | PLAY_CARD | CAST_SKILL | SELECT_DICE | ELEMENTAL_TUNING |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed0 | P1 | 8 | 25 | 51 | 10 | 9 | 51 | 14 |
| seed0 | P2 | 9 | 29 | 53 | 11 | 17 | 52 | 18 |
| seed1 | P1 | 2 | 16 | 7 | 2 | 9 | 6 | 0 |
| seed1 | P2 | 1 | 8 | 3 | 0 | 3 | 3 | 3 |
| seed2 | P1 | 9 | 27 | 57 | 3 | 9 | 57 | 27 |
| seed2 | P2 | 9 | 30 | 42 | 12 | 18 | 42 | 36 |

## 终局血量观察

多个 timeout 对局已经接近可以结束，但策略没有完成最后击杀。例如：

- seed2 P2：对手 P1 保持 3 个存活角色、总血量约 29–30；模型 P2 只剩 1 个角色、2 HP，说明模型处于劣势但仍反复拖延。
- seed0 P1：部分对局模型只剩 1 个角色、1–8 HP，而对手仍有 25–30 总血量，属于明显失败后的继续拖延。
- seed2 P1：多局模型有 2 个存活角色、总血量 4–15，对手 3 个角色、约 29–30 总血量；最后 20 个动作中大量 `END_ROUND`/`SELECT_DICE`，技能只有 9 次。
- 也有对手只剩 1 个角色、1–2 HP 的局面仍 timeout，说明模型缺少终结低血量目标的动作选择能力，而不仅仅是整体实力不足。

## 结论

timeout 的主要模式是：

1. `END_ROUND` 与 `SELECT_DICE` 高频出现，消耗行动机会但没有产生击杀；
2. `CAST_SKILL` 在失败 seed0/2 中明显偏少；
3. `SWAP_CHARACTER` 也偏多，但它不是唯一原因，核心是换人/结束回合后没有转化为攻击；
4. 在对手低血量时仍不能完成终结，说明目标选择、技能选择或费用支付链存在学习不足；
5. seed1 timeout 很少，说明同一网络结构在部分初始化下能找到有效攻击策略，问题仍表现为训练不稳定和动作信用分配不足。

## 下一步

- 在训练中记录每个 COMMIT 的实际状态变化：伤害、击杀、血量差变化和势函数变化；
- 对 `CAST_SKILL` 后的目标、伤害和支付费用做细分统计，确认是技能选择还是目标/费用子 choice 出错；
- 对低血量终结场景单独建立评估集；
- 保持 choice 级 GAE 作为当前基线，暂不采用“把 PlayerAction advantage 复制给所有 choice”的方案。
