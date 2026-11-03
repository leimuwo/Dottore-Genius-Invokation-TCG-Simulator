# M2 评估核验与新 seed 复核（2026-11-03）

## 训练前同任务对照

三个对应未训练模型使用与训练后相同的 `decks/test` 3×3 配对、seeds `0..4`、Random/Attack-first、P1/P2 交换和关闭 shaping 协议，共 540 局。

训练前结果：

| 模型 | 对手 | P1 胜/45 | P2 胜/45 | P1 timeout | P2 timeout |
|---|---|---:|---:|---:|---:|
| seed0 | Random | 41 | 39 | 4 | 6 |
| seed0 | Attack-first | 45 | 45 | 0 | 0 |
| seed1 | Random | 45 | 45 | 0 | 0 |
| seed1 | Attack-first | 45 | 45 | 0 | 0 |
| seed2 | Random | 0 | 0 | 33 | 39 |
| seed2 | Attack-first | 0 | 0 | 30 | 30 |

训练后 100k 模型在同一协议下为 538/540 胜，且训练前 seed2 的 0 胜与训练后明显不同，说明结果不是未训练策略也能轻易达到。

## 终局与对手核验

540 局逐局记录按 `terminal_reason` 重算后与模型 payoff 胜负一致：模型胜局均为 `kill:P1` 或 `kill:P2`；没有把 timeout 计为胜利。seed2 的两次异常分别为一局 Random/P1 timeout 和一局 Attack-first/P1 被 `kill:P2`。

独立抽查 Attack-first 自身控制的自然对局：3 局中执行 33–35 次技能，32–35 次造成伤害，单局技能伤害总量 55–57，证明对手不是只结束回合的空策略。

## 新 seed 复核

固定 100k checkpoint，协议不变，新增 seeds `5..14`；每个 checkpoint/对手/席位 90 局，共 1080 局：

| checkpoint | 对手 | P1 胜/90 | P2 胜/90 | P1 timeout | P2 timeout |
|---|---|---:|---:|---:|---:|
| seed0 | Random | 89 | 87 | 1 | 3 |
| seed0 | Attack-first | 90 | 90 | 0 | 0 |
| seed1 | Random | 89 | 90 | 1 | 0 |
| seed1 | Attack-first | 90 | 90 | 0 | 0 |
| seed2 | Random | 87 | 90 | 2 | 0 |
| seed2 | Attack-first | 89 | 90 | 0 | 0 |

新 seed 复核仍显示模型明显高于 80% 目标；seed2 的弱点主要集中在 Random/P1，未表现为普遍 timeout 崩溃。

原始逐局记录：`runs/m2-baseline-eval/`、`runs/m2-540-eval/`、`runs/m2-newseed-eval/`。
