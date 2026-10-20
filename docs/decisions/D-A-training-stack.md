# D-A：训练栈

status: proposed
date: 2026-10
范围：只决定「用哪套代码做训练」，不决定超参、不决定算法细节。

选择依据不是「哪个框架更强」，而是第 6 讲的那把尺子：**需求改一处，系统要跟着改几处**。
所以本记录冻结的是一个**接口**（§2 的 4 个函数），训练器只是可替换的后端。

---

## 1. 候选与依赖

| 候选 | 新增第三方依赖 | 还要自己写 | 判定 |
|---|---|---|---|
| **A. 自写 PPO（torch）** | `torch`（PyPI cp312-win_amd64 wheel **124.1 MB**；本机**尚未安装**，`import torch` → ModuleNotFoundError） | env 适配 + 策略头 + PPO ≈ 680 行 | **推荐** |
| B. 内联 CleanRL 单文件 PPO | `torch`（`gymnasium` 只为 `spaces`，可去掉） | 接到本仓库 env ≈ 300 行 | 备选：A 跑不动时一周内可切 |
| C. SB3 / sb3-contrib | `torch`、`gymnasium`、`stable-baselines3`、`sb3-contrib` | 胶水 + 把动作空间拍平 | **否**：见 §1.1 |
| D. 纯 numpy REINFORCE | 无（本机已有 numpy） | ≈ 200 行 | 只作 env 自检基线，不作训练器 |

### 1.1 三条硬约束（都来自本仓库实测）

1. **动作空间是分层的**：一个落子要经过若干次 `ActionGenerator.choose()` 才 `filled()`，
   而 `choices()` 的返回类型是**异质**的（`tuple` / `ActualDice` / `AbstractDice` / `Cards`，
   见 `action_generator.py:77` 的 docstring）。要求「定长 Discrete」的框架（SB3）必须先把
   这棵树拍平，等于丢掉分层结构 —— 而分层恰好是这套引擎最值得利用的东西。
2. **随机源没有分离**：引擎抽牌走进程级 `random`，没有「引擎 RNG / 策略 RNG」之分。
   复现、CRN 配对评估都得自己兜，框架帮不上。
3. **吞吐不是瓶颈**：实测 300 局 / 123 s（`analysis/probe_side_effect.py`）= **0.41 s/局**，
   测试套件另报 0.75 s/局；每局 2.1e3 次状态迁移、约 120–145 个决策 →
   **2.4e3–4.3e3 迁移/秒**，1e5 步 ≈ **23–42 s**。所以选框架的理由不是性能。

---

## 2. 摩擦点：`CardEnv` 必须提供的 4 个函数

```python
class CardEnv(Protocol):
    def reset(self, deck_p1: Deck, deck_p2: Deck, seed: int) -> Obs: ...
    def obs(self, seat: Pid) -> Obs: ...                  # 双方视角都要能取
    def step(self, choice_index: int) -> StepOut: ...    # 推进「一层 choice」
    def fork(self) -> "CardEnv": ...                      # 分叉（rollout / 搜索）
```

- `Obs` 必须携带：`dense`（动态标量）、`tokens`（实体序列）、`choices`（当前层候选项，
  **每个候选项自带内容特征**）、`filled`（是否已能生成 `PlayerAction`）、`seat`。
- `StepOut` = `(Obs | None, reward_of_seat, done, info{"terminal_reason"})`；
  `filled=False` 的 `step()` 不推进游戏（只推进 choice），reward 为 0。

**现有 `LinearEnv` 缺什么**（逐条读代码所得）：

| 需要 | 现状 |
|---|---|
| 合法动作枚举 | 无。`step()` 只吃编码后的 `PlayerAction`；`choices()` 在 `ActionGenerator`/phase 里，`LinearEnv` 没暴露 |
| 席位对称奖励 | `real_reward()` 只按 P1 给 ±1，P2 侧从不翻转 |
| 双席位观测 | `view()` 只给**当前等待者**的视角，取不到"同一状态两个视角" |
| 分叉 | `copy()` 语义是"共享同一个 state"，没有独立随机流的 fork |

两个必须先处理的坑：

1. `ActionGenerator.choices()` 的**顺序不稳定**（C2b 未修）→ 策略头必须对**候选项内容**打分，
   不能依赖 index 的语义，否则重跑就漂。
2. 异质 choice 类型：朴素的 `choices()[0]` 会在几局内触发 `AssertionError`
   （`card.py:622` 要求传 `ActualDice`）—— 适配层必须按类型分支。

---

## 3. 最小实现的行数估计与风险

| 文件 | 估行 | 内容 |
|---|---|---|
| `dgisim_ext/env_api.py` | ~180 | 上面 4 个函数 + 候选项内容特征 |
| `dgisim_ext/policy.py` | ~120 | 实体分词 → 编码器 → 候选项打分头 |
| `trainers/ppo.py` | ~260 | rollout / GAE / 掩码采样 / 自对弈 |
| `evaluation/report.py` | ~120 | T7 两张表 + 固定评估套件 |
| 合计 | **~680** | 其中「跑通一局并打印形状」的最小闭环 ≈ 300 行 |

风险（按「会不会让项目返工」排序）：
1. **动作顺序不稳定** → 策略头依赖 index 就会学出无法复现的策略。缓解：内容打分（已定为契约）。
2. **70% 超时**（实测 71.3%，超时双方判负）→ 奖励极稀疏，PPO 先学会「拖」。缓解：见 D-D，
   shaping 只在 Φ 通过「早期区分度 + 泄漏检测」后才启用。
3. **无 GPU / 单进程**：不是吞吐瓶颈，但自对弈需要两个策略 → 先做「当前 vs 随机/历史快照」，
   别一上来做纯自对弈。
4. **依赖体积**：torch wheel 124.1 MB；装不上的退路是 D，代价是手写梯度（不做反传）。

---

## 4. 推荐与反悔成本

**推荐 A：torch 上自写 PPO，接口按 §2 冻结。**
唯一能同时满足「分层动作 + 内容打分 + 双席位 + 可分叉」的选项，且反悔最便宜。

| 换到 | 要重写 | 成本 |
|---|---|---|
| B（CleanRL 单文件） | 只换 `trainers/ppo.py` | ~1 天 |
| C（SB3） | env 压成定长 Discrete + 丢分层 | ~3 天，且要重新论证还学不学得动 |
| D（numpy） | 只换 `trainers/` | ~2 天（无自动微分） |

**D-A 落地判据**（M1 第一个 PR 必须全绿，缺一条就打回）：
- A1 一局内能枚举 ≥100 个 choice 阶段，每阶段 `len(choices) ≥ 1`；
- A2 同 seed 两次运行，动作序列逐位一致（证明 RNG 真的分离了）；
- A3 构造一次 `Obs` ≤ 1 ms（实测 2.8 ms/决策含编码，留 3× 余量）；
- A4 1e5 步 rollout ≤ 5 分钟。

## 5. 需要人拍板的一句话

**「A」还是「B」**（两者差别只是"PPO 自己写还是抄单文件"）。C 不建议，D 只在装不上 torch 时启用。
