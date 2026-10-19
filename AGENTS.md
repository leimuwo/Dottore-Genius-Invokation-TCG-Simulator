# AGENTS.md

本仓库是 **dgisim 的 fork**（上游已 archive，基线 commit `fd39ddc`）。目标是把它变成可复现的两人零和不完全信息博弈训练环境。

## 绝对边界

- 只准用公开面 `src.dgisim.*`（本仓库测试的既有约定）；禁止 `src.dgisim._core`。
- `src/dgisim/**` 的任何改动必须：`enginefix(<ID>):` 前缀 + `deviations.md` 登记 + 一个不变量测试。
- 禁止删除 / 跳过 `src/tests/**` 中的任何断言或测试。
- 禁止修改 `.github/workflows/**`、`pyproject.toml`、`scripts/verify_*.py`（需人工批准，见 CODEOWNERS）。

## 每轮工作方式

先检查当前状态 → 只推进一个明确步骤 → 跑 `python -m unittest discover src/tests` → 失败就修，不许改断言 → 通过后一个 commit。

## 命令

```
# 全量测试（Windows 上必须用绝对路径的解释器；PATH 里的 python 是存根）
& 'C:\Users\wenxi\miniconda3\python.exe' -m unittest discover src/tests

# 单条改动验收（红 → 绿）；护栏在代码里，不在 prompt 里
& 'C:\Users\wenxi\miniconda3\python.exe' scripts\verify_item.py B14 --phase red
& 'C:\Users\wenxi\miniconda3\python.exe' scripts\verify_item.py B14 --phase green

# 编码覆盖率 / 文档预算 / 测试完整性
& 'C:\Users\wenxi\miniconda3\python.exe' scripts\verify_encoding_coverage.py
& 'C:\Users\wenxi\miniconda3\python.exe' scripts\verify_doc_budget.py
& 'C:\Users\wenxi\miniconda3\python.exe' scripts\verify_test_integrity.py
```

## 硬禁令（都有实测依据）

- **禁止 `LinearEnv.copy()`**：旧实现会走 `__init__` → `reset()`，白烧一整局随机数。现已修好并有不变量测试，但仍建议在新局用 `LinearEnv()` + 显式 seed。
- **禁止把零向量/补零向量传进 `step()`**：`decoding([0]*81)` = `EndRoundAction`，在 ActionPhase 里等于"结束回合"。
- **每次 accepted 决策必须产生状态迁移**；`hash` 与 `waiting_for` 都没变就是 noop，必须记录并终止该局。
- 不许把状态表 `print` 到 stdout；诊断走 logging。
- 不用 `GameStateMachine`（每步 `random.seed()`，无法复现）。

## 现在不要做的事

- 不要实现新卡/新角色（论文口径固定为"v0.6.0 子集"）。
- 不要拆 `card.py` / `status.py` 巨型声明文件。
- 不要改 `pyproject.toml` 的包映射结构。
- 不要重写 `EffectStack` 语义。
- 不要写超过 8 KB 的文档；文档写"为什么"，不写"我做了什么"。
