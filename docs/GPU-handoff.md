# GPU 环境接续开发

本 fork 已完成 M1 的 CardEnv、跨进程复现、性能及 10,000 局随机稳定性验收。
M2 的 Policy-Value 网络、PPO learner 和训练 CLI 尚未实现，当前没有可直接启动的 GPU 训练任务。
M2 首先需要实现训练闭环；完整路线图原文保存在 `analysis/project-context/milestone.md`。

## 拉取与环境自检（Linux）

使用 Python 3.12；从仓库根目录运行命令，以保持 `src.dgisim` 与 `dgisim_ext` 导入一致。
不要用 PyPI 的上游 `dgisim` 代替本 fork。

```bash
git clone --branch master https://github.com/leimuwo/Dottore-Genius-Invokation-TCG-Simulator.git
cd Dottore-Genius-Invokation-TCG-Simulator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'typing-extensions>=4.7.1'
python -B -m unittest discover src/tests
python -B analysis/cardenv_acceptance.py --games 162 --workers 4 --determinism --out analysis/gpu-env-smoke
```

环境仿真使用 CPU。PyTorch 的 CUDA wheel 需根据 GPU 主机的驱动与支持版本选择，
在该主机按 PyTorch 官方安装指引安装：<https://pytorch.org/get-started/locally/>。
若平台已预装 PyTorch，先检查以下输出，再决定是否安装，避免覆盖平台配置。

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

本机 Windows 的验收数据不能代替 Linux GPU 主机上的环境自检；异平台复现及吞吐需重新记录。
`requirements.txt` 包含上游完整开发工具，CardEnv 的最小自检仅需 typing-extensions。

## M2 的约束

- 使用 `reset / obs / step / fork`，按候选项内容评分；分层 choice 包含选择前缀。
- 训练用 `decks/train` 六套牌组，`decks/test` 三套只用于固定评估；先后手均衡。
- 终局读取 `info['rewards']` 中的双方奖励；超时双方 -1，不能令 P2 奖励等于 P1 的负数。
- 引擎 RNG 与策略 RNG 分离，记录环境版本、配置、种子、模型和优化器状态。
- 非法、异常、空转使该局失效，丢弃该局训练样本并保存诊断，不当成正常终局。
- shaping 默认关闭，开启规则见 `docs/decisions/D-D-reward-shaping.md`。
- 按 AGENTS.md 保持规则边界、测试完整性及每次修改后的全量回归。

M2 首个阶段验证采样、GAE、更新、保存/恢复和固定评估；路线图 Gate 2 还要求数值稳定，
在固定测试牌组与先后手均衡的独立评估中对 Random 达到目标 80% 胜率，并优于训练前模型。
80% 是待验证目标，不是当前结果。

## M1 证据位置

验收解释见 `docs/M1-acceptance-2026-10-21.md`。
可迁移的原始摘要、源码指纹、跨进程结果及账本校验和位于
`analysis/evidence/cardenv-m1-20261021/`。
逐步压缩账本约 374 MB，保留于原 Windows 主机，不随 Git 提交；可用验收脚本重新生成。
manifest 保留历史 Windows 路径和 dirty 状态，它描述验收时的内容，不是 GPU 主机路径。
