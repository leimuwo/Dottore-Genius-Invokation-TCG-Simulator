# CardEnv M1 验收依据（2026-10-21）

本次关闭的是接口、性能及随机稳定性验收；不代表 PPO 或训练效果已完成。
规格见 `docs/superpowers/specs/2026-10-21-cardenv-m1-design.md`；原接口与门槛见
`docs/decisions/D-A-training-stack.md` §2、§4。

## 接口边界

重掷、换牌及付款按种类逐个选择数量，覆盖全部多重集子集；付款候选仅保留有合法完成方案的分支。
观察携带当前选择前缀，另一席位不见私有前缀。每环境独立保存引擎随机流，fork 复制随机状态。
非法动作、异常和提交后无迁移均分类计数、使该局失效并要求 reset；终局奖励明确标注双方和执行席位。

性能开销来自重复编码及 encode_item 的导入/类型分派，使用等价映射查找和按状态、席位缓存。
跨进程验证还发现 E8：Status 字段原来通过 set 遍历，Support 编码随进程变化；
改为 dataclass 声明顺序，保留 usages 优先，并用不变量测试锁定。这是编码修复，不改变对局规则。

## 固定条件与结果

Windows、Python 3.12.12，9 套现有牌表（6 train +3 test）；固定引擎/策略 seed，预热后串行性能验收。
测试牌组仅用于环境验证。策略是各 choice 层均匀随机，不等于完整动作均匀随机。

| 验收 | 结果 |
|---|---|
| 三轮新状态编码平均 | 0.588 / 0.618 / 0.628 ms |
| 完整冷 obs()，编码及候选缓存均清空 | 0.789 / 0.786 / 0.796 ms；各轮 384 样本 |
| 完整冷 obs() p95 | 1.389 / 1.363 / 1.451 ms |
| 实际 100,000 choice 步，串行 | 77.107 秒，1,296.89 choice/s |
| 同 seed 跨进程 | 18 条完整轨迹一致，比较双方观测、候选、动作及奖励；子进程换 hash seed |
| 8 进程压力测试 | 10,000 / 10,000 局完成；735.010 秒 |
| 压测吞吐 | 11,478.08 choice/s；2,083.38 commit/s；13.61 局/s |
| 步数 | 8,436,504 choice；1,531,303 commit |
| 非法 / 异常 / 空转 | 均为 0，三率均为 0 |
| 终局 | P1 击杀 876；P2 击杀 931；15 回合超时 8,193（81.93%） |
| 独立逐步日志审计 | 10,000 局、81 种有序组合，索引/候选唯一性/奖励/终局/席位均衡/校验和通过 |
| 全量回归 | 651 tests，OK |

完整冷 obs() 的 ≤1 ms 是平均门槛，不能解读为所有样本或 p95 ≤1 ms。
曾与日志审计并行补测，平均为 1.398–1.467 ms；该次受资源争用影响，保留在 cold-obs.json，
正式采用独立串行复测 cold-obs-isolated.json。串行性能与 8 进程压测吞吐不可直接混比。
高超时率说明随机策略经常打满回合，不能据此推断训练胜率或策略质量。
已知 E5（Cards 负计数）仍在 deviations.md 中，当前三率为零不代表引擎所有已知缺陷消失。

## 可复核证据

产物目录：`C:/Users/wenxi/Desktop/seven/analysis/cardenv-m1-10000-20261021/`。
manifest.json 保存运行前 HEAD、dirty 状态、全部引擎/扩展源码与牌表 SHA256；
summary.json 确认运行期间源码未变，并保存逐局及逐步账本 SHA256。
benchmark.json、cold-obs-isolated.json、determinism.json、ledger-audit.json 分别对应上表。
events-*.jsonl.gz 包含 seed 对应的逐步动作、候选内容、前缀和双方奖励；games-*.jsonl 保存逐局结果。
验收时 HEAD 为 4539f29898655d496712a4293727d8ac75c92de7，包含本次未提交源码；
后续提交不改变已验收源码，版本应以 manifest 中的内容指纹为准。

源码内容指纹：`3c36a042a298535797ba777e686e4e0d8c9e75b811de3387a900b535fe869a0f`。

重跑：`python -B analysis/cardenv_acceptance.py --games 10000 --workers 8 --determinism --benchmark --out <新目录>`。
独立串行冷观察：`python -B analysis/cardenv_cold_obs.py <目录>/cold-obs-isolated.json`。
日志审计：`python -B analysis/cardenv_ledger_audit.py <目录>`。
Windows 使用 AGENTS.md 中的绝对解释器路径。每次重跑必须新建输出目录，保留失败证据。
