# ContextWorld Public v1 发布准备与多模型验证矩阵

**状态：草案；不构成发布、预注册、Public Test 授权或正式 scoreboard。** 机器可读状态见
[`contextworld_public_v1_release_readiness_draft_v1.yaml`](../configs/benchmark/contextworld_public_v1_release_readiness_draft_v1.yaml)。

本文用于把已经冻结的本地技术候选推进为可对外下载、可独立复现的 Public v1。现有
Suite v2 decision、Cube reference 行和
[`ContextWorld_ICL_Benchmark.md`](ContextWorld_ICL_Benchmark.md) 保持不变；完成下列工作后，
再通过新的版本化发布决定把本文中的有效内容合入正式 Public v1 文档。
这里的“Public v1”是对外分发版本，不会把内部已经冻结的 Suite v2 协议重命名为 Suite v1。

## 1. 当前结论

Cube 已经证明当前 Benchmark 能区分正、负参考结果：LeWM 在三个训练种子上通过
Development、一次性 Public Test 和预注册 CEM 留存门，PLDM 使用同一批合成训练数据但在
Development 为 0/3 未通过。它支持“当前 LeWM 配方学会了 History=3 二值夹爪携带规则，
且规划退化未超过预注册边界”，但不能外推为“所有模型都会学会”或“训练后 CEM 完全没有
下降”。

### 1.1 Cube 当前可核验对照

| 模型 | 训练状态 | ICL 数据口径 | ICL 主分数 | ICL 判定 | 原 Cube CEM | 规划判定 |
|---|---|---|---:|---|---:|---|
| LeWM | 原始 checkpoint | 未按 v4r1 ICL 协议评测 | 未评测 | 未判定 | 66.00%（198/300） | 参考值 |
| LeWM | 固定图像编码器，拟合配对真实未来 | Public Test | 78.45%；77.73%、79.10%、78.52% | 通过（3/3） | 平均 61.56%；62.00%、61.00%、61.67% | 保持 |
| PLDM | 使用相同合成数据，`mixed_pldm_joint` | Development | 50.13%；50.20%、50.20%、50.00% | 未通过（0/3） | 未评测 | 未判定 |

PLDM 行明确是 Development 负结果，不得混入正式 Public scoreboard。原始 LeWM 没有
v4r1 ICL 分数，因此只报告真实存在的 CEM baseline，不能补写一个推测的 50%。正式
scoreboard 目前仍只有一条 Cube LeWM Public 行。

## 2. 为什么 Public v1 还需要更多开源模型

单个正参考和单个负参考已经足以证明评测链能够工作，但不足以支撑较强的跨架构外部有效性。
对外 Public v1 建议在当前 LeWM、PLDM 之外增加三个开源模型槽位，并至少覆盖两个独立上游
仓库和两类不同架构或训练目标。这里要求的是覆盖面和诚实报告，不要求所有模型通过：

- 每个可训练方法运行三个独立训练种子；
- 所有模型使用相同的数据和 query identity、History=3、可见像素与动作字段；评测期间禁止
  更新参数；
- 模型选择、配方选择和 checkpoint 选择只能使用 Training/Development；
- 在运行前冻结共同训练预算口径，并报告总参数、可训练参数、optimizer steps、样本量、GPU
  和 wall time；模型专属 best-effort 配方必须与共同预算轨分开列出；
- 首次读取 Public 前冻结上游 commit、许可证、适配器、训练配方和 checkpoint SHA；
- Development 未通过的模型保留为负结果，不进入 Public；
- 跨模型只比较相同 query 上的任务正确率，不比较不同 latent 空间的原始 MSE；
- 支持规划的模型使用同一冻结 query 做训练前后 CEM；不支持规划的模型可以报告 ICL，
  但不能用于“保持规划能力”的结论；
- 至少一个新增外部模型应具备可比较的 Cube 规划接口，以检验 CEM 留存结论是否超出 LeWM
  单一配方。

### 2.1 多开源模型对比表（待补齐）

以下是发布前工作表，不是成绩表；空白项必须由真实运行和冻结证据填写。

| 外部模型槽位 | 模型与上游 commit | 开源许可证 | 架构/目标类别 | 参数量与训练预算 | 适配器 | 原始 Development | 训练后 Development（3 seeds） | Public（3 seeds） | CEM baseline → trained | 状态 |
|---|---|---|---|---|---|---:|---:|---:|---|---|
| External-01 | 待选 | 待核验 | 待分类 | 待冻结 | 待实现 | 待运行 | 待运行 | 未授权 | 待运行 | 待补齐 |
| External-02 | 待选 | 待核验 | 待分类 | 待冻结 | 待实现 | 待运行 | 待运行 | 未授权 | 待运行 | 待补齐 |
| External-03 | 待选 | 待核验 | 待分类 | 待冻结 | 待实现 | 待运行 | 待运行 | 未授权 | 待运行 | 待补齐 |

## 3. Public v1 发布门槛

| 发布门 | 当前状态 | Public v1 通过条件 |
|---|---|---|
| Cube 科学评测链 | 已通过当前 LeWM reference 范围 | 保留现有冻结结果和因果/泄漏审计 |
| 外部开源模型覆盖 | **阻断：未运行** | 完成上节建议矩阵；失败模型同样归档 |
| 源码许可证 | **阻断：未声明** | 仓库根目录提供明确 LICENSE，并与依赖兼容 |
| 合成数据许可证 | **阻断：未声明** | 明确自产数据许可、用途和再分发边界 |
| 上游再分发与署名 | **阻断：未清理** | 分别核验数据、环境、checkpoint 和代码的权利与 NOTICE |
| 稳定公共下载 | **阻断：未配置** | 发布不可变 URL、字节数和 SHA256；禁止依赖本机绝对路径 |
| 下载 manifest | **阻断：未发布** | 提供机器可读 manifest、分片恢复和全量校验命令 |
| 干净环境复现 | **阻断：未记录** | 从全新 checkout/环境完成安装、下载、audit 和至少一个外部模型评测 |
| 外部提交治理 | **阻断：未授权** | 定义提交 schema、身份校验、重复提交和 scoreboard 纳入规则 |
| 跨模型共同训练预算 | **阻断：未冻结** | 定义共同预算轨；best-effort 结果单独报告，禁止混表 |
| 引用信息 | **阻断：缺失** | 增加 CITATION 元数据和版本 DOI/归档标识（如采用） |
| 正式文档修订 | **等待上述门** | 走版本化 amendment，更新哈希和 release decision，不原地改冻结证据 |

模型没有通过 ICL 门槛本身不是发布阻断；缺少运行、身份、许可证或诚实的负结果记录才是。

## 4. 执行顺序

1. 先确定外部模型选择准则、共同训练预算和三个候选，不读取 Public；
2. 核验每个上游的 commit、许可证、checkpoint 来源和输入输出接口；
3. 完成适配器 smoke、Training/Development 三种子训练和评分；
4. 对通过 Development 的冻结方法另行预注册一次性 Public；
5. 对支持规划的模型运行配对 CEM 留存；
6. 并行补齐许可证、公开下载 manifest、干净环境复现和提交治理；
7. 所有阻断门通过后创建 Public v1 的版本化正式文档与 release decision。

在第 7 步之前，本文件可以持续更新，但不得修改或冒充现有冻结的 Cube reference 结果。
