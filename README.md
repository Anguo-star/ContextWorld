# ContextWorld

ContextWorld 是用于评测 latent 世界模型上下文学习能力的 benchmark。评测时，模型只能
看到最近的图像和动作，不能读取速度、延迟、质量、接触属性或空间转移规则等隐藏变量。
模型需要从已经观察到的物理响应中判断当前规律，并在不更新参数的情况下预测下一状态。

评测直接在参评模型自己的 latent 空间中进行，不要求图像解码器或像素重建。因此，
JEPA、LeWM、PLDM、PreJEPA 以及其他 latent 世界模型都可以使用同一套协议。

## 从这里开始

- **[ContextWorld 统一技术报告](docs/ContextWorld_ICL_Benchmark.md)**：任务、数据版本、评分协议、九任务三模型训练方式对比与主要发现。
- [直接查看最新总表](docs/ContextWorld_ICL_Benchmark.md#training-comparison)：按任务横向比较原始模型、ICL 从头训练，以及原始权重初始化（含冻结 Encoder 对照）的 ICL / CEM。
- [文档导航](docs/README.md)：数据生成、模型接入、训练和结果复现。

## 九项任务

任务按模型需要识别的隐藏动力学类型组织。环境只是任务的物理载体，不构成类别权重。

| 能力类型 | 任务 | 环境 | 历史长度 | 模型需要判断什么 |
|---|---|---|---:|---|
| 即时连续响应 | 速度 | TwoRoom | 3 帧 | 相同动作会移动多远 |
| 即时连续响应 | 推手移动幅度 | PushT | 3 帧 | 相同动作会让推手移动较短还是较远 |
| 即时连续响应 | 机械臂质量 | Reacher | 3 帧 | 质量如何改变机械臂对力矩的响应 |
| 时间延迟动力学 | 动作延迟 | TwoRoom | 7 帧 | 动作等待多少步才生效 |
| 接触或附着条件动力学 | 接触摩擦 | PushT | 3 帧 | 接触时摩擦如何改变运动 |
| 接触或附着条件动力学 | 运动阻尼 | PushT | 3 帧 | 脱离接触后运动衰减得快还是慢 |
| 接触或附着条件动力学 | Cube 夹爪携带规则 | Cube | 3 帧 | 闭合夹爪能否携带方块 |
| 隐藏结构转移 | 门通行规则 | TwoRoom | 3 帧 | 外观相同的门能否通过 |
| 隐藏结构转移 | 传送门出口位置 | TwoRoom | 3 帧 | 进入同一入口后会从哪里离开 |

九项任务分别报告 ICL 评分、条件响应和原环境 CEM，便于比较具体能力与规划表现。

## 数据与评测

`ContextWorld-v3-hf` 是面向 Hugging Face 的发布候选目录，包含九项任务的 Training、
Development 和 Test 数据、任务注册表、数据卡与文件完整性清单。它组合 v3 已冻结的
`ContextWorld-v1` Training/Development 与 `ContextWorld-v1-full` Test，保留原始数据字节。
稳定的公共下载地址与 revision 尚未发布。

这些样本来自环境模拟器的连续真实轨迹：生成器改变待识别的隐藏规律，连续执行历史与
查询动作，并保存模拟器产生的真实未来。图像不是由生成式模型合成或编辑的。配对规则、
拆分隔离和九项任务的生成入口见[数据生成方法](docs/Data_Generation.md)。

Development 用于实现检查、模型开发、训练配方选择和消融；Test 只用于选定方法后的最终
报告，不应反馈到调参或模型选择。两者均随数据包公开，并可用冻结评分器离线复现。当前不
提供托管提交服务，因此离线 Test 结果不是由服务器集中验证的排行榜条目。

主文档用横向主表概览九任务、三模型和各训练方案的 ICL / CEM；各任务详情提供历史利用、响应幅值与误差、泛化条件和重复数。相同配置的全部可用训练重复参与均值，详情报告标准差；数据扩量另作比较。原环境 CEM 衡量规划能力保持，不替代隐藏规律下的规划评测。

## 快速开始

以下命令使用本地 `ContextWorld-v3-hf` 候选包，不会自动下载数据或模型检查点。

```bash
# Benchmark 配置和通用 Adapter 接口
pip install -e .

# 数据读取与评分
pip install -e ".[eval]"

# 内置 LeWM、PLDM 和 PreJEPA 集成
pip install -e ".[stablewm]"

# 完整数据放在仓库外的原 data 目录；按实际挂载路径设置。
export CONTEXTWORLD_DATASET_ROOT=/absolute/path/data/world_model
export CONTEXTWORLD_BENCHMARK_ROOT="$CONTEXTWORLD_DATASET_ROOT/ContextWorld-v3-hf"

contextworld-benchmark info

python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --evaluation-split development \
  --task action_strength \
  --adapter prejepa \
  --checkpoint /path/to/model.ckpt \
  --model-name my-model \
  --output /path/to/development-result.json

# 方法与训练配方固定后，才运行公开 Test：
python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --evaluation-split test \
  --task action_strength \
  --adapter prejepa \
  --checkpoint /path/to/model.ckpt \
  --model-name my-model \
  --output /path/to/test-result.json
```

这些 Python extras 只安装软件依赖；数据和模型检查点单独分发。

### Public Test 最终报告

单个模型的临时评测用上面的 `external_model_cli`；成批、需要留下证据的最终报告用
`contextworld-public-test-report`，输入一份冻结的检查点清单（JSON，逐单元声明组件、
检查点、输出路径与 `admission.cleared_development`），按 `plan` → `run` → `verify`
三步执行：

```bash
# 事前核对：列出将要执行的单元与跳过原因，不跑任何评测
contextworld-public-test-report plan --manifest frozen_checkpoints.json

# 执行全部单元的公开 Test 评测，并写出回执（含各文件 sha256）
contextworld-public-test-report run --manifest frozen_checkpoints.json

# 事后核对：按回执复核产出文件是否仍在且 sha256 一致
contextworld-public-test-report verify --receipt public_test_report_<report_id>.json
```

准入闸门：未清过 Development 的单元会被拒绝执行（回执记为 `skipped_not_admitted`），
公开 Test 只用于最终报告，不参与模型选择或调参。

当前参考使用 `schema_version: 2` 的清单。每个单元的 `admission` 除
`cleared_development` 外，还须提供 `development_result: {"path": "/absolute/result.json",
"sha256": "..."}`。入口核对任务、训练种子、检查点和源文件哈希，再按统一门槛重新判定；
手填通过标志不能绕过检查。旧 v1 清单保留历史语义，不代表已经经过当前参考的准入核验。

### 结果如何组织

[统一技术报告](docs/ContextWorld_ICL_Benchmark.md)区分两类结果：冻结 v3 保留既有数据、评分与三种子参考；较新的训练方式研究报告扩量数据、完整原始初始化和冻结 Encoder 的 Development 结果，并按各配置可用的训练重复汇总。后者不覆盖前者，也不自动成为新的正式 Test 排行榜。

已有研究显示，固定数据和原始初始化时，冻结 Encoder 可明显改善部分任务，却使另一些任务退步。数据量、模型和表示更新方式需要分别检验；条件预测与原环境规划也可能不同步。具体分数、响应幅值、反例和解释边界均在技术报告维护。

```bash
# 检查冻结参考与当前研究表分别是否和各自数据源一致
python scripts/render_current_reference_tables.py --check
python scripts/render_training_comparison.py --check
```

固定 v3 的验收与复现细节见[参考结果附录](docs/reference/Benchmark_Result_Provenance.md)。新方法与新的训练数据使用独立结果身份；任务定义、评分合同或评测数据变化需建立新版本，不能回写冻结数字。

## 接入其他模型

数据包不限制模型必须属于内置模型族。外部模型只需实现
`contextworld.benchmarks.adapters.LatentWorldModelAdapter`，把自己的编码器和 latent
rollout 接口转换为统一输入格式。评分器不要求解码器，也不会提供隐藏模拟器状态。

接口方法、数组形状和运行示例见
[外部模型 Adapter 规范](docs/External_Model_Adapter_Contract.md)。

## 文档

- [统一技术报告](docs/ContextWorld_ICL_Benchmark.md)：任务、数据、全部结果、发现与报告规则；
- [数据生成方法](docs/Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
- [HF 数据集指南](docs/HF_Dataset_Export.md)：v3 发布目录、加载方式和维护者打包流程；
- [Stable-WorldModel 训练](docs/StableWM_Training.md)：内置参考模型的可复现训练入口；
- [文档导航](docs/README.md)：公开指南、结果复现附录和历史协议。

`docs/protocols/` 和 `docs/archive/` 保存详细实验协议与历史材料，用于复核已经报告的结果，
不是运行公开 Training 或 Development 工作流的前置要求。

## 发布状态

v3 研究基线已封板；HF 发布候选沿用该基线，发布准备不修改数据、评分或参考结果。
正式发布需确定 HF 数据集仓库、上传后固定 revision，并完成从该 revision 下载与加载的
验证。Test 使用公开离线评测合同；外部独立复现作为单独的证据工作记录。

剩余发布条件见
[v3 发布需求与验收](docs/ContextWorld_Public_v1_Release_Readiness.md)。
