# ContextWorld

ContextWorld 是用于评测 latent 世界模型上下文学习能力的 benchmark。评测时，模型只能
看到最近的图像和动作，不能读取速度、延迟、质量、接触属性或空间转移规则等隐藏变量。
模型需要从已经观察到的物理响应中判断当前规律，并在不更新参数的情况下预测下一状态。

评测直接在参评模型自己的 latent 空间中进行，不要求图像解码器或像素重建。因此，
JEPA、LeWM、PLDM、PreJEPA 以及其他 latent 世界模型都可以使用同一套协议。

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

九项任务分别计分，不计算跨任务平均分或统一总分。这样，某个环境承载的任务较多也不会
获得更高权重。

## 数据与评测

`ContextWorld-v1` 是 benchmark 数据的统一分发包，包含九项任务的 Training、Development
和 Test 数据、任务注册表、组件说明和文件完整性清单。该数据包已在本地完成组装，但尚未
公布稳定的公共下载版本。

这些样本来自环境模拟器的连续真实轨迹：生成器改变待识别的隐藏规律，连续执行历史与
查询动作，并保存模拟器产生的真实未来。图像不是由生成式模型合成或编辑的。配对规则、
拆分隔离和九项任务的生成入口见[数据生成方法](docs/Data_Generation.md)。

Development 用于实现检查、模型开发、训练配方选择和消融；Test 只用于选定方法后的最终
报告，不应反馈到调参或模型选择。两者均随数据包公开，并可用冻结评分器离线复现。当前不
提供托管提交服务，因此离线 Test 结果不是由服务器集中验证的排行榜条目。

ContextWorld 分别报告两项互补指标：

- **ICL 正确率**：模型是否利用交互历史识别了隐藏规律；
- **原任务 CEM**：使用组件数据训练后，模型原有的规划能力是否保持。

两项指标不合成一个分数。模型可能学会隐藏规律但损害规划能力，也可能保持规划能力却
没有学会目标规律。

## 快速开始

以下命令使用本地 `ContextWorld-v1` 数据包，不会自动下载数据或模型检查点。

```bash
# Benchmark 配置和通用 Adapter 接口
pip install -e .

# 数据读取与评分
pip install -e ".[eval]"

# 内置 LeWM、PLDM 和 PreJEPA 集成
pip install -e ".[stablewm]"

export CONTEXTWORLD_BENCHMARK_ROOT=/path/to/ContextWorld-v1

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

### 固定研究参考

当前参考升级为 v3，修复 v2 校核发现的判定与来源绑定缺口。验收证据与明确的能力范围见
[最终验收记录](docs/reference/Baseline_Final_Acceptance_2026-09-10.md)。

后续方法以
[`contextworld_joint_scratch_v1_reference_results_freeze_v3.json`](configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v3.json)
为比较起点。该记录绑定逐单元源结果快照、主分数、门槛决策、数据选择、检查点与代码身份；
历史运行时未自动记录的部分明确披露，新补评记录实际运行时。

```bash
# 核对冻结来源、分数和门槛决策
python scripts/freeze_current_reference_baseline.py verify

# 核对主文档和 Development 表是否仍与冻结记录一致
python scripts/render_current_reference_tables.py --check

# 运行固定正/负控制
python scripts/verify_reference_capability_controls.py

# 新方法使用同一判定器，原始评测 JSON 保持不变
python -m contextworld.benchmarks.reference_decision single \
  --component action_delay --split development --input result.json \
  --output reference_decision.json
```

新方法追加独立结果并引用该 freeze ID。改变任务数据、主指标或门槛语义时建立新版本，
不覆盖已有参考，也不以提高 baseline 分数作为封板条件。
方法级判定使用同模块的 `method` 子命令，重复三次 `--input`；三个独立训练种子、不同
检查点哈希及一致配方/adapter 身份必须齐全，且各自通过。旧单组件 scorer 的历史通过字段
保留原义，当前结论以 `reference_decision` 为准。

## 接入其他模型

数据包不限制模型必须属于内置模型族。外部模型只需实现
`contextworld.benchmarks.adapters.LatentWorldModelAdapter`，把自己的编码器和 latent
rollout 接口转换为统一输入格式。评分器不要求解码器，也不会提供隐藏模拟器状态。

接口方法、数组形状和运行示例见
[外部模型 Adapter 规范](docs/External_Model_Adapter_Contract.md)。

## 文档

- [Benchmark 规范](docs/ContextWorld_ICL_Benchmark.md)：任务、数据、指标、参考结果和报告规则；
- [数据生成方法](docs/Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
- [ContextWorld-v1 数据集指南](docs/HF_Dataset_Export.md)：分发目录、加载方式和维护者导出流程；
- [Stable-WorldModel 训练](docs/StableWM_Training.md)：内置参考模型的可复现训练入口；
- [文档导航](docs/README.md)：公开指南、结果复现附录和历史协议。

`docs/protocols/` 和 `docs/archive/` 保存详细实验协议与历史材料，用于复核已经报告的结果，
不是运行公开 Training 或 Development 工作流的前置要求。

## 发布状态

软件接口和九项任务的 Training/Development/Test 数据包已在本地完成发布候选组装。Public
v1 正式发布仍需稳定的数据集修订、最终分发元数据和干净环境验证；Test 已进入公开离线
评测合同，不再依赖维护方代跑模型。

剩余发布条件见
[Public v1 发布清单](docs/ContextWorld_Public_v1_Release_Readiness.md)。
