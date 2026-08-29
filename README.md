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
