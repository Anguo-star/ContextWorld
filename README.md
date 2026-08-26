# ContextWorld

ContextWorld 是面向 latent 世界模型的上下文规则学习基准（Benchmark）。模型只看到最近
的画面和动作，需要在不更新参数的情况下判断当前环境的隐藏规则，并据此预测之后会发生
什么。JEPA、LeWM、PLDM 等没有图像解码器的模型可以直接参加；评测不要求重建像素。

面向外部使用和后续 Hugging Face 发布的唯一 ContextWorld 数据根是 `ContextWorld-v1`：
它包含九项任务的 Training 和 Development 数据、任务注册表及完整性清单。当前生成物仍是
本地 staging，并不等同于已经完成正式发布。协议中的 Public Test 是用于最终报告的冻结
测试划分，当前不随数据包分发；Development 结果可用于开发和诊断，但不进入正式
scoreboard，也不构成 Public Test 结果。

## 能力与任务

ContextWorld 按模型需要从历史中识别的因果变化组织任务，而不是按模拟器数量组织：

| 能力类型 | 任务 | 环境 | 历史 | 模型需要从历史中判断什么 |
|---|---|---|---:|---|
| 即时连续响应 | 速度 | TwoRoom | 3 帧 | 相同动作会移动多远 |
| 即时连续响应 | 推手移动幅度 | PushT | 3 帧 | 同一指令让推手移动较短还是较远 |
| 即时连续响应 | 机械臂质量 | Reacher | 3 帧 | 相同力矩会产生较快还是较慢的响应 |
| 时间延迟动力学 | 动作延迟 | TwoRoom | 7 帧 | 动作等待多久才生效 |
| 接触或附着条件动力学 | 接触摩擦 | PushT | 3 帧 | 接触时的摩擦较小还是较大 |
| 接触或附着条件动力学 | 运动阻尼 | PushT | 3 帧 | 离开接触后减速较快还是较慢 |
| 接触或附着条件动力学 | 夹爪携带规则 | Cube | 3 帧 | 相同夹爪动作能否携带方块 |
| 隐藏结构转移规则 | 门通行规则 | TwoRoom | 3 帧 | 外观相同的门能否通过 |
| 隐藏结构转移规则 | 传送门出口位置 | TwoRoom | 3 帧 | 穿过同一入口后从哪里出现 |

TwoRoom 的四项任务分属即时响应、时间延迟和结构转移三类能力，不是对同一能力的
重复计数。环境只是任务的实现载体；对外报告始终以九项能力任务为单位。

`ContextWorld-v1` 为每项任务提供训练集（Training）、开发集（Development）和模拟器生成的
真实未来；本仓库提供相应评分代码与完整性审计。Public Test 是协议中的冻结测试划分，当前
保持关闭。参考结果按任务分别报告；Development 分数用于说明模型在开发数据上的表现，
不用多个种子的平均数包装成正式通过结论。
九项任务分别计分，不计算能力类型分数、跨任务平均分或统一总分。能力分类只帮助读者
理解各任务提出的因果问题，不代表权重或排行榜分组。

每项任务把模型在相同 query 上的判断换算为 0/1 正确结果，因此不同模型可以在同一任务
内直接比较正确率。不同模型的 raw latent loss 不可直接比较，也不会跨任务求平均。

原始环境 CEM 与 ContextWorld ICL 是不同的评测：前者使用原始环境数据，后者使用
`ContextWorld-v1`。CEM 的标准预算为 6 个评测 seed × 每 seed 50 次；这不是 Development
ICL 的抽样规则。原始 baseline、训练后 reference 与 Public v1 发布状态不能互相替代。

## 快速开始

以下命令安装软件并读取本地的公开 Development bundle；它们不会下载数据或模型检查点。

```bash
# 读取 YAML 配置并使用通用 adapter 接口
pip install -e .

# 读取冻结 Lance/H5 数据、重评分并运行审计
pip install -e ".[eval]"

# 使用随仓库提供的 Stable-WorldModel LeWM / PLDM / PreJEPA adapter（包含 eval 依赖）
pip install -e ".[stablewm]"

# 开发与运行测试
pip install -e ".[dev]"

export CONTEXTWORLD_BENCHMARK_ROOT=/path/to/ContextWorld-v1

contextworld-benchmark info

# 对公开 Development 运行内置或外部模型评测。
python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --evaluation-split development \
  --task action_strength \
  --adapter prejepa \
  --checkpoint /path/to/model.ckpt \
  --model-name my-model \
  --output /path/to/development-result.json
```

这些 extras 只解决 Python 软件依赖；数据和 checkpoint 不随 pip 包分发。`.[stablewm]`
会安装可导入的 Stable-WorldModel 运行时。原始环境训练与 CEM 另需原始 `quentinll` 数据；
它不是 ContextWorld 发布包的一部分。若要复现历史冻结运行，才需要显式设置
`CONTEXTWORLD_ARTIFACT_ROOT` 指向私有 `context_world` archive，并使用历史 release 指定的
源码 checkout 与 commit；该路径不是公开 Training 或 Development 工作流的依赖。

九个独立评测入口是：

```text
contextworld-speed
contextworld-action-strength
contextworld-reacher-arm-mass
contextworld-action-delay
contextworld-contact-friction
contextworld-motion-damping
contextworld-cube-gripper-carry
contextworld-door
contextworld-portal-exit
```

九个任务专用命令内置 Stable-WorldModel LeWM / PLDM；通用评测模块
`python -m contextworld.benchmarks.external_model_cli` 还内置 PreJEPA。命令行入口存在，表示
该任务的数据与评分接口可以运行，不表示参考模型已经通过该任务。公开 CLI 默认产生
Development-only 结果；它不能读取或声称 Public Test 分数。其他模型可以实现通用 Python
adapter，通过 `package.module:ClassName` 或 `contextworld.adapters` entry point 交给同一 CLI
加载，也可以直接调用 scorer API。

## 文档

任务定义、数据规模、评分规则、参考结果、结果报告规范和完整使用方法统一见：

> [ContextWorld ICL Benchmark](docs/ContextWorld_ICL_Benchmark.md)

`docs/protocols/` 保存复现协议，`docs/archive/` 保存历史材料；它们不是并列的发布文档。

## 发布状态

`ContextWorld-v1` 是唯一面向对外分发的 ContextWorld-specific Training 与 Development
数据包；其当前元数据状态仍为 `staging_not_public_release`。Public Test、正式 scoreboard
和 Public v1 最终发布决定仍保持关闭。每项任务是否已有通过门槛的历史参考方法，以
Benchmark 文档的明确证据范围为准；Development 结果不改变这些结论。发布状态与尚缺条件见
[Public v1 发布准备清单](docs/ContextWorld_Public_v1_Release_Readiness.md)。
