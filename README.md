# ContextWorld

ContextWorld 是面向 latent 世界模型的上下文规则学习基准（Benchmark）。模型只看到最近
的画面和动作，需要在不更新参数的情况下判断当前环境的隐藏规则，并据此预测之后会发生
什么。JEPA、LeWM、PLDM 等没有图像解码器的模型可以直接参加；评测不要求重建像素。

当前状态：九项任务的源码、配置、评分器和本地审计接口已具备技术运行条件；协议中的
Public Test 是用于最终报告的冻结测试划分，当前不可公开下载。完整评测只面向已获授权并
持有冻结工件的使用者；Public v1 的许可证、可下载工件和正式发布决定尚未完成。

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

每项任务都包含训练集（Training）、开发集（Development）、Public Test（协议中的冻结测试划分，
当前并非公开下载）、
模拟器生成的真实未来、评分代码和完整性审计。Public Test 是协议名称；当前公共下载和
部分参考运行授权仍受发布门限制。参考结果按任务分别报告：有些方法稳定通过，有些只得到
负结果；没有通过门槛的任务不会用多个种子的平均数包装成成功。
九项任务分别计分，不计算能力类型分数、跨任务平均分或统一总分。能力分类只帮助读者
理解各任务提出的因果问题，不代表权重或排行榜分组。

每项任务把模型在相同 query 上的判断换算为 0/1 正确结果，因此不同模型可以在同一任务
内直接比较正确率。不同模型的 raw latent loss 不可直接比较，也不会跨任务求平均。

训练后 Public 与 CEM 是否执行，均由各任务预先规定的流程决定；未通过 Development 的方法
通常不进入 Public，具体例外和证据范围见正式结果表。原始 baseline、训练后 reference 与
Public v1 发布状态不能互相替代。

## 快速开始

以下命令安装软件并检查本地已有工件；它们不会下载 Benchmark 数据或正式 checkpoint。
完整评测需要使用者自备或获授权取得冻结工件，公共下载将在 Public v1 正式发布后另行提供。

```bash
# 读取 YAML 配置并使用通用 adapter 接口
pip install -e .

# 读取冻结 Lance/H5 数据、重评分并运行审计
pip install -e ".[eval]"

# 使用随仓库提供的 Stable-WorldModel LeWM / PLDM adapter（包含 eval 依赖）
pip install -e ".[stablewm]"

# 开发与运行测试
pip install -e ".[dev]"

export CONTEXTWORLD_ARTIFACT_ROOT=/path/to/benchmark
export CONTEXTWORLD_TWOROOM_H5=/path/to/tworoom.h5
export CONTEXTWORLD_TWOROOM_LANCE=/path/to/lewm_tworoom.lance
export CONTEXTWORLD_PUSHT_H5=/path/to/pusht_expert_train.h5
export CONTEXTWORLD_PUSHT_LANCE=/path/to/lewm_pusht.lance
export CONTEXTWORLD_PUSHT_INIT_CHECKPOINT=/path/to/pusht_lewm_baseline_weights.ckpt
export CONTEXTWORLD_REACHER_H5=/path/to/reacher.h5
export CONTEXTWORLD_REACHER_LANCE=/path/to/lewm_reacher.lance
export CONTEXTWORLD_REACHER_LEWM_INIT_CHECKPOINT=/path/to/reacher_lewm_weights.ckpt
export CONTEXTWORLD_REACHER_PLDM_INIT_CHECKPOINT=/path/to/reacher_pldm_baseline_weights.ckpt

contextworld-benchmark info
contextworld-benchmark results
contextworld-benchmark audit --full
```

这些 extras 只解决 Python 软件依赖；数据和 checkpoint 不随 pip 包分发。`.[stablewm]`
会安装可导入的 Stable-WorldModel 运行时，但正式冻结 checkpoint 的审计仍须提供各任务
release 指定的源码 checkout 和 commit，因为历史 checkpoint 需要由对应训练配置加载。

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

当前命令行内置 Stable-WorldModel LeWM / PLDM。命令行入口存在，表示该任务的数据与评分
接口可以运行，不表示参考模型已经通过该任务。其他模型可以实现通用 Python adapter 并
直接调用 scorer API，复用同一份冻结数据和评分规则；自定义 adapter 的 CLI 插件加载尚未
提供。

## 文档

任务定义、数据规模、评分规则、参考结果、结果报告规范和完整使用方法统一见：

> [ContextWorld ICL Benchmark](docs/ContextWorld_ICL_Benchmark.md)

`docs/protocols/` 保存复现协议，`docs/archive/` 保存历史材料；它们不是并列的发布文档。

## 发布状态

本仓库提供九项任务的本地技术接口，但不提供可下载的 Public v1 数据发布。源码许可证、
自产数据许可证和稳定公共下载地址仍待补齐。每项任务
是否已有通过门槛的参考方法，以正式 Benchmark 文档中的结果表为准。Public v1 的发布状态与
尚缺条件见 [Public v1 发布准备清单](docs/ContextWorld_Public_v1_Release_Readiness.md)；该文档不授予
新的 Public Test 访问或正式发布身份。
