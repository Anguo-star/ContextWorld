# ContextWorld

ContextWorld 是面向 latent 世界模型的上下文规则学习基准（Benchmark）。模型只看到最近
的画面和动作，需要在不更新参数的情况下判断当前环境的隐藏规则，并据此预测之后会发生
什么。JEPA、LeWM、PLDM 等没有图像解码器的模型可以直接参加；评测不要求重建像素。

## 任务

| 任务 | 环境 | 历史 | 模型需要判断的隐藏规则 |
|---|---|---:|---|
| 速度 | TwoRoom | 3 帧 | 每个动作步移动多远 |
| 门通行规则 | TwoRoom | 3 帧 | 外观相同的门能否通过 |
| 动作延迟 | TwoRoom | 7 帧 | 动作等待多久才执行 |
| 推手移动幅度 | PushT | 3 帧 | 相同指令让推手移动得较短还是较远 |
| 接触摩擦 | PushT | 3 帧 | 推手与物体接触时摩擦较小还是较大 |
| 运动阻尼 | PushT | 3 帧 | 物体离开推手后减速较快还是较慢 |
| 机械臂质量 | Reacher | 3 帧 | 相同力矩下机械臂响应较快还是较慢 |
| 传送门出口位置 | TwoRoom | 3 帧 | 穿过同一入口后从哪个位置出现 |

每项任务都包含训练集（Training）、开发集（Development）、冻结的公开测试集（Public
Test）、模拟器生成的真实未来、评分代码和完整性审计。参考结果按任务分别报告：有些方法
稳定通过，有些只得到负结果；没有通过门槛的任务不会用多个种子的平均数包装成成功。
八项任务分别计分，不计算统一总分。

每项任务把模型在相同 query 上的判断换算为 0/1 正确结果，因此不同模型可以在同一任务
内直接比较正确率。不同模型的 raw latent loss 不可直接比较，也不会跨任务求平均。

Cube 夹爪携带规则（History=3）目前作为未发布研发候选记录在主 Benchmark 文档 6.9：
LeWM 已通过三种子 Development 和原 Cube CEM 留存，PLDM 未通过 Development；一次性
Public 前冻结已通过，但 Public 数据、分数和判定仍未生成。它尚未计入上述八项任务、
统一 Suite 或公开命令。

## 快速开始

```bash
pip install -e .

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

八个独立评测入口是：

```text
contextworld-speed
contextworld-door
contextworld-action-delay
contextworld-action-strength
contextworld-contact-friction
contextworld-motion-damping
contextworld-reacher-arm-mass
contextworld-portal-exit
```

当前参考运行时支持 Stable-WorldModel LeWM / PLDM。命令行入口存在，表示该任务的数据与
评分接口可以运行，不表示参考模型已经通过该任务。其他模型工程可以实现公开适配接口，
复用同一份冻结数据和评分规则。

## 文档

任务定义、数据规模、评分规则、参考结果、结果报告规范和完整使用方法统一见：

> [ContextWorld ICL Benchmark](docs/ContextWorld_ICL_Benchmark.md)

`docs/protocols/` 保存复现协议，`docs/archive/` 保存历史材料；它们不是并列的发布文档。

## 发布状态

八项任务已统一登记在同一套数据、评分和审计接口中。正式公共分发仍需补充源码许可证、
自产数据许可证和稳定的公共下载地址，因此当前版本称为“本地技术发布候选”。每项任务
是否已有通过门槛的参考方法，以正式 Benchmark 文档中的结果表为准。
