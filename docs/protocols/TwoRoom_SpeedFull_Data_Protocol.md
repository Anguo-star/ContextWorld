# TwoRoom-SpeedFull-v1 数据、训练与评测协议

> **文档角色**：历史数据与训练协议，不维护当前结果。当前数据比较和正式结论
> 统一见[Benchmark 主文档中的速度章节](../ContextWorld_ICL_Benchmark.md#41-速度)。

**版本**：v1.5
**日期**：2026-07-19  
**状态**：正式生成、训练、validation 评测与四模型归因均已完成

> 当前阶段结论统一见
> [ContextWorld ICL Benchmark：速度](../ContextWorld_ICL_Benchmark.md#41-速度)；
> 本文只保留 SpeedFull 数据、训练与 checkpoint 审计。
> 旧机器字段 `correct`、`wrong`、`wrong-slow` 和 `wrong-fast` 分别按同速、
> 另一档、慢速和快速历史解释，不表示速度正确或错误。

## 1. 目标

`TwoRoom-SpeedFull-v1` 用于联合修复 `TwoRoom-SpeedTask-v1` 中三个已量化的
实验缺口：同一 reset geometry 被 32 个速度重复使用、50/50 固定总预算使
original exposure 减半，以及原始 ID 从训练所用 H5 的 clip-level split 抽样。

本版本是集成协议修复，不是三个因素的独立因果消融。固定内容包括
history-3 LeWM、32 个 train speeds、全部 E4 speed support、ExpertPolicy、
lossless PNG、seed 3072、original/synthetic 50/50 batch mixture 和冻结 E4。

## 2. 独立覆盖

train 由 512 scenarios、每 scenario 32 episodes 构成，共 16,384 episodes。
`paired_cycle` 使每个 reset seed group 只属于一个 speed，避免把 factor cross
误计为独立任务覆盖。每个 speed 恰有 16 scenarios、512 episodes。

| Stratum | Scenarios | Episodes | 约束 |
|---|---:|---:|---|
| broad cross-room | 192 | 6,144 | 起点与目标异房间 |
| broad same-room | 192 | 6,144 | 起点与目标同房间 |
| E4 templates s0–s3 | 128 | 4,096 | 每模板 32 scenarios，覆盖全部 train speeds |

正式 preflight 观测到 16,384 个唯一 reset seeds、start states 和 goal states，
14 px start-goal grid pairs 为 9,892；same/cross-room 各 8,192。四个模板的
reset counts 均不低于 1,024。val 另含 96 scenarios、1,536 episodes，与 train
factor values 隔离。

## 3. 数据真实性与加载

数据使用 PNG level-1 无损编码。全量产物包含 608 scenarios、17,920 episodes、
1,203,993 rows；validator 精确重放 1,186,073 个 transitions，pixel bytes、
decoded pixels、state transitions、goal 和 termination mismatch 均为 0。

history-3 train 产生 795,156 个 synthetic raw clips。loader 不直接等权所有
scenario：先在同一 speed 内按 raw clip 数进行零复制拼接，再在 32 个 speeds
之间等权采样。该结构既保证 factor exposure 平衡，也避免短轨迹 scenario 被
过度重复。

## 4. Additive exposure

历史正式 original-only recipe 的完整预算为 6,574,080 draws。SpeedFull 保留
这部分 original exposure，并追加 6,574,080 synthetic draws：

| 参数 | 值 |
|---|---:|
| Logical epochs | 5 |
| Global draws / epoch | 2,629,632 |
| Original draws / run | 6,574,080 |
| Synthetic draws / run | 6,574,080 |
| Effective global batch | 1,024 |
| Optimizer steps | 12,840 |
| Warmup steps | 128 |

训练仍通过 Stable-WorldModel 原生 Trainer、checkpoint 和 resume 语义执行；
ContextWorld 只提供数据配置、DataModule 注入、冻结参数与启动入口。

## 5. Episode-held-out original ID

原始 `tworoom.h5` 的 10,000 episodes 使用 NumPy seed 3072 固定划分为 9,000
train episodes 与 1,000 heldout episodes。归一化统计只由 train episode rows
拟合；heldout 部分通过 HDF5 Virtual Dataset 建立只读视图，不复制图像数据。

该 ID 与历史同-H5 clip retention 协议具有不同含义，不把两者的百分比直接作为
模型优劣比较。正式 heldout ID 使用统一 `50×6` 预算。

## 6. 正式结果

最终 checkpoint 为
`artifacts/training/runs/checkpoints/h3_speedfull_s3072/weights_final_step_12840.pt`，
SHA-256 为
`79e2b2d17445a21d1685759679c6ab82f9fda834584168d7e3e02bf50b78a778`。

| Gate | 结果 | 结论 |
|---|---:|---|
| E1 K=0 latent MSE | 0.06362 | 一般 OOD prediction 基线 |
| E1 K=2 同速历史 latent MSE | 0.02839 | 同速历史明显降低误差 |
| E1 同速历史收益 | 0.03523，95% CI [0.01282, 0.06255] | prediction ICL 成立 |
| E1 另一档−同速历史 | 0.12753，95% CI [0.07568, 0.18115] | 历史内容强分离 |
| Episode-heldout ID | 289/300（96.33%） | 严格 ID 泛化保持 |
| 旧 E4 同速 / 另一档历史 | 73/300 / 73/300 | 该模板集未显示规划历史效应 |
| E4 discordant success | 0 / 0 | success effect=0 pp |

上述 E4 及跨模型配对属于历史六模型协议，使用 full-H5 normalizer。SpeedFull
相对 SpeedTask 的同速/另一档历史 final distance 分别减少 13.75/13.63 px，
但成功集合逐条不变。相对 legacy H3-Orig，SpeedFull 的同速/另一档历史 success
分别少 39/42 次，final distance 分别高 28.90/28.55 px；由于 H3-Orig 不是
episode-heldout 公平基线，这一差距只保留为历史观察。

同一个 SpeedFull checkpoint 在公平四模型协议中复用相同 E4 query 与 CEM
seed、改用 original 9,000-train-only normalizer 重评。两次同速/另一档历史
success 都是 73/300，但连续距离如下：

| E4 协议 | 同速历史距离 | 另一档历史距离 |
|---|---:|---:|
| 历史六模型，full-H5 normalizer | 103.64 px | 104.03 px |
| 公平四模型，train-only normalizer | 103.10 px | 104.26 px |

公平能力对照中，SpeedFull 在 original-heldout 为 289/300（96.33%），高于
OrigHeldout 的 273/300；在 speed=5 matched 为 285/300（95.00%），也通过
预注册 non-inferiority。因此当前不再把 legacy H3-Orig 差距正式归因为
synthetic 数据质量或多速度竞争。

因此本协议修复已经使两步 context 进入模型预测，并改善部分连续规划质量；
其收益尚未转化为冻结 CEM E4 的任务成功率。固定 speed=5 补充实验已经证明
E4 低分不是速度混合主导，而是 s0/s1/s2 全失败的 planner/template 地板效应。
双向 heldout Eval 得到
“慢速历史 50.67% < 同速历史 58.33% < 快速历史 64.67%”，正式确认当前
速度条件化规划 ICL 已生效，但快速历史的有限 CEM 成功率最高。三个单速对照在
相同 directional v2 中均未通过冻结门，只有 SpeedFull 通过快速−慢速门。该结果
支持当前固定训练 seed 下的多速度集成配方归因，但不把
速度 support 写成已完成的单变量因果证明。规划机制和后续训练安排不属于本
数据协议，完整结论与路线图见
[ContextWorld ICL Benchmark：速度](../ContextWorld_ICL_Benchmark.md#41-速度)。

## 7. 执行入口与产物

- 数据：`bash scripts/run_tworoom_speedfull_data.sh all`
- 训练 preflight：`bash scripts/run_h3_speedfull_train.sh preflight`
- 正式训练/自动续训：`CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_h3_speedfull_train.sh train`
- E1、heldout ID 与 E4：`bash scripts/run_h3_speedfull_eval.sh all`
- 合成配置：`configs/synthesis/tworoom_speed_full_v1.yaml`
- Benchmark 配置：`configs/benchmark/tworoom_speed_full_v1.yaml`
- 合成报告：`artifacts/synthesis/reports/tworoom_speed_full_v1.json`
- 训练报告：`artifacts/training/reports/h3_speedfull_s3072.json`
- 评测目录：`artifacts/evaluation/history3/h3_speedfull_s3072/`

默认 `artifacts/...` 映射到
`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/`；可通过
`CONTEXTWORLD_ARTIFACT_ROOT` 更换产物根目录。仓库只保留代码、配置、测试与
文档。
