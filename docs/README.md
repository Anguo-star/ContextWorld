# ContextWorld 文档导航

**先读 [ContextWorld 统一技术报告](ContextWorld_ICL_Benchmark.md)。** 任务定义、数据版本、完整模型与训练方案对比、主要发现均在这一页维护。其他文档只承担数据、训练、接入和复现操作说明；历史协议不是使用 benchmark 的前置知识。

## 公开使用

1. [项目首页](../README.md)：任务概览、安装和最短运行示例；
2. [统一技术报告](ContextWorld_ICL_Benchmark.md)：数据、指标、九任务最新总表、冻结参考和结果解释；
3. [数据生成方法](Data_Generation.md)：连续仿真、配对构造、拆分隔离和九项任务的生成来源；
4. [HF 数据集指南](HF_Dataset_Export.md)：v3 发布候选目录、下载与加载方式；
5. [外部模型 Adapter 规范](External_Model_Adapter_Contract.md)：接入新模型所需的统一接口；
6. [Stable-WorldModel 训练](StableWM_Training.md)：内置 LeWM、PLDM 和 PreJEPA 的训练入口。

九项任务在 Benchmark 规范中按隐藏动力学类型组织，而不是按环境数量组织。每项任务均
报告自己的 ICL 分数，不计算跨任务总分；原任务规划保持分析单列附录。

最新结果直接见[九任务训练方式比较](ContextWorld_ICL_Benchmark.md#training-comparison)。总览、完整指标和 Speed 分布结果由同一份 [JSON](research/data/icl_training_study_v2.json) 生成，也提供 [CSV](research/data/icl_training_study_v2.csv)。不再单独维护另一份当前研究报告。

## 任务导航

| 能力类型 | 任务 | 说明 | 机器配置 | 任务命令 |
|---|---|---|---|---|
| 即时连续响应 | 速度 | [速度](ContextWorld_ICL_Benchmark.md#611-速度) | `configs/benchmark/tworoom_speed_icl_release_v1.yaml` | `contextworld-speed` |
| 即时连续响应 | 推手移动幅度 | [推手移动幅度](ContextWorld_ICL_Benchmark.md#612-推手移动幅度) | `configs/benchmark/pusht_action_strength_icl_release_v1.yaml` | `contextworld-action-strength` |
| 即时连续响应 | 机械臂质量 | [机械臂质量](ContextWorld_ICL_Benchmark.md#613-机械臂质量) | `configs/benchmark/reacher_arm_mass_icl_release_v1.yaml` | `contextworld-reacher-arm-mass` |
| 时间延迟动力学 | 动作延迟 | [动作延迟](ContextWorld_ICL_Benchmark.md#621-动作延迟) | `configs/benchmark/tworoom_action_delay_icl_release_v1.yaml` | `contextworld-action-delay` |
| 接触或附着条件动力学 | 接触摩擦 | [接触摩擦](ContextWorld_ICL_Benchmark.md#631-接触摩擦) | `configs/benchmark/pusht_contact_friction_icl_release_v1.yaml` | `contextworld-contact-friction` |
| 接触或附着条件动力学 | 运动阻尼 | [运动阻尼](ContextWorld_ICL_Benchmark.md#632-运动阻尼) | `configs/benchmark/pusht_motion_damping_icl_release_v1.yaml` | `contextworld-motion-damping` |
| 接触或附着条件动力学 | Cube 夹爪携带规则 | [6.3.3 Cube 夹爪携带规则](ContextWorld_ICL_Benchmark.md#633-cube-夹爪携带规则) | `configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml` | `contextworld-cube-gripper-carry` |
| 隐藏结构转移 | 门通行规则 | [门通行规则](ContextWorld_ICL_Benchmark.md#641-门通行规则) | `configs/benchmark/tworoom_door_icl_release_v1.yaml` | `contextworld-door` |
| 隐藏结构转移 | 传送门出口位置 | [传送门出口位置](ContextWorld_ICL_Benchmark.md#642-传送门出口位置) | `configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml` | `contextworld-portal-exit` |

## 结果复现

- [固定基准验收](reference/Baseline_Final_Acceptance_2026-09-10.md)：v3 的覆盖范围、验证证据与后续比较规则；
- [参考结果复现附录](reference/Benchmark_Result_Provenance.md)：检查点来源、训练种子、评测预算和机器可读结果；
- `protocols/`：执行前确定的任务协议；
- `archive/`：已经结束的实验阶段材料；
- `reference/`：第三方工程、运行环境和结果来源说明。

这些材料用于复核已报告结果，不是新的排行榜。Training/Development 使用 `ContextWorld-v1`，
Test 使用 `ContextWorld-v1-full`；发布候选 `ContextWorld-v3-hf` 汇集这两个冻结来源，
保持所选数据字节与评分语义不变。历史结果目录不承担当前数据分发职责。

## 仓库维护与发布

[v3 发布需求与验收](ContextWorld_Public_v1_Release_Readiness.md) 记录候选包、数据卡、
文件校验和上传后固定 revision 的下载验证。独立外部复现与新增模型覆盖单独维护，
不重开已封板的 baseline。

目前 Training、Development 和 Test 数据包已在本地完成发布候选组装，但稳定的公开数据集
修订尚未公布；Test 通过离线评分入口用于最终报告。

扩量训练数据的打包与发布准备见[发布说明](Expanded_Training_Release.md)。研究结果仍以统一技术报告为准。
